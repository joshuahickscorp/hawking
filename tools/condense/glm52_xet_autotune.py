#!/usr/bin/env python3.12
"""Offline planner and fail-closed authority gate for GLM-5.2 Xet autotuning.

This module deliberately contains no model download implementation.  The
``preflight``, ``plan``, and ``verify`` commands are control-plane-only: they
verify sealed campaign evidence, inspect local package configuration in fresh
Python processes, and create a deterministic bounded plan.  A separate live
executor exists in ``glm52_xet_live.py``.  This planner's ``run`` command never
executes body trials and the sealed offline plan alone grants no authority.

Keeping planning separate from execution makes three properties testable:

* no planning command can accidentally read a model body;
* the network and disk envelope is fixed before the first body byte; and
* the separate live executor must enter through authenticated controller authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    atomic_json,
    atomic_text,
    canonical,
    read_sealed_json,
    seal,
    sha256_file,
    verify_sealed,
)
from glm52_state import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    EXPECTED_CONTRACT_SCHEMA,
    GENESIS_HASH,
    TELEGRAM_RECEIPT_SCHEMA,
    TERMINAL_EVIDENCE_SCHEMA,
    TRANSITION_EVENT_KINDS,
    TRANSITION_INTENT_SCHEMA,
    StateError,
    validate_transition_intent,
)


REPO_ID = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
EXPECTED_SOURCE_BYTES = 1_506_693_036_946
EXPECTED_WEIGHT_SHARDS = 282
EXPECTED_TENSORS = 59_585
EXPECTED_LARGEST_SHARD = "model-00200-of-00282.safetensors"
EXPECTED_LARGEST_SHARD_BYTES = 5_368_361_544
EXPECTED_LARGEST_SHARD_LFS_SHA256 = (
    "8d058b156900395078dd8350f24c16d492d9854d54d019d357bb583bd9a23d3a"
)
EXPECTED_LARGEST_SHARD_XET_HASH = (
    "e427925541b9f7cac39cd11b1f3682b69710caf31550c755c58662f559773f75"
)

RANGE_BYTES = 64 * 1024**2
RANGE_ALIGNMENT = 64 * 1024
RANGE_SHARDS = 48
NETWORK_CAP_NUMERATOR = 2
NETWORK_CAP_DENOMINATOR = 100
NETWORK_CAP_BYTES = EXPECTED_SOURCE_BYTES * NETWORK_CAP_NUMERATOR // NETWORK_CAP_DENOMINATOR
SMALL_CACHE_BYTES = 1 * 1024**3
CONDITIONAL_CACHE_BYTES = 10 * 1024**3
MAX_HEAVY_REGRESSION = 0.05
SUSTAINED_BINS = 3
REQUIRED_FILE_SETTINGS = (8, 16, 24, 32, 48)
MAX_TESTED_FILE_SETTING = max(REQUIRED_FILE_SETTINGS)
PRELIMINARY_SCHEDULE_MAX_DISPATCH = 23

PLAN_SCHEMA = "hawking.glm52.xet_autotune_plan.v2"
PREFLIGHT_SCHEMA = "hawking.glm52.xet_autotune_preflight.v1"
EXECUTION_AUTHORITY_SCHEMA = "hawking.glm52.xet_execution_authority.v3"
COMMITTED_CHECKPOINT_REF_SCHEMA = (
    "hawking.glm52.xet_committed_controller_checkpoint_ref.v1"
)
CONTROLLER_EPOCH = "glm52-controller-v2"
TOOLCHAIN_BINDING_SCHEMA = "hawking.glm52.xet_autotune_toolchain_binding.v1"
PINNED_VERSIONS = {
    "huggingface_hub": "1.24.0",
    "hf_xet": "1.5.2",
}

INPUT_CONTRACTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "GLM52_OFFICIAL_MANIFEST.json",
        "hawking.glm52.official_manifest.v1",
        ("PASS_",),
    ),
    (
        "GLM52_SOURCE_FORMAT_LEDGER.json",
        "hawking.glm52.source_format_ledger.v1",
        ("PASS_",),
    ),
    (
        "GLM52_SHARD_DEPENDENCY_GRAPH.json",
        "hawking.glm52.shard_dependency_graph.v1",
        ("PASS_",),
    ),
    (
        "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
        "hawking.glm52.streaming_schedule.v1",
        ("PRELIMINARY_", "PASS_"),
    ),
    (
        "GLM52_SOURCE_ADMISSION.json",
        "hawking.glm52.source_admission.v1",
        ("ADMITTED_", "PASS_"),
    ),
    (
        "GLM52_ADAPTER_TWIN.json",
        "hawking.glm52.adapter_twin.v1",
        ("PASS_", "PASS"),
    ),
    (
        "GLM52_REFERENCE_PARITY.json",
        "hawking.glm52.reference_parity.v1",
        ("PASS_", "PASS"),
    ),
    (
        "GLM52_CORPUS_INTEGRITY.json",
        "hawking.glm52.corpus_integrity.v2",
        ("PASS_", "PASS"),
    ),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_GLOB_CHARS = frozenset("*?[")
_RUNTIME_FIELDS = (
    "data.max_concurrent_file_downloads",
    "chunk_cache.size_bytes",
    "client.enable_adaptive_concurrency",
    "client.ac_min_download_concurrency",
    "client.ac_initial_download_concurrency",
    "client.ac_max_download_concurrency",
    "reconstruction.min_reconstruction_fetch_size",
    "reconstruction.max_reconstruction_fetch_size",
    "reconstruction.download_buffer_size",
    "reconstruction.download_buffer_perfile_size",
    "reconstruction.download_buffer_limit",
)
LIVE_OBSERVATION_FIELDS = frozenset({
    "swap_used_bytes",
    "swapouts",
    "thermal_warning",
    "free_disk_bytes",
    "available_ram_bytes",
    "cpu_percent",
    "disk_write_bytes_per_second",
    "reconstruction_latency_seconds",
    "retry_rate",
    "temporary_amplification_ratio",
    "actual_network_bytes",
})
SELECTABLE_TRIAL_MEASUREMENTS = frozenset({
    "peak_cpu_percent",
    "peak_disk_write_bytes_per_second",
    "maximum_reconstruction_latency_seconds",
    "maximum_retry_rate",
    "maximum_temporary_amplification_ratio",
    "actual_network_bytes",
    "trial_network_cap_bytes",
})


class ExecutionAuthorityVerifier(Protocol):
    """Trusted, executor-owned checks for untrusted serialized authority."""

    def verify_prepared_transition_intent_hmac(
        self,
        transition_intent: Mapping[str, Any],
        *,
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        """Authenticate the prepared intent with live, non-serialized key material."""

    def verify_telegram_delivery_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        transition_intent: Mapping[str, Any],
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        """Authenticate the exact Bot API v3 receipt and prepared-intent binding."""

    def verify_committed_controller_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        transition_intent: Mapping[str, Any],
        telegram_receipt: Mapping[str, Any],
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        """Read/replay the live checkpoint and match this committed transition."""

    def verify_live_singleton_lease(
        self,
        checkpoint: Mapping[str, Any],
        *,
        transition_intent: Mapping[str, Any],
        plan_seal: str,
        expected_contract_sha256: str,
    ) -> bool:
        """Inspect the current lease owner/epoch; never trust a serialized flag."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _status_has_prefix(value: Any, prefixes: Sequence[str]) -> bool:
    return isinstance(value, str) and any(value == prefix or value.startswith(prefix) for prefix in prefixes)


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Glm52Error(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) < minimum \
            or (maximum is not None and float(value) > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise Glm52Error(f"{label} must be finite and >= {minimum}{suffix}")
    return float(value)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _runtime_probe_program() -> str:
    return (
        "import json\n"
        "from hf_xet import XetConfig\n"
        f"wanted={list(_RUNTIME_FIELDS)!r}\n"
        "items=dict(XetConfig().items())\n"
        "print(json.dumps({k: items.get(k) for k in wanted}, default=str, sort_keys=True))\n"
    )


def toolchain_binding() -> dict[str, Any]:
    """Bind plans to the exact planner, shared helpers, lock, and probe instrument."""
    files = (
        ("planner_generator", HERE / "glm52_xet_autotune.py", "tools/condense/glm52_xet_autotune.py"),
        ("shared_common", HERE / "glm52_common.py", "tools/condense/glm52_common.py"),
        ("controller_state", HERE / "glm52_state.py", "tools/condense/glm52_state.py"),
        ("live_executor", HERE / "glm52_xet_live.py", "tools/condense/glm52_xet_live.py"),
        ("requirements_lock", HERE / "requirements-glm52.txt", "tools/condense/requirements-glm52.txt"),
    )
    rows: list[dict[str, str]] = []
    for role, path, relative in files:
        if not path.is_file():
            raise Glm52Error(f"Xet autotune toolchain file is missing: {relative}")
        rows.append({"role": role, "path": relative, "sha256": sha256_file(path)})
    program = _runtime_probe_program()
    return {
        "schema": TOOLCHAIN_BINDING_SCHEMA,
        "files": rows,
        "runtime_probe_program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "runtime_probe_fields_sha256": _canonical_sha256(list(_RUNTIME_FIELDS)),
    }


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _input_refs(inputs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "path": name,
            "schema": str(inputs[name]["schema"]),
            "status": str(inputs[name]["status"]),
            "seal_sha256": str(inputs[name]["seal_sha256"]),
        }
        for name, _schema, _prefixes in INPUT_CONTRACTS
    ]


def load_and_validate_inputs(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Load the eight sealed prerequisites and reject any identity drift."""
    result: dict[str, dict[str, Any]] = {}
    for name, expected_schema, status_prefixes in INPUT_CONTRACTS:
        value = read_sealed_json(root / name)
        if value.get("schema") != expected_schema:
            raise Glm52Error(f"{name} schema mismatch")
        if not _status_has_prefix(value.get("status"), status_prefixes):
            raise Glm52Error(f"{name} is not green: {value.get('status')!r}")
        result[name] = value

    official_names = (
        "GLM52_OFFICIAL_MANIFEST.json",
        "GLM52_SOURCE_FORMAT_LEDGER.json",
        "GLM52_SHARD_DEPENDENCY_GRAPH.json",
        "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json",
        "GLM52_SOURCE_ADMISSION.json",
    )
    for name in official_names:
        value = result[name]
        if value.get("repo") != REPO_ID or value.get("revision") != REVISION:
            raise Glm52Error(f"{name} immutable source identity mismatch")

    manifest = result["GLM52_OFFICIAL_MANIFEST.json"]
    source_format = result["GLM52_SOURCE_FORMAT_LEDGER.json"]
    graph = result["GLM52_SHARD_DEPENDENCY_GRAPH.json"]
    schedule = result["GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json"]
    admission = result["GLM52_SOURCE_ADMISSION.json"]
    adapter = result["GLM52_ADAPTER_TWIN.json"]
    reference = result["GLM52_REFERENCE_PARITY.json"]
    corpus = result["GLM52_CORPUS_INTEGRITY.json"]

    if manifest.get("source_logical_bytes") != EXPECTED_SOURCE_BYTES \
            or manifest.get("weight_shards") != EXPECTED_WEIGHT_SHARDS:
        raise Glm52Error("official source totals changed")
    if source_format.get("weight_shards") != EXPECTED_WEIGHT_SHARDS \
            or len(source_format.get("per_shard", [])) != EXPECTED_WEIGHT_SHARDS:
        raise Glm52Error("source-format shard coverage changed")
    if graph.get("shard_count") != EXPECTED_WEIGHT_SHARDS \
            or graph.get("tensor_count") != EXPECTED_TENSORS:
        raise Glm52Error("dependency graph totals changed")
    if schedule.get("source_shards_scheduled") != EXPECTED_WEIGHT_SHARDS:
        raise Glm52Error("streaming schedule does not cover 282 shards")
    if admission.get("xet", {}).get("body_bytes_read") != 0 \
            or admission.get("source", {}).get("complete_source_resident") is not False:
        raise Glm52Error("source admission is not the body-pending baseline")
    if manifest.get("one_copy", {}).get("weight_body_copies") != 0:
        raise Glm52Error("offline autotune plan requires zero admitted weight-body copies")

    binding = adapter.get("binding")
    if not isinstance(binding, dict) or binding.get("official_repo") != REPO_ID \
            or binding.get("official_revision") != REVISION:
        raise Glm52Error("adapter twin is not bound to the immutable source")
    sweep = adapter.get("checks", {}).get("official_schema_sweep", {})
    if sweep.get("status") != "PASS" or sweep.get("graph_seal_sha256") != graph["seal_sha256"]:
        raise Glm52Error("adapter schema sweep is not bound to the current graph")
    claim = reference.get("claim_boundary")
    if not isinstance(claim, dict) or claim.get("capability") != "NOT_CLAIMED" \
            or not str(claim.get("official_bf16_parent_forward", "")).startswith("PENDING"):
        raise Glm52Error("reference parity overclaims the unread BF16 parent")
    tokenizer = corpus.get("official_tokenizer")
    scope = corpus.get("scope")
    if not isinstance(tokenizer, dict) or tokenizer.get("repository") != REPO_ID \
            or tokenizer.get("revision") != REVISION:
        raise Glm52Error("corpus is not bound to the official tokenizer revision")
    if not isinstance(scope, dict) or scope.get("model_payload_downloaded") is not False \
            or scope.get("network_access_used") is not False:
        raise Glm52Error("corpus integrity artifact is not offline/body-free")

    reconcile_schedule(schedule)
    return result


def _clean_probe_environment(updates: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HF_XET_")
        and key not in {"HF_HUB_ENABLE_HF_TRANSFER", "HF_HUB_DISABLE_XET"}
    }
    environment.update(dict(updates or {}))
    return environment


def probe_xet_config(updates: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read effective Xet config in a fresh process; this performs no I/O."""
    program = _runtime_probe_program()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            env=_clean_probe_environment(updates),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Glm52Error(f"cannot inspect hf_xet configuration: {exc}") from exc
    if completed.returncode != 0:
        raise Glm52Error(f"hf_xet configuration probe failed: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Glm52Error("hf_xet configuration probe returned invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != set(_RUNTIME_FIELDS):
        raise Glm52Error("hf_xet configuration probe returned an incomplete field set")
    return value


def classify_runtime_snapshots(
    versions: Mapping[str, str], snapshots: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Classify pinned Xet semantics from no-body configuration snapshots."""
    missing = {key for key in ("default", "legacy64", "fixed16", "hp", "cache1g") if key not in snapshots}
    if missing:
        raise Glm52Error(f"runtime snapshots missing: {sorted(missing)}")
    exact_versions = dict(versions) == PINNED_VERSIONS
    default = dict(snapshots["default"])
    legacy = dict(snapshots["legacy64"])
    fixed = dict(snapshots["fixed16"])
    hp = dict(snapshots["hp"])
    cache = dict(snapshots["cache1g"])
    legacy_ignored = legacy == default
    fixed_effective = [
        fixed.get("client.ac_min_download_concurrency"),
        fixed.get("client.ac_initial_download_concurrency"),
        fixed.get("client.ac_max_download_concurrency"),
    ] == [16, 16, 16]
    hp_effective = [
        hp.get("client.ac_min_download_concurrency"),
        hp.get("client.ac_initial_download_concurrency"),
        hp.get("client.ac_max_download_concurrency"),
        hp.get("reconstruction.download_buffer_limit"),
    ] == [4, 16, 124, 64_000_000_000]
    cache_parsed = cache.get("chunk_cache.size_bytes") == SMALL_CACHE_BYTES
    default_bounds = [
        default.get("client.ac_min_download_concurrency"),
        default.get("client.ac_initial_download_concurrency"),
        default.get("client.ac_max_download_concurrency"),
    ]
    gates = {
        "pinned_versions_exact": exact_versions,
        "default_bounds_1_4_64": default_bounds == [1, 4, 64],
        "legacy_range_get_variable_ignored": legacy_ignored,
        "fixed_download_alias_effective": fixed_effective,
        "high_performance_profile_effective": hp_effective,
        "chunk_cache_size_configuration_parsed": cache_parsed,
    }
    if not all(gates.values()):
        raise Glm52Error(f"pinned Xet runtime compatibility failed: {gates}")
    return {
        "status": "PASS_WITH_PINNED_RUNTIME_LIMITATIONS",
        "versions": dict(versions),
        "gates": gates,
        "effective_default": default,
        "effective_fixed_16": fixed,
        "effective_high_performance": hp,
        "effective_cache_1g": cache,
        "semantics": {
            "adaptive_default": (
                "configured by default; enable/disable flag is parsed but not consulted by "
                "the pinned 1.5.2 production controller"
            ),
            "fixed_download_concurrency": (
                "HF_XET_FIXED_DOWNLOAD_CONCURRENCY pins min/initial/max; use for 16/32/64 trials"
            ),
            "legacy_range_get_variable": "HF_XET_NUM_CONCURRENT_RANGE_GETS is ignored",
            "high_performance": (
                "applied after fixed/custom settings and therefore tested as a separate 4/16/124 profile"
            ),
            "chunk_cache": (
                "configured but inert: pinned 1.5.2 production download constructors do not attach the cache"
            ),
            "direct_stream_file_limit": (
                "direct range streams bypass HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS; "
                "the campaign caller must enforce the declared stream cap"
            ),
        },
        "primary_sources": [
            "https://github.com/huggingface/xet-core/blob/v1.5.2/xet_runtime/src/config/groups/client.rs",
            "https://github.com/huggingface/xet-core/blob/v1.5.2/xet_runtime/src/config/aliases.rs",
            "https://github.com/huggingface/xet-core/blob/v1.5.2/xet_runtime/src/config/xet_config.rs",
            "https://huggingface.co/docs/hub/xet/using-xet-storage",
        ],
    }


def runtime_compatibility() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in PINNED_VERSIONS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Glm52Error(f"required package is absent: {package}") from exc
    snapshots = {
        "default": probe_xet_config(),
        "legacy64": probe_xet_config({"HF_XET_NUM_CONCURRENT_RANGE_GETS": "64"}),
        "fixed16": probe_xet_config({"HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "16"}),
        "hp": probe_xet_config({"HF_XET_HIGH_PERFORMANCE": "1"}),
        "cache1g": probe_xet_config({"HF_XET_CHUNK_CACHE_SIZE_BYTES": str(SMALL_CACHE_BYTES)}),
    }
    return classify_runtime_snapshots(versions, snapshots)


def deterministic_shard_indices() -> list[int]:
    indices = [1 + (index * 280) // 47 for index in range(RANGE_SHARDS)]
    if len(indices) != RANGE_SHARDS or len(set(indices)) != RANGE_SHARDS \
            or indices[0] != 1 or indices[-1] != 281:
        raise AssertionError("deterministic range-shard selection is invalid")
    return indices


def build_ranges(
    manifest: Mapping[str, Any],
    source_format: Mapping[str, Any],
    *,
    range_bytes: int = RANGE_BYTES,
) -> list[dict[str, Any]]:
    _require_int(range_bytes, "range_bytes", minimum=RANGE_ALIGNMENT)
    if range_bytes % RANGE_ALIGNMENT:
        raise Glm52Error("range_bytes must be aligned to 64 KiB")
    manifest_rows = {
        row["path"]: row
        for row in manifest.get("files", [])
        if isinstance(row, dict) and row.get("is_weight") is True
    }
    format_rows = {
        row["path"]: row
        for row in source_format.get("per_shard", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    if set(manifest_rows) != set(format_rows) or len(format_rows) != EXPECTED_WEIGHT_SHARDS:
        raise Glm52Error("manifest/source-format shard sets differ")
    result: list[dict[str, Any]] = []
    for shard_index in deterministic_shard_indices():
        path = f"model-{shard_index:05d}-of-00282.safetensors"
        metadata = format_rows[path]
        manifest_row = manifest_rows[path]
        file_bytes = _require_int(metadata.get("file_bytes"), f"{path}.file_bytes", minimum=1)
        data_start = _require_int(metadata.get("data_start"), f"{path}.data_start", minimum=8)
        available = file_bytes - data_start
        if available < range_bytes:
            raise Glm52Error(f"selected shard cannot hold bounded body range: {path}")
        slots = (available - range_bytes) // RANGE_ALIGNMENT + 1
        digest = hashlib.sha256(
            f"{manifest['seal_sha256']}:{path}".encode("utf-8")
        ).digest()
        relative = (int.from_bytes(digest[:8], "big") % slots) * RANGE_ALIGNMENT
        start = data_start + relative
        end = start + range_bytes
        if start < data_start or end > file_bytes:
            raise AssertionError("bounded range escaped safetensors payload")
        range_identity = {
            "schema": "hawking.glm52.xet_body_range_identity.v1",
            "path": path,
            "xet_hash": metadata["xet_hash"],
            "lfs_sha256": metadata["lfs_sha256"],
            "start": start,
            "end": end,
            "length": range_bytes,
        }
        result.append({
            "range_id_sha256": _canonical_sha256(range_identity),
            "path": path,
            "shard_index": shard_index,
            "xet_hash": metadata["xet_hash"],
            "lfs_sha256": metadata["lfs_sha256"],
            "file_bytes": file_bytes,
            "data_start": data_start,
            "start": start,
            "end": end,
            "length": range_bytes,
            "offset_alignment_relative_to_data_start": RANGE_ALIGNMENT,
            "selection_digest_sha256": hashlib.sha256(digest).hexdigest(),
            "manifest_identity_matches": (
                manifest_row.get("xet_hash") == metadata.get("xet_hash")
                and manifest_row.get("lfs_sha256") == metadata.get("lfs_sha256")
                and manifest_row.get("logical_bytes") == file_bytes
            ),
        })
    if not all(row["manifest_identity_matches"] for row in result):
        raise Glm52Error("selected range identity differs between sealed ledgers")
    return result


def build_trial_matrix(
    body_ranges: Sequence[Mapping[str, Any]],
    *,
    range_bytes: int = RANGE_BYTES,
) -> list[dict[str, Any]]:
    if any(not isinstance(row, Mapping) for row in body_ranges):
        raise Glm52Error("trial matrix body-range identities must be objects")
    range_ids = [row.get("range_id_sha256") for row in body_ranges]
    if len(range_ids) != RANGE_SHARDS \
            or any(not _is_sha256(value) for value in range_ids) \
            or any(row.get("length") != range_bytes for row in body_ranges):
        raise Glm52Error("trial matrix requires the exact ordered sealed body-range identities")
    if len(set(range_ids)) != RANGE_SHARDS:
        raise Glm52Error("trial matrix body-range identities must be unique")
    templates = [
        ("DEFAULT_UNSET", 8, {}, "DEFAULT_UNPINNED", "ZERO"),
        ("FILES_08", 8, {"HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "8"}, "DEFAULT_UNPINNED", "ZERO"),
        ("FILES_16", 16, {"HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "16"}, "DEFAULT_UNPINNED", "ZERO"),
        ("FILES_24", 24, {"HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "24"}, "DEFAULT_UNPINNED", "ZERO"),
        ("FILES_32", 32, {"HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "32"}, "DEFAULT_UNPINNED", "ZERO"),
        ("FILES_48", 48, {"HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "48"}, "DEFAULT_UNPINNED", "ZERO"),
        ("FIXED_16", 8, {"HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "16"}, "FIXED_16", "ZERO"),
        ("FIXED_32", 8, {"HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "32"}, "FIXED_32", "ZERO"),
        ("FIXED_64", 8, {"HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "64"}, "FIXED_64", "ZERO"),
        ("HIGH_PERFORMANCE", 8, {"HF_XET_HIGH_PERFORMANCE": "1"}, "HP_4_16_124", "ZERO"),
        (
            "CACHE_1G_COLD",
            8,
            {"HF_XET_CHUNK_CACHE_SIZE_BYTES": str(SMALL_CACHE_BYTES)},
            "DEFAULT_UNPINNED",
            "ONE_GIB_CONFIGURED_INERT",
        ),
        (
            "CACHE_1G_REPLAY",
            8,
            {"HF_XET_CHUNK_CACHE_SIZE_BYTES": str(SMALL_CACHE_BYTES)},
            "DEFAULT_UNPINNED",
            "ONE_GIB_CONFIGURED_INERT_REPLAY",
        ),
    ]
    result: list[dict[str, Any]] = []
    for ordinal, (trial_id, streams, environment, transfer, cache_policy) in enumerate(templates):
        ordered_range_ids = list(range_ids[:streams])
        result.append({
            "ordinal": ordinal,
            "trial_id": trial_id,
            "kind": "BOUNDED_XET_BODY_RANGE",
            "caller_concurrent_shard_streams": streams,
            "configured_file_download_limit": (
                int(environment["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
                if "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS" in environment
                else "UNSET_DEFAULT_8"
            ),
            "transfer_profile": transfer,
            "high_performance": environment.get("HF_XET_HIGH_PERFORMANCE") == "1",
            "chunk_cache_policy": cache_policy,
            "environment": environment,
            "range_count": streams,
            "ordered_range_ids": ordered_range_ids,
            "ordered_range_ids_sha256": _canonical_sha256(ordered_range_ids),
            "planned_payload_bytes": streams * range_bytes,
            # The schedule is preliminary until this measurement runs. Every
            # requested setting must be eligible to win; pre-excluding 32/48
            # from preliminary geometry would make the result circular.
            "selectable_after_schedule_refreeze": True,
            "diagnostic_only_reason": None,
        })
    if sum(row["range_count"] for row in result) != 184:
        raise AssertionError("autotune matrix must contain exactly 184 bounded ranges")
    return result


def _largest_shard(manifest: Mapping[str, Any]) -> dict[str, Any]:
    largest = manifest.get("largest_weight_shard")
    if largest != {"path": EXPECTED_LARGEST_SHARD, "bytes": EXPECTED_LARGEST_SHARD_BYTES}:
        raise Glm52Error("official largest shard identity changed")
    row = next(
        (
            item for item in manifest.get("files", [])
            if isinstance(item, dict) and item.get("path") == EXPECTED_LARGEST_SHARD
        ),
        None,
    )
    if not isinstance(row, dict) or row.get("logical_bytes") != EXPECTED_LARGEST_SHARD_BYTES \
            or row.get("lfs_sha256") != EXPECTED_LARGEST_SHARD_LFS_SHA256 \
            or row.get("xet_hash") != EXPECTED_LARGEST_SHARD_XET_HASH:
        raise Glm52Error("largest shard hashes differ from the immutable manifest")
    return {
        "path": EXPECTED_LARGEST_SHARD,
        "bytes": EXPECTED_LARGEST_SHARD_BYTES,
        "lfs_sha256": EXPECTED_LARGEST_SHARD_LFS_SHA256,
        "xet_hash": EXPECTED_LARGEST_SHARD_XET_HASH,
    }


def reconcile_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    windows = schedule.get("windows")
    if not isinstance(windows, list) or not windows:
        raise Glm52Error("streaming schedule has no windows")
    fetched: Counter[str] = Counter()
    previous_carry: list[str] = []
    max_new = 0
    max_resident = 0
    adjacent_union = 0
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or window.get("window_id") != f"W{index:03d}":
            raise Glm52Error("streaming window order/identity mismatch")
        fields: dict[str, list[str]] = {}
        for key in (
            "source_shards", "carry_in_shards", "new_fetch_shards",
            "carry_out_shards", "evict_after_seal_shards", "refetch_shards",
        ):
            value = window.get(key)
            if not isinstance(value, list) or len(value) != len(set(value)) \
                    or any(not isinstance(item, str) or not item for item in value):
                raise Glm52Error(f"{window.get('window_id')}.{key} is not an exact unique path list")
            fields[key] = value
        if fields["carry_in_shards"] != previous_carry:
            raise Glm52Error("schedule carry chain is broken")
        if fields["refetch_shards"]:
            raise Glm52Error("preliminary schedule unexpectedly contains a refetch")
        if set(fields["source_shards"]) != set(fields["carry_in_shards"]) | set(fields["new_fetch_shards"]):
            raise Glm52Error("window source partition is invalid")
        if set(fields["source_shards"]) != set(fields["carry_out_shards"]) | set(fields["evict_after_seal_shards"]):
            raise Glm52Error("window carry/eviction partition is invalid")
        if set(fields["carry_out_shards"]) & set(fields["evict_after_seal_shards"]):
            raise Glm52Error("window carry and eviction sets overlap")
        fetched.update(fields["new_fetch_shards"])
        previous_carry = fields["carry_out_shards"]
        max_new = max(max_new, len(fields["new_fetch_shards"]))
        max_resident = max(max_resident, len(fields["source_shards"]))
    if previous_carry:
        raise Glm52Error("final schedule window retains source shards")
    if len(fetched) != EXPECTED_WEIGHT_SHARDS or set(fetched.values()) != {1}:
        raise Glm52Error("schedule violates exact one-fetch coverage")
    for left, right in zip(windows, windows[1:]):
        adjacent_union = max(
            adjacent_union,
            len(set(left["source_shards"]) | set(right["source_shards"])),
        )
    if max_new != PRELIMINARY_SCHEDULE_MAX_DISPATCH or max_resident != 26 or adjacent_union != 45:
        raise Glm52Error(
            f"sealed schedule geometry changed: new={max_new} resident={max_resident} adjacent={adjacent_union}"
        )
    return {
        "status": "PASS_PRELIMINARY_SCHEDULE_ONLY",
        "schedule_seal_sha256": schedule.get("seal_sha256"),
        "window_count": len(windows),
        "source_shards_exactly_once": len(fetched),
        "planned_refetches": 0,
        "maximum_new_fetch_shards": max_new,
        "maximum_resident_shards_one_window": max_resident,
        "maximum_actual_adjacent_active_prefetch_union": adjacent_union,
        "conservative_active_prefetch_upper_bound": schedule.get(
            "maximum_simultaneous_shards_active_plus_prefetch_upper_bound"
        ),
        "required_file_settings_tested": list(REQUIRED_FILE_SETTINGS),
        "largest_useful_tested_file_setting": MAX_TESTED_FILE_SETTING,
        "preliminary_schedule_maximum_caller_dispatch": PRELIMINARY_SCHEDULE_MAX_DISPATCH,
        "diagnostic_file_settings_not_selectable": [],
        "freeze_boundary": "requires measured GLM52_XET_AUTOTUNE; no freeze is claimed by this plan",
    }


def build_plan(
    root: Path = REPO_ROOT,
    *,
    range_bytes: int = RANGE_BYTES,
    network_cap_bytes: int = NETWORK_CAP_BYTES,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = load_and_validate_inputs(root)
    _require_int(network_cap_bytes, "network_cap_bytes", minimum=1)
    if network_cap_bytes > NETWORK_CAP_BYTES:
        raise Glm52Error("offline planner refuses a cap above 2% of official source bytes")
    compatibility = dict(runtime) if runtime is not None else runtime_compatibility()
    if compatibility.get("status") != "PASS_WITH_PINNED_RUNTIME_LIMITATIONS":
        raise Glm52Error("runtime compatibility is not green")
    manifest = inputs["GLM52_OFFICIAL_MANIFEST.json"]
    source_format = inputs["GLM52_SOURCE_FORMAT_LEDGER.json"]
    schedule = inputs["GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json"]
    ranges = build_ranges(manifest, source_format, range_bytes=range_bytes)
    matrix = build_trial_matrix(ranges, range_bytes=range_bytes)
    range_payload = sum(row["planned_payload_bytes"] for row in matrix)
    largest = _largest_shard(manifest)
    full_validation = 2 * largest["bytes"]
    planned = range_payload + full_validation
    if planned > network_cap_bytes:
        raise Glm52Error(f"planned Xet evidence exceeds network cap: {planned} > {network_cap_bytes}")
    if len(ranges) < max(row["range_count"] for row in matrix):
        raise Glm52Error("trial matrix requests more distinct ranges than the sealed range set")
    reconciliation = reconcile_schedule(schedule)
    if reconciliation["planned_refetches"] != 0:
        raise Glm52Error("10 GiB cache condition unexpectedly became eligible")
    plan = seal({
        "schema": PLAN_SCHEMA,
        "status": "PASS_OFFLINE_PLAN_BODY_NOT_READ",
        "repo": REPO_ID,
        "revision": REVISION,
        "source_mode": "VULTURE_XET_STREAMING",
        "inputs": _input_refs(inputs),
        "toolchain_binding": toolchain_binding(),
        "body_read_boundary": {
            "planner_network_access": False,
            "planner_model_body_bytes_read": 0,
            "planner_model_body_files_created": 0,
            "planner_cli_live_execution_implemented": False,
            "separate_live_executor_implemented": True,
            "separate_live_executor_path": "tools/condense/glm52_xet_live.py",
            "offline_plan_alone_authorizes_execution": False,
            "planner_live_run_default": "REFUSE",
        },
        "runtime_compatibility": compatibility,
        "live_observation_schema": {
            "required_sample_fields": {
                "swap_used_bytes": "nonnegative integer bytes",
                "swapouts": "nonnegative cumulative integer count",
                "thermal_warning": "boolean",
                "free_disk_bytes": "nonnegative integer bytes",
                "available_ram_bytes": "nonnegative integer bytes",
                "cpu_percent": "finite nonnegative percent; aggregate cores may exceed 100",
                "disk_write_bytes_per_second": "finite nonnegative bytes/second",
                "reconstruction_latency_seconds": "finite nonnegative seconds",
                "retry_rate": "finite ratio in [0,1]",
                "temporary_amplification_ratio": "finite nonnegative ratio",
                "actual_network_bytes": "nonnegative cumulative trial byte counter",
            },
            "actual_network_counter_must_be_monotonic": True,
            "actual_network_delta_must_be_positive": True,
            "trial_network_cap_enforced_before_selection": True,
            "selection_requires_aggregates": sorted(SELECTABLE_TRIAL_MEASUREMENTS),
        },
        "range_strategy": {
            "selection": "indices = [1 + floor(i*280/47) for i in range(48)]",
            "selected_shards": RANGE_SHARDS,
            "excluded_anomalous_small_final_shard": "model-00282-of-00282.safetensors",
            "range_bytes": range_bytes,
            "alignment_bytes": RANGE_ALIGNMENT,
            "offset_seed": "sha256(manifest_seal_sha256 + ':' + shard_path)",
            "trial_assignment_policy": "CANONICAL_ORDERED_PREFIX_BY_CALLER_STREAM_COUNT",
            "same_stream_count_has_identical_range_ids_and_order": True,
            "different_stream_counts_share_the_exact_canonical_prefix": True,
            "body_ranges": ranges,
        },
        "trial_matrix": matrix,
        "network_budget": {
            "official_source_bytes": EXPECTED_SOURCE_BYTES,
            "hard_cap_fraction": "2/100",
            "hard_cap_bytes": network_cap_bytes,
            "bounded_range_count": sum(row["range_count"] for row in matrix),
            "bounded_range_payload_bytes": range_payload,
            "largest_shard_full_validations": 2,
            "largest_shard_validation_bytes": full_validation,
            "planned_maximum_bytes": planned,
            "remaining_guard_bytes": network_cap_bytes - planned,
            "planned_amplification_contribution": {
                "numerator": planned,
                "denominator": EXPECTED_SOURCE_BYTES,
            },
            "next_trial_must_fit_before_start": True,
            "retries_and_protocol_overhead_count_against_cap": True,
        },
        "largest_shard_validation": {
            **largest,
            "passes": [
                "selected acquisition-burst profile without Metal-heavy work",
                "selected steady-pipeline profile with the frozen Metal probe",
            ],
            "hash_before_promotion": True,
            "seal_evidence_before_exact_eviction": True,
            "repeat_attribution": "AUTOTUNE_LANE_VALIDATION",
        },
        "cache_policy": {
            "selected_for_one_pass": 0,
            "small_cache_trial_bytes": SMALL_CACHE_BYTES,
            "small_cache_expected_runtime_semantics": "CONFIGURED_BUT_INERT_IN_PINNED_1_5_2",
            "conditional_10_gib_bytes": CONDITIONAL_CACHE_BYTES,
            "conditional_10_gib_status": "SKIPPED_CONDITION_FALSE",
            "reason": "sealed schedule has zero planned refetches and pinned cache path is inert",
        },
        "selection_policy": {
            "acquisition_burst": "fastest safe profile; high-performance may compete",
            "steady_pipeline": "fastest safe profile with no sustained heavy-lane regression above 5%",
            "confidence_tie_break": [
                "lower retry rate",
                "lower reconstruction latency",
                "lower Xet temporary amplification",
                "lower actual network bytes",
                "lower peak CPU use",
                "lower peak disk write throughput",
                "lower peak RSS",
                "lower effective transfer concurrency",
                "fewer caller shard streams",
                "lexicographically smaller trial id",
            ],
            "required_selectable_file_settings": list(REQUIRED_FILE_SETTINGS),
            "maximum_selectable_file_setting": MAX_TESTED_FILE_SETTING,
            "preliminary_schedule_maximum_dispatch": PRELIMINARY_SCHEDULE_MAX_DISPATCH,
            "settings_32_and_48": "REQUIRED_AND_SELECTABLE_BEFORE_POST_AUTOTUNE_REFREEZE",
        },
        "schedule_reconciliation": reconciliation,
        "execution_authority": {
            "required_environment": {"HAWKING_GLM52_XET_EXECUTE": "1"},
            "live_executor_path": "tools/condense/glm52_xet_live.py",
            "live_executor_implemented": True,
            "required_authority_schema": EXECUTION_AUTHORITY_SCHEMA,
            "required_controller_state": "AUTOTUNE_XET",
            "required_controller_epoch": CONTROLLER_EPOCH,
            "required_prepared_transition_intent_schema": TRANSITION_INTENT_SCHEMA,
            "required_committed_checkpoint_schema": CHECKPOINT_SCHEMA,
            "required_committed_checkpoint_ref_schema": COMMITTED_CHECKPOINT_REF_SCHEMA,
            "exact_expected_campaign_contract_binding_required": True,
            "exact_plan_seal_in_prepared_terminal_evidence_required": True,
            "independent_prepared_transition_intent_hmac_verifier_required": True,
            "independent_live_singleton_lease_verifier_required": True,
            "independent_committed_controller_checkpoint_verifier_required": True,
            "self_reported_controller_booleans_are_authority": False,
            "authenticated_telegram_receipt_schema": TELEGRAM_RECEIPT_SCHEMA,
            "independent_exact_telegram_delivery_receipt_verifier_required": True,
            "credentials_must_not_be_serialized": True,
            "current_plan_authorizes_execution": False,
            "offline_plan_alone_authorizes_execution": False,
        },
        "gc_policy": {
            "exact_manifest_paths_only": True,
            "shell_globs_allowed": False,
            "symlink_targets_allowed": False,
            "outside_scratch_root_allowed": False,
            "deletion_implemented_by_this_offline_module": False,
        },
        "claims": {
            "xet_autotune_complete": False,
            "source_stream_started": False,
            "source_shards_fetched": 0,
            "full_parent_parity": False,
            "schedule_frozen": False,
        },
    })
    verify_plan(plan, root=root, rebuild=False)
    return plan


def render_plan_markdown(plan: Mapping[str, Any]) -> str:
    verify_sealed(dict(plan), label="GLM52_XET_AUTOTUNE_PLAN")
    budget = plan["network_budget"]
    schedule = plan["schedule_reconciliation"]
    return "\n".join([
        "# GLM-5.2 Xet autotune offline plan",
        "",
        f"Status: **{plan['status']}**",
        "",
        f"- Immutable source: `{plan['repo']}@{plan['revision']}`",
        f"- Body bytes read by planner: `{plan['body_read_boundary']['planner_model_body_bytes_read']}`",
        f"- Bounded ranges: `{budget['bounded_range_count']}` × `{plan['range_strategy']['range_bytes']}` bytes",
        f"- Planned maximum network accounting: `{budget['planned_maximum_bytes']}` / `{budget['hard_cap_bytes']}` bytes",
        (
            "- Preliminary-schedule maximum new-fetch shards (informational, not a "
            f"selection cap): `{schedule['maximum_new_fetch_shards']}`"
        ),
        f"- Maximum resident shards in one window: `{schedule['maximum_resident_shards_one_window']}`",
        f"- Maximum actual adjacent active+prefetch union: `{schedule['maximum_actual_adjacent_active_prefetch_union']}`",
        (
            "- Required file settings `8`, `16`, `24`, `32`, and `48` are all selectable; "
            "the winning profile is applied only during post-autotune schedule refreeze."
        ),
        (
            "- A separate live executor exists at `tools/condense/glm52_xet_live.py`; "
            "this offline plan alone does not authorize execution, which remains "
            "controller/Telegram gated."
        ),
        "- The 10 GiB cache trial is skipped because planned refetches are zero and the pinned cache path is inert.",
        "",
        f"Plan seal: `{plan['seal_sha256']}`.",
        "",
    ])


def verify_plan(plan: Mapping[str, Any], *, root: Path = REPO_ROOT, rebuild: bool = True) -> dict[str, Any]:
    value = verify_sealed(dict(plan), label="GLM52_XET_AUTOTUNE_PLAN")
    if value.get("schema") != PLAN_SCHEMA or value.get("status") != "PASS_OFFLINE_PLAN_BODY_NOT_READ":
        raise Glm52Error("Xet autotune plan schema/status mismatch")
    if value.get("repo") != REPO_ID or value.get("revision") != REVISION:
        raise Glm52Error("Xet autotune plan immutable identity mismatch")
    boundary = value.get("body_read_boundary", {})
    if boundary.get("planner_network_access") is not False \
            or boundary.get("planner_model_body_bytes_read") != 0 \
            or boundary.get("planner_model_body_files_created") != 0 \
            or boundary.get("planner_cli_live_execution_implemented") is not False \
            or boundary.get("separate_live_executor_implemented") is not True \
            or boundary.get("separate_live_executor_path") != \
            "tools/condense/glm52_xet_live.py" \
            or boundary.get("offline_plan_alone_authorizes_execution") is not False:
        raise Glm52Error("Xet autotune plan crosses the no-body boundary")
    inputs = load_and_validate_inputs(root)
    if value.get("inputs") != _input_refs(inputs):
        raise Glm52Error("Xet autotune plan input seals are stale")
    if value.get("toolchain_binding") != toolchain_binding():
        raise Glm52Error("Xet autotune plan toolchain binding is stale")
    ranges = value.get("range_strategy", {}).get("body_ranges")
    if not isinstance(ranges, list) or len(ranges) != RANGE_SHARDS:
        raise Glm52Error("Xet autotune plan range set is incomplete")
    if len({row.get("path") for row in ranges if isinstance(row, dict)}) != RANGE_SHARDS:
        raise Glm52Error("Xet autotune range paths are not unique")
    for row in ranges:
        if not isinstance(row, dict):
            raise Glm52Error("Xet autotune range identity is invalid")
        identity = {
            "schema": "hawking.glm52.xet_body_range_identity.v1",
            "path": row.get("path"),
            "xet_hash": row.get("xet_hash"),
            "lfs_sha256": row.get("lfs_sha256"),
            "start": row.get("start"),
            "end": row.get("end"),
            "length": row.get("length"),
        }
        if not _is_sha256(row.get("xet_hash")) or not _is_sha256(row.get("lfs_sha256")) \
                or not _is_sha256(row.get("range_id_sha256")) \
                or row["range_id_sha256"] != _canonical_sha256(identity):
            raise Glm52Error("Xet autotune range identity digest is invalid")
    expected_ranges = build_ranges(
        inputs["GLM52_OFFICIAL_MANIFEST.json"],
        inputs["GLM52_SOURCE_FORMAT_LEDGER.json"],
        range_bytes=value.get("range_strategy", {}).get("range_bytes"),
    )
    if canonical(ranges) != canonical(expected_ranges):
        raise Glm52Error("Xet autotune exact body-range identities changed")
    matrix = value.get("trial_matrix")
    if not isinstance(matrix, list) or any(not isinstance(row, dict) for row in matrix) \
            or any(
                isinstance(row.get("range_count"), bool)
                or not isinstance(row.get("range_count"), int)
                for row in matrix
            ) \
            or sum(row["range_count"] for row in matrix) != 184:
        raise Glm52Error("Xet autotune trial matrix changed")
    expected_matrix = build_trial_matrix(
        ranges,
        range_bytes=value.get("range_strategy", {}).get("range_bytes"),
    )
    if canonical(matrix) != canonical(expected_matrix):
        raise Glm52Error("Xet autotune trial range identities/order changed")
    budget = value.get("network_budget", {})
    if budget.get("planned_maximum_bytes", NETWORK_CAP_BYTES + 1) > budget.get("hard_cap_bytes", -1) \
            or budget.get("hard_cap_bytes", NETWORK_CAP_BYTES + 1) > NETWORK_CAP_BYTES:
        raise Glm52Error("Xet autotune network cap is invalid")
    reconcile_schedule(inputs["GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json"])
    if rebuild:
        expected = build_plan(
            root,
            range_bytes=value["range_strategy"]["range_bytes"],
            network_cap_bytes=value["network_budget"]["hard_cap_bytes"],
            runtime=value["runtime_compatibility"],
        )
        if canonical(value) != canonical(expected):
            raise Glm52Error("Xet autotune plan is not the deterministic rebuild")
    return value


def evaluate_resource_trial(
    before: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    after: Mapping[str, Any],
    *,
    required_free_bytes: int,
    required_available_ram_bytes: int,
    trial_network_cap_bytes: int,
    heavy_lane_regressions: Sequence[float] = (),
    complete_source_views: int = 0,
) -> dict[str, Any]:
    """Return a strict selection verdict from already measured resource samples."""
    _require_int(required_free_bytes, "required_free_bytes")
    _require_int(required_available_ram_bytes, "required_available_ram_bytes")
    _require_int(trial_network_cap_bytes, "trial_network_cap_bytes", minimum=1)
    _require_int(complete_source_views, "complete_source_views")
    reasons: list[str] = []
    observations = [before, *samples, after]
    if not samples:
        reasons.append("NO_RUNTIME_SAMPLES")
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            reasons.append(f"SAMPLE_{index}_NOT_AN_OBJECT")
            continue
        missing = LIVE_OBSERVATION_FIELDS - set(observation)
        if missing:
            reasons.append(f"SAMPLE_{index}_MISSING_{'_'.join(sorted(missing)).upper()}")
            continue
        for key in {
            "swap_used_bytes",
            "swapouts",
            "free_disk_bytes",
            "available_ram_bytes",
            "actual_network_bytes",
        }:
            try:
                _require_int(observation[key], f"sample[{index}].{key}")
            except Glm52Error:
                reasons.append(f"SAMPLE_{index}_{key.upper()}_INVALID")
        if not isinstance(observation["thermal_warning"], bool):
            reasons.append(f"SAMPLE_{index}_THERMAL_INVALID")
        for key in {
            "cpu_percent",
            "disk_write_bytes_per_second",
            "reconstruction_latency_seconds",
            "temporary_amplification_ratio",
        }:
            try:
                _require_finite_number(observation[key], f"sample[{index}].{key}")
            except Glm52Error:
                reasons.append(f"SAMPLE_{index}_{key.upper()}_INVALID")
        try:
            _require_finite_number(
                observation["retry_rate"],
                f"sample[{index}].retry_rate",
                maximum=1.0,
            )
        except Glm52Error:
            reasons.append(f"SAMPLE_{index}_RETRY_RATE_INVALID")
    if reasons:
        return {"status": "FAIL", "reasons": sorted(set(reasons))}

    baseline_swap = int(before["swap_used_bytes"])
    baseline_swapouts = int(before["swapouts"])
    peak_swap = max(int(item["swap_used_bytes"]) for item in observations)
    peak_swapouts = max(int(item["swapouts"]) for item in observations)
    min_disk = min(int(item["free_disk_bytes"]) for item in observations)
    min_ram = min(int(item["available_ram_bytes"]) for item in observations)
    network_counters = [int(item["actual_network_bytes"]) for item in observations]
    actual_network_bytes = network_counters[-1] - network_counters[0]
    peak_cpu = max(float(item["cpu_percent"]) for item in observations)
    peak_disk_write = max(float(item["disk_write_bytes_per_second"]) for item in observations)
    maximum_latency = max(float(item["reconstruction_latency_seconds"]) for item in observations)
    maximum_retry_rate = max(float(item["retry_rate"]) for item in observations)
    maximum_temporary_amplification = max(
        float(item["temporary_amplification_ratio"]) for item in observations
    )
    if peak_swap > baseline_swap:
        reasons.append("SWAP_GROWTH")
    if peak_swapouts > baseline_swapouts:
        reasons.append("NEW_SWAPOUTS")
    if any(item["thermal_warning"] is True for item in observations):
        reasons.append("THERMAL_WARNING")
    if min_disk < required_free_bytes:
        reasons.append("DISK_FLOOR_RISK")
    if min_ram < required_available_ram_bytes:
        reasons.append("RAM_FLOOR_RISK")
    if complete_source_views > 1:
        reasons.append("DUPLICATED_COMPLETE_SOURCE_VIEW")
    if any(right < left for left, right in zip(network_counters, network_counters[1:])):
        reasons.append("ACTUAL_NETWORK_COUNTER_REGRESSION")
    if actual_network_bytes <= 0:
        reasons.append("NO_ACTUAL_NETWORK_BYTES")
    if actual_network_bytes > trial_network_cap_bytes:
        reasons.append("TRIAL_NETWORK_CAP_EXCEEDED")
    consecutive = 0
    sustained = False
    for value in heavy_lane_regressions:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or value < 0:
            reasons.append("HEAVY_LANE_SAMPLE_INVALID")
            break
        if float(value) > MAX_HEAVY_REGRESSION:
            consecutive += 1
            sustained = sustained or consecutive >= SUSTAINED_BINS
        else:
            consecutive = 0
    if sustained:
        reasons.append("SUSTAINED_HEAVY_LANE_REGRESSION_GT_5_PERCENT")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "measured": {
            "baseline_swap_used_bytes": baseline_swap,
            "peak_swap_used_bytes": peak_swap,
            "baseline_swapouts": baseline_swapouts,
            "peak_swapouts": peak_swapouts,
            "minimum_free_disk_bytes": min_disk,
            "minimum_available_ram_bytes": min_ram,
            "peak_cpu_percent": peak_cpu,
            "peak_disk_write_bytes_per_second": peak_disk_write,
            "maximum_reconstruction_latency_seconds": maximum_latency,
            "maximum_retry_rate": maximum_retry_rate,
            "maximum_temporary_amplification_ratio": maximum_temporary_amplification,
            "actual_network_bytes": actual_network_bytes,
            "trial_network_cap_bytes": trial_network_cap_bytes,
            "maximum_complete_source_views": complete_source_views,
            "sustained_heavy_lane_regression": sustained,
        },
    }


def select_profile(trials: Sequence[Mapping[str, Any]], *, lane: str) -> dict[str, Any]:
    if lane not in {"acquisition", "steady"}:
        raise Glm52Error("lane must be acquisition or steady")
    eligible: list[dict[str, Any]] = []
    for raw in trials:
        if not isinstance(raw, Mapping):
            continue
        trial = dict(raw)
        if not isinstance(trial.get("trial_id"), str) or not trial["trial_id"]:
            continue
        lanes = trial.get("eligible_lanes")
        if not isinstance(lanes, list) or lane not in lanes:
            continue
        if trial.get("resource_verdict") != "PASS":
            continue
        streams = trial.get("caller_concurrent_shard_streams")
        if isinstance(streams, bool) or not isinstance(streams, int) \
                or not 1 <= streams <= MAX_TESTED_FILE_SETTING:
            continue
        throughput = trial.get("throughput_bytes_per_second")
        if isinstance(throughput, bool) or not isinstance(throughput, (int, float)) \
                or not math.isfinite(float(throughput)) or throughput <= 0:
            continue
        if lane == "steady" and trial.get("sustained_heavy_lane_regression") is not False:
            continue
        lower = trial.get("throughput_ci95_low", throughput)
        upper = trial.get("throughput_ci95_high", throughput)
        if isinstance(lower, bool) or isinstance(upper, bool) \
                or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)) \
                or not math.isfinite(float(lower)) or not math.isfinite(float(upper)) \
                or lower <= 0 or upper < lower or not lower <= throughput <= upper:
            continue
        peak_rss = trial.get("peak_rss_bytes")
        transfer = trial.get("effective_transfer_concurrency")
        if isinstance(peak_rss, bool) or not isinstance(peak_rss, int) or peak_rss < 0 \
                or isinstance(transfer, bool) or not isinstance(transfer, int) or transfer < 1:
            continue
        continuous_measurements = SELECTABLE_TRIAL_MEASUREMENTS - {
            "actual_network_bytes", "trial_network_cap_bytes",
        }
        if any(
            isinstance(trial.get(key), bool)
            or not isinstance(trial.get(key), (int, float))
            or not math.isfinite(float(trial[key]))
            or float(trial[key]) < 0
            for key in continuous_measurements
        ):
            continue
        retry_rate = float(trial["maximum_retry_rate"])
        if retry_rate > 1.0:
            continue
        actual_network = trial.get("actual_network_bytes")
        network_cap = trial.get("trial_network_cap_bytes")
        if isinstance(actual_network, bool) or not isinstance(actual_network, int) \
                or actual_network <= 0 \
                or isinstance(network_cap, bool) or not isinstance(network_cap, int) \
                or network_cap < actual_network:
            continue
        trial["throughput_ci95_low"] = float(lower)
        trial["throughput_ci95_high"] = float(upper)
        eligible.append(trial)
    if not eligible:
        raise Glm52Error(f"no safe selectable {lane} Xet trial")
    fastest = max(eligible, key=lambda item: (float(item["throughput_bytes_per_second"]), item["trial_id"]))
    statistically_tied = [
        item for item in eligible
        if item["throughput_ci95_high"] >= fastest["throughput_ci95_low"]
        and fastest["throughput_ci95_high"] >= item["throughput_ci95_low"]
    ]
    selected = min(
        statistically_tied,
        key=lambda item: (
            item["maximum_retry_rate"],
            item["maximum_reconstruction_latency_seconds"],
            item["maximum_temporary_amplification_ratio"],
            item["actual_network_bytes"],
            item["peak_cpu_percent"],
            item["peak_disk_write_bytes_per_second"],
            item["peak_rss_bytes"],
            item["effective_transfer_concurrency"],
            item["caller_concurrent_shard_streams"],
            str(item["trial_id"]),
        ),
    )
    return {
        "lane": lane,
        "status": "SELECTED",
        "trial_id": selected["trial_id"],
        "selection_pool": sorted(item["trial_id"] for item in statistically_tied),
        "selected_caller_concurrent_shard_streams": selected[
            "caller_concurrent_shard_streams"
        ],
        "preliminary_schedule_maximum_dispatch": PRELIMINARY_SCHEDULE_MAX_DISPATCH,
        "post_autotune_schedule_refreeze_required": True,
        "selected_trial": selected,
    }


def validate_gc_manifest(
    scratch_root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    inspect_existing: bool = False,
) -> list[dict[str, Any]]:
    """Validate exact GC targets without deleting anything."""
    root = scratch_root.resolve(strict=False)
    if not scratch_root.is_absolute():
        raise Glm52Error("GC scratch root must be absolute")
    if root == Path(root.anchor):
        raise Glm52Error("GC scratch root may not be a filesystem root")
    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise Glm52Error(f"GC entry {index} is not an object")
        raw = entry.get("path")
        if not isinstance(raw, str) or not raw or any(char in raw for char in _GLOB_CHARS):
            raise Glm52Error(f"GC entry {index} is not an exact path")
        path = Path(raw)
        if not path.is_absolute():
            raise Glm52Error(f"GC entry {index} path must be absolute")
        lexical = Path(os.path.abspath(path))
        if lexical == root or not _path_within(lexical, root):
            raise Glm52Error(f"GC entry {index} escapes or names the scratch root")
        resolved = path.resolve(strict=False)
        if not _path_within(resolved, root):
            raise Glm52Error(f"GC entry {index} resolves outside scratch root")
        if lexical in seen:
            raise Glm52Error(f"GC entry {index} duplicates a target")
        seen.add(lexical)
        cursor = lexical
        while cursor != root:
            if cursor.is_symlink():
                raise Glm52Error(f"GC entry {index} traverses a symlink")
            cursor = cursor.parent
            if not _path_within(cursor, root) and cursor != root:
                raise Glm52Error(f"GC entry {index} parent traversal escaped root")
        expected_size = _require_int(entry.get("expected_size"), f"GC entry {index}.expected_size")
        expected_inode = _require_int(entry.get("expected_inode"), f"GC entry {index}.expected_inode", minimum=1)
        expected_sha = entry.get("expected_sha256")
        if not _is_sha256(expected_sha):
            raise Glm52Error(f"GC entry {index}.expected_sha256 is invalid")
        inspected = False
        if inspect_existing:
            try:
                metadata = path.lstat()
            except FileNotFoundError as exc:
                raise Glm52Error(f"GC target is missing: {path}") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise Glm52Error(f"GC target is not a regular file: {path}")
            if metadata.st_size != expected_size or metadata.st_ino != expected_inode:
                raise Glm52Error(f"GC target size/inode changed: {path}")
            if sha256_file(path) != expected_sha:
                raise Glm52Error(f"GC target hash changed: {path}")
            inspected = True
        result.append({
            "path": str(lexical),
            "expected_size": expected_size,
            "expected_inode": expected_inode,
            "expected_sha256": expected_sha,
            "inside_scratch_root": True,
            "symlink_free": True,
            "inspected_existing_file": inspected,
            "deletion_performed": False,
        })
    return result


def _require_independent_authority_check(
    callback: Any,
    *args: Any,
    label: str,
    **kwargs: Any,
) -> None:
    try:
        accepted = callback(*args, **kwargs)
    except Exception:
        raise Glm52Error(f"independent {label} verification failed") from None
    if accepted is not True:
        raise Glm52Error(f"independent {label} verification refused authority")


def _validate_prepared_autotune_intent(
    raw: Any,
    *,
    plan_seal: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise Glm52Error("Xet authority lacks the prepared AUTOTUNE_XET intent")
    try:
        intent = validate_transition_intent(dict(raw))
    except (StateError, Glm52Error):
        raise Glm52Error("prepared AUTOTUNE_XET transition intent is invalid") from None
    if intent["source_revision"] != REVISION \
            or intent["controller_epoch"] != CONTROLLER_EPOCH \
            or intent["expected_contract_sha256"] != expected_contract_sha256 \
            or intent["from_state"] != "BUILD_DEPENDENCY_GRAPH" \
            or intent["to_state"] != "AUTOTUNE_XET" \
            or intent["event_kind"] != TRANSITION_EVENT_KINDS["AUTOTUNE_XET"]:
        raise Glm52Error("prepared transition is not the exact controller-v2 AUTOTUNE_XET entry")
    anchor = intent["controller_anchor"]
    if anchor["campaign_id"] != intent["campaign_id"] \
            or anchor["source_revision"] != REVISION \
            or anchor["controller_epoch"] != CONTROLLER_EPOCH \
            or anchor["expected_contract_sha256"] != expected_contract_sha256:
        raise Glm52Error("prepared AUTOTUNE_XET controller anchor identity mismatch")
    anchor_checkpoint = anchor["checkpoint"]
    if anchor_checkpoint["event_count"] < 6 \
            or not _is_sha256(anchor_checkpoint["checkpoint_seal_sha256"]) \
            or anchor_checkpoint["window_event_count"] != 0 \
            or anchor_checkpoint["window_event_head_hash"] != GENESIS_HASH:
        raise Glm52Error("prepared AUTOTUNE_XET intent lacks the exact pre-window checkpoint")
    terminal_raw = intent["state_payload"].get("terminal_evidence")
    if not isinstance(terminal_raw, Mapping):
        raise Glm52Error("prepared AUTOTUNE_XET intent lacks controller-generated evidence")
    try:
        terminal = verify_sealed(dict(terminal_raw), label="AUTOTUNE_XET terminal evidence")
    except Glm52Error:
        raise Glm52Error("prepared AUTOTUNE_XET terminal evidence is invalid") from None
    terminal_fields = {
        "schema", "state", "expected_contract_sha256", "controller_anchor_sha256",
        "artifact_seals", "checklist", "phone_status", "created_at", "seal_sha256",
    }
    if set(terminal) != terminal_fields \
            or terminal.get("schema") != TERMINAL_EVIDENCE_SCHEMA \
            or terminal.get("state") != "AUTOTUNE_XET" \
            or terminal.get("expected_contract_sha256") != expected_contract_sha256 \
            or terminal.get("controller_anchor_sha256") != anchor["anchor_sha256"]:
        raise Glm52Error("prepared AUTOTUNE_XET terminal evidence binding mismatch")
    artifact_seals = terminal.get("artifact_seals")
    plan_evidence = (
        artifact_seals.get("xet_autotune_plan")
        if isinstance(artifact_seals, dict)
        else None
    )
    plan_evidence_fields = {"path", "file_sha256", "seal_sha256", "schema", "status"}
    if not isinstance(plan_evidence, dict) or set(plan_evidence) != plan_evidence_fields \
            or plan_evidence.get("path") != "GLM52_XET_AUTOTUNE_PLAN.json" \
            or plan_evidence.get("schema") != PLAN_SCHEMA \
            or plan_evidence.get("status") != "PASS_OFFLINE_PLAN_BODY_NOT_READ" \
            or plan_evidence.get("seal_sha256") != plan_seal \
            or not _is_sha256(plan_evidence.get("file_sha256")):
        raise Glm52Error("prepared AUTOTUNE_XET intent does not bind the exact Xet plan seal")
    return intent


def _validate_bot_api_receipt_v3(
    raw: Any,
    *,
    transition_intent: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise Glm52Error("Xet authority lacks the Telegram Bot API delivery receipt")
    receipt = dict(raw)
    receipt_fields = {
        "schema", "status", "algorithm", "event_kind", "claim_id", "from_state",
        "to_state", "dedupe_key", "canonical_status", "canonical_status_sha256",
        "rendered_message", "rendered_message_sha256", "controller_anchor",
        "controller_anchor_sha256", "transition_intent", "transition_intent_sha256",
        "message_id", "delivered_at", "bot_api_http_status", "bot_api_response_sha256",
        "chat_identity_digest", "response_validated", "hmac_sha256",
    }
    if set(receipt) != receipt_fields \
            or receipt.get("schema") != TELEGRAM_RECEIPT_SCHEMA \
            or receipt.get("status") != "DELIVERED" \
            or receipt.get("algorithm") != "HMAC-SHA256" \
            or receipt.get("bot_api_http_status") != 200 \
            or receipt.get("response_validated") is not True \
            or isinstance(receipt.get("message_id"), bool) \
            or not isinstance(receipt.get("message_id"), int) \
            or receipt["message_id"] <= 0 \
            or not isinstance(receipt.get("delivered_at"), str) \
            or not receipt["delivered_at"] \
            or any(not _is_sha256(receipt.get(key)) for key in (
                "dedupe_key", "canonical_status_sha256", "rendered_message_sha256",
                "controller_anchor_sha256", "transition_intent_sha256",
                "bot_api_response_sha256", "chat_identity_digest", "hmac_sha256",
            )):
        raise Glm52Error("Telegram authority lacks an exact successful Bot API v3 receipt")
    intent = dict(transition_intent)
    bindings = {
        "event_kind": intent["event_kind"],
        "claim_id": intent["claim_id"],
        "from_state": intent["from_state"],
        "to_state": intent["to_state"],
        "dedupe_key": intent["dedupe_key"],
        "canonical_status": intent["canonical_status"],
        "canonical_status_sha256": intent["canonical_status_sha256"],
        "rendered_message": intent["rendered_message"],
        "rendered_message_sha256": intent["rendered_message_sha256"],
        "controller_anchor": intent["controller_anchor"],
        "controller_anchor_sha256": intent["controller_anchor"]["anchor_sha256"],
        "transition_intent": intent,
        "transition_intent_sha256": intent["seal_sha256"],
    }
    if any(receipt.get(key) != expected for key, expected in bindings.items()):
        raise Glm52Error(
            "Telegram Bot API v3 receipt does not bind the exact prepared intent/status/message/anchor"
        )
    return receipt


def _validate_committed_checkpoint_ref(
    raw: Any,
    *,
    transition_intent: Mapping[str, Any],
    telegram_receipt: Mapping[str, Any],
    expected_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise Glm52Error("Xet authority lacks a committed controller checkpoint reference")
    checkpoint = dict(raw)
    checkpoint_fields = {
        "schema", "checkpoint_schema", "campaign_id", "source_revision",
        "controller_epoch", "expected_contract_sha256", "state", "last_claim_id",
        "transition_intent_sha256", "telegram_receipt_hmac_sha256",
        "checkpoint_seal_sha256", "event_count", "event_head_hash",
        "window_event_count", "window_event_head_hash",
    }
    intent = dict(transition_intent)
    anchor_checkpoint = intent["controller_anchor"]["checkpoint"]
    if set(checkpoint) != checkpoint_fields \
            or checkpoint.get("schema") != COMMITTED_CHECKPOINT_REF_SCHEMA \
            or checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA \
            or checkpoint.get("campaign_id") != intent["campaign_id"] \
            or checkpoint.get("source_revision") != REVISION \
            or checkpoint.get("controller_epoch") != CONTROLLER_EPOCH \
            or checkpoint.get("expected_contract_sha256") != expected_contract_sha256 \
            or checkpoint.get("state") != "AUTOTUNE_XET" \
            or checkpoint.get("last_claim_id") != intent["claim_id"] \
            or checkpoint.get("transition_intent_sha256") != intent["seal_sha256"] \
            or checkpoint.get("telegram_receipt_hmac_sha256") != telegram_receipt["hmac_sha256"] \
            or checkpoint.get("event_count") != anchor_checkpoint["event_count"] + 1 \
            or checkpoint.get("window_event_count") != anchor_checkpoint["window_event_count"] \
            or checkpoint.get("window_event_head_hash") != anchor_checkpoint[
                "window_event_head_hash"
            ] \
            or any(not _is_sha256(checkpoint.get(key)) for key in (
                "checkpoint_seal_sha256", "event_head_hash", "window_event_head_hash",
            )) \
            or checkpoint.get("event_head_hash") == anchor_checkpoint["event_head_hash"] \
            or checkpoint.get("checkpoint_seal_sha256") == anchor_checkpoint[
                "checkpoint_seal_sha256"
            ]:
        raise Glm52Error("committed checkpoint does not bind the exact AUTOTUNE_XET transition")
    return checkpoint


def validate_execution_authority(
    authority: Mapping[str, Any],
    *,
    plan_seal: str,
    expected_contract_sha256: str,
    verifier: ExecutionAuthorityVerifier | None = None,
) -> dict[str, Any]:
    """Bind one committed controller-v2 transition, then require trusted live checks."""
    if not _is_sha256(plan_seal):
        raise Glm52Error("trusted Xet plan seal is invalid")
    if not _is_sha256(expected_contract_sha256):
        raise Glm52Error("trusted expected campaign contract seal is invalid")
    value = verify_sealed(dict(authority), label="Xet execution authority")
    authority_fields = {
        "schema", "status", "repo", "revision", "campaign_id", "controller_epoch",
        "expected_contract_sha256", "plan_seal_sha256", "transition_intent",
        "telegram_delivery_receipt", "committed_controller_checkpoint",
        "credentials_serialized", "seal_sha256",
    }
    if set(value) != authority_fields:
        raise Glm52Error("live Xet authority fields are incomplete or unknown")
    if value.get("schema") != EXECUTION_AUTHORITY_SCHEMA \
            or value.get("status") != "AUTHORIZED_BY_COMMITTED_AUTOTUNE_XET_TRANSITION":
        raise Glm52Error("live Xet authority schema/status mismatch")
    if value.get("repo") != REPO_ID or value.get("revision") != REVISION \
            or value.get("controller_epoch") != CONTROLLER_EPOCH \
            or value.get("expected_contract_sha256") != expected_contract_sha256 \
            or value.get("plan_seal_sha256") != plan_seal:
        raise Glm52Error("live Xet authority identity mismatch")
    intent = _validate_prepared_autotune_intent(
        value.get("transition_intent"),
        plan_seal=plan_seal,
        expected_contract_sha256=expected_contract_sha256,
    )
    if value.get("campaign_id") != intent["campaign_id"]:
        raise Glm52Error("live Xet authority campaign identity mismatch")
    receipt = _validate_bot_api_receipt_v3(
        value.get("telegram_delivery_receipt"), transition_intent=intent
    )
    checkpoint = _validate_committed_checkpoint_ref(
        value.get("committed_controller_checkpoint"),
        transition_intent=intent,
        telegram_receipt=receipt,
        expected_contract_sha256=expected_contract_sha256,
    )
    if value.get("credentials_serialized") is not False:
        raise Glm52Error("execution authority must not serialize credentials")
    if verifier is None:
        raise Glm52Error(
            "independent intent/checkpoint/lease/Telegram authority verifier is required"
        )
    _require_independent_authority_check(
        verifier.verify_prepared_transition_intent_hmac,
        intent,
        plan_seal=plan_seal,
        expected_contract_sha256=expected_contract_sha256,
        label="prepared transition intent HMAC",
    )
    _require_independent_authority_check(
        verifier.verify_telegram_delivery_receipt,
        receipt,
        transition_intent=intent,
        plan_seal=plan_seal,
        expected_contract_sha256=expected_contract_sha256,
        label="Telegram Bot API v3 receipt",
    )
    _require_independent_authority_check(
        verifier.verify_committed_controller_checkpoint,
        checkpoint,
        transition_intent=intent,
        telegram_receipt=receipt,
        plan_seal=plan_seal,
        expected_contract_sha256=expected_contract_sha256,
        label="committed controller checkpoint",
    )
    _require_independent_authority_check(
        verifier.verify_live_singleton_lease,
        checkpoint,
        transition_intent=intent,
        plan_seal=plan_seal,
        expected_contract_sha256=expected_contract_sha256,
        label="live singleton lease",
    )
    return value


def preflight_receipt(root: Path = REPO_ROOT) -> dict[str, Any]:
    inputs = load_and_validate_inputs(root)
    runtime = runtime_compatibility()
    disk = shutil.disk_usage(root)
    return seal({
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS_OFFLINE_PLANNING_LIVE_EXECUTION_AUTHORITY_REQUIRED",
        "repo": REPO_ID,
        "revision": REVISION,
        "inputs": _input_refs(inputs),
        "toolchain_binding": toolchain_binding(),
        "runtime_compatibility": runtime,
        "observation": {
            "filesystem_free_bytes": disk.free,
            "filesystem_total_bytes": disk.total,
            "dynamic_resource_verdict_deferred_to_each_live_trial": True,
        },
        "body_boundary": {
            "network_access": False,
            "model_body_bytes_read": 0,
            "live_controller_state_created": False,
            "run_authorized": False,
        },
    })


def _load_plan(path: Path, root: Path) -> dict[str, Any]:
    plan = read_sealed_json(path)
    return verify_plan(plan, root=root)


def _command_preflight(args: argparse.Namespace) -> int:
    receipt = preflight_receipt(args.root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _command_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args.root, range_bytes=args.range_bytes, network_cap_bytes=args.network_cap_bytes)
    atomic_json(args.output, plan)
    atomic_text(args.markdown_output, render_plan_markdown(plan))
    print(json.dumps({
        "status": plan["status"],
        "plan": str(args.output),
        "markdown": str(args.markdown_output),
        "seal_sha256": plan["seal_sha256"],
        "model_body_bytes_read": 0,
    }, indent=2, sort_keys=True))
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    plan = _load_plan(args.plan, args.root)
    print(json.dumps({
        "status": "PASS",
        "schema": plan["schema"],
        "seal_sha256": plan["seal_sha256"],
        "model_body_bytes_read": 0,
        "network_access": False,
    }, indent=2, sort_keys=True))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    if os.environ.get("HAWKING_GLM52_XET_EXECUTE") != "1":
        raise Glm52Error("live Xet run refused: HAWKING_GLM52_XET_EXECUTE=1 is required")
    if args.authority is None:
        raise Glm52Error("live Xet run refused: sealed controller/Telegram authority is required")
    plan = _load_plan(args.plan, args.root)
    expected_contract = read_sealed_json(args.expected_contract)
    if expected_contract.get("schema") != EXPECTED_CONTRACT_SCHEMA \
            or expected_contract.get("source_revision") != REVISION:
        raise Glm52Error("live Xet run refused: expected campaign contract identity mismatch")
    authority = read_sealed_json(args.authority)
    validate_execution_authority(
        authority,
        plan_seal=plan["seal_sha256"],
        expected_contract_sha256=expected_contract["seal_sha256"],
    )
    raise Glm52Error(
        "offline planner run refused: live trials are implemented only by "
        "tools/condense/glm52_xet_live.py with executor-owned authority verification"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(root=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="verify no-body prerequisites")
    preflight.add_argument("--root", type=Path, default=REPO_ROOT)
    preflight.add_argument("--no-body", action="store_true", required=True)
    preflight.set_defaults(handler=_command_preflight)

    plan = subparsers.add_parser("plan", help="write deterministic offline autotune plan")
    plan.add_argument("--root", type=Path, default=REPO_ROOT)
    plan.add_argument("--range-bytes", type=int, default=RANGE_BYTES)
    plan.add_argument("--network-cap-bytes", type=int, default=NETWORK_CAP_BYTES)
    plan.add_argument("--output", type=Path, default=REPO_ROOT / "GLM52_XET_AUTOTUNE_PLAN.json")
    plan.add_argument(
        "--markdown-output", type=Path, default=REPO_ROOT / "GLM52_XET_AUTOTUNE_PLAN.md"
    )
    plan.set_defaults(handler=_command_plan)

    verify = subparsers.add_parser("verify", help="offline deterministic plan verification")
    verify.add_argument("--root", type=Path, default=REPO_ROOT)
    verify.add_argument("--plan", type=Path, default=REPO_ROOT / "GLM52_XET_AUTOTUNE_PLAN.json")
    verify.add_argument("--offline", action="store_true", required=True)
    verify.set_defaults(handler=_command_verify)

    run = subparsers.add_parser(
        "run", help="validate the offline authority boundary; never execute live trials"
    )
    run.add_argument("--root", type=Path, default=REPO_ROOT)
    run.add_argument("--plan", type=Path, default=REPO_ROOT / "GLM52_XET_AUTOTUNE_PLAN.json")
    run.add_argument(
        "--expected-contract",
        type=Path,
        default=REPO_ROOT / "GLM52_EXPECTED_CAMPAIGN_CONTRACT.json",
    )
    run.add_argument("--authority", type=Path)
    run.set_defaults(handler=_command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Glm52Error as exc:
        print(f"GLM52_XET_AUTOTUNE_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
