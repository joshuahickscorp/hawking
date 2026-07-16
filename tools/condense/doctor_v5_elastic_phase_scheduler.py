#!/usr/bin/env python3.12
"""Unbound, default-off single-device elastic phase scheduler contract.

The active Doctor queue does not import this module.  It defines a future
pending-only state machine that pipelines at most one heavy prepare, one primary
encoder, one serial finalizer, and one opportunistic companion.  Companion work
is admitted only from authenticated idle CPU/RAM samples and a qualified exact
8/12/16/20 vendor profile.  Encoder return closes lending immediately and makes
an existing companion checkpoint/preempt before finalization.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
import sysconfig
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

import doctor_v5_aggressive_admission_policy as aggressive
import doctor_v5_local_observer as local_observer


ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
AGGRESSIVE_OVERLAY = (
    ULTRA_ROOT / "staged_acceleration/aggressive_v2/aggressive_admission_overlay.json"
)
STAGE_ROOT = ULTRA_ROOT / "staged_acceleration/elastic_v1"
DEFAULT_CONTRACT = STAGE_ROOT / "elastic_phase_contract.json"
DEFAULT_STATE = STAGE_ROOT / "elastic_phase_state.json"
DEFAULT_INVOCATION_MANIFEST = STAGE_ROOT / "phase_invocation_manifest.json"
DEFAULT_HOST_PLAN = (
    ULTRA_ROOT / "staged_acceleration/host_sprint_v1/host_sprint_plan.json"
)

CONTRACT_SCHEMA = "hawking.doctor_v5_elastic_phase_contract.v1"
STATE_SCHEMA = "hawking.doctor_v5_elastic_phase_state.v1"
DECISION_SCHEMA = "hawking.doctor_v5_elastic_lend_decision.v1"
RECOVERY_SCHEMA = "hawking.doctor_v5_elastic_crash_recovery.v1"
ROLLBACK_SCHEMA = "hawking.doctor_v5_elastic_rollback.v1"
OVERLAP_SCHEMA = "hawking.doctor_v5_elastic_overlap_envelope.v1"
SWAP_DECISION_SCHEMA = "hawking.doctor_v5_elastic_swap_decision_binding.v1"
PROCESS_IDENTITY_SCHEMA = "hawking.doctor_v5_elastic_process_identity.v1"
OWNER_LEASE_SCHEMA = "hawking.doctor_v5_elastic_owner_lease.v1"
EXIT_OBSERVATION_SCHEMA = "hawking.doctor_v5_elastic_exit_observation.v1"
PHASE_OUTPUT_SCHEMA = "hawking.doctor_v5_elastic_phase_output_receipt.v1"
WORKER_COMPLETION_SCHEMA = "hawking.doctor_v5_elastic_worker_completion.v1"
INVOCATION_MANIFEST_SCHEMA = "hawking.doctor_v5_phase_invocation_manifest.v1"
INVOCATION_ENTRY_SCHEMA = "hawking.doctor_v5_phase_invocation_entry.v1"
LAUNCH_REQUEST_SCHEMA = "hawking.doctor_v5_inert_phase_launch_request.v1"
INERT_HANDSHAKE_SCHEMA = "hawking.doctor_v5_inert_phase_handshake.v1"
PHASE_VALIDATOR_RECEIPT_SCHEMA = (
    "hawking.doctor_v5_phase_semantic_validator_receipt.v1"
)
INERT_RELEASE_SCHEMA = "hawking.doctor_v5_inert_phase_release.v1"
TARGET_CLAIM_HANDSHAKE_SCHEMA = (
    "hawking.doctor_v5_target_resource_claim_handshake.v1"
)
TARGET_CLAIM_ACK_SCHEMA = "hawking.doctor_v5_target_resource_claim_ack.v1"
TARGET_CLAIM_FAILURE_SCHEMA = (
    "hawking.doctor_v5_target_resource_claim_failure.v1"
)
TARGET_RESOURCE_GUARD_SCHEMA = "hawking.doctor_v5_target_resource_guard.v1"
PYTHON_MODULE_CLOSURE_SCHEMA = "hawking.python_module_source_closure.v1"
TARGET_DEPENDENCY_CLOSURE_SCHEMA = (
    "hawking.doctor_v5_target_dependency_closure.v1"
)
CAS_SCHEMA = "hawking.doctor_v5_elastic_cas_commit.v1"
VERSION = "2026-07-14.1"
MIN_IDLE_SAMPLES = 3
MIN_SAMPLE_SPACING_SECONDS = 5.0
MAX_SAMPLE_AGE_SECONDS = 60.0
SHA256_LENGTH = 64
REVIEWED_TOPOLOGY = {
    "physical_cores": 28,
    "logical_cores": 28,
    "performance_cores": 20,
    "efficiency_cores": 8,
}
TOPOLOGY_SYSCTLS = {
    "physical_cores": "hw.physicalcpu",
    "logical_cores": "hw.logicalcpu",
    "performance_cores": "hw.perflevel0.physicalcpu",
    "efficiency_cores": "hw.perflevel1.physicalcpu",
}
INVOCATION_PHASES = ("prepare", "encoder", "finalizer", "companion")
INERT_LAUNCHER = HERE / "doctor_v5_inert_phase_launcher.py"
CONTROL_ENVIRONMENT_PREFIXES = (
    "HAWKING_", "DOCTOR_", "STRAND_", "RAYON_", "OMP_", "METAL_",
    "MLX_", "TOKENIZERS_",
)
TARGET_CLAIM_PROTOCOL = {
    "schema": "hawking.doctor_v5_target_resource_claim_protocol.v1",
    "transport": "source-bound-argv-plus-two-phase-file-handshake",
    "request_flag": "--elastic-launch-request",
    "handshake_flag": "--elastic-claim-handshake",
    "ack_flag": "--elastic-claim-ack",
    "selected_threads_flag": "--elastic-selected-threads",
    "reservation_bytes_flag": "--elastic-reservation-bytes",
    "resource_spec_flag": "--elastic-resource-spec-sha256",
    "tier_flag": "--elastic-tier",
    "rate_flag": "--elastic-rate",
    "thread_selection_flag": "--elastic-thread-selection-sha256",
    "lend_decision_flag": "--elastic-lend-decision-sha256",
    "heavy_work_before_ack_permitted": False,
}


class ElasticError(RuntimeError):
    """The elastic contract, state, or measured lend decision is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _file_reference(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ElasticError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ElasticError(f"JSON root is not an object: {path}")
    return value


def _stable_json_reference(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse one confined JSON artifact from the bytes hashed on one fd."""
    try:
        confined = _safe_stage_input(Path(path))
        descriptor = os.open(
            confined, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise ElasticError(f"cannot open elastic JSON artifact {path}: {exc}") \
            from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ElasticError(f"elastic JSON artifact is not regular: {confined}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise ElasticError(f"elastic JSON artifact is oversized: {confined}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_stat = os.stat(confined, follow_symlinks=False)
    identity = lambda row: (
        row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns
    )
    raw = b"".join(chunks)
    if identity(before) != identity(after) or identity(after) != identity(path_stat) \
            or len(raw) != after.st_size:
        raise ElasticError(f"elastic JSON artifact changed during read: {confined}")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ElasticError(f"cannot parse elastic JSON artifact {confined}: {exc}") \
            from exc
    if not isinstance(value, dict):
        raise ElasticError(f"elastic JSON root is not an object: {confined}")
    try:
        display = str(confined.relative_to(ROOT.resolve(strict=True)))
    except ValueError:
        display = str(confined)
    return value, {
        "path": display, "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _regular_reference_matches(reference: Any) -> bool:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"} \
            or not _valid_sha(reference.get("sha256")) \
            or isinstance(reference.get("bytes"), bool) \
            or not isinstance(reference.get("bytes"), int) \
            or reference["bytes"] < 0:
        return False
    try:
        raw_path = Path(reference["path"])
        lexical = raw_path if raw_path.is_absolute() else ROOT / raw_path
        cursor = Path(lexical.anchor) if lexical.is_absolute() else Path()
        components = lexical.parts[1:] if lexical.is_absolute() else lexical.parts
        for component in components:
            cursor = cursor / component
            if cursor.is_symlink():
                return False
        path = lexical.resolve(strict=True)
        if not path.is_file():
            return False
        raw = path.read_bytes()
    except (OSError, TypeError, ValueError):
        return False
    return len(raw) == reference["bytes"] \
        and hashlib.sha256(raw).hexdigest() == reference["sha256"]


def _bound_workspace_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a source file without allowing escape or any symlink component."""
    root = ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path if path.is_absolute() else Path.cwd() / path))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ElasticError("invocation manifest must remain inside the workspace") from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ElasticError("invocation manifest path cannot contain a symlink")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_file():
        raise ElasticError("invocation manifest is not a regular file")
    value = _read_json(resolved)
    return value, _file_reference(resolved)


def _workspace_artifact_reference(path: Path, *, cwd: Path | None = None) \
        -> dict[str, Any]:
    root = ROOT.resolve(strict=True)
    raw = Path(path)
    lexical = raw if raw.is_absolute() else (cwd or root) / raw
    absolute = Path(os.path.abspath(lexical))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ElasticError("invocation code/request/config artifact escaped workspace") \
            from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ElasticError("invocation code/request/config path contains a symlink")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_file():
        raise ElasticError("invocation code/request/config artifact is not regular")
    return local_observer.stable_artifact_reference(resolved)


def _canonical_stage_path(path: Path, *, must_exist: bool = False) -> str:
    """Return one lexical, symlink-free path below the inert elastic stage."""
    stage = STAGE_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path if path.is_absolute() else Path.cwd() / path))
    try:
        relative = absolute.relative_to(stage)
    except ValueError as exc:
        raise ElasticError("inert launcher artifact escaped elastic_v1") from exc
    cursor = stage
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ElasticError("inert launcher artifact path contains a symlink")
        if cursor.exists() and cursor != absolute and not cursor.is_dir():
            raise ElasticError("inert launcher artifact parent is not a directory")
    if must_exist:
        resolved = absolute.resolve(strict=True)
        if resolved != absolute or not resolved.is_file():
            raise ElasticError("inert launcher artifact is not an exact regular file")
    else:
        parent = absolute.parent.resolve(strict=True)
        parent.relative_to(stage)
    return str(absolute)


def _environment_hashes(environment: dict[str, str]) -> dict[str, str]:
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in environment.items()):
        raise ElasticError("launch environment is invalid")
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sorted(environment.items())
    }


def _cwd_identity(path: Path) -> dict[str, Any]:
    root = ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    relative = absolute.relative_to(root)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ElasticError("launch cwd contains a symlink component")
    resolved = absolute.resolve(strict=True)
    if resolved != absolute or not resolved.is_dir():
        raise ElasticError("launch cwd is not one exact workspace directory")
    details = os.stat(resolved, follow_symlinks=False)
    return {"path": str(resolved), "device": details.st_dev,
            "inode": details.st_ino}


def _command_artifacts(argv: list[str], *, cwd: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for position, token in enumerate(argv[1:], start=1):
        candidate = Path(token)
        path = candidate if candidate.is_absolute() else cwd / candidate
        if not path.is_file():
            continue
        role = "script" if candidate.suffix.lower() in {".py", ".pyw"} \
            else "config" if candidate.suffix.lower() in {
                ".json", ".yaml", ".yml", ".toml"
            } else "semantic"
        artifacts.append({
            "position": position, "argv_value": token, "role": role,
            "reference": _workspace_artifact_reference(candidate, cwd=cwd),
        })
    return artifacts


def _named_semantic_artifacts(paths: dict[str, Path] | None, *, cwd: Path) \
        -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for name, path in sorted((paths or {}).items()):
        if not isinstance(name, str) \
                or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", name):
            raise ElasticError("launch semantic artifact name is invalid")
        artifacts.append({
            "name": name,
            "reference": _workspace_artifact_reference(Path(path), cwd=cwd),
        })
    return artifacts


def _resource_claim_errors(claim: Any, *, phase: str) -> list[str]:
    required = {
        "selected_threads", "reservation_bytes", "resource_spec_sha256",
        "tier", "rate", "thread_selection_sha256", "lend_decision_sha256",
    }
    if not isinstance(claim, dict) or set(claim) != required:
        return ["inert launch resource claim keys are invalid"]
    errors: list[str] = []
    threads = claim.get("selected_threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        errors.append("inert launch selected thread claim is invalid")
    if phase in {"prepare", "finalizer"}:
        ram = claim.get("reservation_bytes")
        if isinstance(ram, bool) or not isinstance(ram, int) or ram <= 0 \
                or not _valid_sha(claim.get("resource_spec_sha256")) \
                or any(claim.get(name) is not None for name in (
                    "tier", "rate", "thread_selection_sha256",
                    "lend_decision_sha256",
                )):
            errors.append("inert prepare/finalizer resource claim is invalid")
    elif phase == "encoder":
        if not isinstance(claim.get("tier"), str) or not claim.get("tier") \
                or not isinstance(claim.get("rate"), str) or not claim.get("rate") \
                or not _valid_sha(claim.get("thread_selection_sha256")) \
                or any(claim.get(name) is not None for name in (
                    "reservation_bytes", "resource_spec_sha256",
                    "lend_decision_sha256",
                )):
            errors.append("inert encoder resource claim is invalid")
    elif phase == "companion":
        ram = claim.get("reservation_bytes")
        if isinstance(ram, bool) or not isinstance(ram, int) or ram <= 0 \
                or not isinstance(claim.get("tier"), str) or not claim.get("tier") \
                or not isinstance(claim.get("rate"), str) or not claim.get("rate") \
                or not _valid_sha(claim.get("thread_selection_sha256")) \
                or not _valid_sha(claim.get("lend_decision_sha256")) \
                or claim.get("resource_spec_sha256") is not None:
            errors.append("inert companion resource claim is invalid")
    else:
        errors.append("inert launch resource-claim phase is invalid")
    return errors


def _claim_token(value: Any) -> str:
    return "none" if value is None else str(value)


def _resource_claim_environment(claim: dict[str, Any], *, request_path: str) \
        -> dict[str, str]:
    threads = _claim_token(claim.get("selected_threads"))
    return {
        "DOCTOR_V5_ELASTIC_REQUEST_PATH": request_path,
        "DOCTOR_V5_ELASTIC_RESOURCE_CLAIM_SHA256": _hash_value(claim),
        "DOCTOR_V5_ELASTIC_SELECTED_THREADS": threads,
        "DOCTOR_V5_ELASTIC_RESERVATION_BYTES": _claim_token(
            claim.get("reservation_bytes")
        ),
        "DOCTOR_V5_ELASTIC_RESOURCE_SPEC_SHA256": _claim_token(
            claim.get("resource_spec_sha256")
        ),
        "DOCTOR_V5_ELASTIC_TIER": _claim_token(claim.get("tier")),
        "DOCTOR_V5_ELASTIC_RATE": _claim_token(claim.get("rate")),
        "DOCTOR_V5_ELASTIC_THREAD_SELECTION_SHA256": _claim_token(
            claim.get("thread_selection_sha256")
        ),
        "DOCTOR_V5_ELASTIC_LEND_DECISION_SHA256": _claim_token(
            claim.get("lend_decision_sha256")
        ),
        "RAYON_NUM_THREADS": threads,
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
    }


def _claim_protocol_argv_tail(claim: dict[str, Any], *, request_path: str,
                              handshake_path: str, ack_path: str) -> list[str]:
    protocol = TARGET_CLAIM_PROTOCOL
    return [
        protocol["request_flag"], request_path,
        protocol["handshake_flag"], handshake_path,
        protocol["ack_flag"], ack_path,
        protocol["selected_threads_flag"], _claim_token(
            claim.get("selected_threads")
        ),
        protocol["reservation_bytes_flag"], _claim_token(
            claim.get("reservation_bytes")
        ),
        protocol["resource_spec_flag"], _claim_token(
            claim.get("resource_spec_sha256")
        ),
        protocol["tier_flag"], _claim_token(claim.get("tier")),
        protocol["rate_flag"], _claim_token(claim.get("rate")),
        protocol["thread_selection_flag"], _claim_token(
            claim.get("thread_selection_sha256")
        ),
        protocol["lend_decision_flag"], _claim_token(
            claim.get("lend_decision_sha256")
        ),
    ]


def _claim_protocol_errors(request: dict[str, Any]) -> list[str]:
    claim = request.get("resource_claim")
    paths = request.get("paths")
    target = request.get("target")
    if not isinstance(claim, dict) or not isinstance(paths, dict) \
            or not isinstance(target, dict):
        return ["target resource-claim protocol inputs are invalid"]
    errors: list[str] = []
    claim_sha = _hash_value(claim)
    if request.get("resource_claim_sha256") != claim_sha:
        errors.append("target resource-claim hash differs")
    if request.get("target_claim_protocol") != TARGET_CLAIM_PROTOCOL:
        errors.append("target resource-claim protocol contract differs")
    expected_tail = _claim_protocol_argv_tail(
        claim, request_path=paths.get("request", ""),
        handshake_path=paths.get("target_claim_handshake", ""),
        ack_path=paths.get("target_claim_ack", ""),
    )
    argv = target.get("argv")
    if not isinstance(argv, list) or len(argv) < len(expected_tail) \
            or argv[-len(expected_tail):] != expected_tail:
        errors.append("target argv is not exactly resource-claim bound")
    elif any(
            token in {"-t", "--threads", "--num-threads", "--n-threads",
                      "--workers", "--num-workers"}
            or token.startswith(("--threads=", "--num-threads=", "--n-threads=",
                                 "--workers=", "--num-workers="))
            for token in argv[:-len(expected_tail)]):
        errors.append("target argv contains a conflicting unbound thread control")
    expected_environment = _resource_claim_environment(
        claim, request_path=paths.get("request", "")
    )
    if target.get("resource_environment") != expected_environment:
        errors.append("target injected environment is not resource-claim bound")
    timeout = request.get("target_claim_handshake_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
            or not math.isfinite(float(timeout)) \
            or not 0.25 <= float(timeout) <= 30.0:
        errors.append("target resource-claim handshake timeout is invalid")
    return errors


def _applied_resource_controls(request: dict[str, Any]) -> dict[str, Any]:
    environment = request["target"]["resource_environment"]
    thread_keys = (
        "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    claim = request["resource_claim"]
    return {
        "selected_threads": claim["selected_threads"],
        "reservation_bytes": claim["reservation_bytes"],
        "thread_environment": {key: environment[key] for key in thread_keys},
        "resource_environment_sha256": _hash_value(environment),
        "claim_argv_tail_sha256": _hash_value(_claim_protocol_argv_tail(
            claim, request_path=request["paths"]["request"],
            handshake_path=request["paths"]["target_claim_handshake"],
            ack_path=request["paths"]["target_claim_ack"],
        )),
        "enforcement": (
            "pre-process-authoritative-thread-env+source-bound-adapter+tree-rss-guard"
        ),
    }


def _isolated_python_argv(argv: list[str]) -> list[str]:
    if "python" not in Path(argv[0]).name.lower():
        return list(argv)
    if len(argv) < 2 or argv[1] in {"-", "-c", "-m"}:
        raise ElasticError(
            "production Python target/validator must be one isolated bound script"
        )
    if argv[1:3] == ["-I", "-S"]:
        return list(argv)
    if argv[1] in {"-I", "-S"}:
        raise ElasticError("production Python isolation flags are incomplete/ambiguous")
    return [argv[0], "-I", "-S", *argv[1:]]


def build_inert_launch_request(*, phase: str, cell_id: str,
                               request_path: Path, contract_path: Path,
                               state_path: Path, expected_state_generation: int,
                               target_argv: list[str], target_cwd: Path,
                               target_environment: dict[str, str],
                               validator_argv: list[str], validator_cwd: Path,
                               validator_environment: dict[str, str],
                               handshake_path: Path, release_path: Path,
                               target_claim_handshake_path: Path,
                               target_claim_ack_path: Path,
                               target_claim_failure_path: Path,
                               target_resource_guard_path: Path,
                               output_path: Path, semantic_receipt_path: Path,
                               worker_receipt_path: Path,
                               resource_claim: dict[str, Any],
                               target_semantic_artifact_paths:
                               dict[str, Path] | None = None,
                               validator_semantic_artifact_paths:
                               dict[str, Path] | None = None,
                               launch_nonce: str | None = None,
                               target_claim_handshake_timeout_seconds: float = 5.0,
                               commit_wait_timeout_seconds: float = 30.0) \
        -> dict[str, Any]:
    """Build, but do not execute, one generation-bound inert launch request."""
    if phase not in INVOCATION_PHASES or not isinstance(cell_id, str) or not cell_id:
        raise ElasticError("inert launch phase/cell is invalid")
    if isinstance(expected_state_generation, bool) \
            or not isinstance(expected_state_generation, int) \
            or expected_state_generation < 0:
        raise ElasticError("inert launch state generation is invalid")
    if not isinstance(commit_wait_timeout_seconds, (int, float)) \
            or isinstance(commit_wait_timeout_seconds, bool) \
            or not math.isfinite(float(commit_wait_timeout_seconds)) \
            or not 1.0 <= float(commit_wait_timeout_seconds) <= 300.0:
        raise ElasticError("inert launch commit timeout is invalid")
    if not target_argv or not validator_argv \
            or any(not isinstance(row, str) or not row
                   for row in [*target_argv, *validator_argv]):
        raise ElasticError("inert target/validator argv is invalid")
    claim_errors = _resource_claim_errors(resource_claim, phase=phase)
    if claim_errors:
        raise ElasticError("invalid inert resource claim: " + "; ".join(claim_errors))
    authoritative_threads = str(resource_claim["selected_threads"])
    conflicts = {
        key: value for key, value in target_environment.items()
        if key in {
            "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        } and value != authoritative_threads
    }
    if conflicts:
        raise ElasticError("target base environment conflicts with admitted threads")
    request = _canonical_stage_path(Path(request_path))
    contract = _canonical_stage_path(Path(contract_path))
    state = _canonical_stage_path(Path(state_path))
    paths = {
        "request": request, "contract": contract, "state": state,
        "handshake": _canonical_stage_path(Path(handshake_path)),
        "release": _canonical_stage_path(Path(release_path)),
        "target_claim_handshake": _canonical_stage_path(
            Path(target_claim_handshake_path)
        ),
        "target_claim_ack": _canonical_stage_path(Path(target_claim_ack_path)),
        "target_claim_failure": _canonical_stage_path(
            Path(target_claim_failure_path)
        ),
        "target_resource_guard": _canonical_stage_path(
            Path(target_resource_guard_path)
        ),
        "output": _canonical_stage_path(Path(output_path)),
        "semantic_receipt": _canonical_stage_path(Path(semantic_receipt_path)),
        "worker_receipt": _canonical_stage_path(Path(worker_receipt_path)),
        "commit": _canonical_stage_path(Path(state).with_name(
            f"{Path(state).name}.commit.{expected_state_generation + 1}.json"
        )),
    }
    if len(set(paths.values())) != len(paths):
        raise ElasticError("inert launch paths are not distinct")
    target_base_argv = _isolated_python_argv(target_argv)
    validator_bound_argv = _isolated_python_argv(validator_argv)
    bound_target_argv = [*target_base_argv, *_claim_protocol_argv_tail(
        resource_claim, request_path=request,
        handshake_path=paths["target_claim_handshake"],
        ack_path=paths["target_claim_ack"],
    )]
    target_cwd_path = Path(target_cwd).resolve(strict=True)
    validator_cwd_path = Path(validator_cwd).resolve(strict=True)
    for cwd in (target_cwd_path, validator_cwd_path):
        cwd.relative_to(ROOT.resolve(strict=True))
    target_executable = local_observer.stable_artifact_reference(
        Path(target_base_argv[0]).resolve(strict=True)
    )
    validator_executable = local_observer.stable_artifact_reference(
        Path(validator_bound_argv[0]).resolve(strict=True)
    )
    if target_base_argv[0] != target_executable["path"] \
            or validator_bound_argv[0] != validator_executable["path"]:
        raise ElasticError("target/validator argv[0] is not canonical executable path")
    nonce = launch_nonce or secrets.token_hex(32)
    action = {
        "prepare": "prepare_start", "encoder": "encoder_start",
        "finalizer": "finalizer_start", "companion": "companion_launch",
    }[phase]
    value: dict[str, Any] = {
        "schema": LAUNCH_REQUEST_SCHEMA, "version": VERSION,
        "phase": phase, "action": action, "cell_id": cell_id,
        "resource_claim": copy.deepcopy(resource_claim),
        "launch_nonce": nonce,
        "expected_state_generation": expected_state_generation,
        "commit_wait_timeout_seconds": float(commit_wait_timeout_seconds),
        "launcher": _file_reference(INERT_LAUNCHER), "paths": paths,
        "target": {
            "executable": target_executable, "argv": bound_target_argv,
            "argv_artifacts": _command_artifacts(
                bound_target_argv, cwd=target_cwd_path
            ),
            "semantic_artifacts": _named_semantic_artifacts(
                target_semantic_artifact_paths, cwd=target_cwd_path
            ),
            "cwd": str(target_cwd_path),
            "cwd_identity": _cwd_identity(target_cwd_path),
            "environment_value_sha256s": _environment_hashes(target_environment),
            "resource_environment": _resource_claim_environment(
                resource_claim, request_path=request
            ),
        },
        "validator": {
            "executable": validator_executable, "argv": validator_bound_argv,
            "argv_artifacts": _command_artifacts(
                validator_bound_argv, cwd=validator_cwd_path
            ),
            "semantic_artifacts": _named_semantic_artifacts(
                validator_semantic_artifact_paths, cwd=validator_cwd_path
            ),
            "cwd": str(validator_cwd_path),
            "cwd_identity": _cwd_identity(validator_cwd_path),
            "environment": dict(sorted(validator_environment.items())),
            "environment_value_sha256s": _environment_hashes(
                validator_environment
            ),
            "receipt_schema": PHASE_VALIDATOR_RECEIPT_SCHEMA,
        },
        "required_semantics": {
            "exact_output": True, "parity_verified": True,
            "zero_skips": True, "skipped_count": 0,
        },
        "resource_claim_sha256": _hash_value(resource_claim),
        "target_claim_protocol": copy.deepcopy(TARGET_CLAIM_PROTOCOL),
        "target_claim_handshake_timeout_seconds": float(
            target_claim_handshake_timeout_seconds
        ),
        "caller_selected_completion_paths_permitted": False,
        "heavy_work_before_commit_permitted": False,
    }
    value["request_sha256"] = _hash_value(value)
    errors = validate_inert_launch_request(value)
    if errors:
        raise ElasticError("invalid inert launch request: " + "; ".join(errors))
    return value


def _command_binding_errors(command: Any, *, validator: bool) -> list[str]:
    required = {"executable", "argv", "argv_artifacts", "semantic_artifacts", "cwd",
                "cwd_identity",
                "environment_value_sha256s"}
    if validator:
        required |= {"environment", "receipt_schema"}
    else:
        required |= {"resource_environment"}
    if not isinstance(command, dict) or set(command) != required:
        return ["launch command keys are invalid"]
    errors: list[str] = []
    executable, argv, artifacts = (
        command.get("executable"), command.get("argv"),
        command.get("argv_artifacts"),
    )
    if not _regular_reference_matches(executable):
        errors.append("launch command executable is absent or drifted")
    if not isinstance(argv, list) or not argv \
            or any(not isinstance(row, str) or not row for row in argv) \
            or not isinstance(executable, dict) \
            or argv[0] != executable.get("path"):
        errors.append("launch command argv/executable binding is invalid")
    cwd = command.get("cwd")
    try:
        cwd_path = Path(cwd).resolve(strict=True)
        cwd_path.relative_to(ROOT.resolve(strict=True))
        if str(cwd_path) != cwd or not cwd_path.is_dir():
            raise ValueError
    except (OSError, TypeError, ValueError):
        cwd_path = None
        errors.append("launch command cwd is invalid")
    try:
        current_cwd_identity = _cwd_identity(Path(cwd))
    except (ElasticError, OSError, TypeError, ValueError):
        current_cwd_identity = None
    if command.get("cwd_identity") != current_cwd_identity:
        errors.append("launch command cwd inode/path binding is absent or drifted")
    bound_positions: set[int] = set()
    if not isinstance(artifacts, list):
        errors.append("launch command artifact inventory is invalid")
    else:
        for row in artifacts:
            if not isinstance(row, dict) or set(row) != {
                    "position", "argv_value", "role", "reference"}:
                errors.append("launch command artifact row is invalid")
                continue
            position = row.get("position")
            if isinstance(position, bool) or not isinstance(position, int) \
                    or position <= 0 or position in bound_positions \
                    or not isinstance(argv, list) or position >= len(argv) \
                    or argv[position] != row.get("argv_value") \
                    or row.get("role") not in {"script", "config", "semantic"}:
                errors.append("launch command artifact position/role is invalid")
                continue
            bound_positions.add(position)
            reference = row.get("reference")
            try:
                current = _workspace_artifact_reference(
                    Path(row["argv_value"]), cwd=cwd_path
                )
            except (ElasticError, OSError, TypeError, ValueError):
                current = None
            if current != reference or not _regular_reference_matches(reference):
                errors.append("launch command source artifact is absent/drifted")
    semantic = command.get("semantic_artifacts")
    semantic_names: set[str] = set()
    semantic_by_name: dict[str, Any] = {}
    if not isinstance(semantic, list):
        errors.append("launch command semantic artifact inventory is invalid")
    else:
        for row in semantic:
            if not isinstance(row, dict) or set(row) != {"name", "reference"} \
                    or not isinstance(row.get("name"), str) \
                    or not re.fullmatch(
                        r"[a-z][a-z0-9_.-]{0,63}", row.get("name", "")
                    ) or row["name"] in semantic_names:
                errors.append("launch command semantic artifact row is invalid")
                continue
            semantic_names.add(row["name"])
            reference = row.get("reference")
            semantic_by_name[row["name"]] = reference
            if not _regular_reference_matches(reference):
                errors.append("launch command semantic artifact is absent/drifted")
                continue
            try:
                semantic_path = Path(reference["path"])
                semantic_path = semantic_path if semantic_path.is_absolute() \
                    else ROOT / semantic_path
                current = _workspace_artifact_reference(semantic_path)
            except (ElasticError, OSError, TypeError, ValueError, KeyError):
                current = None
            if current != reference:
                errors.append("launch command semantic artifact escaped workspace")
    python_command = isinstance(executable, dict) \
        and "python" in Path(str(executable.get("path", ""))).name.lower()
    if python_command:
        if not isinstance(argv, list) or len(argv) < 4 \
                or argv[1:3] != ["-I", "-S"] \
                or 3 not in bound_positions:
            errors.append("Python launch command is not isolated/source-bound")
        if not validator:
            closure = semantic_by_name.get("target.dependency_closure")
            errors.extend(_target_dependency_closure_errors(
                closure, command=command
            ))
    if isinstance(argv, list):
        for position, token in enumerate(argv[:-1]):
            if token in {"--config", "--config-path"} \
                    and position + 1 not in bound_positions:
                errors.append("launch request/config argument is not source-bound")
    if python_command and not semantic_names:
        errors.append("Python launch command lacks explicit semantic source binding")
    if not semantic_names:
        errors.append("launch command lacks explicit semantic artifact binding")
    hashes = command.get("environment_value_sha256s")
    if not isinstance(hashes, dict) or any(
            not isinstance(key, str) or not _valid_sha(value)
            for key, value in hashes.items()):
        errors.append("launch command environment hashes are invalid")
    forbidden_injection = {
        "PYTHONHOME", "PYTHONPATH", "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    }
    if isinstance(hashes, dict) and forbidden_injection.intersection(hashes):
        errors.append("launch environment contains Python/dynamic-loader injection")
    if validator:
        environment = command.get("environment")
        try:
            expected_hashes = _environment_hashes(environment)
        except (ElasticError, AttributeError):
            expected_hashes = None
        if expected_hashes != hashes \
                or command.get("receipt_schema") \
                != PHASE_VALIDATOR_RECEIPT_SCHEMA:
            errors.append("phase validator environment/receipt binding is invalid")
        if isinstance(environment, dict) and any(
                re.search(r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|PRIVATE_KEY)",
                          key, re.IGNORECASE)
                for key in environment):
            errors.append("phase validator environment contains secret-like keys")
    elif not isinstance(command.get("resource_environment"), dict) \
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in command["resource_environment"].items()):
        errors.append("target injected resource environment is invalid")
    return errors


def validate_inert_launch_request(request: Any, *,
                                  request_reference: dict[str, Any] | None = None) \
        -> list[str]:
    required = {
        "schema", "version", "phase", "action", "cell_id", "launch_nonce",
        "resource_claim", "resource_claim_sha256", "target_claim_protocol",
        "target_claim_handshake_timeout_seconds",
        "expected_state_generation", "commit_wait_timeout_seconds", "launcher",
        "paths", "target", "validator", "required_semantics",
        "caller_selected_completion_paths_permitted",
        "heavy_work_before_commit_permitted", "request_sha256",
    }
    if not isinstance(request, dict) or set(request) != required:
        return ["inert launch request keys are invalid"]
    errors: list[str] = []
    try:
        hash_matches = request.get("request_sha256") == _hash_value(
            _without(request, "request_sha256")
        )
    except (TypeError, ValueError):
        hash_matches = False
    if request.get("schema") != LAUNCH_REQUEST_SCHEMA \
            or request.get("version") != VERSION or not hash_matches:
        errors.append("inert launch request schema/hash is invalid")
    phase = request.get("phase")
    action = request.get("action")
    if phase not in INVOCATION_PHASES or action != {
            "prepare": "prepare_start", "encoder": "encoder_start",
            "finalizer": "finalizer_start", "companion": "companion_launch",
    }.get(phase) or not isinstance(request.get("cell_id"), str) \
            or not request.get("cell_id"):
        errors.append("inert launch phase/action/cell is invalid")
    errors.extend(_resource_claim_errors(request.get("resource_claim"), phase=phase))
    if not _valid_sha(request.get("launch_nonce")):
        errors.append("inert launch nonce is invalid")
    generation = request.get("expected_state_generation")
    timeout = request.get("commit_wait_timeout_seconds")
    if isinstance(generation, bool) or not isinstance(generation, int) \
            or generation < 0 or isinstance(timeout, bool) \
            or not isinstance(timeout, (int, float)) \
            or not math.isfinite(float(timeout)) or not 1.0 <= float(timeout) <= 300.0:
        errors.append("inert launch generation/timeout is invalid")
    if request.get("launcher") != _file_reference(INERT_LAUNCHER) \
            or not _regular_reference_matches(request.get("launcher")):
        errors.append("inert launcher source artifact is absent/drifted")
    paths = request.get("paths")
    path_names = {
        "request", "contract", "state", "handshake", "release", "output",
        "semantic_receipt", "worker_receipt", "commit",
        "target_claim_handshake", "target_claim_ack", "target_claim_failure",
        "target_resource_guard",
    }
    if not isinstance(paths, dict) or set(paths) != path_names \
            or len(set(paths.values())) != len(path_names):
        errors.append("inert launch path inventory is invalid")
    else:
        for value in paths.values():
            try:
                if _canonical_stage_path(Path(value)) != value:
                    raise ElasticError("noncanonical")
            except (ElasticError, OSError, TypeError, ValueError):
                errors.append("inert launch path is unconfined/noncanonical")
        expected_commit = str(Path(paths["state"]).with_name(
            f"{Path(paths['state']).name}.commit.{generation + 1}.json"
        )) if isinstance(generation, int) and not isinstance(generation, bool) else None
        if paths.get("commit") != expected_commit:
            errors.append("inert launch commit path/generation binding is invalid")
        if request_reference is not None:
            try:
                current_request = _workspace_artifact_reference(Path(paths["request"]))
            except (ElasticError, OSError, TypeError, ValueError):
                current_request = None
            if current_request != request_reference:
                errors.append("inert launch request artifact binding is invalid")
    errors.extend(_command_binding_errors(request.get("target"), validator=False))
    errors.extend(_command_binding_errors(request.get("validator"), validator=True))
    errors.extend(_claim_protocol_errors(request))
    validator_argv = request.get("validator", {}).get("argv", []) \
        if isinstance(request.get("validator"), dict) else []
    if isinstance(paths, dict) and (
            "--request" not in validator_argv
            or paths.get("request") not in validator_argv
            or "--output" not in validator_argv
            or paths.get("output") not in validator_argv):
        errors.append("phase validator argv lacks exact request/output paths")
    if request.get("required_semantics") != {
            "exact_output": True, "parity_verified": True,
            "zero_skips": True, "skipped_count": 0,
    } or request.get("caller_selected_completion_paths_permitted") is not False \
            or request.get("heavy_work_before_commit_permitted") is not False:
        errors.append("inert launch semantic/completion boundary is invalid")
    return errors


def build_python_module_closure(module: str,
                                sources: Iterable[Path]) -> dict[str, Any]:
    if not isinstance(module, str) \
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
        raise ElasticError("Python module closure name is invalid")
    references = [
        local_observer.stable_artifact_reference(Path(path).resolve(strict=True))
        for path in sources
    ]
    if not references:
        raise ElasticError("Python module closure has no source artifacts")
    value: dict[str, Any] = {
        "schema": PYTHON_MODULE_CLOSURE_SCHEMA, "version": VERSION,
        "module": module,
        "sources": sorted(references, key=lambda row: row["path"]),
    }
    value["closure_sha256"] = _hash_value(value)
    return value


def _python_module_closure_errors(reference: Any, *, module: str) -> list[str]:
    if not _regular_reference_matches(reference):
        return ["Python module closure artifact is absent or drifted"]
    try:
        path = Path(reference["path"])
        path = path if path.is_absolute() else ROOT / path
        workspace_reference = _workspace_artifact_reference(path)
        if workspace_reference != reference:
            raise ElasticError("closure reference differs")
        value = _read_json(path.resolve(strict=True))
        sources = value.get("sources")
        if value.get("schema") != PYTHON_MODULE_CLOSURE_SCHEMA \
                or value.get("version") != VERSION \
                or value.get("module") != module \
                or value.get("closure_sha256") != _hash_value(
                    _without(value, "closure_sha256")
                ) or not isinstance(sources, list) or not sources \
                or any(not _regular_reference_matches(row) for row in sources):
            raise ElasticError("closure content is invalid")
    except (ElasticError, OSError, TypeError, ValueError, KeyError):
        return ["Python module closure content/source binding is invalid"]
    return []


def _python_imports(path: Path) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ElasticError(f"cannot parse target dependency source {path}: {exc}") \
            from exc
    imports: set[tuple[str, int]] = set()
    forbidden_modules = {
        "ctypes", "importlib", "pkgutil", "runpy", "site", "subprocess",
        "zipimport",
    }
    forbidden_calls = {
        "__import__", "compile", "eval", "exec", "getattr", "globals",
        "locals", "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                raise ElasticError(
                    f"target dependency source uses wildcard import: {path}"
                )
            module = node.module or ""
            imports.add((module, int(node.level or 0)))
            if not module:
                imports.update((alias.name, int(node.level or 0))
                               for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in forbidden_calls:
                raise ElasticError(
                    f"target dependency source uses dynamic code/import call {name}: {path}"
                )
            if isinstance(node.func, ast.Attribute):
                root = node.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in forbidden_modules:
                    raise ElasticError(
                        f"target dependency source uses dynamic loader module "
                        f"{root.id}: {path}"
                    )
    if any(module.split(".", 1)[0] in forbidden_modules
           for module, _level in imports):
        raise ElasticError(
            f"target dependency source imports forbidden loader/subprocess module: {path}"
        )
    return sorted(imports)


def _workspace_import_path(source: Path, module: str, level: int) -> Path | None:
    components = [row for row in module.split(".") if row]
    bases: list[Path] = []
    if level:
        base = source.parent
        for _ in range(max(0, level - 1)):
            base = base.parent
        bases.append(base)
    else:
        bases.extend((source.parent, ROOT.resolve(strict=True)))
    for base in bases:
        candidate = base.joinpath(*components) if components else base
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(ROOT.resolve(strict=True))
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
    return None


def _runtime_module_binding(module: str) -> dict[str, Any]:
    top = module.split(".", 1)[0]
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ElasticError(f"cannot resolve standard-library import {top}: {exc}") \
            from exc
    if spec is None:
        raise ElasticError(f"cannot resolve standard-library import: {top}")
    origin = spec.origin
    value: dict[str, Any] = {"module": top, "origin": origin}
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        path = Path(origin).resolve(strict=True)
        value["reference"] = local_observer.stable_artifact_reference(path)
    else:
        value["reference"] = None
    value["binding_sha256"] = _hash_value(value)
    return value


def build_target_dependency_closure(entrypoint: Path, *, interpreter: Path) \
        -> dict[str, Any]:
    """Inventory direct/workspace AST imports for a structural Python target.

    This is deliberately not production qualification: Python/native transitive
    runtime closure is not proven here.  The production protocol remains
    structural-only via ``_closed_target_runtime_errors``.
    """
    entry = Path(entrypoint).resolve(strict=True)
    entry.relative_to(ROOT.resolve(strict=True))
    python = Path(interpreter).resolve(strict=True)
    observed_executable, _, _ = local_observer._darwin_procargs(os.getpid())
    observed_python = Path(observed_executable).resolve(strict=True)
    reviewed_interpreters = {
        Path(sys.executable).resolve(strict=True), observed_python,
    }
    if python not in reviewed_interpreters:
        raise ElasticError("target closure interpreter is not the reviewed runtime")
    pending = [entry]
    seen: set[Path] = set()
    import_rows: list[dict[str, Any]] = []
    stdlib_modules: set[str] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        imports = _python_imports(source)
        import_rows.append({
            "source": _workspace_artifact_reference(source),
            "imports": [
                {"module": module, "level": level}
                for module, level in imports
            ],
        })
        for module, level in imports:
            workspace = _workspace_import_path(source, module, level)
            if workspace is not None:
                pending.append(workspace)
                continue
            top = module.split(".", 1)[0] if module else ""
            if level or not top or top not in sys.stdlib_module_names:
                raise ElasticError(
                    f"target import is neither workspace-bound nor stdlib: {module}"
                )
            stdlib_modules.add(top)
    runtime_artifacts = [
        local_observer.stable_artifact_reference(path)
        for path in reviewed_interpreters
    ]
    libdir, library = (
        sysconfig.get_config_var("LIBDIR"),
        sysconfig.get_config_var("LDLIBRARY"),
    )
    if isinstance(libdir, str) and isinstance(library, str):
        libpython = Path(libdir) / library
        if libpython.is_file():
            runtime_artifacts.append(
                local_observer.stable_artifact_reference(libpython.resolve(strict=True))
            )
    value: dict[str, Any] = {
        "schema": TARGET_DEPENDENCY_CLOSURE_SCHEMA, "version": VERSION,
        "entrypoint": _workspace_artifact_reference(entry),
        "workspace_sources": sorted(
            (_workspace_artifact_reference(path) for path in seen),
            key=lambda row: row["path"],
        ),
        "machine_derived_imports": sorted(
            import_rows, key=lambda row: row["source"]["path"]
        ),
        "stdlib_modules": sorted(
            (_runtime_module_binding(module) for module in stdlib_modules),
            key=lambda row: row["module"],
        ),
        "runtime_artifacts": sorted(runtime_artifacts, key=lambda row: row["path"]),
        "isolated_argv_prefix": [str(python), "-I", "-S"],
        "inventory_complete": False,
        "qualification_authority": False,
        "system_site_packages_excluded": False,
        "native_dependency_closure_proven": False,
        "forbidden_environment_keys": [
            "PYTHONHOME", "PYTHONPATH", "DYLD_FRAMEWORK_PATH",
            "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
        ],
    }
    value["closure_sha256"] = _hash_value(value)
    return value


def _target_dependency_closure_errors(reference: Any, *, command: dict[str, Any]) \
        -> list[str]:
    if not _regular_reference_matches(reference):
        return ["target dependency closure is absent or drifted"]
    try:
        raw = Path(reference["path"])
        path = raw if raw.is_absolute() else ROOT / raw
        closure = _read_json(path.resolve(strict=True))
        argv = command["argv"]
        if closure.get("schema") != TARGET_DEPENDENCY_CLOSURE_SCHEMA \
                or closure.get("version") != VERSION \
                or closure.get("closure_sha256") != _hash_value(
                    _without(closure, "closure_sha256")
                ) or not isinstance(argv, list) or len(argv) < 4 \
                or closure.get("inventory_complete") is not False \
                or closure.get("qualification_authority") is not False \
                or closure.get("system_site_packages_excluded") is not False \
                or closure.get("native_dependency_closure_proven") is not False \
                or argv[:3] != closure.get("isolated_argv_prefix"):
            raise ElasticError("target dependency closure identity differs")
        entry_raw = Path(closure.get("entrypoint", {}).get("path", ""))
        entry_path = entry_raw if entry_raw.is_absolute() else ROOT / entry_raw
        if entry_path.resolve(strict=True) != Path(argv[3]).resolve(strict=True):
            raise ElasticError("target dependency closure entrypoint differs")
        current = build_target_dependency_closure(
            Path(argv[3]), interpreter=Path(argv[0])
        )
        if current != closure:
            raise ElasticError("target dependency closure is incomplete or stale")
    except (ElasticError, OSError, TypeError, ValueError, KeyError):
        return ["target dependency closure content/import/runtime binding is invalid"]
    return []


def _entry_production_protocol(entry: dict[str, Any]) -> str:
    artifacts = entry.get("argv_artifacts")
    template = entry.get("argv_template")
    if not isinstance(artifacts, list) or not isinstance(template, list):
        return "structural-only"
    request_rows = [
        row for row in artifacts if isinstance(row, dict)
        and row.get("role") == "request"
    ]
    script_rows = [
        row for row in artifacts if isinstance(row, dict)
        and row.get("role") == "script"
    ]
    executable = entry.get("executable")
    if len(request_rows) != 1 or len(script_rows) != 1 \
            or not isinstance(executable, dict) \
            or "python" not in Path(str(executable.get("path", ""))).name.lower() \
            or script_rows[0].get("position") != 1 \
            or request_rows[0].get("position") != 3 \
            or script_rows[0].get("reference") != _file_reference(INERT_LAUNCHER):
        return "structural-only"
    request_row = request_rows[0]
    try:
        request_path = Path(request_row["argv_value"])
        request_path = request_path if request_path.is_absolute() \
            else Path(entry["cwd"]) / request_path
        request = _read_json(request_path.resolve(strict=True))
    except (OSError, TypeError, ValueError, KeyError, ElasticError):
        return "structural-only"
    if validate_inert_launch_request(
            request, request_reference=request_row.get("reference")):
        return "structural-only"
    if _closed_target_runtime_errors(request):
        return "structural-only"
    literals = [row.get("literal") if isinstance(row, dict) else None
                for row in template]
    if literals != [
            executable.get("path"), str(INERT_LAUNCHER), "--request",
            request["paths"]["request"],
    ]:
        return "structural-only"
    return "inert-commit-v1-qualified"


def _closed_target_runtime_errors(request: dict[str, Any]) -> list[str]:
    """Return unresolved production proofs; never infer them from an echoed claim.

    The staged two-phase gate, process-group cleanup, authoritative thread env,
    and RSS monitor are useful scaffolding.  They are not yet a closed Python
    runtime or a measured thread-count proof, so production qualification stays
    fail-closed.  Tests may mock this helper to exercise the otherwise inert
    receipt chain; no runtime/default path can bypass it.
    """
    target = request.get("target", {})
    executable = target.get("executable", {}) if isinstance(target, dict) else {}
    path = Path(str(executable.get("path", ""))).name.lower() \
        if isinstance(executable, dict) else ""
    errors: list[str] = []
    if "python" in path:
        errors.append("reviewed closed Python target runtime is absent")
    reservation = request.get("resource_claim", {}).get("reservation_bytes")
    if isinstance(reservation, bool) or not isinstance(reservation, int) \
            or reservation <= 0:
        errors.append("mandatory admitted RAM ceiling is absent for target phase")
    errors.append("machine-enforced measured target thread-count proof is absent")
    return errors


def build_invocation_entry(*, phase: str, executable: dict[str, Any],
                           argv_template: list[dict[str, str]],
                           allowed_substitutions: dict[str, list[str]],
                           cwd: str,
                           environment_allowlist: dict[str, list[str]],
                           argv_artifacts: list[dict[str, Any]] | None = None,
                           semantic_artifacts: list[dict[str, Any]] | None = None) \
        -> dict[str, Any]:
    """Build one exact phase launcher allowlist entry."""
    value: dict[str, Any] = {
        "schema": INVOCATION_ENTRY_SCHEMA, "version": VERSION,
        "phase": phase, "executable": copy.deepcopy(executable),
        "argv_template": copy.deepcopy(argv_template),
        "allowed_substitutions": copy.deepcopy(allowed_substitutions),
        "argv_artifacts": copy.deepcopy(argv_artifacts or []),
        "semantic_artifacts": copy.deepcopy(semantic_artifacts or []),
        "cwd": cwd,
        "environment_allowlist": copy.deepcopy(environment_allowlist),
        "control_environment_prefixes": list(CONTROL_ENVIRONMENT_PREFIXES),
        "unlisted_environment_forbidden": True,
        "environment_is_exact": True,
        "worker_completion_schema": WORKER_COMPLETION_SCHEMA,
        "phase_output_receipt_schema": PHASE_OUTPUT_SCHEMA,
        "stable_output_hash_required": True,
        "inert_precommit_handshake_required": True,
        "production_execution_protocol": "structural-only",
        "nearest_or_fallback_phase_permitted": False,
    }
    value["production_execution_protocol"] = _entry_production_protocol(value)
    value["entry_sha256"] = _hash_value(value)
    errors = _invocation_entry_errors(value)
    if errors:
        raise ElasticError("invalid invocation entry: " + "; ".join(errors))
    return value


def build_invocation_entry_from_observation(*, phase: str,
                                            observation: dict[str, Any],
                                            substitution_positions: dict[int, str]
                                            | None = None,
                                            allowed_substitutions:
                                            dict[str, list[str]] | None = None,
                                            environment_allowlist:
                                            dict[str, list[str]] | None = None,
                                            argv_artifact_positions:
                                            dict[int, str] | None = None,
                                            semantic_artifact_paths:
                                            dict[str, Path] | None = None) \
        -> dict[str, Any]:
    """Freeze a directly observed fixture invocation into a manifest entry."""
    if not isinstance(observation, dict) \
            or observation.get("schema") \
            != local_observer.INVOCATION_OBSERVATION_SCHEMA \
            or observation.get("invocation_observation_sha256") != _hash_value(
                _without(observation, "invocation_observation_sha256")
            ):
        raise ElasticError("cannot freeze an invalid invocation observation")
    argv = observation.get("argv")
    if not isinstance(argv, list) or any(not isinstance(row, str) for row in argv):
        raise ElasticError("observed invocation argv is invalid")
    environment_hashes = observation.get("environment_value_sha256s")
    if not isinstance(environment_hashes, dict) \
            or any(not isinstance(key, str) or not _valid_sha(value)
                   for key, value in environment_hashes.items()):
        raise ElasticError("observed invocation environment hashes are invalid")
    positions = substitution_positions or {}
    if any(isinstance(index, bool) or not isinstance(index, int)
           or index < 0 or index >= len(argv) or not isinstance(name, str)
           or not name for index, name in positions.items()):
        raise ElasticError("invocation substitution positions are invalid")
    template = [
        {"substitution": positions[index]} if index in positions
        else {"literal": token}
        for index, token in enumerate(argv)
    ]
    cwd_path = Path(observation["cwd"])
    explicit_positions = argv_artifact_positions or {}
    auto_positions: dict[int, str] = {}
    for index, token in enumerate(argv[1:], start=1):
        candidate = Path(token)
        resolved_candidate = candidate if candidate.is_absolute() else cwd_path / candidate
        if resolved_candidate.is_file():
            if index > 0 and argv[index - 1] in {"--request", "--request-path"}:
                role = "request"
            elif candidate.suffix.lower() in {".json", ".yaml", ".yml", ".toml"}:
                role = "config"
            elif candidate.suffix.lower() in {".py", ".pyw"}:
                role = "script"
            else:
                role = "semantic"
            auto_positions[index] = role
    positions_to_bind = {**auto_positions, **explicit_positions}
    argv_artifacts = []
    for index, role in sorted(positions_to_bind.items()):
        if isinstance(index, bool) or not isinstance(index, int) \
                or index <= 0 or index >= len(argv) \
                or role not in {"script", "module", "request", "config", "semantic"}:
            raise ElasticError("invocation argv artifact position/role is invalid")
        argv_artifacts.append({
            "position": index, "argv_value": argv[index], "role": role,
            "reference": _workspace_artifact_reference(
                Path(argv[index]), cwd=cwd_path
            ),
        })
    semantic_artifacts = []
    for name, path in sorted((semantic_artifact_paths or {}).items()):
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", name):
            raise ElasticError("invocation semantic artifact name is invalid")
        semantic_artifacts.append({
            "name": name,
            "reference": _workspace_artifact_reference(Path(path), cwd=cwd_path),
        })
    return build_invocation_entry(
        phase=phase, executable=observation["executable"],
        argv_template=template,
        allowed_substitutions=allowed_substitutions or {},
        cwd=observation["cwd"],
        environment_allowlist=(
            environment_allowlist if environment_allowlist is not None else {
                key: [value]
                for key, value in environment_hashes.items()
            }
        ),
        argv_artifacts=argv_artifacts,
        semantic_artifacts=semantic_artifacts,
    )


def _invocation_entry_errors(entry: Any) -> list[str]:
    required = {
        "schema", "version", "phase", "executable", "argv_template",
        "allowed_substitutions", "argv_artifacts", "semantic_artifacts",
        "cwd", "environment_allowlist",
        "control_environment_prefixes", "unlisted_environment_forbidden",
        "environment_is_exact", "worker_completion_schema",
        "phase_output_receipt_schema", "stable_output_hash_required",
        "inert_precommit_handshake_required", "production_execution_protocol",
        "nearest_or_fallback_phase_permitted", "entry_sha256",
    }
    if not isinstance(entry, dict) or set(entry) != required:
        return ["invocation entry keys are invalid"]
    errors: list[str] = []
    try:
        hash_matches = entry.get("entry_sha256") == _hash_value(
            _without(entry, "entry_sha256")
        )
    except (TypeError, ValueError):
        hash_matches = False
    if entry.get("schema") != INVOCATION_ENTRY_SCHEMA \
            or entry.get("version") != VERSION or not hash_matches:
        errors.append("invocation entry schema/hash is invalid")
    if entry.get("phase") not in INVOCATION_PHASES:
        errors.append("invocation entry phase is invalid")
    executable = entry.get("executable")
    if not _regular_reference_matches(executable):
        errors.append("invocation executable artifact is absent or drifted")
    template = entry.get("argv_template")
    substitutions = entry.get("allowed_substitutions")
    names: list[str] = []
    if not isinstance(template, list) or not template:
        errors.append("invocation argv template is empty")
    else:
        for token in template:
            if not isinstance(token, dict) or len(token) != 1 \
                    or set(token) not in ({"literal"}, {"substitution"}):
                errors.append("invocation argv token is invalid")
                continue
            key, value = next(iter(token.items()))
            if not isinstance(value, str) or not value:
                errors.append("invocation argv token value is invalid")
            elif key == "substitution":
                names.append(value)
        first = template[0] if template else None
        executable_path = executable.get("path") if isinstance(executable, dict) else None
        if not isinstance(first, dict) or first != {"literal": executable_path}:
            errors.append("invocation argv[0] is not the frozen executable path")
    if not isinstance(substitutions, dict):
        errors.append("invocation allowed substitutions are invalid")
    else:
        valid_names = all(isinstance(name, str) for name in substitutions)
        if not valid_names or set(names) != set(substitutions):
            errors.append("invocation template substitutions differ from allowlist")
        for name, values in substitutions.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) \
                    or not isinstance(values, list) or not values \
                    or any(not isinstance(row, str) or not row for row in values) \
                    or values != sorted(set(values)):
                errors.append("invocation substitution values are invalid")
    cwd = entry.get("cwd")
    try:
        cwd_path = Path(cwd).resolve(strict=True) if isinstance(cwd, str) else None
        if cwd_path is None or not cwd_path.is_dir():
            raise ValueError
        cwd_path.relative_to(ROOT.resolve(strict=True))
        if str(cwd_path) != cwd:
            raise ValueError
    except (OSError, TypeError, ValueError):
        errors.append("invocation cwd is not one canonical workspace directory")
        cwd_path = None
    argv_artifacts = entry.get("argv_artifacts")
    artifact_positions: set[int] = set()
    artifact_roles: dict[int, str] = {}
    if not isinstance(argv_artifacts, list):
        errors.append("invocation argv artifact inventory is invalid")
    else:
        for artifact in argv_artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                    "position", "argv_value", "role", "reference"}:
                errors.append("invocation argv artifact row is invalid")
                continue
            position, argv_value, role = (
                artifact.get("position"), artifact.get("argv_value"),
                artifact.get("role"),
            )
            if isinstance(position, bool) or not isinstance(position, int) \
                    or position <= 0 or position in artifact_positions \
                    or not isinstance(template, list) or position >= len(template) \
                    or not isinstance(argv_value, str) or not argv_value \
                    or role not in {
                        "script", "module", "request", "config", "semantic"
                    }:
                errors.append("invocation argv artifact position/role is invalid")
                continue
            artifact_positions.add(position)
            artifact_roles[position] = role
            if template[position] != {"literal": argv_value}:
                errors.append("invocation argv artifact is not a frozen literal")
            reference = artifact.get("reference")
            if not _regular_reference_matches(reference):
                errors.append("invocation argv artifact is absent or drifted")
                continue
            try:
                current = _workspace_artifact_reference(
                    Path(argv_value), cwd=cwd_path
                )
            except (ElasticError, OSError, TypeError, ValueError):
                current = None
            if current != reference:
                errors.append("invocation argv artifact path/hash binding is invalid")
    semantic_artifacts = entry.get("semantic_artifacts")
    semantic_names: set[str] = set()
    semantic_by_name: dict[str, Any] = {}
    if not isinstance(semantic_artifacts, list):
        errors.append("invocation semantic artifact inventory is invalid")
    else:
        for artifact in semantic_artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                    "name", "reference"}:
                errors.append("invocation semantic artifact row is invalid")
                continue
            name, reference = artifact.get("name"), artifact.get("reference")
            if not isinstance(name, str) \
                    or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", name) \
                    or name in semantic_names:
                errors.append("invocation semantic artifact name is invalid")
            else:
                semantic_names.add(name)
                semantic_by_name[name] = reference
            if not _regular_reference_matches(reference):
                errors.append("invocation semantic artifact is absent or drifted")
            else:
                try:
                    path = Path(reference["path"])
                    path = path if path.is_absolute() else ROOT / path
                    if _workspace_artifact_reference(path) != reference:
                        raise ElasticError("semantic reference differs")
                except (ElasticError, OSError, TypeError, ValueError):
                    errors.append(
                        "invocation semantic artifact escaped workspace/path binding"
                    )
    if isinstance(template, list) and template \
            and isinstance(executable, dict) \
            and "python" in Path(str(executable.get("path", ""))).name.lower():
        argv_literals = [
            token.get("literal") if isinstance(token, dict) else None
            for token in template
        ]
        if len(argv_literals) < 2 or argv_literals[1] == "-":
            errors.append("Python invocation has no source-bound program")
        elif argv_literals[1] == "-m":
            module = argv_literals[2] if len(argv_literals) >= 3 else None
            closure_name = f"module.{module}" if module else None
            if not module or closure_name not in semantic_by_name:
                errors.append("Python module invocation lacks semantic artifact binding")
            else:
                errors.extend(_python_module_closure_errors(
                    semantic_by_name[closure_name], module=module
                ))
        elif argv_literals[1] != "-c" \
                and artifact_roles.get(1) not in {"script", "module"}:
            errors.append("Python script invocation lacks argv source binding")
    if isinstance(template, list):
        request_flags = {"--request", "--request-path"}
        config_flags = {"--config", "--config-path"}
        for index, token in enumerate(template[:-1]):
            literal = token.get("literal") if isinstance(token, dict) else None
            if literal in request_flags and artifact_roles.get(index + 1) != "request":
                errors.append("invocation request argument lacks artifact binding")
            if literal in config_flags and artifact_roles.get(index + 1) != "config":
                errors.append("invocation config argument lacks artifact binding")
    environment = entry.get("environment_allowlist")
    if not isinstance(environment, dict):
        errors.append("invocation environment allowlist is invalid")
    else:
        for key, hashes in environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) \
                    or not isinstance(hashes, list) or not hashes \
                    or any(not _valid_sha(row) for row in hashes) \
                    or hashes != sorted(set(hashes)):
                errors.append("invocation environment key/value hashes are invalid")
    if entry.get("control_environment_prefixes") \
            != list(CONTROL_ENVIRONMENT_PREFIXES) \
            or entry.get("unlisted_environment_forbidden") is not True \
            or entry.get("environment_is_exact") is not True:
        errors.append("invocation environment authority boundary is invalid")
    if entry.get("worker_completion_schema") != WORKER_COMPLETION_SCHEMA \
            or entry.get("phase_output_receipt_schema") != PHASE_OUTPUT_SCHEMA \
            or entry.get("stable_output_hash_required") is not True \
            or entry.get("nearest_or_fallback_phase_permitted") is not False:
        errors.append("invocation worker/output/no-fallback contract is invalid")
    if entry.get("inert_precommit_handshake_required") is not True \
            or entry.get("production_execution_protocol") \
            != _entry_production_protocol(entry):
        errors.append("invocation inert pre-commit protocol binding is invalid")
    return errors


def build_invocation_manifest(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    copied = [copy.deepcopy(row) for row in entries]
    errors = [error for row in copied for error in _invocation_entry_errors(row)]
    if errors:
        raise ElasticError("cannot build invalid invocation manifest: "
                           + "; ".join(errors))
    copied.sort(key=lambda row: (row["phase"], row["entry_sha256"]))
    qualified_phases = sorted({
        row["phase"] for row in copied
        if row.get("production_execution_protocol") == "inert-commit-v1-qualified"
    })
    missing = [phase for phase in INVOCATION_PHASES if phase not in qualified_phases]
    blockers = [
        f"phase invocation has no inert-commit-v1-qualified entry: {phase}"
        for phase in missing
    ]
    if missing:
        blockers.extend([
            "reviewed closed target runtime proof is absent",
            "mandatory admitted RAM ceiling is absent for one or more heavy phases",
            "machine-enforced measured target thread-count proof is absent",
        ])
    manifest: dict[str, Any] = {
        "schema": INVOCATION_MANIFEST_SCHEMA, "version": VERSION,
        "mode": "unbound-default-off", "enabled_by_default": False,
        "status": "qualified" if not blockers else "blocked",
        "blockers": blockers, "required_phases": list(INVOCATION_PHASES),
        "entries": copied, "entry_sha256s": [row["entry_sha256"] for row in copied],
        "nearest_or_fallback_phase_permitted": False,
        "caller_declared_command_authority": False,
    }
    manifest["manifest_sha256"] = _hash_value(manifest)
    return manifest


def validate_invocation_manifest(manifest: Any) -> list[str]:
    if not isinstance(manifest, dict):
        return ["phase invocation manifest is not an object"]
    errors: list[str] = []
    required = {
        "schema", "version", "mode", "enabled_by_default", "status",
        "blockers", "required_phases", "entries", "entry_sha256s",
        "nearest_or_fallback_phase_permitted",
        "caller_declared_command_authority", "manifest_sha256",
    }
    try:
        hash_matches = manifest.get("manifest_sha256") == _hash_value(
            _without(manifest, "manifest_sha256")
        )
    except (TypeError, ValueError):
        hash_matches = False
    if set(manifest) != required:
        errors.append("phase invocation manifest keys are invalid")
    if manifest.get("schema") != INVOCATION_MANIFEST_SCHEMA \
            or manifest.get("version") != VERSION \
            or manifest.get("mode") != "unbound-default-off" \
            or manifest.get("enabled_by_default") is not False \
            or not hash_matches:
        errors.append("phase invocation manifest schema/hash/default is invalid")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return errors + ["phase invocation manifest entries are invalid"]
    entry_errors = [error for row in entries for error in _invocation_entry_errors(row)]
    errors.extend(entry_errors)
    sortable = all(
        isinstance(row, dict) and isinstance(row.get("phase"), str)
        and isinstance(row.get("entry_sha256"), str) for row in entries
    )
    sorted_entries = sorted(
        entries, key=lambda row: (row["phase"], row["entry_sha256"])
    ) if sortable else []
    hashes = [row.get("entry_sha256") for row in sorted_entries]
    qualified_phases = sorted({
        row.get("phase") for row in sorted_entries
        if isinstance(row.get("phase"), str)
        and row.get("production_execution_protocol")
        == "inert-commit-v1-qualified"
    })
    missing = [phase for phase in INVOCATION_PHASES if phase not in qualified_phases]
    expected_blockers = [
        f"phase invocation has no inert-commit-v1-qualified entry: {phase}"
        for phase in missing
    ]
    if missing:
        expected_blockers.extend([
            "reviewed closed target runtime proof is absent",
            "mandatory admitted RAM ceiling is absent for one or more heavy phases",
            "machine-enforced measured target thread-count proof is absent",
        ])
    if entries != sorted_entries or hashes != manifest.get("entry_sha256s") \
            or len(hashes) != len(set(hashes)):
        errors.append("phase invocation entry inventory/order is invalid")
    if manifest.get("required_phases") != list(INVOCATION_PHASES) \
            or manifest.get("blockers") != expected_blockers \
            or manifest.get("status") != ("qualified" if not missing else "blocked"):
        errors.append("phase invocation qualification/blockers are invalid")
    if manifest.get("nearest_or_fallback_phase_permitted") is not False \
            or manifest.get("caller_declared_command_authority") is not False:
        errors.append("phase invocation manifest permits fallback/caller authority")
    return errors


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_LENGTH \
        and all(character in "0123456789abcdef" for character in value)


def _valid_epoch(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(float(value)) and float(value) >= 0


def _artifact_identity(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"sha256", "bytes"} \
        and _valid_sha(value.get("sha256")) \
        and not isinstance(value.get("bytes"), bool) \
        and isinstance(value.get("bytes"), int) and value["bytes"] >= 0


def _safe_validate_swap_state(state: Any, baseline: Any) -> list[str]:
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or float(baseline) < 0:
        return ["sealed swap baseline is invalid"]
    try:
        return aggressive.validate_swap_state(
            state, sealed_baseline_swap_mb=float(baseline)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return ["swap state is malformed"]


def build_process_identity(*, pid: int, start_identity: str,
                           command_sha256: str) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 \
            or not isinstance(start_identity, str) or not start_identity \
            or len(start_identity) > 256 or not _valid_sha(command_sha256):
        raise ElasticError("PID/start/command process identity is invalid")
    identity: dict[str, Any] = {
        "schema": PROCESS_IDENTITY_SCHEMA, "version": VERSION,
        "pid": pid, "start_identity": start_identity,
        "command_sha256": command_sha256,
    }
    identity["process_identity_sha256"] = _hash_value(identity)
    return identity


def _process_identity_errors(value: Any) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
            "schema", "version", "pid", "start_identity", "command_sha256",
            "process_identity_sha256",
    } or value.get("schema") != PROCESS_IDENTITY_SCHEMA \
            or value.get("version") != VERSION \
            or value.get("process_identity_sha256") != _hash_value(
                _without(value, "process_identity_sha256")
            ):
        return ["process identity schema/hash is invalid"]
    try:
        expected = build_process_identity(
            pid=value["pid"], start_identity=value["start_identity"],
            command_sha256=value["command_sha256"],
        )
    except (KeyError, ElasticError) as exc:
        return [str(exc)]
    return [] if expected == value else ["process identity differs after reconstruction"]


def _observer_contract_sha256() -> str:
    return _hash_value({
        "schema": EXIT_OBSERVATION_SCHEMA,
        "method": "pid-start-command-exact-negative-membership",
        "requires_fresh_wall_epoch": True,
        "live_exact_identity_blocks_release": True,
    })


def _signed_host_topology(host_probe: Any) -> tuple[dict[str, int] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(host_probe, dict) or host_probe.get("probe_sha256") \
            != _hash_value(_without(host_probe, "probe_sha256")):
        return None, ["signed host probe is absent or invalid"]
    topology = host_probe.get("topology")
    if not isinstance(topology, dict) or topology.get("topology_sha256") \
            != _hash_value(_without(topology, "topology_sha256")):
        return None, ["signed host topology is absent or invalid"]
    if topology.get("verified_for_doctor_v5") is not True:
        errors.append("host topology is not verified for Doctor V5")
    for field, expected in REVIEWED_TOPOLOGY.items():
        if topology.get(field) != expected:
            errors.append(f"host topology {field} is not reviewed value {expected}")
    if all(isinstance(topology.get(field), int) for field in (
            "performance_cores", "efficiency_cores", "physical_cores")) \
            and topology["performance_cores"] + topology["efficiency_cores"] \
            != topology["physical_cores"]:
        errors.append("host performance/efficiency topology is inconsistent")
    if topology.get("source") != "read-only-sysctl":
        errors.append("host topology is not derived from the read-only sysctl probe")
    receipts = topology.get("sysctl_receipts")
    if not isinstance(receipts, dict):
        errors.append("host topology sysctl receipts are missing")
    else:
        for field, key in TOPOLOGY_SYSCTLS.items():
            row = receipts.get(field)
            argv = row.get("argv") if isinstance(row, dict) else None
            if not isinstance(row, dict) or row.get("returncode") != 0 \
                    or row.get("timed_out") is not False \
                    or not isinstance(argv, list) or argv[-2:] != ["-n", key] \
                    or str(row.get("output", "")).strip() \
                    != str(REVIEWED_TOPOLOGY[field]):
                errors.append(f"host topology sysctl receipt is invalid: {field}")
    if errors:
        return None, errors
    return {field: topology[field] for field in REVIEWED_TOPOLOGY}, []


def build_contract(aggressive_overlay: dict[str, Any], *,
                   aggressive_overlay_path: Path | None = None,
                   host_plan_reference: dict[str, Any] | None = None,
                   host_probe: dict[str, Any] | None = None,
                   invocation_manifest_path: Path | None = None) -> dict[str, Any]:
    overlay_errors = aggressive.validate_overlay(aggressive_overlay)
    qualification = aggressive_overlay.get("thread_profile_qualification")
    blockers = [f"aggressive overlay: {row}" for row in overlay_errors]
    invocation_manifest = build_invocation_manifest([])
    invocation_manifest_reference = None
    if invocation_manifest_path is None:
        blockers.append("source-bound phase invocation manifest is absent")
    else:
        try:
            invocation_manifest, invocation_manifest_reference = \
                _bound_workspace_json(invocation_manifest_path)
            invocation_errors = validate_invocation_manifest(invocation_manifest)
            blockers.extend(
                f"phase invocation manifest: {row}" for row in invocation_errors
            )
            if invocation_manifest.get("status") != "qualified":
                blockers.extend(invocation_manifest.get("blockers", [
                    "phase invocation manifest is not qualified"
                ]))
        except (OSError, ElasticError) as exc:
            blockers.append(f"phase invocation manifest binding failed: {exc}")
    if not isinstance(qualification, dict) \
            or qualification.get("qualification_sha256") != aggressive._hash_value(
                aggressive._without(qualification, "qualification_sha256")
            ) or qualification.get("status") != "qualified":
        blockers.append("qualified exact vendor thread profile is absent")
        selections: dict[str, Any] = {}
        qualification_sha = None
    else:
        selections = qualification.get("selections", {})
        qualification_sha = qualification["qualification_sha256"]
        if not isinstance(selections, dict) or not selections:
            blockers.append("qualified vendor thread profile has no selections")
        else:
            try:
                profile_ref, binary_ref = qualification["profile"], qualification["binary"]
                if not aggressive._reference_matches(profile_ref) \
                        or not aggressive._reference_matches(binary_ref):
                    raise ElasticError("thread profile/binary binding changed or escaped workspace")
                profile_path = (ROOT / profile_ref["path"]).resolve(strict=True)
                binary_path = (ROOT / binary_ref["path"]).resolve(strict=True)
                identity_cells = [
                    {"model_label": row["tier"], "rate_id": row["rate"]}
                    for row in selections.values() if isinstance(row, dict)
                ]
                current = aggressive.qualify_thread_profile(
                    identity_cells, profile_path=profile_path, binary_path=binary_path,
                )
                if current != qualification:
                    raise ElasticError("vendor thread profile differs from qualification")
            except (KeyError, OSError, TypeError, ValueError, ElasticError) as exc:
                blockers.append(f"thread profile revalidation failed: {exc}")
    baseline = aggressive_overlay.get("resource_policy", {}).get(
        "sealed_swap_baseline_mb"
    )
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or baseline < 0:
        blockers.append("aggressive swap baseline is invalid")
    if aggressive_overlay_path is None:
        blockers.append("aggressive overlay file binding is absent")
    else:
        try:
            overlay_reference = _file_reference(aggressive_overlay_path)
            if _read_json(aggressive_overlay_path) != aggressive_overlay:
                blockers.append("aggressive overlay argument differs from bound file")
        except (OSError, ElasticError) as exc:
            blockers.append(f"aggressive overlay file binding failed: {exc}")
            overlay_reference = None
    topology, topology_errors = _signed_host_topology(host_probe)
    blockers.extend(topology_errors)
    if not aggressive._reference_matches(host_plan_reference):
        blockers.append("host sprint plan file binding is absent or stale")
    else:
        try:
            host_plan = _read_json((ROOT / host_plan_reference["path"]).resolve(strict=True))
            host_bindings = host_plan.get("source_bindings", {})
            if host_plan.get("plan_sha256") != _hash_value(
                    _without(host_plan, "plan_sha256")
            ) or host_plan.get("enabled_by_default") is not False \
                    or host_plan.get("automatic_execution_permitted") is not False \
                    or host_plan.get("probe") != host_probe \
                    or host_bindings.get("local_observer") \
                    != _file_reference(Path(local_observer.__file__)) \
                    or host_bindings.get("local_observer_authority_tools") \
                    != local_observer.authority_tool_references():
                blockers.append("host sprint plan/probe binding is invalid")
        except (KeyError, OSError, ElasticError) as exc:
            blockers.append(f"host sprint plan binding failed: {exc}")
    topology = topology or {field: None for field in REVIEWED_TOPOLOGY}
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA, "version": VERSION,
        "created_at": _now(), "mode": "unbound-default-off",
        "enabled_by_default": False,
        "status": "qualified" if not blockers else "blocked",
        "blockers": blockers,
        "source_bindings": {
            "elastic_scheduler": _file_reference(Path(__file__)),
            "local_observer": _file_reference(Path(local_observer.__file__)),
            "local_observer_authority_tools": (
                local_observer.authority_tool_references()
            ),
            "phase_invocation_manifest": invocation_manifest_reference,
            "aggressive_policy": _file_reference(Path(aggressive.__file__)),
            "aggressive_overlay": (
                overlay_reference if aggressive_overlay_path is not None else None
            ),
            "vendor_thread_contract": (
                qualification.get("contract") if isinstance(qualification, dict) else None
            ),
            "thread_profile": (
                qualification.get("profile") if isinstance(qualification, dict) else None
            ),
            "thread_binary": (
                qualification.get("binary") if isinstance(qualification, dict) else None
            ),
            "host_sprint_plan": host_plan_reference,
            "host_probe_sha256": (
                host_probe.get("probe_sha256") if isinstance(host_probe, dict) else None
            ),
            "host_topology_sha256": (
                host_probe.get("topology", {}).get("topology_sha256")
                if isinstance(host_probe, dict) else None
            ),
        },
        "aggressive_overlay_sha256": aggressive_overlay.get("overlay_sha256"),
        "phase_invocation_manifest_sha256": invocation_manifest.get(
            "manifest_sha256"
        ),
        "invocation_policy": {
            "required_phases": list(INVOCATION_PHASES),
            "entries": copy.deepcopy(invocation_manifest.get("entries", [])),
            "entry_sha256s": copy.deepcopy(
                invocation_manifest.get("entry_sha256s", [])
            ),
            "nearest_or_fallback_phase_permitted": False,
            "caller_declared_command_authority": False,
            "direct_lock_scoped_observation_required": True,
        },
        "thread_profile_qualification_sha256": qualification_sha,
        "thread_selections": selections,
        "swap": {
            "sealed_baseline_mb": baseline,
            "policy": aggressive.swap_policy(),
            "requires_hash_bound_aggressive_state": True,
            "companion_launch_requires_green": True,
        },
        "phase_policy": {
            "maximum_heavy_prepare": 1,
            "maximum_primary_encoder": 1,
            "maximum_serial_finalizer": 1,
            "maximum_companion": 1,
            "prepare_and_encoder_mutually_exclusive": True,
            "prepare_and_finalizer_may_overlap_if_measured_envelope_fits": True,
            "companion_uses_only_contract_selected_threads": True,
            "idle_sample_count": MIN_IDLE_SAMPLES,
            "minimum_sample_spacing_seconds": MIN_SAMPLE_SPACING_SECONDS,
            "maximum_sample_age_seconds": MAX_SAMPLE_AGE_SECONDS,
            "encoder_return_closes_new_companion_launches": True,
            "encoder_return_requires_companion_checkpoint_preemption": True,
            "finalizer_waits_for_companion_release": True,
            "twenty_thread_encoder_is_exclusive": True,
            "exclusive_primary_thread_count": topology["performance_cores"],
            "efficiency_cores_are_not_homogeneous_heavy_capacity": True,
        },
        "resource_policy": {
            **topology,
            "topology_source": "signed-host-probe",
            "host_probe_sha256": (
                host_probe.get("probe_sha256") if isinstance(host_probe, dict) else None
            ),
            "host_topology_sha256": (
                host_probe.get("topology", {}).get("topology_sha256")
                if isinstance(host_probe, dict) else None
            ),
            "cpu_budget_cores": aggressive.CPU_BUDGET_CORES,
            "ram_admission_ceiling_bytes": aggressive.ADMISSION_CEILING_BYTES,
            "idle_cpu_rule": "minimum measured idle cores across the hysteresis window",
            "idle_ram_rule": "minimum measured idle bytes across the hysteresis window",
            "declared_worker_counts_are_not_idle_measurements": True,
        },
        "persistence": {
            "hash_chained_events": True, "fsync_atomic_state": True,
            "file_lock_required": True,
            "compare_and_swap_generation_required": True,
            "owner_lease_identity_required": True,
            "heavy_work_launch_requires_cas_commit_receipt": True,
            "crash_recovery_receipt_required": True,
            "rollback_receipt_required": True,
            "completed_evidence_mutation_permitted": False,
            "parent_source_deletion_permitted": False,
        },
        "promotion": {
            "automatic_activation_permitted": False,
            "requires_quiescent_checkpoint": True,
            "requires_new_source_bound_generation": True,
            "runtime_defaults_change_permitted": False,
        },
    }
    contract["contract_sha256"] = _hash_value(contract)
    return contract


def validate_contract(contract: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["elastic contract is not an object"]
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("version") != VERSION \
            or contract.get("mode") != "unbound-default-off" \
            or contract.get("enabled_by_default") is not False:
        errors.append("elastic schema/version/default-off identity is invalid")
    if contract.get("contract_sha256") != _hash_value(
            _without(contract, "contract_sha256")):
        errors.append("elastic contract hash mismatch")
    bindings = contract.get("source_bindings")
    try:
        current_authority_tools = local_observer.authority_tool_references()
    except (OSError, local_observer.LocalObserverError):
        current_authority_tools = None
    if not isinstance(bindings, dict) \
            or bindings.get("local_observer_authority_tools") \
            != current_authority_tools:
        errors.append("elastic observer authority-tool binding is absent or stale")
    phase = contract.get("phase_policy")
    if not isinstance(phase, dict) \
            or phase.get("maximum_heavy_prepare") != 1 \
            or phase.get("maximum_primary_encoder") != 1 \
            or phase.get("maximum_serial_finalizer") != 1 \
            or phase.get("maximum_companion") != 1 \
            or phase.get("prepare_and_encoder_mutually_exclusive") is not True \
            or phase.get("prepare_and_finalizer_may_overlap_if_measured_envelope_fits") \
            is not True \
            or phase.get("minimum_sample_spacing_seconds") \
            != MIN_SAMPLE_SPACING_SECONDS \
            or phase.get("encoder_return_closes_new_companion_launches") is not True \
            or phase.get("encoder_return_requires_companion_checkpoint_preemption") is not True \
            or phase.get("finalizer_waits_for_companion_release") is not True \
            or phase.get("twenty_thread_encoder_is_exclusive") is not True \
            or phase.get("efficiency_cores_are_not_homogeneous_heavy_capacity") is not True:
        errors.append("elastic phase envelope is weaker than reviewed")
    resource = contract.get("resource_policy")
    if contract.get("status") == "qualified":
        if not isinstance(resource, dict) \
                or resource.get("topology_source") != "signed-host-probe" \
                or any(resource.get(field) != value
                       for field, value in REVIEWED_TOPOLOGY.items()) \
                or not _valid_sha(resource.get("host_probe_sha256")) \
                or not _valid_sha(resource.get("host_topology_sha256")) \
                or (phase.get("exclusive_primary_thread_count")
                    if isinstance(phase, dict) else None) != 20:
            errors.append("elastic reviewed 28/20/8 host topology binding is invalid")
    elif contract.get("status") != "blocked" or not contract.get("blockers"):
        errors.append("elastic qualification status/blockers are invalid")
    persistence = contract.get("persistence")
    if not isinstance(persistence, dict) \
            or persistence.get("completed_evidence_mutation_permitted") is not False \
            or persistence.get("parent_source_deletion_permitted") is not False \
            or persistence.get("hash_chained_events") is not True \
            or persistence.get("file_lock_required") is not True \
            or persistence.get("compare_and_swap_generation_required") is not True \
            or persistence.get("owner_lease_identity_required") is not True \
            or persistence.get("heavy_work_launch_requires_cas_commit_receipt") \
            is not True:
        errors.append("elastic persistence/rollback contract is invalid")
    promotion = contract.get("promotion")
    if not isinstance(promotion, dict) \
            or promotion.get("automatic_activation_permitted") is not False \
            or promotion.get("runtime_defaults_change_permitted") is not False:
        errors.append("elastic promotion is not default-off")
    invocation = contract.get("invocation_policy")
    entries = invocation.get("entries") if isinstance(invocation, dict) else None
    entry_errors = [error for row in entries for error in _invocation_entry_errors(row)] \
        if isinstance(entries, list) else ["invocation entry inventory is invalid"]
    phases = {row.get("phase") for row in entries if isinstance(row, dict)
              and isinstance(row.get("phase"), str)
              and row.get("production_execution_protocol")
              == "inert-commit-v1-qualified"} \
        if isinstance(entries, list) else set()
    hashes = [row.get("entry_sha256") for row in entries if isinstance(row, dict)] \
        if isinstance(entries, list) else []
    if not isinstance(invocation, dict) \
            or invocation.get("required_phases") != list(INVOCATION_PHASES) \
            or invocation.get("entry_sha256s") != hashes \
            or invocation.get("nearest_or_fallback_phase_permitted") is not False \
            or invocation.get("caller_declared_command_authority") is not False \
            or invocation.get("direct_lock_scoped_observation_required") is not True \
            or entry_errors:
        errors.append("elastic phase invocation policy is invalid")
    if contract.get("status") == "qualified" \
            and any(phase not in phases for phase in INVOCATION_PHASES):
        errors.append(
            "qualified elastic contract lacks an inert-commit-v1-qualified phase entry"
        )
    qualification = contract.get("thread_profile_qualification_sha256")
    if contract.get("status") == "qualified" and not _valid_sha(qualification):
        errors.append("qualified elastic contract lacks a profile qualification hash")
    if contract.get("status") == "qualified":
        required = (
            "elastic_scheduler", "local_observer", "phase_invocation_manifest",
            "aggressive_policy", "aggressive_overlay",
            "vendor_thread_contract", "thread_profile", "thread_binary",
            "host_sprint_plan",
        )
        if not isinstance(bindings, dict) or any(
                not aggressive._reference_matches(bindings.get(name))
                for name in required
        ):
            errors.append("qualified elastic source binding is absent or stale")
        else:
            try:
                overlay = _read_json(
                    (ROOT / bindings["aggressive_overlay"]["path"]).resolve(strict=True)
                )
                host_plan = _read_json(
                    (ROOT / bindings["host_sprint_plan"]["path"]).resolve(strict=True)
                )
                host_bindings = host_plan.get("source_bindings", {})
                invocation_manifest, current_manifest_reference = \
                    _bound_workspace_json(
                        ROOT / bindings["phase_invocation_manifest"]["path"]
                    )
                if aggressive.validate_overlay(overlay) \
                        or overlay.get("overlay_sha256") \
                        != contract.get("aggressive_overlay_sha256"):
                    errors.append("qualified aggressive overlay binding is invalid")
                if host_plan.get("plan_sha256") != _hash_value(
                        _without(host_plan, "plan_sha256")
                ) or host_plan.get("probe", {}).get("probe_sha256") \
                        != resource.get("host_probe_sha256") \
                        or host_plan.get("probe", {}).get("topology", {}).get(
                            "topology_sha256"
                        ) != resource.get("host_topology_sha256") \
                        or host_bindings.get("local_observer") \
                        != _file_reference(Path(local_observer.__file__)) \
                        or host_bindings.get("local_observer_authority_tools") \
                        != local_observer.authority_tool_references():
                    errors.append("qualified host plan/probe binding is invalid")
                if validate_invocation_manifest(invocation_manifest) \
                        or invocation_manifest.get("status") != "qualified" \
                        or current_manifest_reference \
                        != bindings.get("phase_invocation_manifest") \
                        or invocation_manifest.get("manifest_sha256") \
                        != contract.get("phase_invocation_manifest_sha256") \
                        or invocation_manifest.get("entries") != entries:
                    errors.append("qualified phase invocation manifest binding is invalid")
                selections = contract.get("thread_selections")
                if not isinstance(selections, dict) or not selections:
                    errors.append("qualified thread selection inventory is empty")
                else:
                    identity_cells = [
                        {"model_label": row["tier"], "rate_id": row["rate"]}
                        for row in selections.values() if isinstance(row, dict)
                    ]
                    current = aggressive.qualify_thread_profile(
                        identity_cells,
                        profile_path=(ROOT / bindings["thread_profile"]["path"]).resolve(
                            strict=True
                        ),
                        binary_path=(ROOT / bindings["thread_binary"]["path"]).resolve(
                            strict=True
                        ),
                    )
                    if current.get("status") != "qualified" \
                            or current.get("qualification_sha256") != qualification \
                            or current.get("selections") != selections:
                        errors.append("qualified thread profile was not exactly reproduced")
            except (KeyError, OSError, TypeError, ValueError,
                    aggressive.PolicyError, ElasticError) as exc:
                errors.append(f"qualified source revalidation failed: {exc}")
    return errors


def new_state(contract: dict[str, Any]) -> dict[str, Any]:
    errors = validate_contract(contract)
    if errors:
        raise ElasticError("invalid elastic contract: " + "; ".join(errors))
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA, "version": VERSION,
        "contract_sha256": contract["contract_sha256"], "status": "idle",
        "sequence": 0, "state_generation": 0, "phase_generation": 0,
        "lease_generation": 0,
        "prepare_owner": None, "encoder_owner": None,
        "serial_finalizer_owner": None, "companion_owner": None,
        "companion_launch_closed": True,
        "last_event_sha256": None, "events": [],
        "created_at": _now(), "updated_at": _now(),
        "completed_evidence_mutated": False,
        "parent_source_deleted": False,
    }
    state["state_sha256"] = _hash_value(state)
    return state


def validate_state(state: Any, contract: Any) -> list[str]:
    errors = validate_contract(contract)
    if not isinstance(contract, dict):
        return errors + (["elastic state is not an object"]
                         if not isinstance(state, dict) else [])
    if not isinstance(state, dict):
        return errors + ["elastic state is not an object"]
    if state.get("schema") != STATE_SCHEMA or state.get("version") != VERSION \
            or state.get("contract_sha256") != contract.get("contract_sha256"):
        errors.append("elastic state identity mismatch")
    if state.get("status") not in ("idle", "rolled_back"):
        errors.append("elastic state status is invalid")
    if state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        errors.append("elastic state hash mismatch")
    for field in ("sequence", "state_generation", "phase_generation",
                  "lease_generation"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"elastic {field} is invalid")
    if state.get("completed_evidence_mutated") is not False \
            or state.get("parent_source_deleted") is not False:
        errors.append("elastic state violates evidence/source immutability")
    events, previous = state.get("events"), None
    if not isinstance(events, list):
        return errors + ["elastic event journal is invalid"]
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or event.get("sequence") != index \
                or event.get("state_generation") != index \
                or event.get("previous_event_sha256") != previous \
                or event.get("event_sha256") != _hash_value(
                    _without(event, "event_sha256")):
            errors.append(f"elastic event chain is invalid at {index}")
            break
        previous = event["event_sha256"]
    if state.get("sequence") != len(events) \
            or state.get("state_generation") != len(events) \
            or state.get("last_event_sha256") != previous:
        errors.append("elastic event head/sequence mismatch")
    owner_roles = {
        "prepare_owner": "prepare", "encoder_owner": "encoder",
        "serial_finalizer_owner": "finalizer", "companion_owner": "companion",
    }
    for name, role in owner_roles.items():
        owner = state.get(name)
        if owner is not None and (not isinstance(owner, dict)
                                  or not isinstance(owner.get("cell_id"), str)):
            errors.append(f"elastic owner is invalid: {name}")
            continue
        if isinstance(owner, dict):
            identity, lease = owner.get("process_identity"), owner.get("lease")
            if _process_identity_errors(identity):
                errors.append(f"elastic owner process identity is invalid: {name}")
            if not isinstance(lease, dict) \
                    or lease.get("schema") != OWNER_LEASE_SCHEMA \
                    or lease.get("version") != VERSION \
                    or lease.get("lease_sha256") != _hash_value(
                        _without(lease, "lease_sha256")
                    ) or lease.get("contract_sha256") != contract.get("contract_sha256") \
                    or lease.get("role") != role \
                    or lease.get("cell_id") != owner.get("cell_id") \
                    or lease.get("process_identity_sha256") \
                    != (identity or {}).get("process_identity_sha256") \
                    or not isinstance(lease.get("lease_generation"), int) \
                    or lease["lease_generation"] <= 0 \
                    or lease["lease_generation"] > state.get("lease_generation", -1) \
                    or not isinstance(lease.get("state_generation_at_acquire"), int) \
                    or lease["state_generation_at_acquire"] <= 0 \
                    or lease["state_generation_at_acquire"] \
                    > state.get("state_generation", -1) \
                    or not _valid_epoch(lease.get("acquired_epoch")):
                errors.append(f"elastic owner lease is invalid: {name}")
    if state.get("encoder_owner") is None and state.get("companion_launch_closed") is not True:
        errors.append("companion launch is open without a primary encoder")
    if state.get("prepare_owner") is not None and state.get("encoder_owner") is not None:
        errors.append("heavy prepare and primary encoder overlap")
    if state.get("serial_finalizer_owner") is not None \
            and state.get("encoder_owner") is not None:
        errors.append("serial finalizer and primary encoder overlap")
    prepare, finalizer = state.get("prepare_owner"), state.get("serial_finalizer_owner")
    if isinstance(prepare, dict) and isinstance(finalizer, dict) \
            and (not _valid_sha(prepare.get("overlap_envelope_sha256"))
                 or prepare.get("overlap_envelope_sha256")
                 != finalizer.get("overlap_envelope_sha256")):
        errors.append("prepare/finalizer overlap lacks one shared measured envelope")
    return errors


def _event(state: dict[str, Any], action: str, payload: dict[str, Any]) -> None:
    next_generation = state["state_generation"] + 1
    event: dict[str, Any] = {
        "sequence": state["sequence"] + 1,
        "state_generation": next_generation,
        "previous_event_sha256": state["last_event_sha256"],
        "action": action, "payload": payload, "recorded_at": _now(),
    }
    event["event_sha256"] = _hash_value(event)
    state["events"].append(event)
    state["sequence"] = event["sequence"]
    state["state_generation"] = next_generation
    state["last_event_sha256"] = event["event_sha256"]
    state["updated_at"] = event["recorded_at"]


def _new_owner_lease(state: dict[str, Any], contract: dict[str, Any], *,
                     role: str, cell_id: str, process_identity: dict[str, Any],
                     current_wall_epoch: float) -> dict[str, Any]:
    if _process_identity_errors(process_identity) or not _valid_epoch(current_wall_epoch):
        raise ElasticError("owner lease lacks exact process identity/current wall epoch")
    state["lease_generation"] += 1
    lease: dict[str, Any] = {
        "schema": OWNER_LEASE_SCHEMA, "version": VERSION,
        "contract_sha256": contract["contract_sha256"],
        "role": role, "cell_id": cell_id,
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "lease_generation": state["lease_generation"],
        "state_generation_at_acquire": state["state_generation"] + 1,
        "acquired_epoch": float(current_wall_epoch),
    }
    lease["lease_sha256"] = _hash_value(lease)
    return lease


def build_exit_observation(owner: dict[str, Any], contract: dict[str, Any], *,
                           observed_epoch: float,
                           active_process_identities: Iterable[dict[str, Any]],
                           preemption_verified: bool) -> dict[str, Any]:
    """Pure fixture constructor; production CAS replaces this observation."""
    identity, lease = owner.get("process_identity"), owner.get("lease")
    if _process_identity_errors(identity) or not isinstance(lease, dict) \
            or not _valid_sha(lease.get("lease_sha256")) \
            or not _valid_epoch(observed_epoch) \
            or not isinstance(preemption_verified, bool):
        raise ElasticError("cannot build exit observation for invalid owner")
    active = list(active_process_identities)
    if any(_process_identity_errors(row) for row in active):
        raise ElasticError("exit observation contains invalid active identity")
    active_hashes = sorted({row["process_identity_sha256"] for row in active})
    observation: dict[str, Any] = {
        "schema": EXIT_OBSERVATION_SCHEMA, "version": VERSION,
        "observer_contract_sha256": _observer_contract_sha256(),
        "contract_sha256": contract["contract_sha256"],
        "host_probe_sha256": contract["resource_policy"]["host_probe_sha256"],
        "owner_lease_sha256": lease["lease_sha256"],
        "pid": identity["pid"], "start_identity": identity["start_identity"],
        "command_sha256": identity["command_sha256"],
        "process_identity_sha256": identity["process_identity_sha256"],
        "observed_epoch": float(observed_epoch),
        "active_process_identity_sha256s": active_hashes,
        "exact_identity_running": identity["process_identity_sha256"] in active_hashes,
        "exit_verified": identity["process_identity_sha256"] not in active_hashes,
        "preemption_verified": preemption_verified,
        "probe_method": "pid-start-command-exact-negative-membership",
    }
    observation["observation_sha256"] = _hash_value(observation)
    return observation


def build_phase_output_receipt(owner: dict[str, Any], contract: dict[str, Any], *,
                               phase: str, exit_observation: dict[str, Any],
                               output: dict[str, Any], receipt: dict[str, Any],
                               checkpoint: bool,
                               semantic_validator_receipt:
                               dict[str, Any] | None = None) -> dict[str, Any]:
    identity, lease = owner.get("process_identity"), owner.get("lease")
    if phase not in ("prepare", "encoder", "finalizer", "companion_checkpoint") \
            or _process_identity_errors(identity) or not isinstance(lease, dict) \
            or not _artifact_identity(output) or not _artifact_identity(receipt) \
            or not isinstance(checkpoint, bool) \
            or not isinstance(exit_observation, dict) \
            or not _valid_sha(exit_observation.get("observation_sha256")) \
            or (semantic_validator_receipt is not None
                and not _artifact_identity(semantic_validator_receipt)):
        raise ElasticError("cannot build invalid phase output receipt")
    result: dict[str, Any] = {
        "schema": PHASE_OUTPUT_SCHEMA, "version": VERSION,
        "contract_sha256": contract["contract_sha256"],
        "phase": phase, "cell_id": owner["cell_id"],
        "process_identity_sha256": identity["process_identity_sha256"],
        "owner_lease_sha256": lease["lease_sha256"],
        "exit_observation_sha256": exit_observation["observation_sha256"],
        "output": output, "receipt": receipt,
        "semantic_validator_receipt": semantic_validator_receipt,
        "checkpoint": checkpoint, "exact_output": True,
        "complete": True, "source_files_deleted": False,
        "completed_evidence_mutated": False,
    }
    result["phase_receipt_sha256"] = _hash_value(result)
    return result


def build_worker_completion_document(owner: dict[str, Any],
                                     contract: dict[str, Any], *, phase: str,
                                     output_reference: dict[str, Any],
                                     launch_request_sha256: str | None = None,
                                     inert_handshake_sha256: str | None = None,
                                     commit_sha256: str | None = None,
                                     release_sha256: str | None = None,
                                     resource_claim_sha256: str | None = None,
                                     target_claim_handshake_sha256:
                                     str | None = None,
                                     target_claim_ack_sha256: str | None = None,
                                     target_process_identity:
                                     dict[str, Any] | None = None,
                                     target_resource_guard_sha256:
                                     str | None = None,
                                     target_returncode: int | None = None) \
        -> dict[str, Any]:
    """Build the JSON document a worker writes beside its completed output."""
    identity, lease = owner.get("process_identity"), owner.get("lease")
    if phase not in ("prepare", "encoder", "finalizer", "companion_checkpoint") \
            or _process_identity_errors(identity) or not isinstance(lease, dict) \
            or not isinstance(output_reference, dict) \
            or set(output_reference) != {"path", "sha256", "bytes"} \
            or not isinstance(output_reference.get("path"), str) \
            or not _artifact_identity({
                "sha256": output_reference.get("sha256"),
                "bytes": output_reference.get("bytes"),
            }) or isinstance(target_returncode, bool) \
            or (target_returncode is not None
                and not isinstance(target_returncode, int)) \
            or (target_process_identity is not None
                and _process_identity_errors(target_process_identity)) \
            or (target_resource_guard_sha256 is not None
                and not _valid_sha(target_resource_guard_sha256)):
        raise ElasticError("cannot build invalid worker completion document")
    value: dict[str, Any] = {
        "schema": WORKER_COMPLETION_SCHEMA, "version": VERSION,
        "contract_sha256": contract["contract_sha256"],
        "phase": phase, "cell_id": owner["cell_id"],
        "process_identity_sha256": identity["process_identity_sha256"],
        "owner_lease_sha256": lease["lease_sha256"],
        "launch_request_sha256": launch_request_sha256,
        "inert_handshake_sha256": inert_handshake_sha256,
        "commit_sha256": commit_sha256, "release_sha256": release_sha256,
        "resource_claim_sha256": resource_claim_sha256,
        "target_claim_handshake_sha256": target_claim_handshake_sha256,
        "target_claim_ack_sha256": target_claim_ack_sha256,
        "target_process_identity": copy.deepcopy(target_process_identity),
        "target_resource_guard_sha256": target_resource_guard_sha256,
        "target_returncode": target_returncode,
        "output_reference": copy.deepcopy(output_reference),
        "checkpoint": phase == "companion_checkpoint",
        "complete": True, "source_files_deleted": False,
        "completed_evidence_mutated": False,
    }
    value["worker_completion_sha256"] = _hash_value(value)
    return value


def _worker_completion_errors(value: Any, *, owner: dict[str, Any],
                              contract: dict[str, Any], phase: str,
                              output_reference: dict[str, Any],
                              launch_request_sha256: str | None = None,
                              inert_handshake_sha256: str | None = None,
                              commit_sha256: str | None = None,
                              release_sha256: str | None = None,
                              resource_claim_sha256: str | None = None,
                              target_claim_handshake_sha256: str | None = None,
                              target_claim_ack_sha256: str | None = None,
                              target_process_identity:
                              dict[str, Any] | None = None,
                              target_resource_guard_sha256: str | None = None,
                              require_successful_target: bool = False) -> list[str]:
    required = {
        "schema", "version", "contract_sha256", "phase", "cell_id",
        "process_identity_sha256", "owner_lease_sha256", "output_reference",
        "launch_request_sha256", "inert_handshake_sha256", "commit_sha256",
        "release_sha256", "resource_claim_sha256",
        "target_claim_handshake_sha256", "target_claim_ack_sha256",
        "target_process_identity", "target_resource_guard_sha256",
        "target_returncode",
        "checkpoint", "complete", "source_files_deleted",
        "completed_evidence_mutated", "worker_completion_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["worker completion document keys are invalid"]
    expected = build_worker_completion_document(
        owner, contract, phase=phase, output_reference=output_reference,
        launch_request_sha256=launch_request_sha256,
        inert_handshake_sha256=inert_handshake_sha256,
        commit_sha256=commit_sha256, release_sha256=release_sha256,
        resource_claim_sha256=resource_claim_sha256,
        target_claim_handshake_sha256=target_claim_handshake_sha256,
        target_claim_ack_sha256=target_claim_ack_sha256,
        target_process_identity=target_process_identity,
        target_resource_guard_sha256=target_resource_guard_sha256,
        target_returncode=(0 if require_successful_target else value.get(
            "target_returncode"
        )),
    )
    return [] if value == expected else [
        "worker completion document does not bind exact owner/output"
    ]


def _validate_phase_completion(proof: dict[str, Any], *, owner: dict[str, Any],
                               contract: dict[str, Any], phase: str,
                               current_wall_epoch: float,
                               preemption_required: bool) -> tuple[str, str]:
    if not _valid_epoch(current_wall_epoch):
        raise ElasticError("phase completion lacks caller-supplied current wall epoch")
    observation, receipt = proof.get("exit_observation"), proof.get("output_receipt")
    identity, lease = owner.get("process_identity"), owner.get("lease")
    observation_fields = {
        "schema", "version", "observer_contract_sha256", "contract_sha256",
        "host_probe_sha256", "owner_lease_sha256", "pid", "start_identity",
        "command_sha256", "process_identity_sha256", "observed_epoch",
        "active_process_identity_sha256s", "exact_identity_running",
        "exit_verified", "preemption_verified", "probe_method",
        "observation_sha256",
    }
    active_hashes = observation.get("active_process_identity_sha256s") \
        if isinstance(observation, dict) else None
    if not isinstance(observation, dict) or set(observation) != observation_fields \
            or observation.get("observation_sha256") != _hash_value(
                _without(observation, "observation_sha256")
            ) or observation.get("schema") != EXIT_OBSERVATION_SCHEMA \
            or observation.get("version") != VERSION \
            or observation.get("observer_contract_sha256") \
            != _observer_contract_sha256() \
            or observation.get("contract_sha256") != contract["contract_sha256"] \
            or observation.get("host_probe_sha256") \
            != contract["resource_policy"]["host_probe_sha256"] \
            or observation.get("owner_lease_sha256") != lease.get("lease_sha256") \
            or observation.get("pid") != identity.get("pid") \
            or observation.get("start_identity") != identity.get("start_identity") \
            or observation.get("command_sha256") != identity.get("command_sha256") \
            or observation.get("process_identity_sha256") \
            != identity.get("process_identity_sha256") \
            or observation.get("probe_method") \
            != "pid-start-command-exact-negative-membership" \
            or observation.get("exact_identity_running") is not False \
            or observation.get("exit_verified") is not True \
            or not isinstance(active_hashes, list) \
            or any(not _valid_sha(row) for row in active_hashes) \
            or active_hashes != sorted(active_hashes) \
            or len(active_hashes) != len(set(active_hashes)) \
            or identity.get("process_identity_sha256") in observation.get(
                "active_process_identity_sha256s", []
            ) or (preemption_required
                  and observation.get("preemption_verified") is not True) \
            or (not preemption_required
                and observation.get("preemption_verified") is not False):
        raise ElasticError("phase completion does not prove exact process exit/preemption")
    observed = observation.get("observed_epoch")
    if not _valid_epoch(observed) or float(observed) > float(current_wall_epoch) \
            or float(current_wall_epoch) - float(observed) > MAX_SAMPLE_AGE_SECONDS:
        raise ElasticError("phase exit observation is stale or future-dated")
    checkpoint = phase == "companion_checkpoint"
    receipt_fields = {
        "schema", "version", "contract_sha256", "phase", "cell_id",
        "process_identity_sha256", "owner_lease_sha256",
        "exit_observation_sha256", "output", "receipt", "checkpoint",
        "semantic_validator_receipt",
        "exact_output", "complete", "source_files_deleted",
        "completed_evidence_mutated", "phase_receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != receipt_fields \
            or receipt.get("phase_receipt_sha256") != _hash_value(
                _without(receipt, "phase_receipt_sha256")
            ) or receipt.get("schema") != PHASE_OUTPUT_SCHEMA \
            or receipt.get("version") != VERSION \
            or receipt.get("contract_sha256") != contract["contract_sha256"] \
            or receipt.get("phase") != phase \
            or receipt.get("cell_id") != owner["cell_id"] \
            or receipt.get("process_identity_sha256") \
            != identity["process_identity_sha256"] \
            or receipt.get("owner_lease_sha256") != lease["lease_sha256"] \
            or receipt.get("exit_observation_sha256") \
            != observation["observation_sha256"] \
            or not _artifact_identity(receipt.get("output")) \
            or not _artifact_identity(receipt.get("receipt")) \
            or (receipt.get("semantic_validator_receipt") is not None
                and not _artifact_identity(
                    receipt.get("semantic_validator_receipt")
                )) \
            or receipt.get("checkpoint") is not checkpoint \
            or receipt.get("exact_output") is not True \
            or receipt.get("complete") is not True \
            or receipt.get("source_files_deleted") is not False \
            or receipt.get("completed_evidence_mutated") is not False:
        raise ElasticError("phase completion output/checkpoint receipt is invalid")
    return observation["observation_sha256"], receipt["phase_receipt_sha256"]


def _phase_reservation(proof: dict[str, Any], *, phase: str, cell_id: str,
                       contract: dict[str, Any]) -> dict[str, Any]:
    reservation = proof.get("resource_reservation")
    if not isinstance(reservation, dict) \
            or reservation.get("reservation_sha256") != _hash_value(
                _without(reservation, "reservation_sha256")
            ) or reservation.get("contract_sha256") != contract.get("contract_sha256") \
            or reservation.get("phase") != phase \
            or reservation.get("cell_id") != cell_id \
            or not _valid_sha(reservation.get("resource_spec_sha256")):
        raise ElasticError(f"{phase} lacks a hash-bound resource reservation")
    threads, ram = reservation.get("selected_threads"), reservation.get(
        "reservation_bytes"
    )
    performance = contract.get("resource_policy", {}).get("performance_cores")
    ceiling = contract.get("resource_policy", {}).get("ram_admission_ceiling_bytes")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0 \
            or not isinstance(performance, int) or threads > performance \
            or isinstance(ram, bool) or not isinstance(ram, int) or ram <= 0 \
            or not isinstance(ceiling, int) or ram > ceiling:
        raise ElasticError(f"{phase} resource reservation exceeds reviewed host envelope")
    return reservation


def _measured_overlap(proof: dict[str, Any], *, current_owner: dict[str, Any],
                      new_reservation: dict[str, Any],
                      new_process_identity: dict[str, Any],
                      state_generation: int, contract: dict[str, Any],
                      current_wall_epoch: float) -> str:
    envelope = proof.get("overlap_envelope")
    if not _valid_epoch(current_wall_epoch) or not isinstance(envelope, dict) \
            or envelope.get("envelope_sha256") != _hash_value(
                _without(envelope, "envelope_sha256")
            ) or envelope.get("schema") != OVERLAP_SCHEMA \
            or envelope.get("version") != VERSION \
            or envelope.get("contract_sha256") != contract.get("contract_sha256") \
            or envelope.get("host_probe_sha256") \
            != contract.get("resource_policy", {}).get("host_probe_sha256") \
            or envelope.get("state_generation") != state_generation:
        raise ElasticError("prepare/finalizer overlap envelope identity is invalid")
    current_reservation = current_owner.get("resource_reservation")
    current_identity, current_lease = (
        current_owner.get("process_identity"), current_owner.get("lease")
    )
    if not isinstance(current_reservation, dict) \
            or current_reservation.get("reservation_sha256") != _hash_value(
                _without(current_reservation, "reservation_sha256")
            ) or _process_identity_errors(current_identity) \
            or not isinstance(current_lease, dict) \
            or not _valid_sha(current_lease.get("lease_sha256")) \
            or _process_identity_errors(new_process_identity):
        raise ElasticError("existing overlap owner lacks its resource reservation")
    expected = sorted([
        {"cell_id": current_owner["cell_id"],
         "role": current_lease["role"],
         "reservation_sha256": current_reservation["reservation_sha256"],
         "process_identity_sha256": current_identity["process_identity_sha256"],
         "lease_sha256": current_lease["lease_sha256"]},
        {"cell_id": new_reservation["cell_id"],
         "role": new_reservation["phase"],
         "reservation_sha256": new_reservation["reservation_sha256"],
         "process_identity_sha256": new_process_identity[
             "process_identity_sha256"], "lease_sha256": None},
    ], key=lambda row: (row["cell_id"], row["role"]))
    if envelope.get("owners") != expected:
        raise ElasticError("prepare/finalizer overlap owners differ from measured envelope")
    threads = current_reservation["selected_threads"] + new_reservation["selected_threads"]
    ram = current_reservation["reservation_bytes"] + new_reservation["reservation_bytes"]
    policy = contract["resource_policy"]
    if threads > policy["performance_cores"] \
            or ram > policy["ram_admission_ceiling_bytes"]:
        raise ElasticError("prepare/finalizer aggregate reservation exceeds host envelope")
    samples = envelope.get("samples")
    if not isinstance(samples, list) or len(samples) < MIN_IDLE_SAMPLES:
        raise ElasticError("prepare/finalizer measured overlap window is incomplete")
    samples = samples[-MIN_IDLE_SAMPLES:]
    previous = -1.0
    idle_cpu, idle_ram = [], []
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("sample_sha256") != _hash_value(
                _without(sample, "sample_sha256")
        ) or sample.get("pressure_level") != 1 \
                or sample.get("thermal_state") not in ("nominal", "fair") \
                or sample.get("host_probe_sha256") \
                != contract["resource_policy"]["host_probe_sha256"] \
                or sample.get("state_generation") != state_generation \
                or sample.get("current_owner_cell_id") != current_owner["cell_id"] \
                or sample.get("current_owner_role") != current_lease["role"] \
                or sample.get("current_owner_process_identity_sha256") \
                != current_identity["process_identity_sha256"] \
                or sample.get("current_owner_lease_sha256") \
                != current_lease["lease_sha256"]:
            raise ElasticError("prepare/finalizer overlap sample is unauthenticated or red")
        sampled = sample.get("sampled_epoch")
        cpu = sample.get("total_active_cpu_cores")
        rss = sample.get("total_active_tree_rss_bytes")
        if isinstance(sampled, bool) or not isinstance(sampled, (int, float)) \
                or not math.isfinite(float(sampled)) \
                or (previous >= 0 and float(sampled) - previous
                    < MIN_SAMPLE_SPACING_SECONDS) \
                or float(sampled) < float(current_lease["acquired_epoch"]) \
                or float(current_wall_epoch) - float(sampled) > MAX_SAMPLE_AGE_SECONDS \
                or float(sampled) > float(current_wall_epoch) \
                or isinstance(cpu, bool) or not isinstance(cpu, (int, float)) \
                or not math.isfinite(float(cpu)) or cpu < 0 \
                or isinstance(rss, bool) or not isinstance(rss, int) or rss < 0:
            raise ElasticError("prepare/finalizer overlap sample is invalid or stale")
        previous = float(sampled)
        idle_cpu.append(max(0.0, policy["performance_cores"] - float(cpu)))
        idle_ram.append(max(0, policy["ram_admission_ceiling_bytes"] - rss))
    if new_reservation["selected_threads"] > math.floor(min(idle_cpu)) \
            or new_reservation["reservation_bytes"] > min(idle_ram):
        raise ElasticError("new overlap owner exceeds measured idle CPU/RAM")
    return envelope["envelope_sha256"]


def bind_aggressive_swap_decision(swap_state: dict[str, Any],
                                  decision: dict[str, Any]) -> dict[str, Any]:
    """Bind one controller output to its exact hash-sealed successor state."""
    if not isinstance(swap_state, dict):
        raise ElasticError("cannot bind invalid aggressive swap state")
    baseline = swap_state.get("baseline_swap_mb")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or float(baseline) < 0:
        raise ElasticError("cannot bind invalid aggressive swap state")
    try:
        state_errors = _safe_validate_swap_state(swap_state, baseline)
    except (KeyError, TypeError, ValueError, OverflowError):
        state_errors = ["malformed swap state"]
    if state_errors:
        raise ElasticError("cannot bind invalid aggressive swap state")
    mode = swap_state.get("mode")
    expected = {
        "green": (True, aggressive.MAX_LANES, 1.0),
        "soft_throttle": (True, 1, 0.5),
        "hard_stop": (False, 0, 0.0),
        "emergency_shed": (False, 0, 0.0),
    }.get(mode)
    if expected is None:
        raise ElasticError("cannot bind invalid aggressive swap state")
    exact_fields = {
        "mode", "allow_launch", "launch_limit", "cpu_scale", "shed_one",
        "reason", "probe_valid", "swap_growth_mb", "swap_rate_mb_min",
        "green_streak", "hard_until_epoch", "running_evidence_invalidated",
        "emergency_action",
    }
    expected_growth = round(
        float(swap_state["previous_swap_mb"])
        - float(swap_state["baseline_swap_mb"]), 3
    )
    if not isinstance(decision, dict) or set(decision) != exact_fields \
            or decision.get("mode") != mode \
            or (decision.get("allow_launch"), decision.get("launch_limit"),
                decision.get("cpu_scale")) != expected \
            or decision.get("green_streak") != swap_state.get("green_streak") \
            or decision.get("hard_until_epoch") != swap_state.get("hard_until_epoch") \
            or decision.get("running_evidence_invalidated") is not False \
            or not isinstance(decision.get("reason"), str) \
            or not decision.get("reason") \
            or decision.get("swap_growth_mb") != expected_growth \
            or (mode == "green" and (
                decision.get("probe_valid") is not True
                or decision.get("shed_one") is not False
                or decision.get("emergency_action") is not None
            )):
        raise ElasticError("aggressive swap decision differs from exact controller state")
    binding: dict[str, Any] = {
        "schema": SWAP_DECISION_SCHEMA, "version": VERSION,
        "swap_state_sha256": swap_state["state_sha256"],
        "controller_policy_sha256": _hash_value(aggressive.swap_policy()),
        "decision": json.loads(json.dumps(decision)),
    }
    binding["binding_sha256"] = _hash_value(binding)
    return binding


def _bound_swap_decision_errors(swap_state: dict[str, Any], binding: Any) -> list[str]:
    if not isinstance(binding, dict) \
            or binding.get("schema") != SWAP_DECISION_SCHEMA \
            or binding.get("version") != VERSION \
            or binding.get("binding_sha256") != _hash_value(
                _without(binding, "binding_sha256")
            ) or binding.get("swap_state_sha256") != swap_state.get("state_sha256") \
            or binding.get("controller_policy_sha256") \
            != _hash_value(aggressive.swap_policy()):
        return ["aggressive swap decision binding is invalid or stale"]
    try:
        expected = bind_aggressive_swap_decision(swap_state, binding.get("decision"))
    except ElasticError as exc:
        return [str(exc)]
    if expected != binding:
        return ["aggressive swap decision binding differs from controller output"]
    return []


def _fresh_green_swap_proof(proof: dict[str, Any], contract: dict[str, Any], *,
                            current_wall_epoch: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _valid_epoch(current_wall_epoch):
        raise ElasticError("phase admission lacks caller-supplied current wall epoch")
    state = proof.get("aggressive_swap_state")
    binding = proof.get("aggressive_swap_decision")
    baseline = contract.get("swap", {}).get("sealed_baseline_mb")
    errors = _safe_validate_swap_state(state, baseline)
    errors.extend(_bound_swap_decision_errors(state if isinstance(state, dict) else {},
                                               binding))
    sampled = state.get("previous_sample_epoch") if isinstance(state, dict) else None
    decision = binding.get("decision") if isinstance(binding, dict) else None
    if not _valid_epoch(sampled) or float(sampled) > float(current_wall_epoch) \
            or float(current_wall_epoch) - float(sampled) > MAX_SAMPLE_AGE_SECONDS:
        errors.append("aggressive swap controller sample is invalid or stale")
    if not isinstance(state, dict) or state.get("mode") != "green" \
            or not isinstance(decision, dict) or decision.get("mode") != "green" \
            or decision.get("allow_launch") is not True:
        errors.append("aggressive swap controller is not green")
    if errors:
        raise ElasticError("phase swap admission failed: " + "; ".join(errors))
    return state, binding


def transition(state: dict[str, Any], contract: dict[str, Any], *,
               action: str, cell_id: str, proof: dict[str, Any] | None = None,
               current_wall_epoch: float | None = None) \
        -> dict[str, Any]:
    """Pure state-machine reducer; it does not grant production authority.

    The production boundary is :func:`compare_and_swap_transition`, which
    replaces caller process/resource evidence from the lock-scoped observer.
    """
    errors = validate_state(state, contract)
    if errors:
        raise ElasticError("invalid elastic state: " + "; ".join(errors))
    if contract.get("status") != "qualified" and action != "rollback":
        raise ElasticError("blocked/default-off elastic contract cannot transition")
    if state.get("status") == "rolled_back" and action != "rollback":
        raise ElasticError("rolled-back elastic state is terminal")
    if not isinstance(action, str) or not action:
        raise ElasticError("phase transition action is invalid")
    if not isinstance(cell_id, str) or not cell_id:
        raise ElasticError("phase transition cell_id is invalid")
    next_state = json.loads(json.dumps(state))
    if proof is None:
        proof = {}
    elif not isinstance(proof, dict):
        raise ElasticError("phase transition proof is not an object")
    if action == "prepare_start":
        if next_state["prepare_owner"] is not None \
                or next_state["encoder_owner"] is not None:
            raise ElasticError(
                "heavy prepare is single-slot and mutually exclusive with primary encode"
            )
        reservation = _phase_reservation(
            proof, phase="prepare", cell_id=cell_id, contract=contract
        )
        process_identity = proof.get("process_identity")
        if _process_identity_errors(process_identity):
            raise ElasticError("prepare start lacks exact PID/start/command identity")
        _fresh_green_swap_proof(
            proof, contract, current_wall_epoch=current_wall_epoch
        )
        lease = _new_owner_lease(
            next_state, contract, role="prepare", cell_id=cell_id,
            process_identity=process_identity,
            current_wall_epoch=current_wall_epoch,
        )
        owner = {"cell_id": cell_id, "process_identity": process_identity,
                 "lease": lease, "resource_reservation": reservation, "proof": proof}
        if isinstance(next_state["serial_finalizer_owner"], dict):
            envelope_sha = _measured_overlap(
                proof, current_owner=next_state["serial_finalizer_owner"],
                new_reservation=reservation, new_process_identity=process_identity,
                state_generation=state["state_generation"], contract=contract,
                current_wall_epoch=current_wall_epoch,
            )
            owner["overlap_envelope_sha256"] = envelope_sha
            next_state["serial_finalizer_owner"]["overlap_envelope_sha256"] = envelope_sha
        next_state["prepare_owner"] = owner
    elif action == "prepare_complete":
        if not isinstance(next_state["prepare_owner"], dict) \
                or next_state["prepare_owner"]["cell_id"] != cell_id:
            raise ElasticError("heavy prepare completion does not match its owner")
        _validate_phase_completion(
            proof, owner=next_state["prepare_owner"], contract=contract,
            phase="prepare", current_wall_epoch=current_wall_epoch,
            preemption_required=False,
        )
        next_state["prepare_owner"] = None
    elif action == "encoder_start":
        if next_state["encoder_owner"] is not None \
                or next_state["prepare_owner"] is not None \
                or next_state["serial_finalizer_owner"] is not None:
            raise ElasticError(
                "primary encode is mutually exclusive with prepare/finalizer phases"
            )
        tier, rate = proof.get("tier"), proof.get("rate")
        key = json.dumps([tier, rate], separators=(",", ":"), ensure_ascii=False)
        selection = contract.get("thread_selections", {}).get(key)
        if not isinstance(selection, dict) \
                or selection.get("selection_sha256") != aggressive._hash_value(
                    aggressive._without(selection, "selection_sha256")
                ) or proof.get("thread_selection_sha256") \
                != selection.get("selection_sha256") \
                or proof.get("selected_threads") != selection.get("selected_threads"):
            raise ElasticError("primary encoder lacks its exact qualified tier/rate selection")
        process_identity = proof.get("process_identity")
        if _process_identity_errors(process_identity):
            raise ElasticError("primary encoder lacks exact PID/start/command identity")
        _fresh_green_swap_proof(
            proof, contract, current_wall_epoch=current_wall_epoch
        )
        next_state["phase_generation"] += 1
        lease = _new_owner_lease(
            next_state, contract, role="encoder", cell_id=cell_id,
            process_identity=process_identity,
            current_wall_epoch=current_wall_epoch,
        )
        next_state["encoder_owner"] = {
            "cell_id": cell_id, "generation": next_state["phase_generation"],
            "tier": tier, "rate": rate,
            "selected_threads": selection["selected_threads"],
            "thread_selection_sha256": selection["selection_sha256"],
            "process_identity": process_identity, "lease": lease,
            "proof": proof,
        }
        next_state["companion_launch_closed"] = False
    elif action == "companion_launch":
        decision = proof.get("lend_decision")
        encoder = next_state["encoder_owner"]
        process_identity = proof.get("process_identity")
        recorded_epoch = decision.get("recorded_epoch") if isinstance(decision, dict) else None
        swap_state, swap_binding = _fresh_green_swap_proof(
            proof, contract, current_wall_epoch=current_wall_epoch
        )
        if not isinstance(encoder, dict) or next_state["companion_launch_closed"] is not False \
                or next_state["companion_owner"] is not None \
                or not isinstance(decision, dict) or decision.get("allow") is not True \
                or decision.get("decision_sha256") != _hash_value(
                    _without(decision, "decision_sha256")) \
                or decision.get("contract_sha256") != contract["contract_sha256"] \
                or decision.get("state_sha256") != state["state_sha256"] \
                or decision.get("encoder_generation") != encoder["generation"] \
                or decision.get("candidate_cell_id") != cell_id \
                or decision.get("swap_state_sha256") != swap_state["state_sha256"] \
                or decision.get("swap_decision_binding_sha256") \
                != swap_binding["binding_sha256"] \
                or _process_identity_errors(process_identity) \
                or not _valid_epoch(current_wall_epoch) or not _valid_epoch(recorded_epoch) \
                or float(recorded_epoch) > float(current_wall_epoch) \
                or float(current_wall_epoch) - float(recorded_epoch) \
                > MAX_SAMPLE_AGE_SECONDS:
            raise ElasticError("companion launch lacks a current measured lend decision")
        lease = _new_owner_lease(
            next_state, contract, role="companion", cell_id=cell_id,
            process_identity=process_identity,
            current_wall_epoch=current_wall_epoch,
        )
        next_state["companion_owner"] = {
            "cell_id": cell_id, "encoder_generation": encoder["generation"],
            "lend_decision_sha256": decision["decision_sha256"],
            "process_identity": process_identity, "lease": lease,
            "preempt_required": False, "proof": proof,
        }
    elif action == "encoder_return":
        encoder = next_state["encoder_owner"]
        if not isinstance(encoder, dict) or encoder["cell_id"] != cell_id:
            raise ElasticError("encoder return does not match its owner")
        _validate_phase_completion(
            proof, owner=encoder, contract=contract, phase="encoder",
            current_wall_epoch=current_wall_epoch, preemption_required=False,
        )
        next_state["companion_launch_closed"] = True
        next_state["encoder_owner"] = None
        if isinstance(next_state["companion_owner"], dict):
            next_state["companion_owner"]["preempt_required"] = True
            next_state["companion_owner"]["preempt_reason"] = "primary_encoder_returned"
    elif action == "companion_checkpointed":
        companion = next_state["companion_owner"]
        if not isinstance(companion, dict) or companion["cell_id"] != cell_id:
            raise ElasticError("companion checkpoint/release proof is invalid")
        _validate_phase_completion(
            proof, owner=companion, contract=contract,
            phase="companion_checkpoint", current_wall_epoch=current_wall_epoch,
            preemption_required=True,
        )
        next_state["companion_owner"] = None
    elif action == "finalizer_start":
        if next_state["serial_finalizer_owner"] is not None \
                or next_state["encoder_owner"] is not None \
                or next_state["companion_owner"] is not None:
            raise ElasticError("serial finalizer requires encoder return and companion release")
        reservation = _phase_reservation(
            proof, phase="finalizer", cell_id=cell_id, contract=contract
        )
        process_identity = proof.get("process_identity")
        if _process_identity_errors(process_identity):
            raise ElasticError("finalizer start lacks exact PID/start/command identity")
        _fresh_green_swap_proof(
            proof, contract, current_wall_epoch=current_wall_epoch
        )
        lease = _new_owner_lease(
            next_state, contract, role="finalizer", cell_id=cell_id,
            process_identity=process_identity,
            current_wall_epoch=current_wall_epoch,
        )
        owner = {"cell_id": cell_id, "process_identity": process_identity,
                 "lease": lease, "resource_reservation": reservation, "proof": proof}
        if isinstance(next_state["prepare_owner"], dict):
            envelope_sha = _measured_overlap(
                proof, current_owner=next_state["prepare_owner"],
                new_reservation=reservation, new_process_identity=process_identity,
                state_generation=state["state_generation"], contract=contract,
                current_wall_epoch=current_wall_epoch,
            )
            owner["overlap_envelope_sha256"] = envelope_sha
            next_state["prepare_owner"]["overlap_envelope_sha256"] = envelope_sha
        next_state["serial_finalizer_owner"] = owner
    elif action == "finalizer_complete":
        finalizer = next_state["serial_finalizer_owner"]
        if not isinstance(finalizer, dict) or finalizer["cell_id"] != cell_id:
            raise ElasticError("serial finalizer completion does not match its owner")
        _validate_phase_completion(
            proof, owner=finalizer, contract=contract, phase="finalizer",
            current_wall_epoch=current_wall_epoch, preemption_required=False,
        )
        next_state["serial_finalizer_owner"] = None
    elif action == "rollback":
        if any(next_state[name] is not None for name in (
                "prepare_owner", "encoder_owner", "serial_finalizer_owner",
                "companion_owner")):
            raise ElasticError("rollback requires a quiescent elastic state")
        next_state["status"] = "rolled_back"
        next_state["companion_launch_closed"] = True
    else:
        raise ElasticError(f"unknown elastic phase transition: {action}")
    _event(next_state, action, {"cell_id": cell_id, "proof": proof})
    next_state["state_sha256"] = _hash_value(_without(next_state, "state_sha256"))
    errors = validate_state(next_state, contract)
    if errors:
        raise ElasticError("transition produced invalid state: " + "; ".join(errors))
    return next_state


def _validate_idle_samples(samples: Iterable[Any], *, encoder: dict[str, Any],
                           contract: dict[str, Any],
                           now_epoch: float) -> tuple[int, float, int]:
    rows = list(samples)
    if len(rows) < MIN_IDLE_SAMPLES:
        raise ElasticError("companion lending requires three measured idle samples")
    rows = rows[-MIN_IDLE_SAMPLES:]
    previous_time = -1.0
    idle_cpu: list[float] = []
    idle_ram: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("sample_sha256") != _hash_value(
                _without(row, "sample_sha256")) \
                or row.get("encoder_generation") != encoder["generation"] \
                or row.get("primary_cell_id") != encoder["cell_id"] \
                or row.get("primary_process_identity_sha256") \
                != encoder["process_identity"]["process_identity_sha256"] \
                or row.get("primary_lease_sha256") \
                != encoder["lease"]["lease_sha256"] \
                or row.get("state_generation") \
                != encoder["lease"]["state_generation_at_acquire"] \
                or row.get("host_probe_sha256") \
                != contract.get("resource_policy", {}).get("host_probe_sha256"):
            raise ElasticError("idle sample identity/generation is invalid")
        if row.get("pressure_level") != 1 \
                or row.get("thermal_state") not in ("nominal", "fair"):
            raise ElasticError("idle sample pressure/thermal envelope is not green")
        sampled = row.get("sampled_epoch")
        cpu, rss = row.get("total_active_cpu_cores"), row.get("total_active_tree_rss_bytes")
        if isinstance(sampled, bool) or not isinstance(sampled, (int, float)) \
                or not math.isfinite(float(sampled)) \
                or (previous_time >= 0 and float(sampled) - previous_time
                    < MIN_SAMPLE_SPACING_SECONDS) \
                or float(sampled) < float(encoder["lease"]["acquired_epoch"]) \
                or now_epoch - float(sampled) > MAX_SAMPLE_AGE_SECONDS \
                or float(sampled) > float(now_epoch) \
                or isinstance(cpu, bool) or not isinstance(cpu, (int, float)) \
                or not math.isfinite(float(cpu)) or cpu < 0 \
                or isinstance(rss, bool) or not isinstance(rss, int) or rss < 0:
            raise ElasticError("idle sample resource envelope is invalid/stale")
        previous_time = float(sampled)
        cpu_budget = contract.get("resource_policy", {}).get("cpu_budget_cores")
        ram_ceiling = contract.get("resource_policy", {}).get(
            "ram_admission_ceiling_bytes"
        )
        if not isinstance(cpu_budget, int) or not isinstance(ram_ceiling, int):
            raise ElasticError("contract measured-idle resource budget is invalid")
        idle_cpu.append(max(0.0, cpu_budget - float(cpu)))
        idle_ram.append(max(0, ram_ceiling - rss))
    return math.floor(min(idle_cpu)), min(idle_cpu), min(idle_ram)


def lend_decision(state: dict[str, Any], contract: dict[str, Any], *,
                  candidate_cell_id: str, tier: str, rate: str,
                  reservation_bytes: int, idle_samples: Iterable[Any],
                  aggressive_swap_state: dict[str, Any],
                  aggressive_swap_decision: dict[str, Any],
                  now_epoch: float) -> dict[str, Any]:
    blockers = validate_state(state, contract)
    if not _valid_epoch(now_epoch):
        blockers.append("companion lending current wall epoch is invalid")
        wall_epoch = 0.0
    else:
        wall_epoch = float(now_epoch)
    if not isinstance(candidate_cell_id, str) or not candidate_cell_id:
        blockers.append("candidate cell identity is invalid")
    if contract.get("status") != "qualified":
        blockers.append("blocked/default-off elastic contract cannot lend resources")
    encoder = state.get("encoder_owner")
    if not isinstance(encoder, dict) or state.get("companion_launch_closed") is not False:
        blockers.append("primary encoder is not in a lendable phase")
        generation = None
    else:
        generation = encoder["generation"]
    if state.get("companion_owner") is not None:
        blockers.append("the single companion slot is occupied")
    baseline = contract.get("swap", {}).get("sealed_baseline_mb")
    swap_errors = _safe_validate_swap_state(aggressive_swap_state, baseline)
    blockers.extend(f"swap: {row}" for row in swap_errors)
    swap_sampled = aggressive_swap_state.get("previous_sample_epoch")
    if isinstance(swap_sampled, bool) \
            or not isinstance(swap_sampled, (int, float)) \
            or not math.isfinite(float(swap_sampled)) \
            or float(swap_sampled) > wall_epoch \
            or wall_epoch - float(swap_sampled) > MAX_SAMPLE_AGE_SECONDS:
        blockers.append("aggressive swap controller sample is invalid or stale")
    blockers.extend(_bound_swap_decision_errors(
        aggressive_swap_state, aggressive_swap_decision
    ))
    bound_swap_decision = (
        aggressive_swap_decision.get("decision")
        if isinstance(aggressive_swap_decision, dict) else {}
    )
    if aggressive_swap_state.get("mode") != "green" \
            or bound_swap_decision.get("mode") != "green" \
            or bound_swap_decision.get("allow_launch") is not True:
        blockers.append("aggressive swap state is not green for companion lending")
    key = json.dumps([tier, rate], separators=(",", ":"), ensure_ascii=False)
    selection = contract.get("thread_selections", {}).get(key)
    if not isinstance(selection, dict) \
            or selection.get("selection_sha256") != aggressive._hash_value(
                aggressive._without(selection, "selection_sha256")
            ) or selection.get("all_candidates_eligible") is not True:
        blockers.append("candidate lacks an exact qualified tier/rate thread selection")
        threads = None
    else:
        threads = selection.get("selected_threads")
        if threads == 20:
            blockers.append("20-thread profile is exclusive and cannot be a companion")
    if isinstance(encoder, dict) and encoder.get("selected_threads") == 20:
        blockers.append(
            "20-thread primary encoder saturates the 20 performance cores; "
            "efficiency cores are not a heavy companion allowance"
        )
    if isinstance(reservation_bytes, bool) or not isinstance(reservation_bytes, int) \
            or reservation_bytes <= 0:
        blockers.append("candidate RAM reservation is invalid")
    if not isinstance(encoder, dict):
        idle_cores, measured_idle_cpu, idle_ram = 0, 0.0, 0
    else:
        try:
            idle_cores, measured_idle_cpu, idle_ram = _validate_idle_samples(
                idle_samples, encoder=encoder, contract=contract, now_epoch=wall_epoch
            )
        except ElasticError as exc:
            blockers.append(str(exc))
            idle_cores, measured_idle_cpu, idle_ram = 0, 0.0, 0
    if isinstance(threads, int) and threads > idle_cores:
        blockers.append("contract-selected companion threads exceed measured idle CPU")
    if isinstance(reservation_bytes, int) and reservation_bytes > idle_ram:
        blockers.append("candidate reservation exceeds measured idle RAM")
    decision: dict[str, Any] = {
        "schema": DECISION_SCHEMA, "version": VERSION,
        "contract_sha256": contract.get("contract_sha256"),
        "state_sha256": state.get("state_sha256"),
        "encoder_generation": generation,
        "candidate_cell_id": candidate_cell_id, "tier": tier, "rate": rate,
        "allow": not blockers, "blockers": blockers,
        "selected_threads": threads,
        "reservation_bytes": reservation_bytes,
        "measured_idle_cpu_cores": measured_idle_cpu,
        "admissible_idle_cpu_cores": idle_cores,
        "measured_idle_ram_bytes": idle_ram,
        "swap_state_sha256": aggressive_swap_state.get("state_sha256"),
        "swap_decision_binding_sha256": (
            aggressive_swap_decision.get("binding_sha256")
            if isinstance(aggressive_swap_decision, dict) else None
        ),
        "new_launches_close_on_encoder_return": True,
        "recorded_epoch": wall_epoch if _valid_epoch(now_epoch) else None,
        "recorded_at": _now(),
    }
    decision["decision_sha256"] = _hash_value(decision)
    return decision


def crash_recovery_receipt(state: dict[str, Any], contract: dict[str, Any], *,
                           observed_processes: Iterable[dict[str, Any]],
                           current_wall_epoch: float) -> dict[str, Any]:
    """Pure caller-attested fixture; use observed_crash_recovery_receipt live."""
    errors = validate_state(state, contract)
    if errors:
        raise ElasticError("cannot recover invalid state: " + "; ".join(errors))
    if not _valid_epoch(current_wall_epoch):
        raise ElasticError("crash recovery lacks caller-supplied current wall epoch")
    if not isinstance(observed_processes, (list, tuple)):
        raise ElasticError("crash recovery process snapshot is not an inventory")
    observed_rows = list(observed_processes)
    if any(_process_identity_errors(row) for row in observed_rows):
        raise ElasticError("crash recovery process snapshot is invalid")
    observed = {row["process_identity_sha256"] for row in observed_rows}
    if len(observed) != len(observed_rows):
        raise ElasticError("crash recovery process snapshot contains duplicates")
    actions = []
    for role in ("companion_owner", "encoder_owner", "prepare_owner",
                 "serial_finalizer_owner"):
        owner = state.get(role)
        if not isinstance(owner, dict):
            continue
        cell_id = owner["cell_id"]
        identity_sha = owner["process_identity"]["process_identity_sha256"]
        actions.append({
            "role": role, "cell_id": cell_id,
            "process_identity_sha256": identity_sha,
            "owner_lease_sha256": owner["lease"]["lease_sha256"],
            "observed_exact_process_identity": identity_sha in observed,
            "action": ("identity-verify-then-checkpoint-and-preempt"
                       if identity_sha in observed
                       else "record-exact-identity-absent-and-return-pending"),
        })
    receipt: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA, "version": VERSION,
        "contract_sha256": contract["contract_sha256"],
        "state_sha256": state["state_sha256"], "actions": actions,
        "observed_process_identity_sha256s": sorted(observed),
        "recorded_epoch": float(current_wall_epoch),
        "production_authority": "caller-attested-test-only",
        "new_companion_launches_allowed": False,
        "completed_evidence_mutated": False,
        "parent_source_deleted": False, "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    return receipt


def observed_crash_recovery_receipt(path: Path, contract: dict[str, Any]) \
        -> dict[str, Any]:
    """Build production recovery evidence from ps while holding the state lock."""
    target = _safe_stage_output(path)
    if not target.is_file() or target.is_symlink():
        raise ElasticError("observed recovery requires an existing regular state")
    lock_path = _safe_stage_output(target.with_name(target.name + ".lock"))
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ElasticError("observed recovery lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        state = _read_json(target)
        try:
            observed = local_observer.observe_under_lock(target, descriptor)
        except local_observer.LocalObserverError as exc:
            raise ElasticError(f"trusted local observer failed closed: {exc}") from exc
        if action in _COMPLETION_ACTIONS:
            observed_release = observer_receipt.get("extra_json", {}).get(
                "inert_release", {}
            )
            if observed_release.get("value", {}).get("release_sha256") \
                    != (completion_bindings or {}).get("release_sha256") \
                    or observed_release.get("reference") \
                    != (completion_bindings or {}).get("release_reference") \
                    or observed_release.get("value", {}).get(
                        "released_wall_epoch", float("inf")
                    ) > observer_receipt.get("observed_wall_epoch", -1) \
                    or observed_release.get("value", {}).get(
                        "released_monotonic_ns", 2 ** 63
                    ) > observer_receipt.get("observed_monotonic_ns", -1):
                raise ElasticError("inert release changed during completion observation")
        errors = _trusted_observer_errors(observed, state, contract)
        if errors:
            raise ElasticError("trusted recovery observer failed: " + "; ".join(errors))
        owner_rows = {
            row["owner_field"]: row
            for row in observed["persisted_owner_observations"]
        }
        live = [
            state[field]["process_identity"]
            for field, row in owner_rows.items()
            if row["process_observation"].get("exact_identity_running") is True
            and isinstance(state.get(field), dict)
        ]
        receipt = crash_recovery_receipt(
            state, contract, observed_processes=live,
            current_wall_epoch=float(observed["observed_wall_epoch"]),
        )
        receipt["production_authority"] = \
            "trusted-local-observer-under-state-lock"
        receipt["trusted_local_observer_receipt_sha256"] = observed[
            "observer_receipt_sha256"
        ]
        receipt["trusted_local_observer_source"] = observed["observer_source"]
        receipt["lock_lease_sha256"] = observed["lock_lease"][
            "lock_lease_sha256"
        ]
        receipt["observed_monotonic_ns"] = observed["observed_monotonic_ns"]
        receipt["receipt_sha256"] = _hash_value(_without(receipt, "receipt_sha256"))
        return receipt
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def rollback_receipt(state: dict[str, Any], contract: dict[str, Any], *,
                     reason: str, restored_artifacts: Iterable[dict[str, Any]]) \
        -> dict[str, Any]:
    errors = validate_state(state, contract)
    if errors:
        raise ElasticError("cannot rollback invalid state: " + "; ".join(errors))
    if any(state.get(name) is not None for name in (
            "prepare_owner", "encoder_owner", "serial_finalizer_owner",
            "companion_owner")):
        raise ElasticError("rollback receipt requires a quiescent state")
    artifacts = list(restored_artifacts)
    if any(not isinstance(row, dict) or not _valid_sha(row.get("sha256"))
           for row in artifacts):
        raise ElasticError("rollback artifact binding is invalid")
    receipt: dict[str, Any] = {
        "schema": ROLLBACK_SCHEMA, "version": VERSION,
        "contract_sha256": contract["contract_sha256"],
        "state_sha256": state["state_sha256"], "reason": reason,
        "restored_artifacts": artifacts,
        "runtime_defaults_changed": False,
        "completed_evidence_mutated": False,
        "parent_source_deleted": False, "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    return receipt


def _safe_stage_output(path: Path) -> Path:
    # Admission must happen before mkdir: a rejected outside path must have no
    # filesystem side effect.  Walk every existing component so an in-root
    # lexical path cannot escape through a symlink either.
    stage = STAGE_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path if path.is_absolute() else Path.cwd() / path))
    try:
        relative_parent = absolute.parent.relative_to(stage)
    except ValueError as exc:
        raise ElasticError("elastic persistence must remain below elastic_v1") from exc
    cursor = stage
    for component in relative_parent.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ElasticError("elastic persistence parent cannot contain a symlink")
        if cursor.exists() and not cursor.is_dir():
            raise ElasticError("elastic persistence parent is not a directory")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    parent = absolute.parent.resolve(strict=True)
    try:
        parent.relative_to(stage)
    except ValueError as exc:
        raise ElasticError("elastic persistence parent escaped elastic_v1") from exc
    target = parent / absolute.name
    if target.is_symlink():
        raise ElasticError("elastic persistence target cannot be a symlink")
    return target


def _safe_stage_input(path: Path) -> Path:
    """Resolve an existing regular artifact without creating any path."""
    stage = STAGE_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path if path.is_absolute() else Path.cwd() / path))
    try:
        absolute.relative_to(stage)
    except ValueError as exc:
        raise ElasticError("elastic input must remain below elastic_v1") from exc
    cursor = stage
    for component in absolute.relative_to(stage).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ElasticError("elastic input path cannot contain a symlink")
    resolved = absolute.resolve(strict=True)
    try:
        resolved.relative_to(stage)
    except ValueError as exc:
        raise ElasticError("elastic input escaped elastic_v1") from exc
    if not resolved.is_file():
        raise ElasticError("elastic input is not a regular file")
    return resolved


def trusted_process_identity(pid: int) -> dict[str, Any]:
    """Construct the scheduler identity from a direct local ``ps`` observation."""
    observed = local_observer.observe_process_identity(pid)
    return build_process_identity(
        pid=observed["pid"], start_identity=observed["start_identity"],
        command_sha256=observed["command_sha256"],
    )


def _trusted_observer_errors(receipt: Any, state: dict[str, Any],
                             contract: dict[str, Any]) -> list[str]:
    if not isinstance(receipt, dict):
        return ["trusted local observer receipt is absent"]
    errors: list[str] = []
    if receipt.get("schema") != local_observer.OBSERVER_SCHEMA \
            or receipt.get("version") != local_observer.VERSION \
            or receipt.get("authority") \
            != "trusted-local-observer-under-state-lock" \
            or receipt.get("observer_receipt_sha256") != _hash_value(
                _without(receipt, "observer_receipt_sha256")
            ):
        errors.append("trusted local observer receipt schema/hash is invalid")
    binding = contract.get("source_bindings", {}).get("local_observer")
    if receipt.get("observer_source") != binding:
        errors.append("trusted observer source/artifact binding differs from contract")
    if receipt.get("authority_tools") != contract.get(
            "source_bindings", {}).get("local_observer_authority_tools"):
        errors.append("trusted observer authority tools differ from contract")
    if receipt.get("state_sha256") != state.get("state_sha256") \
            or receipt.get("state_generation") != state.get("state_generation"):
        errors.append("trusted observer state hash/generation differs")
    state_reference = receipt.get("state_reference")
    if not isinstance(state_reference, dict) \
            or state_reference.get("sha256") != hashlib.sha256(
                json.dumps(state, indent=2, sort_keys=True,
                           ensure_ascii=False).encode("utf-8") + b"\n"
            ).hexdigest():
        # The persisted byte hash is intentionally distinct from state_sha256.
        errors.append("trusted observer persisted-state artifact binding differs")
    lock = receipt.get("lock_lease")
    if not isinstance(lock, dict) \
            or lock.get("schema") != local_observer.LOCK_LEASE_SCHEMA \
            or lock.get("lock_lease_sha256") != _hash_value(
                _without(lock, "lock_lease_sha256")
            ) or lock.get("state_sha256") != state.get("state_sha256") \
            or lock.get("state_generation") != state.get("state_generation") \
            or lock.get("observer_source_sha256") \
            != (binding or {}).get("sha256"):
        errors.append("trusted observer lock lease is invalid")
    wall, monotonic = (receipt.get("observed_wall_epoch"),
                       receipt.get("observed_monotonic_ns"))
    if not _valid_epoch(wall) or isinstance(monotonic, bool) \
            or not isinstance(monotonic, int) or monotonic <= 0:
        errors.append("trusted observer wall/monotonic time is invalid")
    resources = receipt.get("resources")
    if not isinstance(resources, dict) \
            or resources.get("resource_sha256") != _hash_value(
                _without(resources, "resource_sha256")
            ) or resources.get("source") != "direct-local-subprocess":
        errors.append("trusted observer direct resource receipt is invalid")
    heavy = receipt.get("heavy_owners")
    if not isinstance(heavy, list) \
            or receipt.get("heavy_owner_count") != len(heavy):
        errors.append("trusted observer heavy-owner inventory is inconsistent")
    else:
        heavy_pids: set[int] = set()
        for row in heavy:
            if not isinstance(row, dict) or set(row) != {
                    "pid", "ppid", "start_identity", "command_sha256",
                    "process_generation_sha256", "matched_patterns",
            } or isinstance(row.get("pid"), bool) \
                    or not isinstance(row.get("pid"), int) or row["pid"] <= 0 \
                    or row["pid"] in heavy_pids \
                    or isinstance(row.get("ppid"), bool) \
                    or not isinstance(row.get("ppid"), int) or row["ppid"] < 0 \
                    or not isinstance(row.get("start_identity"), str) \
                    or not row["start_identity"] \
                    or not _valid_sha(row.get("command_sha256")) \
                    or row.get("process_generation_sha256") != _hash_value({
                        key: row[key] for key in (
                            "pid", "start_identity", "command_sha256"
                        )
                    }) or not isinstance(row.get("matched_patterns"), list) \
                    or not row["matched_patterns"] \
                    or any(not isinstance(item, str) or not item
                           for item in row["matched_patterns"]) \
                    or row["matched_patterns"] \
                    != sorted(set(row["matched_patterns"])):
                errors.append("trusted observer heavy-owner row is invalid")
                break
            heavy_pids.add(row["pid"])
    if receipt.get("read_only") is not True \
            or receipt.get("model_or_gpu_work_attempted") is not False \
            or receipt.get("runtime_or_corpus_mutation_attempted") is not False \
            or receipt.get("stable_file_read_method") \
            != "open-fstat-before-after-no-follow":
        errors.append("trusted observer violated its read-only boundary")
    expected = [
        (field, state[field]) for field in local_observer.OWNER_FIELDS
        if isinstance(state.get(field), dict)
    ]
    rows = receipt.get("persisted_owner_observations")
    if not isinstance(rows, list) or len(rows) != len(expected):
        errors.append("trusted observer owner inventory is incomplete")
    else:
        by_field = {row.get("owner_field"): row for row in rows
                    if isinstance(row, dict)}
        for field, owner in expected:
            row = by_field.get(field)
            observation = row.get("process_observation") \
                if isinstance(row, dict) else None
            descendants = row.get("descendants") \
                if isinstance(row, dict) else None
            if not isinstance(row, dict) \
                    or row.get("cell_id") != owner.get("cell_id") \
                    or row.get("lease_sha256") \
                    != owner.get("lease", {}).get("lease_sha256") \
                    or not isinstance(observation, dict) \
                    or observation.get("observation_sha256") != _hash_value(
                        _without(observation, "observation_sha256")
                    ) or observation.get("requested_process_identity_sha256") \
                    != owner.get("process_identity", {}).get(
                        "process_identity_sha256"
                    ):
                errors.append(f"trusted observer owner identity differs: {field}")
                continue
            if not isinstance(descendants, list):
                errors.append(f"trusted observer descendant inventory absent: {field}")
                continue
            admitted = {owner.get("process_identity", {}).get("pid")}
            seen: set[int] = set()
            for descendant in descendants:
                if not isinstance(descendant, dict) or set(descendant) != {
                        "pid", "ppid", "start_identity", "command_sha256",
                        "process_generation_sha256", "descendant_sha256",
                } or descendant.get("descendant_sha256") != _hash_value(
                    _without(descendant, "descendant_sha256")
                ) or isinstance(descendant.get("pid"), bool) \
                        or not isinstance(descendant.get("pid"), int) \
                        or descendant["pid"] <= 0 \
                        or descendant["pid"] in seen \
                        or isinstance(descendant.get("ppid"), bool) \
                        or not isinstance(descendant.get("ppid"), int) \
                        or descendant.get("ppid") not in admitted \
                        or not isinstance(descendant.get("start_identity"), str) \
                        or not descendant["start_identity"] \
                        or not _valid_sha(descendant.get("command_sha256")) \
                        or not _valid_sha(
                            descendant.get("process_generation_sha256")
                        ) or descendant.get("process_generation_sha256") \
                        != _hash_value({
                            key: descendant[key] for key in (
                                "pid", "start_identity", "command_sha256"
                            )
                        }):
                    errors.append(
                        f"trusted observer descendant tree is invalid: {field}"
                    )
                    break
                seen.add(descendant["pid"])
                admitted.add(descendant["pid"])
    return errors


_START_ACTIONS = (
    "prepare_start", "encoder_start", "finalizer_start", "companion_launch",
)
_START_PHASES = {
    "prepare_start": "prepare", "encoder_start": "encoder",
    "finalizer_start": "finalizer", "companion_launch": "companion",
}
_COMPLETION_ACTIONS = {
    "prepare_complete": ("prepare_owner", "prepare", False),
    "encoder_return": ("encoder_owner", "encoder", False),
    "companion_checkpointed": ("companion_owner", "companion_checkpoint", True),
    "finalizer_complete": ("serial_finalizer_owner", "finalizer", False),
}


def _match_phase_invocation(contract: dict[str, Any], *, phase: str,
                            cell_id: str, observation: Any) -> dict[str, Any]:
    """Select exactly one frozen entry from direct process facts; never fallback."""
    if not isinstance(observation, dict) \
            or observation.get("schema") \
            != local_observer.INVOCATION_OBSERVATION_SCHEMA \
            or observation.get("version") != local_observer.VERSION \
            or observation.get("invocation_observation_sha256") != _hash_value(
                _without(observation, "invocation_observation_sha256")
            ) or observation.get("method") \
            != "KERN_PROCARGS2+lsof-cwd+stable-executable-hash":
        raise ElasticError("direct phase invocation observation is invalid")
    argv = observation.get("argv")
    environment = observation.get("environment_value_sha256s")
    keys = observation.get("environment_keys")
    if not isinstance(argv, list) or any(not isinstance(row, str) for row in argv) \
            or observation.get("argv_sha256") != _hash_value(argv) \
            or not isinstance(environment, dict) \
            or keys != sorted(environment) \
            or observation.get("environment_sha256") != _hash_value(environment):
        raise ElasticError("direct phase argv/environment observation is invalid")
    entries = contract.get("invocation_policy", {}).get("entries", [])
    candidates = [row for row in entries if isinstance(row, dict)
                  and row.get("phase") == phase]
    matches: list[dict[str, Any]] = []
    for entry in candidates:
        if _invocation_entry_errors(entry) \
                or entry.get("production_execution_protocol") \
                != "inert-commit-v1-qualified" \
                or observation.get("executable") != entry.get("executable") \
                or observation.get("cwd") != entry.get("cwd"):
            continue
        template = entry["argv_template"]
        if len(template) != len(argv):
            continue
        matched = True
        for token, actual in zip(template, argv, strict=True):
            if "literal" in token:
                matched = matched and actual == token["literal"]
                continue
            name = token["substitution"]
            allowed = entry["allowed_substitutions"].get(name, [])
            matched = matched and actual in allowed
            if name == "cell_id":
                matched = matched and actual == cell_id
        allowlist = entry["environment_allowlist"]
        matched = matched and set(environment) == set(allowlist)
        for key, allowed_hashes in allowlist.items():
            matched = matched and environment.get(key) in allowed_hashes
        if matched:
            matches.append(entry)
    if len(matches) != 1:
        raise ElasticError(
            "direct process invocation does not match exactly one frozen phase entry"
        )
    return matches[0]


def _entry_launch_request(entry: dict[str, Any], *, state_path: Path,
                          contract: dict[str, Any], phase: str,
                          cell_id: str, expected_generation: int) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        row for row in entry.get("argv_artifacts", [])
        if isinstance(row, dict) and row.get("role") == "request"
    ]
    if len(rows) != 1:
        raise ElasticError("inert invocation entry lacks one frozen request")
    row = rows[0]
    raw_path = Path(row["argv_value"])
    request_path = raw_path if raw_path.is_absolute() else Path(entry["cwd"]) / raw_path
    request, reference = _stable_json_reference(request_path)
    if reference != row.get("reference"):
        raise ElasticError("inert launch request artifact differs from invocation entry")
    errors = validate_inert_launch_request(request, request_reference=reference)
    if errors:
        raise ElasticError("inert launch request failed validation: " + "; ".join(errors))
    paths = request["paths"]
    try:
        persisted_contract, _ = _stable_json_reference(Path(paths["contract"]))
    except ElasticError as exc:
        raise ElasticError("inert launch contract artifact is absent or invalid") from exc
    if persisted_contract != contract:
        raise ElasticError("inert launch contract artifact differs from CAS contract")
    if request.get("phase") != phase or request.get("cell_id") != cell_id \
            or request.get("expected_state_generation") != expected_generation \
            or Path(paths["state"]) != Path(state_path):
        raise ElasticError("inert launch request phase/cell/state/generation differs")
    if paths.get("commit") != str(Path(state_path).with_name(
            f"{Path(state_path).name}.commit.{expected_generation + 1}.json"
    )):
        raise ElasticError("inert launch request commit path differs from CAS generation")
    return request, reference


def _resource_claim_from_proof(phase: str, proof: dict[str, Any],
                               contract: dict[str, Any]) -> dict[str, Any]:
    blank = {
        "selected_threads": None, "reservation_bytes": None,
        "resource_spec_sha256": None, "tier": None, "rate": None,
        "thread_selection_sha256": None, "lend_decision_sha256": None,
    }
    if phase in {"prepare", "finalizer"}:
        reservation = proof.get("resource_reservation", {})
        return {
            **blank,
            "selected_threads": reservation.get("selected_threads"),
            "reservation_bytes": reservation.get("reservation_bytes"),
            "resource_spec_sha256": reservation.get("resource_spec_sha256"),
        }
    if phase == "encoder":
        return {
            **blank, "selected_threads": proof.get("selected_threads"),
            "tier": proof.get("tier"), "rate": proof.get("rate"),
            "thread_selection_sha256": proof.get("thread_selection_sha256"),
        }
    decision = proof.get("lend_decision", {})
    key = json.dumps(
        [decision.get("tier"), decision.get("rate")],
        separators=(",", ":"), ensure_ascii=False,
    )
    selection = contract.get("thread_selections", {}).get(key, {})
    return {
        **blank, "selected_threads": decision.get("selected_threads"),
        "reservation_bytes": decision.get("reservation_bytes"),
        "tier": decision.get("tier"), "rate": decision.get("rate"),
        "thread_selection_sha256": selection.get("selection_sha256"),
        "lend_decision_sha256": decision.get("decision_sha256"),
    }


def _handshake_errors(handshake: Any, *, request: dict[str, Any],
                      request_reference: dict[str, Any],
                      process_identity: dict[str, Any],
                      observed_wall_epoch: float,
                      observed_monotonic_ns: int) -> list[str]:
    required = {
        "schema", "version", "request_sha256", "request_artifact_sha256",
        "launch_nonce", "pid", "phase", "cell_id", "action",
        "expected_state_generation", "target_started", "heavy_work_started",
        "created_wall_epoch", "created_monotonic_ns", "launcher",
        "handshake_sha256",
    }
    if not isinstance(handshake, dict) or set(handshake) != required:
        return ["inert pre-commit handshake keys are invalid"]
    errors: list[str] = []
    try:
        hash_matches = handshake.get("handshake_sha256") == _hash_value(
            _without(handshake, "handshake_sha256")
        )
    except (TypeError, ValueError):
        hash_matches = False
    created_wall = handshake.get("created_wall_epoch")
    created_mono = handshake.get("created_monotonic_ns")
    if handshake.get("schema") != INERT_HANDSHAKE_SCHEMA \
            or handshake.get("version") != VERSION or not hash_matches:
        errors.append("inert pre-commit handshake schema/hash is invalid")
    if handshake.get("request_sha256") != request.get("request_sha256") \
            or handshake.get("request_artifact_sha256") \
            != request_reference.get("sha256") \
            or handshake.get("launch_nonce") != request.get("launch_nonce") \
            or handshake.get("pid") != process_identity.get("pid") \
            or handshake.get("phase") != request.get("phase") \
            or handshake.get("cell_id") != request.get("cell_id") \
            or handshake.get("action") != request.get("action") \
            or isinstance(handshake.get("expected_state_generation"), bool) \
            or not isinstance(handshake.get("expected_state_generation"), int) \
            or handshake.get("expected_state_generation") \
            != request.get("expected_state_generation") \
            or handshake.get("launcher") != request.get("launcher"):
        errors.append("inert pre-commit handshake identity/request binding differs")
    if handshake.get("target_started") is not False \
            or handshake.get("heavy_work_started") is not False:
        errors.append("inert launcher reports work before CAS commit")
    if not _valid_epoch(created_wall) \
            or float(created_wall) > float(observed_wall_epoch) \
            or float(observed_wall_epoch) - float(created_wall) \
            > MAX_SAMPLE_AGE_SECONDS \
            or isinstance(created_mono, bool) or not isinstance(created_mono, int) \
            or created_mono <= 0 or created_mono > observed_monotonic_ns \
            or observed_monotonic_ns - created_mono \
            > int(MAX_SAMPLE_AGE_SECONDS * 1_000_000_000):
        errors.append("inert pre-commit handshake is stale or future-dated")
    return errors


def _release_errors(release: Any, *, request: dict[str, Any],
                    handshake_sha256: str, commit_sha256: str,
                    owner_pid: int, handshake: dict[str, Any] | None = None,
                    committed_epoch: float | None = None) -> list[str]:
    required = {
        "schema", "version", "request_sha256", "handshake_sha256",
        "commit_sha256", "state_generation", "pid", "launch_released",
        "target_start_authorized", "released_wall_epoch",
        "released_monotonic_ns", "release_sha256",
    }
    if not isinstance(release, dict) or set(release) != required:
        return ["inert release receipt keys are invalid"]
    try:
        hash_matches = release.get("release_sha256") == _hash_value(
            _without(release, "release_sha256")
        )
    except (TypeError, ValueError):
        hash_matches = False
    if release.get("schema") != INERT_RELEASE_SCHEMA \
            or release.get("version") != VERSION or not hash_matches \
            or release.get("request_sha256") != request.get("request_sha256") \
            or release.get("handshake_sha256") != handshake_sha256 \
            or release.get("commit_sha256") != commit_sha256 \
            or isinstance(release.get("state_generation"), bool) \
            or not isinstance(release.get("state_generation"), int) \
            or release.get("state_generation") \
            != request.get("expected_state_generation", -1) + 1 \
            or release.get("pid") != owner_pid \
            or release.get("launch_released") is not True \
            or release.get("target_start_authorized") is not True \
            or not _valid_epoch(release.get("released_wall_epoch")) \
            or isinstance(release.get("released_monotonic_ns"), bool) \
            or not isinstance(release.get("released_monotonic_ns"), int) \
            or release["released_monotonic_ns"] <= 0 \
            or (isinstance(handshake, dict) and (
                release.get("released_wall_epoch", -1)
                < handshake.get("created_wall_epoch", 0)
                or release.get("released_monotonic_ns", -1)
                < handshake.get("created_monotonic_ns", 0)
            )) or (_valid_epoch(committed_epoch)
                   and release.get("released_wall_epoch", -1)
                   < float(committed_epoch)):
        return ["inert release receipt identity/authorization is invalid"]
    return []


def _target_claim_handshake_errors(value: Any, *, request: dict[str, Any],
                                   owner_pid: int,
                                   release: dict[str, Any]) -> list[str]:
    required = {
        "schema", "version", "request_sha256", "resource_claim_sha256",
        "launch_nonce", "phase", "cell_id", "pid", "parent_pid",
        "target_executable_sha256", "target_argv_sha256",
        "applied_resource_controls",
        "created_wall_epoch", "created_monotonic_ns", "heavy_work_started",
        "awaiting_launcher_ack", "handshake_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["target resource-claim handshake keys are invalid"]
    try:
        valid_hash = value.get("handshake_sha256") == _hash_value(
            _without(value, "handshake_sha256")
        )
    except (TypeError, ValueError):
        valid_hash = False
    pid = value.get("pid")
    wall, monotonic = value.get("created_wall_epoch"), value.get(
        "created_monotonic_ns"
    )
    if value.get("schema") != TARGET_CLAIM_HANDSHAKE_SCHEMA \
            or value.get("version") != VERSION or not valid_hash \
            or value.get("request_sha256") != request.get("request_sha256") \
            or value.get("resource_claim_sha256") \
            != request.get("resource_claim_sha256") \
            or value.get("launch_nonce") != request.get("launch_nonce") \
            or value.get("phase") != request.get("phase") \
            or value.get("cell_id") != request.get("cell_id") \
            or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 \
            or value.get("parent_pid") != owner_pid \
            or value.get("target_executable_sha256") \
            != request.get("target", {}).get("executable", {}).get("sha256") \
            or value.get("target_argv_sha256") \
            != _hash_value(request.get("target", {}).get("argv")) \
            or value.get("applied_resource_controls") \
            != _applied_resource_controls(request) \
            or value.get("heavy_work_started") is not False \
            or value.get("awaiting_launcher_ack") is not True \
            or not _valid_epoch(wall) \
            or isinstance(monotonic, bool) or not isinstance(monotonic, int) \
            or monotonic <= 0 \
            or wall > time.time() + 0.001 \
            or monotonic > time.monotonic_ns() \
            or wall < release.get("released_wall_epoch", float("inf")) \
            or monotonic < release.get("released_monotonic_ns", 2 ** 63):
        return ["target resource-claim handshake identity/timing is invalid"]
    return []


def _target_claim_ack_errors(value: Any, *, request: dict[str, Any],
                             target_handshake: dict[str, Any],
                             commit: dict[str, Any], release: dict[str, Any],
                             owner_pid: int) -> list[str]:
    required = {
        "schema", "version", "request_sha256", "resource_claim_sha256",
        "target_claim_handshake_sha256", "commit_sha256", "release_sha256",
        "target_pid", "target_process_identity", "launcher_pid",
        "heavy_work_authorized",
        "acknowledged_wall_epoch", "acknowledged_monotonic_ns", "ack_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["target resource-claim acknowledgement keys are invalid"]
    try:
        valid_hash = value.get("ack_sha256") == _hash_value(
            _without(value, "ack_sha256")
        )
    except (TypeError, ValueError):
        valid_hash = False
    wall, monotonic = value.get("acknowledged_wall_epoch"), value.get(
        "acknowledged_monotonic_ns"
    )
    if value.get("schema") != TARGET_CLAIM_ACK_SCHEMA \
            or value.get("version") != VERSION or not valid_hash \
            or value.get("request_sha256") != request.get("request_sha256") \
            or value.get("resource_claim_sha256") \
            != request.get("resource_claim_sha256") \
            or value.get("target_claim_handshake_sha256") \
            != target_handshake.get("handshake_sha256") \
            or value.get("commit_sha256") != commit.get("commit_sha256") \
            or value.get("release_sha256") != release.get("release_sha256") \
            or value.get("target_pid") != target_handshake.get("pid") \
            or _process_identity_errors(value.get("target_process_identity")) \
            or value.get("target_process_identity", {}).get("pid") \
            != target_handshake.get("pid") \
            or value.get("launcher_pid") != owner_pid \
            or value.get("heavy_work_authorized") is not True \
            or not _valid_epoch(wall) \
            or isinstance(monotonic, bool) or not isinstance(monotonic, int) \
            or monotonic <= 0 \
            or wall > time.time() + 0.001 \
            or monotonic > time.monotonic_ns() \
            or wall < target_handshake.get("created_wall_epoch", float("inf")) \
            or monotonic < target_handshake.get("created_monotonic_ns", 2 ** 63):
        return ["target resource-claim acknowledgement identity/timing is invalid"]
    return []


def _target_resource_guard_errors(value: Any, *, request: dict[str, Any],
                                  target_handshake: dict[str, Any],
                                  target_ack: dict[str, Any]) -> list[str]:
    required = {
        "schema", "version", "request_sha256", "resource_claim_sha256",
        "target_claim_handshake_sha256", "target_claim_ack_sha256",
        "target_process_identity", "process_group_id", "reservation_bytes",
        "selected_threads", "authoritative_thread_environment",
        "sample_count", "max_tree_rss_bytes", "rss_limit_exceeded",
        "target_returncode", "probe", "guard_complete", "guard_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["target resource guard keys are invalid"]
    try:
        valid_hash = value.get("guard_sha256") == _hash_value(
            _without(value, "guard_sha256")
        )
    except (TypeError, ValueError):
        valid_hash = False
    samples, maximum, returncode = (
        value.get("sample_count"), value.get("max_tree_rss_bytes"),
        value.get("target_returncode"),
    )
    reservation = request.get("resource_claim", {}).get("reservation_bytes")
    controls = _applied_resource_controls(request)
    if value.get("schema") != TARGET_RESOURCE_GUARD_SCHEMA \
            or value.get("version") != VERSION or not valid_hash \
            or value.get("request_sha256") != request.get("request_sha256") \
            or value.get("resource_claim_sha256") \
            != request.get("resource_claim_sha256") \
            or value.get("target_claim_handshake_sha256") \
            != target_handshake.get("handshake_sha256") \
            or value.get("target_claim_ack_sha256") \
            != target_ack.get("ack_sha256") \
            or _process_identity_errors(value.get("target_process_identity")) \
            or value.get("target_process_identity") \
            != target_ack.get("target_process_identity") \
            or value.get("process_group_id") != target_handshake.get("pid") \
            or value.get("reservation_bytes") != reservation \
            or value.get("selected_threads") \
            != request.get("resource_claim", {}).get("selected_threads") \
            or value.get("authoritative_thread_environment") \
            != controls.get("thread_environment") \
            or isinstance(samples, bool) or not isinstance(samples, int) \
            or samples <= 0 \
            or isinstance(maximum, bool) or not isinstance(maximum, int) \
            or maximum < 0 \
            or (reservation is not None and maximum > reservation) \
            or value.get("rss_limit_exceeded") is not False \
            or isinstance(returncode, bool) or not isinstance(returncode, int) \
            or value.get("probe") != "/bin/ps -axo pid=,pgid=,rss=" \
            or value.get("guard_complete") is not True:
        return ["target resource guard claim/enforcement binding is invalid"]
    return []


def _semantic_validator_errors(receipt: Any, *, request: dict[str, Any],
                               output_reference: dict[str, Any],
                               completion_bindings: dict[str, Any]) -> list[str]:
    required = {
        "schema", "version", "validator_profile", "phase", "cell_id",
        "request_sha256", "output", "exact_output", "parity_verified",
        "zero_skips", "skipped_count", "resource_claim_sha256",
        "target_claim_handshake_sha256", "target_claim_ack_sha256",
        "target_process_identity_sha256", "target_resource_guard_sha256",
        "semantic_checks", "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        return ["phase semantic validator receipt keys are invalid"]
    try:
        hash_matches = receipt.get("receipt_sha256") == _hash_value(
            _without(receipt, "receipt_sha256")
        )
    except (TypeError, ValueError):
        hash_matches = False
    checks = receipt.get("semantic_checks")
    if receipt.get("schema") != PHASE_VALIDATOR_RECEIPT_SCHEMA \
            or receipt.get("version") != VERSION or not hash_matches \
            or not isinstance(receipt.get("validator_profile"), str) \
            or not receipt["validator_profile"] \
            or receipt.get("phase") != request.get("phase") \
            or receipt.get("cell_id") != request.get("cell_id") \
            or receipt.get("request_sha256") != request.get("request_sha256") \
            or receipt.get("resource_claim_sha256") \
            != request.get("resource_claim_sha256") \
            or receipt.get("resource_claim_sha256") \
            != completion_bindings.get("resource_claim_sha256") \
            or receipt.get("target_claim_handshake_sha256") \
            != completion_bindings.get("target_claim_handshake_sha256") \
            or receipt.get("target_claim_ack_sha256") \
            != completion_bindings.get("target_claim_ack_sha256") \
            or receipt.get("target_process_identity_sha256") \
            != completion_bindings.get("target_process_identity", {}).get(
                "process_identity_sha256"
            ) \
            or receipt.get("target_resource_guard_sha256") \
            != completion_bindings.get("target_resource_guard_sha256") \
            or receipt.get("output") != output_reference \
            or receipt.get("exact_output") is not True \
            or receipt.get("parity_verified") is not True \
            or receipt.get("zero_skips") is not True \
            or isinstance(receipt.get("skipped_count"), bool) \
            or receipt.get("skipped_count") != 0 \
            or not isinstance(checks, list) or not checks \
            or any(not isinstance(row, str) or not row for row in checks) \
            or len(checks) != len(set(checks)):
        return ["phase semantic validator receipt is not exact/parity/zero-skip"]
    return []


def _persist_immutable_json(path: Path, value: dict[str, Any]) \
        -> dict[str, Any]:
    target = _safe_stage_output(path)
    if target.exists():
        existing, reference = _stable_json_reference(target)
        if existing != value:
            raise ElasticError("immutable semantic receipt path already differs")
        return reference
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            existing, reference = _stable_json_reference(target)
            if existing != value:
                raise ElasticError("immutable semantic receipt race differs")
            return reference
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    _, reference = _stable_json_reference(target)
    return reference


def _run_phase_validator(request: dict[str, Any], *,
                         output_reference: dict[str, Any],
                         completion_bindings: dict[str, Any]) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    command = request["validator"]
    errors = _command_binding_errors(command, validator=True)
    if errors:
        raise ElasticError("phase validator drifted: " + "; ".join(errors))
    try:
        completed = subprocess.run(
            command["argv"], cwd=command["cwd"], env=command["environment"],
            capture_output=True, timeout=60.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ElasticError(f"phase semantic validator execution failed: {exc}") from exc
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        raise ElasticError(
            "phase semantic validator returned nonzero/oversized output"
        )
    try:
        receipt = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ElasticError("phase semantic validator stdout is malformed") from exc
    errors = _semantic_validator_errors(
        receipt, request=request, output_reference=output_reference,
        completion_bindings=completion_bindings,
    )
    if errors:
        raise ElasticError("phase semantic validation failed: " + "; ".join(errors))
    if _command_binding_errors(command, validator=True):
        raise ElasticError("phase validator source changed during execution")
    reference = _persist_immutable_json(
        Path(request["paths"]["semantic_receipt"]), receipt
    )
    persisted, persisted_reference = _stable_json_reference(
        Path(request["paths"]["semantic_receipt"])
    )
    if persisted != receipt or persisted_reference != reference:
        raise ElasticError("phase semantic receipt persistence differs")
    return receipt, reference


def _production_proof(state: dict[str, Any], contract: dict[str, Any], *,
                      action: str, cell_id: str, proof: dict[str, Any] | None,
                      observer_receipt: dict[str, Any],
                      launch_request: dict[str, Any] | None = None,
                      launch_request_reference: dict[str, Any] | None = None,
                      inert_handshake: dict[str, Any] | None = None,
                      inert_handshake_reference: dict[str, Any] | None = None,
                      semantic_validator_reference:
                      dict[str, Any] | None = None,
                      completion_bindings: dict[str, Any] | None = None) \
        -> tuple[dict[str, Any], float]:
    """Replace caller attestations with facts sampled under the state lock."""
    errors = _trusted_observer_errors(observer_receipt, state, contract)
    if errors:
        raise ElasticError("trusted local observer failed: " + "; ".join(errors))
    trusted = copy.deepcopy(proof) if isinstance(proof, dict) else {}
    observed_epoch = float(observer_receipt["observed_wall_epoch"])
    owner_rows = {
        row["owner_field"]: row for row in observer_receipt[
            "persisted_owner_observations"
        ]
    }
    completion = _COMPLETION_ACTIONS.get(action)
    target_field = completion[0] if completion else None
    for field, row in owner_rows.items():
        running = row["process_observation"].get("exact_identity_running")
        if field == target_field:
            if running is not False:
                raise ElasticError(
                    "trusted local observer still sees exact completion process"
                )
        elif running is not True:
            raise ElasticError(
                f"trusted local observer cannot authenticate active owner: {field}"
            )
    if action in _START_ACTIONS:
        proposed = observer_receipt.get("proposed_process_observations")
        identity = trusted.get("process_identity")
        identity_sha = identity.get("process_identity_sha256") \
            if isinstance(identity, dict) else None
        matches = [row for row in proposed if isinstance(row, dict)
                   and row.get("requested_process_identity_sha256") == identity_sha] \
            if isinstance(proposed, list) else []
        if len(matches) != 1 or matches[0].get("exact_identity_running") is not True:
            raise ElasticError(
                "trusted local observer cannot authenticate proposed PID/start/command"
            )
        invocation_entry = _match_phase_invocation(
            contract, phase=_START_PHASES[action], cell_id=cell_id,
            observation=matches[0].get("invocation_observation"),
        )
        if not isinstance(launch_request, dict) \
                or not isinstance(launch_request_reference, dict) \
                or not isinstance(inert_handshake, dict) \
                or not isinstance(inert_handshake_reference, dict):
            raise ElasticError("production start lacks frozen inert request/handshake")
        for caller_key in (
                "invocation", "argv", "cwd", "environment", "executable",
                "phase_invocation_entry_sha256"):
            trusted.pop(caller_key, None)
        trusted["phase_invocation_entry_sha256"] = invocation_entry["entry_sha256"]
        trusted["phase_invocation_observation_sha256"] = matches[0][
            "invocation_observation"
        ]["invocation_observation_sha256"]
        trusted["launch_request_sha256"] = launch_request["request_sha256"]
        trusted["launch_request_reference"] = copy.deepcopy(
            launch_request_reference
        )
        trusted["inert_handshake_sha256"] = inert_handshake[
            "handshake_sha256"
        ]
        trusted["inert_handshake_reference"] = copy.deepcopy(
            inert_handshake_reference
        )
        allowed_process_generations: dict[int, str] = {}
        for row in owner_rows.values():
            if row.get("process_observation", {}).get(
                    "exact_identity_running") is True:
                identity = row.get("process_identity", {})
                allowed_process_generations[identity.get("pid")] = _hash_value({
                    key: identity[key] for key in (
                        "pid", "start_identity", "command_sha256"
                    )
                })
                allowed_process_generations.update({
                    descendant["pid"]: descendant["process_generation_sha256"]
                    for descendant in row.get("descendants", [])
                    if isinstance(descendant, dict)
                })
        proposed_identity = trusted.get("process_identity", {})
        allowed_process_generations[proposed_identity.get("pid")] = _hash_value({
            key: proposed_identity[key] for key in (
                "pid", "start_identity", "command_sha256"
            )
        })
        external_heavy = [
            row for row in observer_receipt.get("heavy_owners", [])
            if isinstance(row, dict) and allowed_process_generations.get(
                row.get("pid")
            ) != row.get("process_generation_sha256")
        ]
        if observer_receipt.get("heavy_owner_count") \
                != len(observer_receipt.get("heavy_owners", [])) \
                or external_heavy:
            raise ElasticError(
                "trusted local observer found an external campaign-wide heavy owner"
            )
        if action == "companion_launch":
            raise ElasticError(
                "production companion launch remains fail-closed until local idle-window "
                "sampling replaces caller lend samples"
            )
        if action == "finalizer_start" and isinstance(state.get("prepare_owner"), dict):
            raise ElasticError(
                "production prepare/finalizer overlap remains fail-closed until its "
                "three-sample local observer window is collected"
            )
        resources = observer_receipt.get("resources", {})
        if resources.get("probe_valid") is not True \
                or resources.get("pressure_level") != 1 \
                or resources.get("thermal_green") is not True \
                or resources.get("ac_power") is not True:
            raise ElasticError("trusted local host resources are not green")
        extra_swap = observer_receipt.get("extra_json", {}).get(
            "aggressive_swap_state", {}
        )
        prior = extra_swap.get("value") if isinstance(extra_swap, dict) else None
        baseline = contract.get("swap", {}).get("sealed_baseline_mb")
        swap_errors = _safe_validate_swap_state(prior, baseline)
        if swap_errors:
            raise ElasticError("trusted swap transition prior is invalid: "
                               + "; ".join(swap_errors))
        updated_swap, decision = aggressive.advance_swap_state(
            prior, {
                "pressure_level": resources.get("pressure_level"),
                "swap_used_mb": resources.get("swap_used_mb"),
            }, now_epoch=observed_epoch,
            sealed_baseline_swap_mb=float(baseline),
        )
        binding = bind_aggressive_swap_decision(updated_swap, decision)
        if decision.get("mode") != "green" or decision.get("allow_launch") is not True:
            raise ElasticError("direct local swap/pressure controller is not green")
        trusted["aggressive_swap_state"] = updated_swap
        trusted["aggressive_swap_decision"] = binding
        # Persist the exact acquisition receipt with the owner.  A hash-valid
        # state synthesized via the pure reducer cannot later claim that a fake
        # PID was ever observed alive by the production CAS.
        trusted["trusted_local_observer_receipt"] = copy.deepcopy(observer_receipt)
    if completion is not None:
        field, phase, preempted = completion
        owner = state.get(field)
        if not isinstance(owner, dict):
            # Transition emits the canonical owner mismatch, but fail here too.
            raise ElasticError("trusted completion owner is absent")
        owner_proof = owner.get("proof") if isinstance(owner.get("proof"), dict) else {}
        acquisition = owner_proof.get("trusted_local_observer_receipt")
        lease = owner.get("lease", {})
        identity_sha = owner.get("process_identity", {}).get(
            "process_identity_sha256"
        )
        proposed_at_acquire = acquisition.get("proposed_process_observations") \
            if isinstance(acquisition, dict) else None
        exact_acquisitions = [
            row for row in proposed_at_acquire
            if isinstance(row, dict)
            and row.get("requested_process_identity_sha256") == identity_sha
            and row.get("exact_identity_running") is True
        ] if isinstance(proposed_at_acquire, list) else []
        acquisition_entry = None
        if len(exact_acquisitions) == 1:
            try:
                acquisition_entry = _match_phase_invocation(
                    contract,
                    phase=("companion" if phase == "companion_checkpoint" else phase),
                    cell_id=owner["cell_id"],
                    observation=exact_acquisitions[0].get("invocation_observation"),
                )
            except ElasticError:
                acquisition_entry = None
        if not isinstance(acquisition, dict) \
                or acquisition.get("observer_receipt_sha256") != _hash_value(
                    _without(acquisition, "observer_receipt_sha256")
                ) or acquisition.get("authority") \
                != "trusted-local-observer-under-state-lock" \
                or acquisition.get("observer_source") \
                != contract.get("source_bindings", {}).get("local_observer") \
                or acquisition.get("authority_tools") \
                != contract.get("source_bindings", {}).get(
                    "local_observer_authority_tools"
                ) \
                or acquisition.get("state_generation") \
                != lease.get("state_generation_at_acquire", 0) - 1 \
                or acquisition.get("observed_wall_epoch") \
                != lease.get("acquired_epoch") \
                or len(exact_acquisitions) != 1 \
                or not isinstance(acquisition_entry, dict) \
                or owner_proof.get("phase_invocation_entry_sha256") \
                != acquisition_entry.get("entry_sha256") \
                or owner_proof.get(
                    "phase_invocation_observation_sha256"
                ) != exact_acquisitions[0].get(
                    "invocation_observation", {}
                ).get("invocation_observation_sha256") \
                or not _valid_sha(owner_proof.get("launch_request_sha256")) \
                or not isinstance(
                    owner_proof.get("launch_request_reference"), dict
                ) or not _valid_sha(
                    owner_proof.get("inert_handshake_sha256")
                ) or not isinstance(
                    owner_proof.get("inert_handshake_reference"), dict
                ):
            raise ElasticError(
                "owner lacks a trusted lock-scoped process acquisition receipt"
            )
        active = [
            state[name]["process_identity"] for name, row in owner_rows.items()
            if name != field and row["process_observation"].get(
                "exact_identity_running"
            ) is True and isinstance(state.get(name), dict)
        ]
        observation = build_exit_observation(
            owner, contract, observed_epoch=observed_epoch,
            active_process_identities=active, preemption_verified=preempted,
        )
        output_reference = observer_receipt.get("extra_artifacts", {}).get(
            "phase_output"
        )
        completion_extra = observer_receipt.get("extra_json", {}).get(
            "worker_completion_receipt", {}
        )
        completion_document = completion_extra.get("value") \
            if isinstance(completion_extra, dict) else None
        completion_reference = completion_extra.get("reference") \
            if isinstance(completion_extra, dict) else None
        if not isinstance(output_reference, dict) \
                or not isinstance(completion_reference, dict):
            raise ElasticError("trusted completion artifact observation is absent")
        claim_handshake_extra = observer_receipt.get("extra_json", {}).get(
            "target_claim_handshake", {}
        )
        claim_ack_extra = observer_receipt.get("extra_json", {}).get(
            "target_claim_ack", {}
        )
        resource_guard_extra = observer_receipt.get("extra_json", {}).get(
            "target_resource_guard", {}
        )
        target_identity = (completion_bindings or {}).get(
            "target_process_identity"
        )
        target_observations = [
            row for row in observer_receipt.get(
                "proposed_process_observations", []
            ) if isinstance(row, dict) and isinstance(target_identity, dict)
            and row.get("requested_process_identity_sha256")
            == target_identity.get("process_identity_sha256")
        ]
        if not isinstance(completion_bindings, dict) \
                or claim_handshake_extra.get("reference") \
                != completion_bindings.get("target_claim_handshake_reference") \
                or claim_ack_extra.get("reference") \
                != completion_bindings.get("target_claim_ack_reference") \
                or claim_handshake_extra.get("value", {}).get(
                    "handshake_sha256"
                ) != completion_bindings.get("target_claim_handshake_sha256") \
                or claim_ack_extra.get("value", {}).get("ack_sha256") \
                != completion_bindings.get("target_claim_ack_sha256") \
                or resource_guard_extra.get("reference") \
                != completion_bindings.get("target_resource_guard_reference") \
                or resource_guard_extra.get("value", {}).get("guard_sha256") \
                != completion_bindings.get("target_resource_guard_sha256") \
                or len(target_observations) != 1 \
                or target_observations[0].get("exact_identity_running") is not False:
            raise ElasticError(
                "trusted target resource-claim/exit observations differ"
            )
        document_errors = _worker_completion_errors(
            completion_document, owner=owner, contract=contract, phase=phase,
            output_reference=output_reference,
            launch_request_sha256=(completion_bindings or {}).get(
                "launch_request_sha256"
            ),
            inert_handshake_sha256=(completion_bindings or {}).get(
                "inert_handshake_sha256"
            ),
            commit_sha256=(completion_bindings or {}).get("commit_sha256"),
            release_sha256=(completion_bindings or {}).get("release_sha256"),
            resource_claim_sha256=(completion_bindings or {}).get(
                "resource_claim_sha256"
            ),
            target_claim_handshake_sha256=(completion_bindings or {}).get(
                "target_claim_handshake_sha256"
            ),
            target_claim_ack_sha256=(completion_bindings or {}).get(
                "target_claim_ack_sha256"
            ),
            target_process_identity=(completion_bindings or {}).get(
                "target_process_identity"
            ),
            target_resource_guard_sha256=(completion_bindings or {}).get(
                "target_resource_guard_sha256"
            ),
            require_successful_target=True,
        )
        if document_errors:
            raise ElasticError("trusted worker completion failed: "
                               + "; ".join(document_errors))
        trusted["exit_observation"] = observation
        trusted["output_receipt"] = build_phase_output_receipt(
            owner, contract, phase=phase, exit_observation=observation,
            output={key: output_reference[key] for key in ("sha256", "bytes")},
            receipt={key: completion_reference[key] for key in ("sha256", "bytes")},
            checkpoint=phase == "companion_checkpoint",
            semantic_validator_receipt=(
                {key: semantic_validator_reference[key]
                 for key in ("sha256", "bytes")}
                if isinstance(semantic_validator_reference, dict) else None
            ),
        )
    trusted["trusted_local_observer_receipt_sha256"] = observer_receipt[
        "observer_receipt_sha256"
    ]
    return trusted, observed_epoch


def persist_state(path: Path, state: dict[str, Any], contract: dict[str, Any]) \
        -> dict[str, Any]:
    errors = validate_state(state, contract)
    if errors:
        raise ElasticError("cannot persist invalid elastic state: " + "; ".join(errors))
    target = _safe_stage_output(path)
    lock_path = _safe_stage_output(target.with_name(target.name + ".lock"))
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ElasticError("elastic initial-state lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if target.exists():
            raise ElasticError("elastic state already exists; use locked compare-and-swap")
        _atomic_json(target, state)
        if _read_json(target) != state:
            raise ElasticError("persisted elastic state readback differs")
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return _file_reference(target)


def compare_and_swap_transition(path: Path, contract: dict[str, Any], *,
                                expected_state_sha256: str,
                                expected_state_generation: int,
                                action: str, cell_id: str,
                                proof: dict[str, Any] | None,
                                current_wall_epoch: float,
                                aggressive_swap_state_path: Path | None = None) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize one production transition from lock-scoped local observations.

    ``current_wall_epoch`` is retained as a compatibility/advisory field only;
    authorization time, owner membership, process identity, host state, pressure,
    and swap are obtained directly by the source-bound observer while this lock
    is held.
    """
    if not _valid_epoch(current_wall_epoch):
        raise ElasticError("elastic CAS current wall epoch is invalid")
    if not isinstance(action, str) or not action:
        raise ElasticError("elastic CAS action is invalid")
    target = _safe_stage_output(path)
    if not target.is_file() or target.is_symlink():
        raise ElasticError("CAS requires an existing regular elastic state")
    lock_path = _safe_stage_output(target.with_name(target.name + ".lock"))
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ElasticError("elastic CAS lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = _read_json(target)
        errors = validate_state(current, contract)
        if errors:
            raise ElasticError("CAS loaded invalid elastic state: " + "; ".join(errors))
        if current.get("state_sha256") != expected_state_sha256 \
                or current.get("state_generation") != expected_state_generation:
            raise ElasticError("elastic CAS conflict: state hash/generation advanced")
        proposed = []
        if action in _START_ACTIONS and isinstance(proof, dict) \
                and isinstance(proof.get("process_identity"), dict):
            proposed = [proof["process_identity"]]
        extra_json_paths: dict[str, Path] = {}
        extra_artifact_paths: dict[str, Path] = {}
        launch_request: dict[str, Any] | None = None
        launch_request_reference: dict[str, Any] | None = None
        inert_handshake: dict[str, Any] | None = None
        inert_handshake_reference: dict[str, Any] | None = None
        completion_bindings: dict[str, Any] | None = None
        semantic_validator_reference: dict[str, Any] | None = None
        if action in _START_ACTIONS:
            if aggressive_swap_state_path is None:
                raise ElasticError(
                    "production start requires a confined persisted swap-controller state"
                )
            extra_json_paths["aggressive_swap_state"] = _safe_stage_input(
                Path(aggressive_swap_state_path)
            )
        if action in _COMPLETION_ACTIONS:
            if isinstance(proof, dict) and any(name in proof for name in (
                    "output_path", "worker_completion_receipt_path",
                    "semantic_receipt_path", "release_path", "commit_path",
                    "launch_request_path")):
                raise ElasticError(
                    "caller-selected completion paths are forbidden"
                )
            owner_field, completed_phase, _ = _COMPLETION_ACTIONS[action]
            owner = current.get(owner_field)
            if not isinstance(owner, dict) or owner.get("cell_id") != cell_id:
                raise ElasticError("trusted completion owner is absent or differs")
            owner_proof = owner.get("proof", {})
            entry_sha = owner_proof.get("phase_invocation_entry_sha256") \
                if isinstance(owner_proof, dict) else None
            entries = [
                row for row in contract.get("invocation_policy", {}).get(
                    "entries", []
                ) if isinstance(row, dict)
                and row.get("entry_sha256") == entry_sha
                and row.get("production_execution_protocol")
                == "inert-commit-v1-qualified"
            ]
            if len(entries) != 1:
                raise ElasticError("completion owner lacks one frozen inert entry")
            request_phase = (
                "companion" if completed_phase == "companion_checkpoint"
                else completed_phase
            )
            acquisition_generation = owner.get("lease", {}).get(
                "state_generation_at_acquire"
            )
            if isinstance(acquisition_generation, bool) \
                    or not isinstance(acquisition_generation, int) \
                    or acquisition_generation <= 0:
                raise ElasticError("completion owner acquisition generation is invalid")
            launch_request, launch_request_reference = _entry_launch_request(
                entries[0], state_path=target, contract=contract,
                phase=request_phase, cell_id=cell_id,
                expected_generation=acquisition_generation - 1,
            )
            if owner_proof.get("launch_request_sha256") \
                    != launch_request.get("request_sha256") \
                    or owner_proof.get("launch_request_reference") \
                    != launch_request_reference:
                raise ElasticError("completion request differs from acquired owner")
            inert_handshake, inert_handshake_reference = _stable_json_reference(
                Path(launch_request["paths"]["handshake"])
            )
            if owner_proof.get("inert_handshake_sha256") \
                    != inert_handshake.get("handshake_sha256") \
                    or owner_proof.get("inert_handshake_reference") \
                    != inert_handshake_reference:
                raise ElasticError("completion handshake differs from acquired owner")
            commit_document, commit_reference = _stable_json_reference(
                Path(launch_request["paths"]["commit"])
            )
            release_document, release_reference = _stable_json_reference(
                Path(launch_request["paths"]["release"])
            )
            if commit_document.get("schema") != CAS_SCHEMA \
                    or commit_document.get("commit_sha256") != _hash_value(
                        _without(commit_document, "commit_sha256")
                    ) or commit_document.get("launch_request_sha256") \
                    != launch_request["request_sha256"] \
                    or commit_document.get("launch_request_artifact_sha256") \
                    != launch_request_reference["sha256"] \
                    or commit_document.get("inert_handshake_sha256") \
                    != inert_handshake["handshake_sha256"]:
                raise ElasticError("completion CAS commit is absent or drifted")
            release_errors = _release_errors(
                release_document, request=launch_request,
                handshake_sha256=inert_handshake["handshake_sha256"],
                commit_sha256=commit_document["commit_sha256"],
                owner_pid=owner["process_identity"]["pid"],
                handshake=inert_handshake,
                committed_epoch=commit_document.get("committed_epoch"),
            )
            if release_errors:
                raise ElasticError("completion release failed: "
                                   + "; ".join(release_errors))
            failure_path = Path(
                launch_request["paths"]["target_claim_failure"]
            )
            if failure_path.exists() or failure_path.is_symlink():
                raise ElasticError(
                    "target resource-claim failure receipt forbids completion"
                )
            target_claim_handshake, target_claim_handshake_reference = \
                _stable_json_reference(Path(
                    launch_request["paths"]["target_claim_handshake"]
                ))
            target_claim_ack, target_claim_ack_reference = \
                _stable_json_reference(Path(
                    launch_request["paths"]["target_claim_ack"]
                ))
            target_resource_guard, target_resource_guard_reference = \
                _stable_json_reference(Path(
                    launch_request["paths"]["target_resource_guard"]
                ))
            claim_errors = _target_claim_handshake_errors(
                target_claim_handshake, request=launch_request,
                owner_pid=owner["process_identity"]["pid"],
                release=release_document,
            )
            claim_errors.extend(_target_claim_ack_errors(
                target_claim_ack, request=launch_request,
                target_handshake=target_claim_handshake,
                commit=commit_document, release=release_document,
                owner_pid=owner["process_identity"]["pid"],
            ))
            claim_errors.extend(_target_resource_guard_errors(
                target_resource_guard, request=launch_request,
                target_handshake=target_claim_handshake,
                target_ack=target_claim_ack,
            ))
            if claim_errors:
                raise ElasticError("target resource-claim binding failed: "
                                   + "; ".join(claim_errors))
            output_path = _safe_stage_input(Path(launch_request["paths"]["output"]))
            receipt_path = _safe_stage_input(
                Path(launch_request["paths"]["worker_receipt"])
            )
            extra_artifact_paths["phase_output"] = output_path
            extra_json_paths["worker_completion_receipt"] = _safe_stage_input(
                receipt_path
            )
            extra_json_paths["inert_release"] = _safe_stage_input(
                Path(launch_request["paths"]["release"])
            )
            extra_json_paths["target_claim_handshake"] = _safe_stage_input(
                Path(launch_request["paths"]["target_claim_handshake"])
            )
            extra_json_paths["target_claim_ack"] = _safe_stage_input(
                Path(launch_request["paths"]["target_claim_ack"])
            )
            extra_json_paths["target_resource_guard"] = _safe_stage_input(
                Path(launch_request["paths"]["target_resource_guard"])
            )
            completion_bindings = {
                "launch_request_sha256": launch_request["request_sha256"],
                "inert_handshake_sha256": inert_handshake["handshake_sha256"],
                "commit_sha256": commit_document["commit_sha256"],
                "release_sha256": release_document["release_sha256"],
                "resource_claim_sha256": launch_request[
                    "resource_claim_sha256"
                ],
                "target_claim_handshake_sha256": target_claim_handshake[
                    "handshake_sha256"
                ],
                "target_claim_ack_sha256": target_claim_ack["ack_sha256"],
                "target_process_identity": target_claim_ack[
                    "target_process_identity"
                ],
                "target_resource_guard_sha256": target_resource_guard[
                    "guard_sha256"
                ],
                "commit_reference": commit_reference,
                "release_reference": release_reference,
                "target_claim_handshake_reference": (
                    target_claim_handshake_reference
                ),
                "target_claim_ack_reference": target_claim_ack_reference,
                "target_resource_guard_reference": (
                    target_resource_guard_reference
                ),
            }
        if action in _COMPLETION_ACTIONS and isinstance(
                completion_bindings, dict
        ) and isinstance(completion_bindings.get("target_process_identity"), dict):
            proposed = [completion_bindings["target_process_identity"]]
        try:
            observer_receipt = local_observer.observe_under_lock(
                target, descriptor, proposed_process_identities=proposed,
                extra_json_paths=extra_json_paths,
                extra_artifact_paths=extra_artifact_paths,
            )
        except local_observer.LocalObserverError as exc:
            raise ElasticError(f"trusted local observer failed closed: {exc}") from exc
        if action in _START_ACTIONS:
            proposed_rows = observer_receipt.get("proposed_process_observations", [])
            identity_sha = proof.get("process_identity", {}).get(
                "process_identity_sha256"
            ) if isinstance(proof, dict) else None
            matched_rows = [
                row for row in proposed_rows if isinstance(row, dict)
                and row.get("requested_process_identity_sha256") == identity_sha
                and row.get("exact_identity_running") is True
            ]
            if len(matched_rows) != 1:
                raise ElasticError("trusted observer lacks exact inert launcher")
            entry = _match_phase_invocation(
                contract, phase=_START_PHASES[action], cell_id=cell_id,
                observation=matched_rows[0].get("invocation_observation"),
            )
            launch_request, launch_request_reference = _entry_launch_request(
                entry, state_path=target, contract=contract,
                phase=_START_PHASES[action], cell_id=cell_id,
                expected_generation=expected_state_generation,
            )
            expected_claim = _resource_claim_from_proof(
                _START_PHASES[action], proof or {}, contract
            )
            if launch_request.get("resource_claim") != expected_claim:
                raise ElasticError(
                    "inert launch resource claim differs from admitted proof"
                )
            for name in ("output", "release", "worker_receipt",
                         "semantic_receipt", "commit", "target_claim_handshake",
                         "target_claim_ack", "target_claim_failure",
                         "target_resource_guard"):
                artifact = Path(launch_request["paths"][name])
                if artifact.exists() or artifact.is_symlink():
                    raise ElasticError(
                        f"inert launch one-shot artifact already exists: {name}"
                    )
            inert_handshake, inert_handshake_reference = _stable_json_reference(
                Path(launch_request["paths"]["handshake"])
            )
            handshake_errors = _handshake_errors(
                inert_handshake, request=launch_request,
                request_reference=launch_request_reference,
                process_identity=(proof or {}).get("process_identity", {}),
                observed_wall_epoch=observer_receipt["observed_wall_epoch"],
                observed_monotonic_ns=observer_receipt["observed_monotonic_ns"],
            )
            if handshake_errors:
                raise ElasticError("inert pre-commit handshake failed: "
                                   + "; ".join(handshake_errors))
        trusted_proof, trusted_wall_epoch = _production_proof(
            current, contract, action=action, cell_id=cell_id, proof=proof,
            observer_receipt=observer_receipt,
            launch_request=launch_request,
            launch_request_reference=launch_request_reference,
            inert_handshake=inert_handshake,
            inert_handshake_reference=inert_handshake_reference,
            completion_bindings=completion_bindings,
        )
        if action in _COMPLETION_ACTIONS:
            output_before = observer_receipt.get("extra_artifacts", {}).get(
                "phase_output"
            )
            _, semantic_validator_reference = _run_phase_validator(
                launch_request, output_reference=output_before,
                completion_bindings=completion_bindings or {},
            )
            output_after = local_observer.stable_artifact_reference(
                Path(launch_request["paths"]["output"])
            )
            if output_after != output_before:
                raise ElasticError("phase output changed during semantic validation")
            # Rebuild only the final semantic-bound phase receipt after the
            # validator succeeds; all other production facts remain identical.
            trusted_proof, trusted_wall_epoch = _production_proof(
                current, contract, action=action, cell_id=cell_id, proof=proof,
                observer_receipt=observer_receipt,
                launch_request=launch_request,
                launch_request_reference=launch_request_reference,
                inert_handshake=inert_handshake,
                inert_handshake_reference=inert_handshake_reference,
                semantic_validator_reference=semantic_validator_reference,
                completion_bindings=completion_bindings,
            )
        updated = transition(
            current, contract, action=action, cell_id=cell_id,
            proof=trusted_proof, current_wall_epoch=trusted_wall_epoch,
        )
        if updated["state_generation"] != expected_state_generation + 1:
            raise ElasticError("elastic CAS transition did not advance exactly one generation")
        if action in _START_ACTIONS:
            current_request, current_request_reference = _stable_json_reference(
                Path(launch_request["paths"]["request"])
            )
            current_handshake, current_handshake_reference = _stable_json_reference(
                Path(launch_request["paths"]["handshake"])
            )
            if current_request != launch_request \
                    or current_request_reference != launch_request_reference \
                    or current_handshake != inert_handshake \
                    or current_handshake_reference != inert_handshake_reference \
                    or _command_binding_errors(
                        launch_request["target"], validator=False
                    ) or _command_binding_errors(
                        launch_request["validator"], validator=True
                    ):
                raise ElasticError("inert launch request/handshake/source drifted before CAS")
        commit_path = _safe_stage_output(target.with_name(
            f"{target.name}.commit.{updated['state_generation']}.json"
        ))
        if commit_path.exists():
            raise ElasticError("elastic CAS commit receipt path already exists")
        _atomic_json(target, updated)
        if _read_json(target) != updated:
            raise ElasticError("elastic CAS readback differs after commit")
        owners = [updated.get(name) for name in (
            "prepare_owner", "encoder_owner", "serial_finalizer_owner",
            "companion_owner",
        )]
        lease_hashes = sorted(
            row["lease"]["lease_sha256"] for row in owners if isinstance(row, dict)
        )
        try:
            commit_display = str(commit_path.relative_to(ROOT.resolve()))
        except ValueError:
            commit_display = str(commit_path)
        receipt: dict[str, Any] = {
            "schema": CAS_SCHEMA, "version": VERSION,
            "contract_sha256": contract["contract_sha256"],
            "state_path": _file_reference(target)["path"],
            "previous_state_sha256": expected_state_sha256,
            "previous_state_generation": expected_state_generation,
            "new_state_sha256": updated["state_sha256"],
            "new_state_generation": updated["state_generation"],
            "commit_receipt_path": commit_display,
            "action": action, "cell_id": cell_id,
            "launch_authorized": action in (
                "prepare_start", "encoder_start", "finalizer_start",
                "companion_launch",
            ),
            "authorization_boundary": (
                "heavy work may begin only after this committed receipt is verified"
            ),
            "active_owner_lease_sha256s": lease_hashes,
            "trusted_local_observer_receipt_sha256": observer_receipt[
                "observer_receipt_sha256"
            ],
            "trusted_local_observer_source": observer_receipt["observer_source"],
            "lock_lease_sha256": observer_receipt["lock_lease"][
                "lock_lease_sha256"
            ],
            "observed_state_sha256": observer_receipt["state_sha256"],
            "observed_state_generation": observer_receipt["state_generation"],
            "observed_monotonic_ns": observer_receipt["observed_monotonic_ns"],
            "committed_epoch": trusted_wall_epoch,
            "caller_wall_epoch_advisory": float(current_wall_epoch),
            "production_authority": "trusted-local-observer-under-state-lock",
            "phase_invocation_entry_sha256": trusted_proof.get(
                "phase_invocation_entry_sha256"
            ),
            "phase_invocation_observation_sha256": trusted_proof.get(
                "phase_invocation_observation_sha256"
            ),
            "launch_request_sha256": (
                launch_request.get("request_sha256")
                if isinstance(launch_request, dict) and action in _START_ACTIONS
                else None
            ),
            "launch_request_artifact_sha256": (
                launch_request_reference.get("sha256")
                if isinstance(launch_request_reference, dict)
                and action in _START_ACTIONS else None
            ),
            "inert_handshake_sha256": (
                inert_handshake.get("handshake_sha256")
                if isinstance(inert_handshake, dict) and action in _START_ACTIONS
                else None
            ),
            "inert_handshake_artifact_sha256": (
                inert_handshake_reference.get("sha256")
                if isinstance(inert_handshake_reference, dict)
                and action in _START_ACTIONS else None
            ),
            "source_files_deleted": False,
            "completed_evidence_mutated": False,
        }
        receipt["commit_sha256"] = _hash_value(receipt)
        _atomic_json(commit_path, receipt)
        if _read_json(commit_path) != receipt:
            raise ElasticError("elastic CAS commit receipt readback differs")
        os.fsync(descriptor)
        return updated, receipt
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def persist_evidence(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, dict) \
            or receipt.get("schema") not in (RECOVERY_SCHEMA, ROLLBACK_SCHEMA) \
            or receipt.get("version") != VERSION \
            or receipt.get("receipt_sha256") != _hash_value(
                _without(receipt, "receipt_sha256")
            ) or receipt.get("completed_evidence_mutated") is not False \
            or receipt.get("parent_source_deleted") is not False:
        raise ElasticError("crash/rollback evidence receipt is invalid")
    target = _safe_stage_output(path)
    lock_path = _safe_stage_output(target.with_name(target.name + ".lock"))
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ElasticError("crash/rollback evidence lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if target.exists():
            raise ElasticError("crash/rollback evidence is immutable and already exists")
        _atomic_json(target, receipt)
        if _read_json(target) != receipt:
            raise ElasticError("persisted crash/rollback evidence readback differs")
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return _file_reference(target)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="read-only default-off contract projection")
    stage = sub.add_parser("stage", help="write only inert elastic contract/state")
    stage.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    stage.add_argument("--state-output", type=Path, default=DEFAULT_STATE)
    stage.add_argument("--invocation-manifest", type=Path,
                       default=DEFAULT_INVOCATION_MANIFEST)
    args = parser.parse_args()
    invocation_manifest_path = (
        args.invocation_manifest if args.command == "stage"
        else DEFAULT_INVOCATION_MANIFEST
    )
    if args.command == "stage" and not invocation_manifest_path.exists():
        target = _safe_stage_output(invocation_manifest_path)
        _atomic_json(target, build_invocation_manifest([]))
    overlay = _read_json(AGGRESSIVE_OVERLAY)
    host_plan_reference, host_probe = None, None
    if DEFAULT_HOST_PLAN.is_file():
        host_plan = _read_json(DEFAULT_HOST_PLAN)
        host_plan_reference = _file_reference(DEFAULT_HOST_PLAN)
        host_probe = host_plan.get("probe")
    contract = build_contract(
        overlay, aggressive_overlay_path=AGGRESSIVE_OVERLAY,
        host_plan_reference=host_plan_reference, host_probe=host_probe,
        invocation_manifest_path=(
            invocation_manifest_path if invocation_manifest_path.is_file() else None
        ),
    )
    state = new_state(contract)
    if args.command == "stage":
        for path in (args.contract_output.resolve(), args.state_output.resolve()):
            try:
                path.relative_to(STAGE_ROOT.resolve())
            except ValueError as exc:
                raise ElasticError("elastic staging must remain below elastic_v1") from exc
        _atomic_json(args.contract_output.resolve(), contract)
        _atomic_json(args.state_output.resolve(), state)
    print(json.dumps({
        "contract_sha256": contract["contract_sha256"],
        "status": contract["status"], "blockers": contract["blockers"],
        "enabled_by_default": contract["enabled_by_default"],
        "validation_errors": validate_contract(contract),
        "promotion_ready": False,
        "promotion_reason": "unbound/default-off; requires qualified profile and quiescent generation",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except ElasticError as exc:
        print(f"doctor_v5_elastic_phase_scheduler: {exc}", file=sys.stderr)
        raise SystemExit(2)
