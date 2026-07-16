#!/usr/bin/env python3.12
"""Source-bound inert launcher for a future Doctor V5 elastic phase.

The launcher performs no target work until an exact, generation-bound CAS
commit receipt appears.  It is intentionally standalone: the scheduler binds
these bytes, the request bytes, and every target artifact before admitting the
waiting PID.  This file does not mutate the live Doctor queue or defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any


REQUEST_SCHEMA = "hawking.doctor_v5_inert_phase_launch_request.v1"
HANDSHAKE_SCHEMA = "hawking.doctor_v5_inert_phase_handshake.v1"
RELEASE_SCHEMA = "hawking.doctor_v5_inert_phase_release.v1"
TARGET_CLAIM_HANDSHAKE_SCHEMA = (
    "hawking.doctor_v5_target_resource_claim_handshake.v1"
)
TARGET_CLAIM_ACK_SCHEMA = "hawking.doctor_v5_target_resource_claim_ack.v1"
TARGET_CLAIM_FAILURE_SCHEMA = (
    "hawking.doctor_v5_target_resource_claim_failure.v1"
)
TARGET_RESOURCE_GUARD_SCHEMA = "hawking.doctor_v5_target_resource_guard.v1"
CAS_SCHEMA = "hawking.doctor_v5_elastic_cas_commit.v1"
VERSION = "2026-07-14.1"
ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/elastic_v1"
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


class LauncherError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _no_symlink_path(path: Path, *, below: Path | None = None,
                     must_exist: bool = True) -> Path:
    absolute = Path(os.path.abspath(path))
    if below is not None:
        root = below.resolve(strict=True)
        relative = absolute.relative_to(root)
        cursor = root
        parts = relative.parts
    else:
        cursor = Path(absolute.anchor)
        parts = absolute.parts[1:]
    for component in parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise LauncherError(f"path contains a symlink component: {absolute}")
    if must_exist:
        resolved = absolute.resolve(strict=True)
        if resolved != absolute:
            raise LauncherError(f"path is not canonical: {absolute}")
    else:
        parent = absolute.parent.resolve(strict=True)
        if below is not None:
            parent.relative_to(below.resolve(strict=True))
    return absolute


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _no_symlink_path(path, must_exist=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LauncherError(f"not a regular JSON artifact: {path}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise LauncherError(f"oversized JSON artifact: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    current = os.stat(path, follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                current.st_dev, current.st_ino, current.st_size,
                current.st_mtime_ns,
            ) \
            or len(raw) != after.st_size:
        raise LauncherError(f"JSON artifact changed during read: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LauncherError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"JSON root is not an object: {path}")
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve(strict=True)))
    except ValueError:
        display = str(resolved)
    return value, {
        "path": display, "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _file_reference(path: Path) -> dict[str, Any]:
    path = _no_symlink_path(path, must_exist=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    current = os.stat(path, follow_symlinks=False)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after) \
            or identity(after) != identity(current) or len(raw) != after.st_size:
        raise LauncherError(f"artifact changed during read: {path}")
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _reference_matches(reference: Any) -> bool:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
        return False
    try:
        raw_path = Path(reference["path"])
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        path = _no_symlink_path(path, must_exist=True)
        if not path.is_file():
            return False
        return _file_reference(path) == reference
    except (LauncherError, OSError, TypeError, ValueError):
        return False


def _cwd_identity(path: Path) -> dict[str, Any]:
    path = _no_symlink_path(path, below=ROOT, must_exist=True)
    if not path.is_dir():
        raise LauncherError("launch cwd is not a directory")
    details = os.stat(path, follow_symlinks=False)
    return {"path": str(path), "device": details.st_dev,
            "inode": details.st_ino}


def _command_errors(command: Any, *, validator: bool) -> list[str]:
    required = {
        "executable", "argv", "argv_artifacts", "semantic_artifacts",
        "cwd", "cwd_identity", "environment_value_sha256s",
    }
    if validator:
        required |= {"environment", "receipt_schema"}
    else:
        required |= {"resource_environment"}
    if not isinstance(command, dict) or set(command) != required:
        return ["launch command keys are invalid"]
    errors: list[str] = []
    argv = command.get("argv")
    executable = command.get("executable")
    if not _reference_matches(executable) or not isinstance(argv, list) \
            or not argv or argv[0] != executable.get("path") \
            or any(not isinstance(row, str) or not row for row in argv):
        errors.append("launch command executable/argv is invalid or drifted")
    try:
        cwd = Path(command["cwd"])
        if command.get("cwd_identity") != _cwd_identity(cwd):
            raise LauncherError("cwd identity differs")
    except (LauncherError, OSError, TypeError, ValueError, KeyError):
        errors.append("launch command cwd identity is invalid or drifted")
        cwd = ROOT
    positions: set[int] = set()
    artifacts = command.get("argv_artifacts")
    if not isinstance(artifacts, list):
        errors.append("launch argv artifact inventory is invalid")
    else:
        for row in artifacts:
            position = row.get("position") if isinstance(row, dict) else None
            if not isinstance(row, dict) or set(row) != {
                    "position", "argv_value", "role", "reference"
            } or isinstance(position, bool) or not isinstance(position, int) \
                    or position <= 0 or position in positions \
                    or not isinstance(argv, list) or position >= len(argv) \
                    or argv[position] != row.get("argv_value") \
                    or row.get("role") not in {"script", "config", "semantic"} \
                    or not _reference_matches(row.get("reference")):
                errors.append("launch argv artifact binding is invalid or drifted")
                continue
            positions.add(position)
    semantic = command.get("semantic_artifacts")
    names: set[str] = set()
    if not isinstance(semantic, list) or not semantic:
        errors.append("launch semantic artifact inventory is empty/invalid")
    else:
        for row in semantic:
            name = row.get("name") if isinstance(row, dict) else None
            if not isinstance(row, dict) or set(row) != {"name", "reference"} \
                    or not isinstance(name, str) or not name \
                    or name in names or not _reference_matches(row.get("reference")):
                errors.append("launch semantic artifact is invalid or drifted")
                continue
            names.add(name)
    if isinstance(argv, list):
        for position, token in enumerate(argv[:-1]):
            if token in {"--config", "--config-path"} \
                    and position + 1 not in positions:
                errors.append("launch request/config argument is not source-bound")
    python_command = isinstance(executable, dict) \
        and "python" in Path(str(executable.get("path", ""))).name.lower()
    if python_command and (
            not isinstance(argv, list) or len(argv) < 4
            or argv[1:3] != ["-I", "-S"] or 3 not in positions):
        errors.append("Python launch command is not isolated/source-bound")
    if python_command and not validator \
            and "target.dependency_closure" not in names:
        errors.append("Python target dependency closure is absent")
    hashes = command.get("environment_value_sha256s")
    if not isinstance(hashes, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            or len(value) != 64 for key, value in hashes.items()):
        errors.append("launch environment hash inventory is invalid")
    if isinstance(hashes, dict) and {
            "PYTHONHOME", "PYTHONPATH", "DYLD_FRAMEWORK_PATH",
            "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    }.intersection(hashes):
        errors.append("launch environment contains Python/dynamic-loader injection")
    if validator:
        environment = command.get("environment")
        if not isinstance(environment, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
        ) or {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in sorted(environment.items())
        } != hashes:
            errors.append("validator environment is invalid or drifted")
        if isinstance(environment, dict) and any(
                re.search(
                    r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|PRIVATE_KEY)",
                    key, re.IGNORECASE,
                ) for key in environment
        ):
            errors.append("validator environment contains secret-like keys")
    elif not isinstance(command.get("resource_environment"), dict) \
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in command["resource_environment"].items()):
        errors.append("target injected resource environment is invalid")
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
    claim, paths, target = (
        request.get("resource_claim"), request.get("paths"), request.get("target")
    )
    if not isinstance(claim, dict) or not isinstance(paths, dict) \
            or not isinstance(target, dict):
        return ["target resource-claim protocol inputs are invalid"]
    errors: list[str] = []
    if request.get("resource_claim_sha256") != _hash_value(claim):
        errors.append("target resource-claim hash differs")
    if request.get("target_claim_protocol") != TARGET_CLAIM_PROTOCOL:
        errors.append("target resource-claim protocol contract differs")
    tail = _claim_protocol_argv_tail(
        claim, request_path=paths.get("request", ""),
        handshake_path=paths.get("target_claim_handshake", ""),
        ack_path=paths.get("target_claim_ack", ""),
    )
    argv = target.get("argv")
    if not isinstance(argv, list) or len(argv) < len(tail) \
            or argv[-len(tail):] != tail:
        errors.append("target argv is not exactly resource-claim bound")
    elif any(
            token in {"-t", "--threads", "--num-threads", "--n-threads",
                      "--workers", "--num-workers"}
            or token.startswith(("--threads=", "--num-threads=", "--n-threads=",
                                 "--workers=", "--num-workers="))
            for token in argv[:-len(tail)]):
        errors.append("target argv contains a conflicting unbound thread control")
    if target.get("resource_environment") != _resource_claim_environment(
            claim, request_path=paths.get("request", "")):
        errors.append("target injected environment is not resource-claim bound")
    timeout = request.get("target_claim_handshake_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
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


def _atomic_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path = _no_symlink_path(path, below=STAGE_ROOT, must_exist=False)
    if path.exists() or path.is_symlink():
        raise LauncherError(f"one-shot receipt already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                              dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_path.unlink(missing_ok=True)


def _environment_hashes() -> dict[str, str]:
    return {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in sorted(os.environ.items())
    }


def _request_errors(request: Any, *, request_path: Path,
                    request_reference: dict[str, Any]) -> list[str]:
    required = {
        "schema", "version", "phase", "action", "cell_id", "resource_claim",
        "resource_claim_sha256", "target_claim_protocol",
        "target_claim_handshake_timeout_seconds",
        "launch_nonce", "expected_state_generation",
        "commit_wait_timeout_seconds", "launcher", "paths", "target",
        "validator", "required_semantics",
        "caller_selected_completion_paths_permitted",
        "heavy_work_before_commit_permitted", "request_sha256",
    }
    if not isinstance(request, dict) or set(request) != required:
        return ["launch request keys are invalid"]
    errors: list[str] = []
    try:
        valid_hash = request.get("request_sha256") == _hash_value(
            _without(request, "request_sha256")
        )
    except (TypeError, ValueError):
        valid_hash = False
    phases = {
        "prepare": "prepare_start", "encoder": "encoder_start",
        "finalizer": "finalizer_start", "companion": "companion_launch",
    }
    if request.get("schema") != REQUEST_SCHEMA \
            or request.get("version") != VERSION or not valid_hash \
            or phases.get(request.get("phase")) != request.get("action") \
            or not isinstance(request.get("cell_id"), str) \
            or not request.get("cell_id"):
        errors.append("launch request schema/hash/phase identity is invalid")
    if not isinstance(request.get("resource_claim"), dict) \
            or set(request["resource_claim"]) != {
                "selected_threads", "reservation_bytes", "resource_spec_sha256",
                "tier", "rate", "thread_selection_sha256",
                "lend_decision_sha256",
            }:
        errors.append("launch request resource claim is invalid")
    generation = request.get("expected_state_generation")
    timeout = request.get("commit_wait_timeout_seconds")
    if isinstance(generation, bool) or not isinstance(generation, int) \
            or generation < 0 or isinstance(timeout, bool) \
            or not isinstance(timeout, (int, float)) \
            or not 1.0 <= float(timeout) <= 300.0:
        errors.append("launch request generation/timeout is invalid")
    if request.get("launcher") != _file_reference(Path(__file__)) \
            or not _reference_matches(request.get("launcher")):
        errors.append("launcher source binding is invalid or drifted")
    paths = request.get("paths")
    names = {
        "request", "contract", "state", "handshake", "release", "output",
        "semantic_receipt", "worker_receipt", "commit",
        "target_claim_handshake", "target_claim_ack", "target_claim_failure",
        "target_resource_guard",
    }
    if not isinstance(paths, dict) or set(paths) != names \
            or len(set(paths.values())) != len(names):
        errors.append("launch request path inventory is invalid")
    else:
        for name, value in paths.items():
            try:
                current = _no_symlink_path(
                    Path(value), below=STAGE_ROOT,
                    must_exist=name in {"request", "contract", "state"},
                )
                if str(current) != value:
                    raise LauncherError("path is noncanonical")
            except (LauncherError, OSError, TypeError, ValueError):
                errors.append(f"launch request path is unconfined: {name}")
        reference_path = Path(request_reference.get("path", ""))
        reference_path = reference_path if reference_path.is_absolute() \
            else ROOT / reference_path
        if paths.get("request") != str(request_path) \
                or reference_path.resolve(strict=True) != request_path:
            errors.append("launch request file path/reference differs")
        if isinstance(generation, int) and not isinstance(generation, bool):
            expected_commit = str(Path(paths["state"]).with_name(
                f"{Path(paths['state']).name}.commit.{generation + 1}.json"
            ))
            if paths.get("commit") != expected_commit:
                errors.append("launch commit path/generation differs")
    errors.extend(_command_errors(request.get("target"), validator=False))
    errors.extend(_command_errors(request.get("validator"), validator=True))
    errors.extend(_claim_protocol_errors(request))
    validator_argv = request.get("validator", {}).get("argv", []) \
        if isinstance(request.get("validator"), dict) else []
    if isinstance(paths, dict) and (
            "--request" not in validator_argv or paths.get("request") not in validator_argv
            or "--output" not in validator_argv or paths.get("output") not in validator_argv):
        errors.append("validator argv lacks exact request/output paths")
    if request.get("required_semantics") != {
            "exact_output": True, "parity_verified": True,
            "zero_skips": True, "skipped_count": 0,
    } or request.get("caller_selected_completion_paths_permitted") is not False \
            or request.get("heavy_work_before_commit_permitted") is not False:
        errors.append("launch semantic/authority boundary is invalid")
    if request.get("target", {}).get("environment_value_sha256s") \
            != _environment_hashes():
        errors.append("launcher environment differs from frozen target environment")
    selected = str(request.get("resource_claim", {}).get("selected_threads"))
    if any(os.environ.get(key) not in (None, selected) for key in (
            "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )):
        errors.append("launcher base environment conflicts with admitted threads")
    return errors


def _owner_field(phase: str) -> str:
    return {
        "prepare": "prepare_owner", "encoder": "encoder_owner",
        "finalizer": "serial_finalizer_owner", "companion": "companion_owner",
    }[phase]


def _verify_committed_owner(request: dict[str, Any], state: dict[str, Any],
                            *, commit: dict[str, Any]) -> dict[str, Any]:
    if state.get("state_sha256") != _hash_value(_without(state, "state_sha256")) \
            or state.get("contract_sha256") != commit.get("contract_sha256") \
            or isinstance(state.get("state_generation"), bool) \
            or not isinstance(state.get("state_generation"), int) \
            or state["state_generation"] < commit.get("new_state_generation", -1):
        raise LauncherError("committed state is invalid or older than launch commit")
    generation = commit["new_state_generation"]
    events = state.get("events")
    if not isinstance(events, list) or len(events) < generation:
        raise LauncherError("committed launch event is absent")
    event = events[generation - 1]
    proof = event.get("payload", {}).get("proof", {}) \
        if isinstance(event, dict) else {}
    if event.get("state_generation") != generation \
            or event.get("action") != request["action"] \
            or event.get("payload", {}).get("cell_id") != request["cell_id"] \
            or proof.get("launch_request_sha256") != request["request_sha256"] \
            or proof.get("inert_handshake_sha256") \
            != commit["inert_handshake_sha256"]:
        raise LauncherError("committed launch event differs from request/handshake")
    owner = state.get(_owner_field(request["phase"]))
    if not isinstance(owner, dict) or owner.get("cell_id") != request["cell_id"] \
            or owner.get("process_identity", {}).get("pid") != os.getpid() \
            or owner.get("proof", {}).get("launch_request_sha256") \
            != request["request_sha256"] \
            or owner.get("proof", {}).get("inert_handshake_sha256") \
            != commit["inert_handshake_sha256"]:
        raise LauncherError("active owner does not bind this inert launcher")
    return owner


def _verify_commit(request: dict[str, Any], request_reference: dict[str, Any],
                   handshake: dict[str, Any], commit: dict[str, Any]) \
        -> dict[str, Any]:
    if commit.get("schema") != CAS_SCHEMA or commit.get("version") != VERSION \
            or commit.get("commit_sha256") != _hash_value(
                _without(commit, "commit_sha256")
            ) or commit.get("launch_authorized") is not True:
        raise LauncherError("CAS commit is invalid or does not authorize launch")
    expected = request["expected_state_generation"]
    if isinstance(commit.get("previous_state_generation"), bool) \
            or not isinstance(commit.get("previous_state_generation"), int) \
            or isinstance(commit.get("new_state_generation"), bool) \
            or not isinstance(commit.get("new_state_generation"), int) \
            or commit.get("previous_state_generation") != expected \
            or commit.get("new_state_generation") != expected + 1 \
            or commit.get("action") != request["action"] \
            or commit.get("cell_id") != request["cell_id"] \
            or commit.get("launch_request_sha256") != request["request_sha256"] \
            or commit.get("launch_request_artifact_sha256") \
            != request_reference["sha256"] \
            or commit.get("inert_handshake_sha256") \
            != handshake["handshake_sha256"]:
        raise LauncherError("CAS commit differs from the frozen launch generation/request")
    contract, _ = _read_json(Path(request["paths"]["contract"]))
    if contract.get("contract_sha256") != _hash_value(
            _without(contract, "contract_sha256")) \
            or contract.get("contract_sha256") != commit.get("contract_sha256"):
        raise LauncherError("CAS commit contract is absent, drifted, or unrelated")
    entries = contract.get("invocation_policy", {}).get("entries", [])
    if not any(
            isinstance(entry, dict)
            and entry.get("entry_sha256") == commit.get(
                "phase_invocation_entry_sha256"
            )
            and any(
                isinstance(row, dict)
                and row.get("role") == "request"
                and row.get("reference") == request_reference
                for row in entry.get("argv_artifacts", [])
            )
            and entry.get("production_execution_protocol")
            == "inert-commit-v1-qualified"
            for entry in entries
    ):
        raise LauncherError("contract does not bind this request/entry")
    state, _ = _read_json(Path(request["paths"]["state"]))
    if state.get("state_generation") == expected + 1 \
            and state.get("state_sha256") != commit.get("new_state_sha256"):
        raise LauncherError("committed state hash differs at exact generation")
    return _verify_committed_owner(request, state, commit=commit)


def _target_claim_handshake_errors(value: Any, *, request: dict[str, Any],
                                   target_pid: int, launcher_pid: int,
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
    wall = value.get("created_wall_epoch")
    monotonic = value.get("created_monotonic_ns")
    if value.get("schema") != TARGET_CLAIM_HANDSHAKE_SCHEMA \
            or value.get("version") != VERSION or not valid_hash \
            or value.get("request_sha256") != request.get("request_sha256") \
            or value.get("resource_claim_sha256") \
            != request.get("resource_claim_sha256") \
            or value.get("launch_nonce") != request.get("launch_nonce") \
            or value.get("phase") != request.get("phase") \
            or value.get("cell_id") != request.get("cell_id") \
            or value.get("pid") != target_pid \
            or value.get("parent_pid") != launcher_pid \
            or value.get("target_executable_sha256") \
            != request.get("target", {}).get("executable", {}).get("sha256") \
            or value.get("target_argv_sha256") \
            != _hash_value(request.get("target", {}).get("argv")) \
            or value.get("applied_resource_controls") \
            != _applied_resource_controls(request) \
            or value.get("heavy_work_started") is not False \
            or value.get("awaiting_launcher_ack") is not True \
            or isinstance(wall, bool) or not isinstance(wall, (int, float)) \
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
                             target_pid: int) -> list[str]:
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
    wall, monotonic = (
        value.get("acknowledged_wall_epoch"),
        value.get("acknowledged_monotonic_ns"),
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
            or value.get("target_pid") != target_pid \
            or not isinstance(value.get("target_process_identity"), dict) \
            or value["target_process_identity"].get("pid") != target_pid \
            or value["target_process_identity"].get(
                "process_identity_sha256"
            ) != _hash_value(_without(
                value["target_process_identity"], "process_identity_sha256"
            )) \
            or value.get("launcher_pid") != os.getpid() \
            or value.get("heavy_work_authorized") is not True \
            or isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or isinstance(monotonic, bool) or not isinstance(monotonic, int) \
            or wall > time.time() + 0.001 \
            or monotonic > time.monotonic_ns() \
            or wall < target_handshake.get("created_wall_epoch", float("inf")) \
            or monotonic < target_handshake.get("created_monotonic_ns", 2 ** 63):
        return ["target resource-claim acknowledgement identity/timing is invalid"]
    return []


def _target_process_identity(pid: int) -> dict[str, Any]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,lstart=,command="],
        capture_output=True, text=True, timeout=5.0, check=False,
    )
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+"
        r"(\S+\s+\S+\s+\d+\s+\d\d:\d\d:\d\d\s+\d{4})\s+(.*)$"
    )
    row = None
    for line in completed.stdout.splitlines():
        match = pattern.match(line)
        if match is not None and int(match.group(1)) == pid:
            row = match
            break
    if completed.returncode != 0 or row is None:
        raise LauncherError("cannot bind exact target PID/start/command identity")
    started = "ps-lstart:" + " ".join(row.group(3).split())
    command_sha = hashlib.sha256(row.group(4).strip().encode("utf-8")).hexdigest()
    value: dict[str, Any] = {
        "schema": "hawking.doctor_v5_elastic_process_identity.v1",
        "version": VERSION, "pid": pid, "start_identity": started,
        "command_sha256": command_sha,
    }
    value["process_identity_sha256"] = _hash_value(value)
    return value


def _process_group_rss_bytes(pgid: int) -> tuple[int, int]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,rss="], capture_output=True,
        text=True, timeout=5.0, check=False,
    )
    if completed.returncode != 0:
        raise LauncherError("target tree RSS guard ps probe failed")
    total, members = 0, 0
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            _, row_pgid, rss_kib = map(int, fields)
        except ValueError:
            continue
        if row_pgid == pgid:
            members += 1
            total += rss_kib * 1024
    return total, members


def _start_orphan_watchdog(parent_pid: int, target_pgid: int) -> int:
    watchdog = os.fork()
    if watchdog != 0:
        return watchdog
    try:
        while True:
            try:
                os.killpg(target_pgid, 0)
            except ProcessLookupError:
                os._exit(0)
            if os.getppid() != parent_pid:
                try:
                    os.killpg(target_pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os._exit(0)
            time.sleep(0.05)
    except BaseException:
        try:
            os.killpg(target_pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        os._exit(70)


def _terminate_target(process: subprocess.Popen[Any]) -> int | None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)
    result = process.returncode
    return result if isinstance(result, int) and not isinstance(result, bool) else None


def _write_claim_failure(request: dict[str, Any], *, commit: dict[str, Any],
                         release: dict[str, Any], target_pid: int,
                         failure_code: str,
                         termination_returncode: int | None,
                         heavy_work_authorized: bool = False) -> None:
    value: dict[str, Any] = {
        "schema": TARGET_CLAIM_FAILURE_SCHEMA, "version": VERSION,
        "request_sha256": request["request_sha256"],
        "resource_claim_sha256": request["resource_claim_sha256"],
        "commit_sha256": commit["commit_sha256"],
        "release_sha256": release["release_sha256"],
        "target_pid": target_pid, "launcher_pid": os.getpid(),
        "failure_code": failure_code,
        "termination_returncode": termination_returncode,
        "failed_wall_epoch": time.time(),
        "failed_monotonic_ns": time.monotonic_ns(),
        "heavy_work_authorized": heavy_work_authorized,
    }
    value["failure_sha256"] = _hash_value(value)
    _atomic_exclusive_json(Path(request["paths"]["target_claim_failure"]), value)


def _run_claim_bound_target(request: dict[str, Any], *, commit: dict[str, Any],
                            release: dict[str, Any]) \
        -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any],
                 dict[str, Any]]:
    target = request["target"]
    environment = dict(os.environ)
    environment.update(target["resource_environment"])
    process: subprocess.Popen[Any] | None = None
    watchdog_pid: int | None = None
    failure_code = "target-claim-handshake-runtime-error"
    old_handlers: dict[int, Any] = {}
    try:
        process = subprocess.Popen(
            target["argv"], cwd=target["cwd"], env=environment,
            start_new_session=True,
        )
        watchdog_pid = _start_orphan_watchdog(os.getpid(), process.pid)

        def forward_signal(signum: int, _frame: Any) -> None:
            if process is not None:
                _terminate_target(process)
            raise LauncherError(f"launcher signal {signum} forwarded to target group")

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)

        target_handshake_path = Path(request["paths"]["target_claim_handshake"])
        target_handshake = None
        failure_code = "target-claim-handshake-timeout"
        deadline = time.monotonic() + float(
            request["target_claim_handshake_timeout_seconds"]
        )
        while time.monotonic() < deadline:
            if target_handshake_path.is_file() \
                    and not target_handshake_path.is_symlink():
                try:
                    candidate, _ = _read_json(target_handshake_path)
                except (LauncherError, OSError, TypeError, ValueError):
                    failure_code = "target-claim-handshake-malformed"
                    raise LauncherError(
                        "target resource-claim handshake is malformed/partial"
                    )
                failures = _target_claim_handshake_errors(
                    candidate, request=request, target_pid=process.pid,
                    launcher_pid=os.getpid(), release=release,
                )
                if failures:
                    failure_code = "target-claim-handshake-invalid"
                    raise LauncherError("; ".join(failures))
                target_handshake = candidate
                break
            if process.poll() is not None:
                failure_code = "target-exited-before-claim-handshake"
                raise LauncherError("target exited before resource-claim handshake")
            time.sleep(0.01)
        if target_handshake is None:
            raise LauncherError("target resource-claim handshake timed out")

        target_identity = _target_process_identity(process.pid)
        ack: dict[str, Any] = {
            "schema": TARGET_CLAIM_ACK_SCHEMA, "version": VERSION,
            "request_sha256": request["request_sha256"],
            "resource_claim_sha256": request["resource_claim_sha256"],
            "target_claim_handshake_sha256": target_handshake[
                "handshake_sha256"
            ],
            "commit_sha256": commit["commit_sha256"],
            "release_sha256": release["release_sha256"],
            "target_pid": process.pid,
            "target_process_identity": target_identity,
            "launcher_pid": os.getpid(), "heavy_work_authorized": True,
            "acknowledged_wall_epoch": time.time(),
            "acknowledged_monotonic_ns": time.monotonic_ns(),
        }
        ack["ack_sha256"] = _hash_value(ack)
        _atomic_exclusive_json(Path(request["paths"]["target_claim_ack"]), ack)

        max_rss, sample_count = 0, 0
        reservation = request["resource_claim"]["reservation_bytes"]
        while True:
            rss, _members = _process_group_rss_bytes(process.pid)
            max_rss = max(max_rss, rss)
            sample_count += 1
            if reservation is not None and rss > reservation:
                failure_code = "target-tree-rss-limit-exceeded"
                raise LauncherError("target tree RSS exceeded admitted reservation")
            if process.poll() is not None:
                break
            time.sleep(0.025)
        returncode = process.wait()
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            failure_code = "target-returncode-invalid"
            raise LauncherError("target returncode is invalid")
        guard: dict[str, Any] = {
            "schema": TARGET_RESOURCE_GUARD_SCHEMA, "version": VERSION,
            "request_sha256": request["request_sha256"],
            "resource_claim_sha256": request["resource_claim_sha256"],
            "target_claim_handshake_sha256": target_handshake[
                "handshake_sha256"
            ],
            "target_claim_ack_sha256": ack["ack_sha256"],
            "target_process_identity": target_identity,
            "process_group_id": process.pid,
            "reservation_bytes": reservation,
            "selected_threads": request["resource_claim"]["selected_threads"],
            "authoritative_thread_environment": _applied_resource_controls(
                request
            )["thread_environment"],
            "sample_count": sample_count, "max_tree_rss_bytes": max_rss,
            "rss_limit_exceeded": False,
            "target_returncode": returncode,
            "probe": "/bin/ps -axo pid=,pgid=,rss=",
            "guard_complete": True,
        }
        guard["guard_sha256"] = _hash_value(guard)
        _atomic_exclusive_json(
            Path(request["paths"]["target_resource_guard"]), guard
        )
        return returncode, target_handshake, ack, target_identity, guard
    except BaseException:
        terminated = _terminate_target(process) if process is not None else None
        failure_path = Path(request["paths"]["target_claim_failure"])
        if process is not None and not failure_path.exists():
            try:
                _write_claim_failure(
                    request, commit=commit, release=release,
                    target_pid=process.pid, failure_code=failure_code,
                    termination_returncode=terminated,
                    heavy_work_authorized=Path(
                        request["paths"]["target_claim_ack"]
                    ).is_file(),
                )
            except (LauncherError, OSError, TypeError, ValueError):
                pass
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if watchdog_pid is not None:
            try:
                os.waitpid(watchdog_pid, os.WNOHANG)
            except ChildProcessError:
                pass


def run(request_path: Path) -> int:
    request_path = _no_symlink_path(
        request_path, below=STAGE_ROOT, must_exist=True
    )
    request, request_file_reference = _read_json(request_path)
    request_errors = _request_errors(
        request, request_path=request_path,
        request_reference=request_file_reference,
    )
    if request_errors:
        raise LauncherError("launch request failed: " + "; ".join(request_errors))

    handshake: dict[str, Any] = {
        "schema": HANDSHAKE_SCHEMA, "version": VERSION,
        "request_sha256": request["request_sha256"],
        "request_artifact_sha256": request_file_reference["sha256"],
        "launch_nonce": request["launch_nonce"], "pid": os.getpid(),
        "phase": request["phase"], "cell_id": request["cell_id"],
        "action": request["action"],
        "expected_state_generation": request["expected_state_generation"],
        "target_started": False, "heavy_work_started": False,
        "created_wall_epoch": time.time(),
        "created_monotonic_ns": time.monotonic_ns(),
        "launcher": request["launcher"],
    }
    handshake["handshake_sha256"] = _hash_value(handshake)
    handshake_path = Path(request["paths"]["handshake"])
    _atomic_exclusive_json(handshake_path, handshake)

    commit_path = Path(request["paths"]["commit"])
    deadline = time.monotonic() + float(request["commit_wait_timeout_seconds"])
    commit = None
    while time.monotonic() < deadline:
        if commit_path.is_file() and not commit_path.is_symlink():
            commit, _ = _read_json(commit_path)
            break
        time.sleep(0.025)
    if commit is None:
        raise LauncherError("generation-bound CAS commit did not arrive before timeout")
    owner = _verify_commit(request, request_file_reference, handshake, commit)

    # The commit wait is an adversarial drift window.  Re-read the exact
    # request bytes and every command/cwd/semantic/environment binding now,
    # immediately before authorizing a target spawn.
    current_request, current_request_reference = _read_json(request_path)
    if current_request != request or current_request_reference != request_file_reference:
        raise LauncherError("launch request changed while waiting for CAS commit")
    request_errors = _request_errors(
        current_request, request_path=request_path,
        request_reference=current_request_reference,
    )
    if request_errors:
        raise LauncherError("post-CAS launch revalidation failed: "
                            + "; ".join(request_errors))
    current_commit, current_commit_reference = _read_json(commit_path)
    if current_commit != commit:
        raise LauncherError("CAS commit changed before target spawn")
    owner = _verify_commit(
        request, request_file_reference, handshake, current_commit
    )

    release: dict[str, Any] = {
        "schema": RELEASE_SCHEMA, "version": VERSION,
        "request_sha256": request["request_sha256"],
        "handshake_sha256": handshake["handshake_sha256"],
        "commit_sha256": commit["commit_sha256"],
        "state_generation": commit["new_state_generation"],
        "pid": os.getpid(), "launch_released": True,
        "target_start_authorized": True, "released_wall_epoch": time.time(),
        "released_monotonic_ns": time.monotonic_ns(),
    }
    release["release_sha256"] = _hash_value(release)
    _atomic_exclusive_json(Path(request["paths"]["release"]), release)

    target = request["target"]
    (completed_returncode, target_handshake, ack, target_identity,
     resource_guard) = _run_claim_bound_target(
        request, commit=commit, release=release
    )
    # Bind a non-child-observable target result to the leased wrapper.  This is
    # written even for nonzero exits so the scheduler can fail closed on it.
    current_request, current_request_reference = _read_json(request_path)
    if current_request != request or current_request_reference != request_file_reference \
            or _command_errors(request["target"], validator=False) \
            or _command_errors(request["validator"], validator=True) \
            or request["target"]["environment_value_sha256s"] \
            != _environment_hashes():
        raise LauncherError("launch source/request/environment drifted during target")
    state, _ = _read_json(Path(request["paths"]["state"]))
    owner = _verify_committed_owner(request, state, commit=commit)
    output_path = _no_symlink_path(
        Path(request["paths"]["output"]), below=STAGE_ROOT, must_exist=True
    )
    if not output_path.is_file():
        raise LauncherError("target did not produce one regular output artifact")
    output_reference = _file_reference(output_path)
    release_document, release_reference = _read_json(
        Path(request["paths"]["release"])
    )
    if release_document != release:
        raise LauncherError("inert release receipt changed during target")
    persisted_target_handshake, target_handshake_reference = _read_json(
        Path(request["paths"]["target_claim_handshake"])
    )
    persisted_ack, target_ack_reference = _read_json(
        Path(request["paths"]["target_claim_ack"])
    )
    persisted_guard, resource_guard_reference = _read_json(
        Path(request["paths"]["target_resource_guard"])
    )
    if persisted_target_handshake != target_handshake \
            or _target_claim_handshake_errors(
                persisted_target_handshake, request=request,
                target_pid=target_identity["pid"], launcher_pid=os.getpid(),
                release=release,
            ) or persisted_ack != ack or _target_claim_ack_errors(
                persisted_ack, request=request,
                target_handshake=target_handshake, commit=commit,
                release=release, target_pid=target_identity["pid"],
            ) or persisted_guard != resource_guard \
            or resource_guard.get("guard_sha256") != _hash_value(
                _without(resource_guard, "guard_sha256")
            ) or Path(request["paths"]["target_claim_failure"]).exists():
        raise LauncherError("target resource-claim handshake/ack drifted")
    phase = ("companion_checkpoint" if request["phase"] == "companion"
             else request["phase"])
    worker: dict[str, Any] = {
        "schema": "hawking.doctor_v5_elastic_worker_completion.v1",
        "version": VERSION, "contract_sha256": commit["contract_sha256"],
        "phase": phase, "cell_id": request["cell_id"],
        "process_identity_sha256": owner["process_identity"][
            "process_identity_sha256"
        ],
        "owner_lease_sha256": owner["lease"]["lease_sha256"],
        "launch_request_sha256": request["request_sha256"],
        "inert_handshake_sha256": handshake["handshake_sha256"],
        "commit_sha256": commit["commit_sha256"],
        "release_sha256": release["release_sha256"],
        "resource_claim_sha256": request["resource_claim_sha256"],
        "target_claim_handshake_sha256": target_handshake["handshake_sha256"],
        "target_claim_ack_sha256": ack["ack_sha256"],
        "target_process_identity": target_identity,
        "target_resource_guard_sha256": resource_guard["guard_sha256"],
        "target_returncode": int(completed_returncode),
        "output_reference": output_reference,
        "checkpoint": phase == "companion_checkpoint", "complete": True,
        "source_files_deleted": False, "completed_evidence_mutated": False,
    }
    worker["worker_completion_sha256"] = _hash_value(worker)
    _atomic_exclusive_json(Path(request["paths"]["worker_receipt"]), worker)
    persisted_worker, _ = _read_json(Path(request["paths"]["worker_receipt"]))
    if persisted_worker != worker or not current_commit_reference.get("sha256") \
            or not release_reference.get("sha256") \
            or not target_handshake_reference.get("sha256") \
            or not target_ack_reference.get("sha256") \
            or not resource_guard_reference.get("sha256"):
        raise LauncherError("worker completion receipt persistence differs")
    return int(completed_returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.request)
    except (LauncherError, OSError, TypeError, ValueError, KeyError) as exc:
        print(f"inert launcher blocked: {exc}", file=os.sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
