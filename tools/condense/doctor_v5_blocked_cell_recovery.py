#!/usr/bin/env python3.12
"""Fail-closed recovery scaffold for one checkpointed Doctor V5 cell.

The normal commands are read-only ``status`` and ``verify`` plus ``stage``,
which may write only below ``staged_acceleration/blocked_cell_recovery_v1``.
The opt-in ``apply`` command exists for a later, explicitly authorized,
owner-free checkpoint.  It cannot pause or drain the live campaign.  It needs
two independent activation keys, both campaign locks, an exact staged state
generation, and a full re-hash of every completed worker-checkpoint artifact.

This module is intentionally specific to
``qwen2-5-14b__4bpw__codec-control``.  General retry/reset behavior belongs in
the reviewed queue, not in an incident-recovery utility.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
from typing import Any, Callable, IO


ROOT = Path(__file__).resolve().parents[2]
ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
TARGET_CELL_ID = "qwen2-5-14b__4bpw__codec-control"
SCHEMA = "hawking.doctor_v5_blocked_cell_recovery.v1"
PACKET_SCHEMA = "hawking.doctor_v5_blocked_cell_recovery_packet.v1"
INTENT_SCHEMA = "hawking.doctor_v5_blocked_cell_recovery_intent.v1"
RECEIPT_SCHEMA = "hawking.doctor_v5_blocked_cell_recovery_receipt.v1"
RESUME_SCHEMA = "hawking.doctor_v5_blocked_cell_recovery_resume.v1"
VERSION = "2026-07-14.1"
STAGE_DIRNAME = "blocked_cell_recovery_v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")

# Exact incident-generation pins.  A later campaign generation must get a new,
# reviewed recovery tool rather than silently inheriting this exception.
EXPECTED_PLAN_SHA256 = "3d254b5f7fcc5f02b55f2a71f306f7f6852839b699fd14ab4ddf5a05dbaa0106"
EXPECTED_CELL_IDENTITY_SHA256 = (
    "0a70063f67e268d05c355bfcaa0b8c8127ba6146ba557761355350f71922c179"
)
EXPECTED_RUNTIME_FILE_SHA256 = (
    "a0b9dc67124af41f9994f9ca24061fb4db5f8b598a94f503365eae332fc52441"
)
EXPECTED_REQUEST_SHA256 = (
    "0a3b5f535ae9aebd3e57ac82a0d0acc233bdfc700a5262e751b3ae2eaee721ae"
)
EXPECTED_REGISTRY_SHA256 = (
    "27c8daeef9abb20dafe7a44d080914e34ff1fea87e763963ff7a3f0bc8f5209f"
)
EXPECTED_ADAPTER_CHECKPOINT_FILE_SHA256 = (
    "950c52cb958385d0e869b45c427fcfea07b5e171d9bfdea1fedaf2da5a00225c"
)
EXPECTED_WORKER_CHECKPOINT_FILE_SHA256 = (
    "b1b78a131d22e06fe6a1c9ea29286809510d91e9eb35f14ac9be9be39dd859ff"
)
EXPECTED_WORKER_REQUEST_FILE_SHA256 = (
    "16974a341131142ba038daeb0c0b93ffd354cc48de61c0f011be7ce7b99d91ef"
)
EXPECTED_AUTORESUME_SHA256 = (
    "831b19e90257d867377b9c1d2ef43626dfaa154e58bffd608167cbe7b47eb758"
)
EXPECTED_ACCEL_QUEUE_SHA256 = (
    "4ceeb89d147989c04ccb7bd8ee0fb1744d5777f8d124d2b345e4a07f6235e57c"
)
EXPECTED_BINARIES = {
    "attestor": {
        "sha256": "d431a04f37ee45cb899f691bc5bae913e2ad8a9271d6db94b027bbe24c85787b",
        "bytes": 827_904,
    },
    "decoder": {
        "sha256": "e1cec500c39fef02a02e63ed00c7a0971484da125454e793f06e9ba37054f676",
        "bytes": 729_472,
    },
}

HEAVY_COMMAND_PATTERNS = (
    "doctor_v5_ultra_queue.py run",
    "doctor_v5_ultra_accelerated_queue.py run",
    "doctor_v5_strand_ladder_adapter.py run",
    "doctor_v5_strand_ladder_block_parallel_adapter.py run",
    "doctor_v5_strand_ladder_worker.py",
    "doctor_v5_strand_ladder_block_parallel_worker.py",
    "quantize-model",
    "strand-quant",
    "hawking-quant",
    "condense_ladder",
    "audit_ladder.py",
    "processing_queue.py run",
    "studio_run.py",
    "appendix_device_runner.py",
    "spec_tq_runner.py",
    "hawking-tq-device-probe",
    "hawking-tq-spec-probe",
    "probe-metal-rht",
    "native_probe.py",
    "mop_generation1_campaign.py",
    "generation1_cognitive_corpus.py",
)


class RecoveryError(RuntimeError):
    """Controlled fail-closed refusal."""


@dataclass(frozen=True)
class RecoveryPaths:
    root: Path
    ultra: Path
    plan: Path
    state: Path
    control: Path
    pid_file: Path
    queue_lock: Path
    heavy_lock: Path
    runtime_spec: Path
    result_dir: Path
    request: Path
    registry_snapshot: Path
    live_registry: Path
    adapter_checkpoint: Path
    worker_checkpoint: Path
    worker_request: Path
    execution_log: Path
    result: Path
    execution_receipt: Path
    active_marker: Path
    accelerated_autoresume: Path
    accelerated_queue: Path
    stage_root: Path
    packet: Path
    recovery_lock: Path
    intent: Path
    receipt: Path
    resume_receipt: Path
    resume_log: Path


@dataclass(frozen=True)
class RecoveryPins:
    plan_sha256: str
    cell_identity_sha256: str
    runtime_file_sha256: str
    request_sha256: str
    registry_sha256: str
    adapter_checkpoint_file_sha256: str
    worker_checkpoint_file_sha256: str
    worker_request_file_sha256: str
    autoresume_sha256: str
    accelerated_queue_sha256: str
    binaries: dict[str, dict[str, Any]]
    attempts: int = 14


PRODUCTION_PINS = RecoveryPins(
    plan_sha256=EXPECTED_PLAN_SHA256,
    cell_identity_sha256=EXPECTED_CELL_IDENTITY_SHA256,
    runtime_file_sha256=EXPECTED_RUNTIME_FILE_SHA256,
    request_sha256=EXPECTED_REQUEST_SHA256,
    registry_sha256=EXPECTED_REGISTRY_SHA256,
    adapter_checkpoint_file_sha256=EXPECTED_ADAPTER_CHECKPOINT_FILE_SHA256,
    worker_checkpoint_file_sha256=EXPECTED_WORKER_CHECKPOINT_FILE_SHA256,
    worker_request_file_sha256=EXPECTED_WORKER_REQUEST_FILE_SHA256,
    autoresume_sha256=EXPECTED_AUTORESUME_SHA256,
    accelerated_queue_sha256=EXPECTED_ACCEL_QUEUE_SHA256,
    binaries=EXPECTED_BINARIES,
)


def production_paths(root: Path = ROOT) -> RecoveryPaths:
    ultra = root / "reports/condense/doctor_v5_ultra"
    result_dir = ultra / "results" / TARGET_CELL_ID
    stage_root = ultra / "staged_acceleration" / STAGE_DIRNAME
    return RecoveryPaths(
        root=root, ultra=ultra,
        plan=ultra / "campaign_plan.json",
        state=ultra / "queue_state.json",
        control=ultra / "control.json",
        pid_file=ultra / "queue.pid.json",
        queue_lock=ultra / "queue.lock",
        heavy_lock=root / "reports/cron/studio_heavy.lock",
        runtime_spec=ultra / "runtime_specs" / f"{TARGET_CELL_ID}.json",
        result_dir=result_dir,
        request=result_dir / "request.json",
        registry_snapshot=result_dir / "adapter_registry.json",
        live_registry=ultra / "adapter_registry.json",
        adapter_checkpoint=result_dir / "checkpoint.json",
        worker_checkpoint=result_dir / "strand_ladder/checkpoint.json",
        worker_request=result_dir / "strand_ladder/request.json",
        execution_log=result_dir / "execution.log",
        result=result_dir / "result.json",
        execution_receipt=result_dir / "execution_receipt.json",
        active_marker=ultra / "staged_acceleration/active_stack.json",
        accelerated_autoresume=root / "tools/condense/doctor_v5_ultra_accelerated_autoresume.py",
        accelerated_queue=root / "tools/condense/doctor_v5_ultra_accelerated_queue.py",
        stage_root=stage_root,
        packet=stage_root / "recovery_packet.json",
        recovery_lock=stage_root / "recovery.lock",
        intent=stage_root / "apply_intent.json",
        receipt=stage_root / "apply_receipt.json",
        resume_receipt=stage_root / "resume_receipt.json",
        resume_log=stage_root / "autoresume.log",
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: row for name, row in value.items() if name != key}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _confined(path: Path, root: Path, *, must_exist: bool = True) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        lexical = Path(os.path.abspath(path))
        relative = lexical.relative_to(resolved_root)
        cursor = resolved_root
        for component in relative.parts:
            cursor /= component
            if cursor.is_symlink():
                raise RecoveryError(f"symlink component is not permitted: {cursor}")
        resolved = path.resolve(strict=must_exist)
        resolved.relative_to(resolved_root)
    except RecoveryError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise RecoveryError(f"path is not confined to workspace: {path}") from exc
    return resolved


def _hash_regular(path: Path, root: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    resolved = _confined(path, root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise RecoveryError(f"cannot open bound file: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryError(f"bound path is not a regular file: {path}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise RecoveryError(f"bound file exceeds size ceiling: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        if identity(before) != identity(after):
            raise RecoveryError(f"bound file changed while hashing: {path}")
        return digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def _read_json(path: Path, root: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    resolved = _confined(path, root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise RecoveryError(f"JSON is not a bounded regular file: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk); total += len(chunk)
            if total > max_bytes:
                raise RecoveryError(f"JSON exceeds size ceiling: {path}")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RecoveryError(f"cannot read JSON: {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or total != after.st_size:
        raise RecoveryError(f"JSON changed while reading: {path}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = child
        return result

    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON root is not an object: {path}")
    return value


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    digest, size = _hash_regular(path, root)
    return {"path": str(path.resolve(strict=True)), "sha256": digest, "bytes": size}


def _stat_artifact(path: Path, root: Path) -> dict[str, Any]:
    resolved = _confined(path, root)
    row = resolved.stat()
    if not stat.S_ISREG(row.st_mode):
        raise RecoveryError(f"artifact is not regular: {path}")
    return {"path": str(resolved), "bytes": row.st_size}


def _artifact_matches(row: Any, expected: Path, root: Path, *, hash_payload: bool = True) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    try:
        if Path(row["path"]).resolve(strict=True) != expected.resolve(strict=True):
            return False
        if hash_payload:
            digest, size = _hash_regular(expected, root)
            return (digest, size) == (row["sha256"], row["bytes"])
        observed = _stat_artifact(expected, root)
        return observed["bytes"] == row["bytes"]
    except (OSError, TypeError, ValueError, RecoveryError):
        return False


def _semantic_hash_errors(document: dict[str, Any], field: str, label: str) -> list[str]:
    value = document.get(field)
    if not _valid_sha(value) or value != _hash_value(_without(document, field)):
        return [f"{label} semantic hash is invalid"]
    return []


def _process_identity(pid: Any) -> tuple[str, str] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    try:
        command = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        started = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return (command, started) if command and started else None


def _supervisor_alive(record: Any) -> bool:
    if not isinstance(record, dict) or record.get("schema") \
            != "hawking.doctor_v5_ultra_queue_pid.v1" \
            or record.get("pid_record_sha256") \
            != _hash_value(_without(record, "pid_record_sha256")):
        return False
    identity = _process_identity(record.get("pid"))
    if identity is None:
        return False
    command, started = identity
    return started == record.get("process_started") \
        and hashlib.sha256(command.encode("utf-8")).hexdigest() \
        == record.get("process_command_sha256")


def active_heavy_owners() -> list[dict[str, Any]]:
    """Trusted local ps observation.  It never sends a signal."""
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,lstart=,command="], capture_output=True,
            text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError(f"cannot observe local process inventory: {exc}") from exc
    own_pid = os.getpid()
    owners: list[dict[str, Any]] = []
    for raw in output.splitlines():
        match = re.match(r"\s*(\d+)\s+(.{24})\s+(.*)$", raw)
        if match is None:
            raise RecoveryError("local process inventory contains an unparsed row")
        pid, started, command = int(match.group(1)), match.group(2).strip(), match.group(3)
        lowered = command.lower()
        if pid == own_pid or not any(pattern in lowered for pattern in HEAVY_COMMAND_PATTERNS):
            continue
        owners.append({
            "pid": pid, "process_started": started, "command": command,
            "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        })
    return sorted(owners, key=lambda row: (row["pid"], row["command_sha256"]))


def _lock_available(path: Path, root: Path) -> bool:
    resolved = _confined(path, root)
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def _validate_worker_checkpoint(value: dict[str, Any], paths: RecoveryPaths,
                                *, full: bool) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    plan, completed, units = value.get("plan"), value.get("completed_units"), value.get("units")
    if value.get("schema") != "hawking.doctor_v5_strand_ladder_checkpoint.v1":
        errors.append("worker checkpoint schema mismatch")
    if not isinstance(plan, list) or any(not isinstance(row, str) for row in plan) \
            or len(plan) != len(set(plan)):
        return errors + ["worker checkpoint plan is invalid"], {}
    if not isinstance(completed, list) or completed != plan[:len(completed)] \
            or len(completed) != len(set(completed)):
        errors.append("worker completed units are not an exact plan prefix")
        completed = []
    expected_encodes = {f"encode:{index:05d}" for index in range(8)}
    expected_attests = {f"attest:{index:05d}" for index in range(7)}
    expected_decodes = {f"decode:{index:05d}" for index in range(7)}
    observed = set(completed)
    if not expected_encodes <= observed or not expected_attests <= observed \
            or not expected_decodes <= observed:
        errors.append("worker checkpoint lacks the exact 8-encode/7-attest/7-decode prefix")
    if {"attest:00007", "decode:00007"} & observed:
        errors.append("worker checkpoint unexpectedly advanced beyond encode:00007")
    if not completed or completed[-1] != "encode:00007" \
            or (len(completed) >= len(plan)) or plan[len(completed)] != "attest:00007":
        errors.append("worker checkpoint next unit is not attest:00007")
    if not isinstance(units, dict) or set(units) != observed \
            or any(not isinstance(units.get(unit), dict) for unit in completed):
        errors.append("worker checkpoint unit evidence is noncanonical")
        units = {}
    refs: dict[tuple[str, str, int], dict[str, Any]] = {}

    def collect(child: Any) -> None:
        if isinstance(child, dict):
            if set(child) >= {"path", "sha256", "bytes"} \
                    and isinstance(child.get("path"), str) \
                    and _valid_sha(child.get("sha256")) \
                    and isinstance(child.get("bytes"), int) \
                    and not isinstance(child.get("bytes"), bool):
                refs[(child["path"], child["sha256"], child["bytes"])] = {
                    "path": child["path"], "sha256": child["sha256"],
                    "bytes": child["bytes"],
                }
            for row in child.values():
                collect(row)
        elif isinstance(child, list):
            for row in child:
                collect(row)

    for unit in completed:
        collect(units.get(unit))
    for row in refs.values():
        try:
            artifact_path = Path(row["path"])
            resolved = _confined(artifact_path, paths.result_dir)
            if resolved.stat().st_size != row["bytes"]:
                errors.append(f"checkpoint artifact byte count changed: {resolved}")
                continue
            if full and _hash_regular(resolved, paths.root)[0] != row["sha256"]:
                errors.append(f"checkpoint artifact hash changed: {resolved}")
        except (OSError, RecoveryError) as exc:
            errors.append(f"checkpoint artifact invalid: {row.get('path')}: {exc}")
    summary = {
        "completed_unit_count": len(completed),
        "completed_units_sha256": _hash_value(completed),
        "last_completed_unit": completed[-1] if completed else None,
        "next_unit": plan[len(completed)] if len(completed) < len(plan) else None,
        "referenced_artifact_count": len(refs),
        "referenced_bytes": sum(row["bytes"] for row in refs.values()),
        "full_payload_hashes_verified": full and not errors,
    }
    return errors, summary


def _validate_marker(paths: RecoveryPaths, pins: RecoveryPins) \
        -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    try:
        marker = _read_json(paths.active_marker, paths.root)
    except RecoveryError as exc:
        return [str(exc)], None
    if marker.get("schema") != "hawking.doctor_v5_acceleration_active_marker.v1":
        errors.append("active acceleration marker schema mismatch")
    errors += _semantic_hash_errors(marker, "marker_sha256", "active marker")
    for field, expected_path, expected_sha in (
        ("accelerated_autoresume", paths.accelerated_autoresume, pins.autoresume_sha256),
        ("accelerated_queue", paths.accelerated_queue, pins.accelerated_queue_sha256),
    ):
        row = marker.get(field)
        if not _artifact_matches(row, expected_path, paths.root):
            errors.append(f"active marker {field} binding changed")
        elif row.get("sha256") != expected_sha:
            errors.append(f"active marker {field} generation mismatch")
    try:
        overlay = Path(marker.get("overlay_path", ""))
        resolved = _confined(overlay, paths.ultra / "staged_acceleration")
        observed = _read_json(resolved, paths.root)
        if observed.get("overlay_sha256") != marker.get("overlay_sha256") \
                or observed.get("overlay_sha256") \
                != _hash_value(_without(observed, "overlay_sha256")):
            errors.append("active marker overlay binding is invalid")
    except (TypeError, RecoveryError) as exc:
        errors.append(f"active marker overlay is invalid: {exc}")
    return errors, marker


def inspect_recovery(paths: RecoveryPaths, pins: RecoveryPins = PRODUCTION_PINS,
                     *, full: bool = False,
                     owner_observer: Callable[[], list[dict[str, Any]]] = active_heavy_owners,
                     probe_locks: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    docs: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("plan", paths.plan), ("state", paths.state), ("control", paths.control),
        ("runtime", paths.runtime_spec), ("request", paths.request),
        ("registry_snapshot", paths.registry_snapshot),
        ("live_registry", paths.live_registry),
        ("adapter_checkpoint", paths.adapter_checkpoint),
        ("worker_checkpoint", paths.worker_checkpoint),
        ("worker_request", paths.worker_request),
    ):
        try:
            docs[name] = _read_json(path, paths.root)
        except RecoveryError as exc:
            errors.append(str(exc))
    if errors:
        return {
            "schema": SCHEMA, "target_cell_id": TARGET_CELL_ID,
            "inspected_at": _now(), "structurally_ready": False,
            "activation_permitted": False, "errors": errors,
            "blockers": ["required recovery input is unreadable"],
        }

    plan, state, control = docs["plan"], docs["state"], docs["control"]
    runtime, request = docs["runtime"], docs["request"]
    registry, live_registry = docs["registry_snapshot"], docs["live_registry"]
    outer, worker, worker_request = (
        docs["adapter_checkpoint"], docs["worker_checkpoint"], docs["worker_request"]
    )
    # Decide whether a high-bandwidth verification is admissible before reading
    # a single model/checkpoint payload.  A mistaken ``--full`` during the live
    # campaign therefore remains a cheap status probe.
    try:
        owner_record = _read_json(paths.pid_file, paths.root)
    except RecoveryError as exc:
        errors.append(str(exc))
        owner_record = {}
    if owner_record.get("schema") != "hawking.doctor_v5_ultra_queue_pid.v1" \
            or owner_record.get("plan_sha256") != pins.plan_sha256 \
            or owner_record.get("pid_record_sha256") \
            != _hash_value(_without(owner_record, "pid_record_sha256")):
        errors.append("queue owner record identity is invalid")
    supervisor_alive = _supervisor_alive(owner_record)
    try:
        owners = owner_observer()
    except RecoveryError as exc:
        errors.append(str(exc)); owners = []
    if not isinstance(owners, list) or any(not isinstance(row, dict) for row in owners):
        errors.append("heavy-owner observer returned invalid data"); owners = []
    active_children, active_cells = state.get("active_children"), state.get("active_cells")
    if supervisor_alive:
        blockers.append("detached Doctor supervisor is active")
    if state.get("supervisor_pid") is not None:
        blockers.append("queue state still records a supervisor")
    if state.get("status") not in {"drained", "waiting-prerequisites"}:
        blockers.append("queue is not at an admitted quiescent status")
    if not isinstance(active_children, dict) or active_children:
        blockers.append("queue has active children")
    if not isinstance(active_cells, list) or active_cells:
        blockers.append("queue has active cells")
    if owners:
        blockers.append("one or more heavy owners are active")
    if control.get("mode") != "run":
        blockers.append("control is not already in run mode")
    if paths.intent.exists() or paths.receipt.exists():
        blockers.append("recovery has already been prepared or applied")
    queue_lock_available = heavy_lock_available = False
    if probe_locks:
        try:
            queue_lock_available = _lock_available(paths.queue_lock, paths.root)
            heavy_lock_available = _lock_available(paths.heavy_lock, paths.root)
        except RecoveryError as exc:
            errors.append(str(exc))
        if not queue_lock_available:
            blockers.append("queue singleton lock is unavailable")
        if not heavy_lock_available:
            blockers.append("campaign-wide heavy lock is unavailable")
    effective_full = full and not errors and not blockers
    if full and not effective_full:
        blockers.append("full payload hashing refused until the owner-free gate passes")

    errors += _semantic_hash_errors(plan, "plan_sha256", "campaign plan")
    errors += _semantic_hash_errors(state, "state_sha256", "queue state")
    errors += _semantic_hash_errors(control, "control_sha256", "control")
    errors += _semantic_hash_errors(request, "request_sha256", "adapter request")
    errors += _semantic_hash_errors(registry, "registry_sha256", "registry snapshot")
    errors += _semantic_hash_errors(live_registry, "registry_sha256", "live registry")
    errors += _semantic_hash_errors(outer, "checkpoint_sha256", "adapter checkpoint")
    if plan.get("plan_sha256") != pins.plan_sha256 \
            or state.get("plan_sha256") != pins.plan_sha256 \
            or control.get("plan_sha256") != pins.plan_sha256:
        errors.append("plan generation differs from the incident pin")
    try:
        runtime_file = _artifact(paths.runtime_spec, paths.root)
        if runtime_file["sha256"] != pins.runtime_file_sha256:
            errors.append("target runtime spec file changed")
        if _artifact(paths.adapter_checkpoint, paths.root)["sha256"] \
                != pins.adapter_checkpoint_file_sha256:
            errors.append("historical adapter checkpoint file changed")
        if _artifact(paths.worker_checkpoint, paths.root)["sha256"] \
                != pins.worker_checkpoint_file_sha256:
            errors.append("authoritative worker checkpoint file changed")
        if _artifact(paths.worker_request, paths.root)["sha256"] \
                != pins.worker_request_file_sha256:
            errors.append("worker request file changed")
    except RecoveryError as exc:
        errors.append(str(exc))

    cells = plan.get("cells")
    plan_cell = next((row for row in cells if isinstance(row, dict)
                      and row.get("cell_id") == TARGET_CELL_ID), None) \
        if isinstance(cells, list) else None
    state_cells = state.get("cells")
    target_row = state_cells.get(TARGET_CELL_ID) \
        if isinstance(state_cells, dict) else None
    if not isinstance(plan_cell, dict) or plan_cell.get("cell_identity_sha256") \
            != pins.cell_identity_sha256:
        errors.append("target plan cell identity changed")
    if not isinstance(target_row, dict):
        errors.append("target state row is missing")
        target_row = {}
    if target_row.get("status") != "blocked-execution" \
            or target_row.get("attempts") != pins.attempts \
            or target_row.get("last_exit_code") != 2 \
            or target_row.get("error") != "typed adapter exited with status 2" \
            or target_row.get("blockers") != ["typed adapter exited with status 2"]:
        errors.append("target state row is not the exact incident failure")
    if target_row.get("runtime_spec_sha256") != pins.runtime_file_sha256 \
            or target_row.get("request_sha256") != pins.request_sha256 \
            or target_row.get("registry_sha256") != pins.registry_sha256:
        errors.append("target state row source bindings changed")
    if request.get("request_sha256") != pins.request_sha256 \
            or request.get("registry_sha256") != pins.registry_sha256 \
            or registry.get("registry_sha256") != pins.registry_sha256 \
            or live_registry.get("registry_sha256") != pins.registry_sha256:
        errors.append("request/registry generation differs from the incident pin")
    if worker.get("request_sha256") != pins.worker_request_file_sha256:
        errors.append("worker checkpoint request binding changed")
    worker_errors, checkpoint_summary = _validate_worker_checkpoint(
        worker, paths, full=effective_full
    )
    errors += worker_errors

    inputs = runtime.get("inputs")
    roles: dict[str, dict[str, Any]] = {}
    if not isinstance(inputs, list):
        errors.append("runtime input inventory is invalid")
        inputs = []
    for row in inputs:
        role = row.get("role") if isinstance(row, dict) else None
        if not isinstance(role, str) or role in roles:
            errors.append("runtime input roles are invalid or duplicated")
            continue
        roles[role] = row
    for role, expected in pins.binaries.items():
        row = roles.get(role)
        if not isinstance(row, dict) or row.get("sha256") != expected["sha256"] \
                or row.get("bytes") != expected["bytes"]:
            errors.append(f"runtime {role} declaration differs from restored binary pin")
            continue
        try:
            binary_path = Path(row["path"])
            if _hash_regular(binary_path, paths.root) != (expected["sha256"], expected["bytes"]):
                errors.append(f"restored {role} binary changed")
        except (KeyError, TypeError, RecoveryError) as exc:
            errors.append(f"restored {role} binary is invalid: {exc}")
    for role, row in roles.items():
        try:
            input_path = Path(row["path"])
            if role.startswith("source_shard:") and not effective_full:
                observed = _stat_artifact(input_path, paths.root)
                if observed["bytes"] != row.get("bytes"):
                    errors.append(f"runtime input byte count changed: {role}")
            elif _hash_regular(input_path, paths.root) != (row.get("sha256"), row.get("bytes")):
                errors.append(f"runtime input changed: {role}")
        except (KeyError, TypeError, RecoveryError) as exc:
            errors.append(f"runtime input is invalid ({role}): {exc}")

    try:
        log_identity = _hash_regular(
            paths.execution_log, paths.root, max_bytes=16 * 1024 * 1024
        )
        log_lines = paths.execution_log.read_text(encoding="utf-8").splitlines()
        if _hash_regular(paths.execution_log, paths.root,
                         max_bytes=16 * 1024 * 1024) != log_identity:
            raise RecoveryError("execution log changed while reading")
        final_event = json.loads(log_lines[-1]) if log_lines else None
        expected_attestor = str(Path(roles.get("attestor", {}).get("path", "")))
        message = final_event.get("error") if isinstance(final_event, dict) else None
        if not isinstance(final_event, dict) or final_event.get("status") != "refused":
            errors.append("execution log has no final refused event")
        elif not isinstance(message, str) or "No such file or directory" not in message \
                or expected_attestor not in message:
            errors.append("execution log does not bind the missing-attestor incident")
    except (OSError, UnicodeError, json.JSONDecodeError, RecoveryError) as exc:
        errors.append(f"execution log is invalid: {exc}")
    if paths.result.exists() or paths.execution_receipt.exists():
        errors.append("target already has terminal result evidence")

    marker_errors, marker = _validate_marker(paths, pins)
    errors += marker_errors
    bindings: dict[str, Any] = {}
    for name, path in (
        ("recovery_tool", Path(__file__)), ("plan", paths.plan),
        ("state", paths.state), ("control", paths.control),
        ("queue_owner_record", paths.pid_file),
        ("runtime_spec", paths.runtime_spec), ("request", paths.request),
        ("registry_snapshot", paths.registry_snapshot),
        ("live_registry", paths.live_registry),
        ("adapter_checkpoint", paths.adapter_checkpoint),
        ("worker_checkpoint", paths.worker_checkpoint),
        ("worker_request", paths.worker_request),
        ("execution_log", paths.execution_log),
        ("active_marker", paths.active_marker),
        ("accelerated_autoresume", paths.accelerated_autoresume),
        ("accelerated_queue", paths.accelerated_queue),
    ):
        try:
            bindings[name] = _artifact(path, paths.root)
        except RecoveryError as exc:
            errors.append(str(exc))
    for role in ("attestor", "decoder"):
        try:
            bindings[role] = _artifact(Path(roles[role]["path"]), paths.root)
        except (KeyError, TypeError, RecoveryError) as exc:
            errors.append(f"cannot bind {role}: {exc}")
    try:
        if _read_json(paths.state, paths.root) != state:
            errors.append("queue state changed during recovery inspection")
        if _read_json(paths.control, paths.root) != control:
            errors.append("control changed during recovery inspection")
    except RecoveryError as exc:
        errors.append(str(exc))

    structurally_ready = not errors
    activation_permitted = structurally_ready and not blockers
    return {
        "schema": SCHEMA, "version": VERSION, "target_cell_id": TARGET_CELL_ID,
        "inspected_at": _now(), "full_checkpoint_verification_requested": full,
        "full_checkpoint_verification": effective_full,
        "structurally_ready": structurally_ready,
        "activation_permitted": activation_permitted,
        "errors": sorted(set(errors)), "blockers": sorted(set(blockers)),
        "plan_sha256": plan.get("plan_sha256"),
        "state_sha256": state.get("state_sha256"),
        "state_file_sha256": bindings.get("state", {}).get("sha256"),
        "target_row": target_row,
        "target_row_sha256": _hash_value(target_row),
        "checkpoint": checkpoint_summary,
        "supervisor_alive": supervisor_alive,
        "active_children": sorted(active_children) if isinstance(active_children, dict) else None,
        "heavy_owners": owners,
        "queue_lock_available": queue_lock_available if probe_locks else None,
        "heavy_lock_available": heavy_lock_available if probe_locks else None,
        "active_marker_sha256": marker.get("marker_sha256") if marker else None,
        "bindings": bindings,
        "source_deletion_permitted": False,
    }


def _key_commitment(key: str) -> str:
    if not isinstance(key, str) or len(key) < 24:
        raise RecoveryError("each activation key must contain at least 24 characters")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_packet(paths: RecoveryPaths, pins: RecoveryPins = PRODUCTION_PINS, *,
                 key_a: str | None = None, key_b: str | None = None,
                 owner_observer: Callable[[], list[dict[str, Any]]] = active_heavy_owners,
                 probe_locks: bool = True) -> dict[str, Any]:
    if (key_a is None) != (key_b is None):
        raise RecoveryError("both activation keys must be supplied together")
    armed = key_a is not None
    commitments: dict[str, str | None] = {"key_a_sha256": None, "key_b_sha256": None}
    if armed:
        assert key_a is not None and key_b is not None
        commitments = {"key_a_sha256": _key_commitment(key_a),
                       "key_b_sha256": _key_commitment(key_b)}
        if commitments["key_a_sha256"] == commitments["key_b_sha256"]:
            raise RecoveryError("activation keys must be independent")
    snapshot = inspect_recovery(
        paths, pins, full=False, owner_observer=owner_observer,
        probe_locks=probe_locks,
    )
    if not snapshot.get("structurally_ready"):
        raise RecoveryError("recovery inputs are structurally invalid: "
                            + "; ".join(snapshot.get("errors", [])))
    blockers = list(snapshot["blockers"])
    if not armed:
        blockers.append("two activation keys have not been committed")
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA, "version": VERSION, "created_at": _now(),
        "target_cell_id": TARGET_CELL_ID,
        "scope": "single_incident_cell_exact_checkpoint_retry_only",
        "activation": {"mode": "two-independent-explicit-keys", "armed": armed,
                       **commitments},
        "activation_permitted_at_stage": not blockers,
        "blockers_at_stage": sorted(set(blockers)),
        "bindings": snapshot["bindings"],
        "plan_sha256": snapshot["plan_sha256"],
        "state_sha256": snapshot["state_sha256"],
        "state_file_sha256": snapshot["state_file_sha256"],
        "target_row": snapshot["target_row"],
        "target_row_sha256": snapshot["target_row_sha256"],
        "checkpoint": snapshot["checkpoint"],
        "transition": {
            "only_cell": TARGET_CELL_ID,
            "patch": {"status": "pending", "blockers": [], "error": None,
                      "last_exit_code": None},
            "preserve_attempts": pins.attempts,
            "other_cells_must_be_byte_semantically_identical": True,
            "completed_evidence_mutation_permitted": False,
            "control_mutation_permitted": False,
            "source_deletion_permitted": False,
        },
        "resume": {
            "entrypoint": snapshot["bindings"]["accelerated_autoresume"],
            "active_marker": snapshot["bindings"]["active_marker"],
            "selection": "existing active marker; no fallback or marker rewrite permitted",
        },
        "full_checkpoint_rehash_required_at_apply": True,
        "source_deletion_permitted": False,
    }
    packet["packet_sha256"] = _hash_value(packet)
    return packet


def _atomic_json(path: Path, value: dict[str, Any], paths: RecoveryPaths) -> None:
    parent = _confined(path.parent, paths.root)
    if path.exists() and path.is_symlink():
        raise RecoveryError(f"refusing to replace symlink: {path}")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    published = False
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode
        )
        try:
            payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False).encode("utf-8") + b"\n"
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise RecoveryError(f"short write: {temporary}")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        published = True
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if not published:
            try: temporary.unlink()
            except FileNotFoundError: pass


def stage_packet(paths: RecoveryPaths, pins: RecoveryPins = PRODUCTION_PINS, *,
                 key_a: str | None = None, key_b: str | None = None,
                 owner_observer: Callable[[], list[dict[str, Any]]] = active_heavy_owners,
                 probe_locks: bool = True) -> dict[str, Any]:
    paths.stage_root.mkdir(parents=True, exist_ok=True)
    _confined(paths.stage_root, paths.ultra / "staged_acceleration")
    # The apply path never creates lock topology.  Staging creates this inert
    # lock inode inside the staging boundary so a later apply can only open an
    # already-reviewed path.
    paths.recovery_lock.touch(exist_ok=True)
    lock_path = _confined(paths.recovery_lock, paths.stage_root)
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(lock_path, flags), "r+") as stage_lock:
        try:
            fcntl.flock(stage_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryError("recovery staging/apply lock is unavailable") from exc
        try:
            if paths.intent.exists() or paths.receipt.exists() \
                    or paths.resume_receipt.exists():
                raise RecoveryError(
                    "refusing to restage an already prepared/applied recovery"
                )
            packet = build_packet(
                paths, pins, key_a=key_a, key_b=key_b,
                owner_observer=owner_observer, probe_locks=probe_locks,
            )
            _atomic_json(paths.packet, packet, paths)
            return packet
        finally:
            fcntl.flock(stage_lock.fileno(), fcntl.LOCK_UN)


def validate_packet(packet: Any, paths: RecoveryPaths,
                    pins: RecoveryPins = PRODUCTION_PINS, *,
                    require_current: bool = True, full: bool = False,
                    owner_observer: Callable[[], list[dict[str, Any]]] = active_heavy_owners,
                    probe_locks: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, dict):
        return ["packet is not an object"]
    if packet.get("schema") != PACKET_SCHEMA or packet.get("version") != VERSION \
            or packet.get("target_cell_id") != TARGET_CELL_ID:
        errors.append("packet schema/version/target mismatch")
    if packet.get("packet_sha256") != _hash_value(_without(packet, "packet_sha256")):
        errors.append("packet semantic hash mismatch")
    if packet.get("plan_sha256") != pins.plan_sha256:
        errors.append("packet plan generation mismatch")
    transition = packet.get("transition")
    if not isinstance(transition, dict) or transition.get("only_cell") != TARGET_CELL_ID \
            or transition.get("patch") != {
                "status": "pending", "blockers": [], "error": None,
                "last_exit_code": None,
            } or transition.get("preserve_attempts") != pins.attempts \
            or transition.get("completed_evidence_mutation_permitted") is not False \
            or transition.get("control_mutation_permitted") is not False \
            or transition.get("source_deletion_permitted") is not False:
        errors.append("packet transition scope is invalid")
    activation = packet.get("activation")
    if not isinstance(activation, dict) or activation.get("mode") \
            != "two-independent-explicit-keys" or activation.get("armed") not in {True, False}:
        errors.append("packet activation contract is invalid")
    bindings = packet.get("bindings")
    expected_paths = {
        "recovery_tool": Path(__file__), "plan": paths.plan, "state": paths.state,
        "control": paths.control, "queue_owner_record": paths.pid_file,
        "runtime_spec": paths.runtime_spec,
        "request": paths.request, "registry_snapshot": paths.registry_snapshot,
        "live_registry": paths.live_registry,
        "adapter_checkpoint": paths.adapter_checkpoint,
        "worker_checkpoint": paths.worker_checkpoint,
        "worker_request": paths.worker_request, "execution_log": paths.execution_log,
        "active_marker": paths.active_marker,
        "accelerated_autoresume": paths.accelerated_autoresume,
        "accelerated_queue": paths.accelerated_queue,
    }
    if not isinstance(bindings, dict):
        errors.append("packet bindings are missing")
    else:
        for name, expected in expected_paths.items():
            # Mutable state/control are checked against the exact staged
            # generation only when currentness is requested.
            hash_payload = require_current or name not in {"state", "control"}
            if not _artifact_matches(bindings.get(name), expected, paths.root,
                                     hash_payload=hash_payload):
                errors.append(f"packet artifact binding changed: {name}")
    if require_current and not errors:
        snapshot = inspect_recovery(
            paths, pins, full=full, owner_observer=owner_observer,
            probe_locks=probe_locks,
        )
        if snapshot.get("errors"):
            errors.extend(snapshot["errors"])
        if snapshot.get("state_sha256") != packet.get("state_sha256") \
                or snapshot.get("state_file_sha256") != packet.get("state_file_sha256"):
            errors.append("packet is stale relative to queue state")
        if snapshot.get("target_row_sha256") != packet.get("target_row_sha256"):
            errors.append("packet target row changed")
        if full and snapshot.get("blockers"):
            errors.extend(f"activation blocker: {row}" for row in snapshot["blockers"])
    return sorted(set(errors))


def build_after_state(before: dict[str, Any], packet: dict[str, Any], *,
                      now: str | None = None) -> dict[str, Any]:
    if before.get("state_sha256") != packet.get("state_sha256") \
            or _hash_value(before.get("cells", {}).get(TARGET_CELL_ID, {})) \
            != packet.get("target_row_sha256"):
        raise RecoveryError("state generation does not match staged recovery packet")
    after = copy.deepcopy(before)
    target = after.get("cells", {}).get(TARGET_CELL_ID)
    if not isinstance(target, dict):
        raise RecoveryError("target state row is absent")
    original_attempts = target.get("attempts")
    target.update(packet["transition"]["patch"])
    if target.get("attempts") != original_attempts \
            or original_attempts != packet["transition"]["preserve_attempts"]:
        raise RecoveryError("transition changed the attempt history")
    after["updated_at"] = now or _now()
    after["state_sha256"] = _hash_value(_without(after, "state_sha256"))
    for cell_id, row in before.get("cells", {}).items():
        if cell_id != TARGET_CELL_ID and after["cells"].get(cell_id) != row:
            raise RecoveryError(f"transition mutated non-target row: {cell_id}")
    allowed_top = {"cells", "updated_at", "state_sha256"}
    for key in set(before) | set(after):
        if key not in allowed_top and before.get(key) != after.get(key):
            raise RecoveryError(f"transition mutated forbidden top-level field: {key}")
    return after


def _acquire_apply_locks(paths: RecoveryPaths) -> list[IO[str]]:
    handles: list[IO[str]] = []
    try:
        for path in (paths.recovery_lock, paths.queue_lock, paths.heavy_lock):
            resolved = _confined(path, paths.root)
            flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) \
                | getattr(os, "O_NOFOLLOW", 0)
            handle = os.fdopen(os.open(resolved, flags), "r+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RecoveryError(f"required lock is unavailable: {path}") from exc
            handles.append(handle)
        return handles
    except BaseException:
        for handle in reversed(handles):
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError: pass
            handle.close()
        raise


def _release_locks(handles: list[IO[str]]) -> None:
    for handle in reversed(handles):
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError: pass
        handle.close()


def _verify_keys(packet: dict[str, Any], key_a: str, key_b: str) -> None:
    activation = packet.get("activation")
    if not isinstance(activation, dict) or activation.get("armed") is not True:
        raise RecoveryError("recovery packet is not armed")
    observed_a, observed_b = _key_commitment(key_a), _key_commitment(key_b)
    if observed_a == observed_b or observed_a != activation.get("key_a_sha256") \
            or observed_b != activation.get("key_b_sha256"):
        raise RecoveryError("activation keys do not match the staged commitments")


def _default_resume_launcher(paths: RecoveryPaths) -> dict[str, Any]:
    command = [sys.executable, str(paths.accelerated_autoresume)]
    error: str | None = None
    returncode: int | None = None
    try:
        with paths.resume_log.open("ab", buffering=0) as log:
            process = subprocess.run(
                command, cwd=paths.root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False,
            )
        returncode = process.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        owner = _read_json(paths.pid_file, paths.root)
        detached_verified = _supervisor_alive(owner)
    except RecoveryError:
        detached_verified = False
    return {"argv": command, "argv_sha256": _hash_value(command),
            "returncode": returncode, "error": error,
            "detached_supervisor_verified": detached_verified}


def apply_packet(paths: RecoveryPaths, packet_path: Path, *, key_a: str, key_b: str,
                 pins: RecoveryPins = PRODUCTION_PINS,
                 owner_observer: Callable[[], list[dict[str, Any]]] = active_heavy_owners,
                 resume_launcher: Callable[[RecoveryPaths], dict[str, Any]] = _default_resume_launcher
                 ) -> dict[str, Any]:
    packet_path = _confined(packet_path, paths.stage_root)
    if packet_path != paths.packet.resolve(strict=True):
        raise RecoveryError("apply accepts only the canonical staged recovery packet")
    packet = _read_json(packet_path, paths.root)
    _verify_keys(packet, key_a, key_b)
    if packet.get("activation_permitted_at_stage") is not True:
        raise RecoveryError("packet was staged while activation blockers existed")
    handles = _acquire_apply_locks(paths)
    try:
        if paths.intent.exists() or paths.receipt.exists():
            raise RecoveryError("duplicate recovery apply refused")
        errors = validate_packet(
            packet, paths, pins, require_current=True, full=False,
            owner_observer=owner_observer, probe_locks=False,
        )
        if errors:
            raise RecoveryError("recovery packet validation failed: " + "; ".join(errors))
        snapshot = inspect_recovery(
            paths, pins, full=True, owner_observer=owner_observer,
            probe_locks=False,
        )
        dynamic_blockers = [row for row in snapshot["blockers"]
                            if "lock is unavailable" not in row]
        if snapshot["errors"] or dynamic_blockers:
            raise RecoveryError("owner-free activation gate refused: "
                                + "; ".join(snapshot["errors"] + dynamic_blockers))
        # Full verification can take minutes on the real 37 GB checkpoint.
        # Re-observe every mutable owner/state surface after that read and while
        # both campaign locks are still held; the pre-hash observation is not
        # allowed to authorize a post-hash transition.
        post_owners = owner_observer()
        if not isinstance(post_owners, list) \
                or any(not isinstance(row, dict) for row in post_owners):
            raise RecoveryError("post-verification owner observation is invalid")
        if post_owners:
            raise RecoveryError("a heavy owner appeared during full verification")
        owner_record = _read_json(paths.pid_file, paths.root)
        if owner_record.get("schema") != "hawking.doctor_v5_ultra_queue_pid.v1" \
                or owner_record.get("plan_sha256") != pins.plan_sha256 \
                or owner_record.get("pid_record_sha256") \
                != _hash_value(_without(owner_record, "pid_record_sha256")) \
                or not _artifact_matches(
                    packet.get("bindings", {}).get("queue_owner_record"),
                    paths.pid_file, paths.root,
                ):
            raise RecoveryError("queue owner record changed during full verification")
        if _supervisor_alive(owner_record):
            raise RecoveryError("Doctor supervisor appeared during full verification")
        before = _read_json(paths.state, paths.root)
        if before.get("supervisor_pid") is not None \
                or before.get("active_children") != {} \
                or before.get("active_cells") != [] \
                or before.get("status") not in {"drained", "waiting-prerequisites"}:
            raise RecoveryError("queue activity appeared during full verification")
        if before.get("state_sha256") != packet.get("state_sha256") \
                or not _artifact_matches(
                    packet.get("bindings", {}).get("state"), paths.state, paths.root
                ):
            raise RecoveryError("queue state changed during full verification")
        control = _read_json(paths.control, paths.root)
        if control.get("mode") != "run" or not _artifact_matches(
                packet.get("bindings", {}).get("control"), paths.control, paths.root
        ):
            raise RecoveryError("control changed during full verification")
        after = build_after_state(before, packet)
        before_artifact = _artifact(paths.state, paths.root)
        intent: dict[str, Any] = {
            "schema": INTENT_SCHEMA, "version": VERSION, "prepared_at": _now(),
            "packet_sha256": packet["packet_sha256"],
            "before_state": before_artifact,
            "before_state_sha256": before["state_sha256"],
            "after_state_sha256": after["state_sha256"],
            "target_before_sha256": _hash_value(before["cells"][TARGET_CELL_ID]),
            "target_after_sha256": _hash_value(after["cells"][TARGET_CELL_ID]),
            "only_cell": TARGET_CELL_ID, "source_deletion_permitted": False,
        }
        intent["intent_sha256"] = _hash_value(intent)
        _atomic_json(paths.intent, intent, paths)
        _atomic_json(paths.state, after, paths)
        observed_after = _read_json(paths.state, paths.root)
        if observed_after != after:
            raise RecoveryError("atomic state transition did not publish exact after-state")
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA, "version": VERSION, "committed_at": _now(),
            "packet_sha256": packet["packet_sha256"],
            "intent_sha256": intent["intent_sha256"],
            "before_state_sha256": before["state_sha256"],
            "after_state_sha256": after["state_sha256"],
            "after_state": _artifact(paths.state, paths.root),
            "target_cell_id": TARGET_CELL_ID,
            "other_cell_count": len(before["cells"]) - 1,
            "other_cells_unchanged": all(
                before["cells"][cell_id] == after["cells"][cell_id]
                for cell_id in before["cells"] if cell_id != TARGET_CELL_ID
            ),
            "completed_evidence_mutated": False,
            "control_mutated": False, "source_deletion_permitted": False,
        }
        receipt["receipt_sha256"] = _hash_value(receipt)
        _atomic_json(paths.receipt, receipt, paths)
    finally:
        _release_locks(handles)

    launch = resume_launcher(paths)
    resume: dict[str, Any] = {
        "schema": RESUME_SCHEMA, "version": VERSION, "launched_at": _now(),
        "apply_receipt_sha256": receipt["receipt_sha256"],
        "active_marker": packet["resume"]["active_marker"],
        "autoresume_entrypoint": packet["resume"]["entrypoint"],
        "launch": launch, "detached_supervisor_requested": True,
        "source_deletion_permitted": False,
    }
    resume["resume_sha256"] = _hash_value(resume)
    _atomic_json(paths.resume_receipt, resume, paths)
    if launch.get("returncode") != 0 or launch.get("detached_supervisor_verified") is not True:
        raise RecoveryError("state recovery committed but accelerated autoresume refused; "
                            f"see {paths.resume_log}")
    return {"apply": receipt, "resume": resume}


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--full", action="store_true",
                        help="hash every completed checkpoint artifact (owner-free only)")
    stage = commands.add_parser("stage")
    stage.add_argument("--activation-key-a")
    stage.add_argument("--activation-key-b")
    verify = commands.add_parser("verify")
    verify.add_argument("--packet", type=Path, default=production_paths().packet)
    verify.add_argument("--full", action="store_true")
    apply = commands.add_parser("apply")
    apply.add_argument("--packet", type=Path, default=production_paths().packet)
    apply.add_argument("--activation-key-a", required=True)
    apply.add_argument("--activation-key-b", required=True)
    apply.add_argument("--resume-via-active-marker", action="store_true", required=True)
    args = parser.parse_args(argv)
    paths = production_paths()
    try:
        if args.command == "status":
            _print(inspect_recovery(paths, full=args.full)); return 0
        if args.command == "stage":
            packet = stage_packet(
                paths, key_a=args.activation_key_a, key_b=args.activation_key_b
            )
            _print({
                "ok": True, "packet": str(paths.packet),
                "packet_sha256": packet["packet_sha256"],
                "armed": packet["activation"]["armed"],
                "activation_permitted_at_stage": packet["activation_permitted_at_stage"],
                "blockers": packet["blockers_at_stage"],
                "live_campaign_mutated": False,
            }); return 0
        if args.command == "verify":
            packet = _read_json(_confined(args.packet, paths.stage_root), paths.root)
            errors = validate_packet(packet, paths, require_current=True, full=args.full)
            _print({"ok": not errors, "errors": errors,
                    "activation_permitted": not errors and args.full})
            return 0 if not errors else 2
        if args.command == "apply":
            # argparse requires the explicit resume acknowledgement.  Keep the
            # branch for defensive direct-main callers and future refactors.
            if not args.resume_via_active_marker:
                raise RecoveryError("explicit active-marker resume acknowledgement missing")
            _print(apply_packet(
                paths, args.packet, key_a=args.activation_key_a,
                key_b=args.activation_key_b,
            )); return 0
    except (RecoveryError, OSError, ValueError, TypeError) as exc:
        print(f"[doctor-v5-blocked-cell-recovery] {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
