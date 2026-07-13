#!/usr/bin/env python3.12
"""Detached, fail-closed Doctor-v5 Pass-B treatment queue.

This controller consumes the completed Pass-A census index and advances the
fixed 0.5B -> 1.5B -> 7B -> 14B -> 32B -> 72B -> 120B ladder.  It never
constructs a command from shell text.  A pilot injection may select only a
typed adapter/command pair from a reviewed, source-bound registry; the adapter
ABI is responsible for producing the exact argv and validating all execution
artifacts.

The controller deliberately waits when contracts are absent or invalid.  It
does not delete, move, or rewrite any model source.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

PASS_A_ROOT = ROOT / "reports" / "condense" / "doctor_v5_scale"
PASS_A_INDEX = PASS_A_ROOT / "index.json"
PASS_B_ROOT = ROOT / "reports" / "condense" / "doctor_v5_pass_b"
STATE = PASS_B_ROOT / "queue_state.json"
PID_FILE = PASS_B_ROOT / "queue.pid.json"
QUEUE_LOCK = PASS_B_ROOT / "queue.lock"
INJECTION_LOCK = PASS_B_ROOT / "injections.lock"
INJECTIONS = PASS_B_ROOT / "injections.jsonl"
DRAIN_REQUEST = PASS_B_ROOT / "drain.request.json"
LOG_FILE = PASS_B_ROOT / "queue.log"
ADAPTER_REGISTRY = PASS_B_ROOT / "adapter_registry.json"
PARAMETER_MANIFESTS = PASS_B_ROOT / "parameter_manifests"
PILOT_SPECS = PASS_B_ROOT / "pilot_specs"
RESULTS_ROOT = PASS_B_ROOT / "results"
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
QUEUE_SCRIPT = Path(__file__).resolve()
ADAPTER_MODULE_PATH = ROOT / "tools" / "condense" / "doctor_v5_adapter_abi.py"
PARAMETER_MODULE_PATH = ROOT / "tools" / "condense" / "doctor_v5_parameter_manifest.py"

SCHEMA = "hawking.doctor_v5_pass_b_queue.v1"
STATE_SCHEMA = "hawking.doctor_v5_pass_b_queue_state.v1"
PID_SCHEMA = "hawking.doctor_v5_pass_b_queue_pid.v1"
INJECTION_SCHEMA = "hawking.doctor_v5_pass_b_injection.v1"
STATUS_SCHEMA = "hawking.doctor_v5_pass_b_queue_status.v1"
VERSION = "2026-07-13.1"
POLL_SECONDS = 5.0
MIN_DISK_FREE_GB = 150.0
CONTROL_RESERVE_GB = 2.0
MIN_PROCESSING_SCRATCH_GB = 64.0
PAUSE_RC = 131
ADOPT_RC = 132

COHORT = (
    {"label": "0.5B", "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
     "model_dir": "scratch/qwen-05b"},
    {"label": "1.5B", "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
     "model_dir": "scratch/qwen-15b"},
    {"label": "7B", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
     "model_dir": "scratch/qwen-7b"},
    {"label": "14B", "hf_id": "Qwen/Qwen2.5-14B-Instruct",
     "model_dir": "scratch/staging/qwen-14b.partial"},
    {"label": "32B", "hf_id": "Qwen/Qwen2.5-32B-Instruct",
     "model_dir": "scratch/staging/qwen-32b.partial"},
    {"label": "72B", "hf_id": "Qwen/Qwen2.5-72B-Instruct",
     "model_dir": "scratch/staging/qwen-72b.partial"},
    {"label": "120B", "hf_id": "openai/gpt-oss-120b",
     "model_dir": "scratch/staging/gpt-oss-120b.partial"},
)
LABELS = tuple(item["label"] for item in COHORT)
COHORT_BY_LABEL = {item["label"]: item for item in COHORT}
ALLOWED_ACTIONS = {"pause", "resume", "drain", "pilot"}
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX32 = re.compile(r"[0-9a-f]{32}")
_STOP = False

STATE_KEYS = {
    "schema", "version", "queue_identity_sha256", "pass_a_index_sha256",
    "created_at", "updated_at", "status", "control_mode", "pending_order",
    "items", "pilot_requests", "last_consumed_injection_seq", "active_label",
    "active_child", "last_resource_gate", "last_prerequisite_blockers",
    "supervisor_pid", "source_deletion_permitted", "restart_command",
    "error", "drained_at", "completed_at", "state_sha256",
}
ITEM_KEYS = {
    "status", "attempts", "started_at", "completed_at", "last_exit_code",
    "request_sha256", "result_sha256", "execution_receipt_sha256", "error",
}
PILOT_KEYS = {
    "label", "adapter_id", "command_id", "registry_file_sha256",
    "parameter_manifest_file_sha256", "pilot_spec_file_sha256",
    "injection_sha256",
}


class QueueError(RuntimeError):
    """The queue cannot safely continue."""


class ContractBlocked(QueueError):
    """A reviewed Pass-B prerequisite is absent or invalid."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise QueueError(f"not a regular file: {path}")
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or total != after.st_size):
            raise QueueError(f"file changed while hashing: {path}")
    finally:
        os.close(fd)
    return digest.hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_no_duplicate_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError(f"non-finite JSON constant: {value}")))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return default


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _append_durable(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(_canonical(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(path.parent)


def _safe_relative(path: Any) -> Path | None:
    if not isinstance(path, str) or not path or "\x00" in path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("."):
        return None
    return candidate


def _lazy_contract(name: str, source: Path) -> tuple[Any | None, str | None]:
    if not source.is_file() or source.is_symlink():
        return None, f"contract module missing or symlinked: {source.relative_to(ROOT)}"
    try:
        importlib.invalidate_caches()
        module = importlib.import_module(name)
        observed = Path(module.__file__).resolve()
        if observed != source.resolve():
            return None, f"contract module resolved outside expected path: {name}"
        return module, None
    except Exception as exc:
        return None, f"contract module import failed ({name}): {type(exc).__name__}: {exc}"


def _call_validator(module: Any, name: str, *args: Any, **kwargs: Any) -> list[str]:
    validator = getattr(module, name, None)
    if not callable(validator):
        return [f"contract validator unavailable: {module.__name__}.{name}"]
    try:
        result = validator(*args, **kwargs)
    except Exception as exc:
        return [f"contract validator raised ({name}): {type(exc).__name__}: {exc}"]
    if not isinstance(result, list) or any(not isinstance(row, str) for row in result):
        return [f"contract validator returned an invalid result: {name}"]
    return result


def _pass_a_index() -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    index = _read_json(PASS_A_INDEX)
    required = {
        "schema", "queue_identity_sha256", "cohort_labels", "reports",
        "report_count", "pass_a_complete", "pass_b_launch_permitted",
        "report_bundles", "all_at_once_source_cleanup", "generated_at",
        "index_sha256",
    }
    if not isinstance(index, dict):
        return None, ["Pass-A index is missing, unreadable, or non-object"]
    if PASS_A_INDEX.is_symlink():
        errors.append("Pass-A index may not be a symlink")
    if set(index) != required:
        errors.append("Pass-A index keys are not exact")
        return index, errors
    payload = {key: value for key, value in index.items() if key != "index_sha256"}
    if index.get("index_sha256") != _hash_value(payload):
        errors.append("Pass-A index hash mismatch")
    if index.get("schema") != "hawking.doctor_v5_scale_index.v1":
        errors.append("Pass-A index schema mismatch")
    if index.get("cohort_labels") != list(LABELS):
        errors.append("Pass-A cohort/order differs from the fixed Pass-B ladder")
    if index.get("report_count") != len(COHORT) or index.get("pass_a_complete") is not True:
        errors.append("Pass-A is not complete for all fixed rungs")
    if index.get("pass_b_launch_permitted") is not False:
        errors.append("Pass-A index unexpectedly claims Pass-B launch authority")
    cleanup = index.get("all_at_once_source_cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("eligible") is not False:
        errors.append("Pass-A cleanup boundary is missing or unsafe")
    bundles = index.get("report_bundles")
    if not isinstance(bundles, dict) or set(bundles) != {"sub_120B_aggregate", "120B"}:
        errors.append("Pass-A report bundles are invalid")
    else:
        for key in ("sub_120B_aggregate", "120B"):
            bundle = bundles.get(key)
            if (not isinstance(bundle, dict) or bundle.get("complete") is not True
                    or bundle.get("quality_result") is not False):
                errors.append(f"Pass-A bundle {key} is not a complete census-only bundle")
    reports = index.get("reports")
    if not isinstance(reports, list) or len(reports) != len(COHORT):
        errors.append("Pass-A report registry is incomplete")
        return index, errors
    census_module, module_error = _lazy_contract("doctor_v5_census",
                                                 ROOT / "tools" / "condense" /
                                                 "doctor_v5_census.py")
    if module_error:
        errors.append(module_error)
        return index, errors
    for position, (item, row) in enumerate(zip(COHORT, reports, strict=True)):
        expected_path = Path("reports") / "condense" / "doctor_v5_scale" / item["label"] / "census.json"
        row_keys = {"label", "path", "report_sha256", "source_manifest_sha256"}
        if not isinstance(row, dict) or set(row) != row_keys:
            errors.append(f"Pass-A report row {position} keys are invalid")
            continue
        if row.get("label") != item["label"] or row.get("path") != expected_path.as_posix():
            errors.append(f"Pass-A report row {position} identity/path mismatch")
            continue
        if not HEX64.fullmatch(str(row.get("report_sha256", ""))) \
                or not HEX64.fullmatch(str(row.get("source_manifest_sha256", ""))):
            errors.append(f"Pass-A report row {position} hashes are invalid")
            continue
        report_path = ROOT / expected_path
        if report_path.is_symlink():
            errors.append(f"Pass-A census may not be a symlink: {item['label']}")
            continue
        report = _read_json(report_path)
        if not isinstance(report, dict):
            errors.append(f"Pass-A census missing/unreadable: {item['label']}")
            continue
        report_errors = _call_validator(census_module, "validate_report", report)
        if report_errors:
            errors.append(f"Pass-A census invalid ({item['label']}): " + "; ".join(report_errors[:3]))
            continue
        source = report.get("source") if isinstance(report.get("source"), dict) else {}
        if (report.get("label") != item["label"] or report.get("hf_id") != item["hf_id"]
                or report.get("report_sha256") != row["report_sha256"]
                or source.get("source_manifest_sha256") != row["source_manifest_sha256"]):
            errors.append(f"Pass-A census/index binding mismatch: {item['label']}")
        model_dir = ROOT / item["model_dir"]
        if not model_dir.is_dir() or model_dir.is_symlink():
            errors.append(f"model source missing or symlinked: {item['label']}")
    return index, errors


def _identity() -> dict[str, Any]:
    index, errors = _pass_a_index()
    index_sha = index.get("index_sha256") if isinstance(index, dict) else None
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "queue_source_sha256": _sha256_file(QUEUE_SCRIPT),
        "pass_a_index_path": str(PASS_A_INDEX.relative_to(ROOT)),
        "pass_a_index_sha256": index_sha,
        "fixed_cohort": [dict(item) for item in COHORT],
        "adapter_contract_module": str(ADAPTER_MODULE_PATH.relative_to(ROOT)),
        "adapter_contract_module_sha256": _sha256_file(ADAPTER_MODULE_PATH)
        if ADAPTER_MODULE_PATH.is_file() and not ADAPTER_MODULE_PATH.is_symlink() else None,
        "parameter_contract_module": str(PARAMETER_MODULE_PATH.relative_to(ROOT)),
        "parameter_contract_module_sha256": _sha256_file(PARAMETER_MODULE_PATH)
        if PARAMETER_MODULE_PATH.is_file() and not PARAMETER_MODULE_PATH.is_symlink() else None,
        "adapter_registry": str(ADAPTER_REGISTRY.relative_to(ROOT)),
        "parameter_manifest_pattern": str((PARAMETER_MANIFESTS / "<label>.json").relative_to(ROOT)),
        "pilot_spec_pattern": str((PILOT_SPECS / "<label>-q4-control.json").relative_to(ROOT)),
        "pass_a_validation_errors": errors,
        "execution_policy": {
            "shell": False,
            "typed_reviewed_allowlist_only": True,
            "shared_heavy_lease": str(HEAVY_LOCK.relative_to(ROOT)),
            "source_deletion_permitted": False,
        },
    }
    return {**payload, "queue_identity_sha256": _hash_value(payload)}


def _base_state(identity: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "version": VERSION,
        "queue_identity_sha256": identity["queue_identity_sha256"],
        "pass_a_index_sha256": identity["pass_a_index_sha256"],
        "created_at": _now(),
        "updated_at": _now(),
        "status": "new",
        "control_mode": "run",
        "pending_order": list(LABELS),
        "items": {
            label: {
                "status": "pending", "attempts": 0, "started_at": None,
                "completed_at": None, "last_exit_code": None,
                "request_sha256": None, "result_sha256": None,
                "execution_receipt_sha256": None, "error": None,
            } for label in LABELS
        },
        "pilot_requests": {label: None for label in LABELS},
        "last_consumed_injection_seq": 0,
        "active_label": None,
        "active_child": None,
        "last_resource_gate": None,
        "last_prerequisite_blockers": [],
        "supervisor_pid": None,
        "source_deletion_permitted": False,
        "restart_command": "python3.12 tools/condense/doctor_v5_pass_b_queue.py resume",
        "error": None,
        "drained_at": None,
        "completed_at": None,
    }
    state["state_sha256"] = _hash_value(state)
    return state


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(row) for row in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(row) for key, row in value.items())
    return True


def _validate_state(state: Any, identity: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(state, dict):
        return ["state is not an object"]
    if set(state) != STATE_KEYS:
        return ["state keys are not exact"]
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != _hash_value(payload):
        errors.append("state hash mismatch")
    if (state.get("schema") != STATE_SCHEMA or state.get("version") != VERSION
            or state.get("queue_identity_sha256") != identity.get("queue_identity_sha256")
            or state.get("pass_a_index_sha256") != identity.get("pass_a_index_sha256")):
        errors.append("state identity mismatch")
    if not _finite(state):
        errors.append("state contains a non-finite number")
    statuses = {
        "new", "running", "waiting-pilot", "waiting-prerequisites",
        "waiting-resources", "waiting-heavy-lease", "running-pilot",
        "paused-control", "drained", "blocked-execution", "complete",
    }
    if state.get("status") not in statuses:
        errors.append("state status is invalid")
    if state.get("control_mode") not in {"run", "pause", "drain"}:
        errors.append("state control mode is invalid")
    if state.get("source_deletion_permitted") is not False:
        errors.append("source deletion boundary was weakened")
    sequence = state.get("last_consumed_injection_seq")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        errors.append("last consumed injection sequence is invalid")
    pending = state.get("pending_order")
    if not isinstance(pending, list) or any(row not in LABELS for row in pending) \
            or len(pending) != len(set(pending)):
        errors.append("pending order is invalid")
    items = state.get("items")
    pilots = state.get("pilot_requests")
    if not isinstance(items, dict) or set(items) != set(LABELS):
        errors.append("item registry is invalid")
    if not isinstance(pilots, dict) or set(pilots) != set(LABELS):
        errors.append("pilot request registry is invalid")
    if isinstance(items, dict) and set(items) == set(LABELS):
        first_noncomplete = 0
        while first_noncomplete < len(LABELS) \
                and items[LABELS[first_noncomplete]].get("status") == "complete":
            first_noncomplete += 1
        if pending != list(LABELS[first_noncomplete:]):
            errors.append("pending order is not the fixed incomplete suffix")
        for position, label in enumerate(LABELS):
            row = items[label]
            if not isinstance(row, dict) or set(row) != ITEM_KEYS:
                errors.append(f"item keys are invalid: {label}")
                continue
            if row.get("status") not in {"pending", "running", "blocked", "complete"}:
                errors.append(f"item status is invalid: {label}")
            attempts = row.get("attempts")
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
                errors.append(f"item attempts are invalid: {label}")
            if position > first_noncomplete and row.get("status") == "complete":
                errors.append(f"fixed ladder completion order was skipped: {label}")
            if row.get("status") == "complete":
                for field in ("request_sha256", "result_sha256", "execution_receipt_sha256"):
                    if not HEX64.fullmatch(str(row.get(field, ""))):
                        errors.append(f"complete item lacks {field}: {label}")
    if isinstance(pilots, dict) and set(pilots) == set(LABELS):
        for label, pilot in pilots.items():
            if pilot is None:
                continue
            if not isinstance(pilot, dict) or set(pilot) != PILOT_KEYS:
                errors.append(f"pilot request keys are invalid: {label}")
                continue
            if pilot.get("label") != label:
                errors.append(f"pilot label mismatch: {label}")
            for field in ("registry_file_sha256", "parameter_manifest_file_sha256",
                          "pilot_spec_file_sha256", "injection_sha256"):
                if not HEX64.fullmatch(str(pilot.get(field, ""))):
                    errors.append(f"pilot {field} is invalid: {label}")
            for field in ("adapter_id", "command_id"):
                if not isinstance(pilot.get(field), str) or not pilot[field]:
                    errors.append(f"pilot {field} is invalid: {label}")
    return errors


def _load_state(identity: dict[str, Any]) -> dict[str, Any]:
    state = _read_json(STATE)
    if state is None:
        if STATE.exists():
            raise QueueError("queue state exists but is corrupt or unreadable")
        return _base_state(identity)
    errors = _validate_state(state, identity)
    if errors:
        raise QueueError("queue state invalid: " + "; ".join(errors))
    return state


def _save_state(state: dict[str, Any], status: str | None = None, **updates: Any) -> None:
    if status is not None:
        state["status"] = status
    state.update(updates)
    state["updated_at"] = _now()
    state["state_sha256"] = _hash_value({key: value for key, value in state.items()
                                         if key != "state_sha256"})
    errors = _validate_state(state, {
        "queue_identity_sha256": state["queue_identity_sha256"],
        "pass_a_index_sha256": state["pass_a_index_sha256"],
    })
    if errors:
        raise QueueError("refusing to write invalid state: " + "; ".join(errors))
    _atomic_json(STATE, state)


def _ps_identity(pid: Any) -> tuple[str, str] | None:
    try:
        pid_int = int(pid)
        os.kill(pid_int, 0)
        start = subprocess.run(["ps", "-ww", "-p", str(pid_int), "-o", "lstart="],
                               capture_output=True, text=True, timeout=3, check=False)
        command = subprocess.run(["ps", "-ww", "-p", str(pid_int), "-o", "command="],
                                 capture_output=True, text=True, timeout=3, check=False)
        if start.returncode or command.returncode:
            return None
        return start.stdout.strip(), command.stdout.strip()
    except Exception:
        return None


def _pid_record(identity: dict[str, Any], nonce: str) -> dict[str, Any]:
    observed = _ps_identity(os.getpid())
    if observed is None:
        raise QueueError("cannot establish supervisor process identity")
    start, command = observed
    payload = {
        "schema": PID_SCHEMA, "version": VERSION, "pid": os.getpid(),
        "process_start": start, "process_command_sha256": hashlib.sha256(
            command.encode("utf-8")).hexdigest(),
        "ownership_nonce": nonce,
        "queue_identity_sha256": identity["queue_identity_sha256"],
        "queue_source_sha256": _sha256_file(QUEUE_SCRIPT), "created_at": _now(),
    }
    return {**payload, "pid_record_sha256": _hash_value(payload)}


def _owner_alive(record: Any, identity: dict[str, Any] | None = None) -> bool:
    required = {
        "schema", "version", "pid", "process_start", "process_command_sha256",
        "ownership_nonce", "queue_identity_sha256", "queue_source_sha256",
        "created_at", "pid_record_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        return False
    payload = {key: value for key, value in record.items() if key != "pid_record_sha256"}
    if (record.get("schema") != PID_SCHEMA or record.get("version") != VERSION
            or record.get("pid_record_sha256") != _hash_value(payload)
            or record.get("queue_source_sha256") != _sha256_file(QUEUE_SCRIPT)):
        return False
    if identity is None:
        try:
            identity = _identity()
        except Exception:
            return False
    if record.get("queue_identity_sha256") != identity.get("queue_identity_sha256"):
        return False
    observed = _ps_identity(record.get("pid"))
    if observed is None:
        return False
    start, command = observed
    nonce = record.get("ownership_nonce")
    return (
        start == record.get("process_start")
        and hashlib.sha256(command.encode("utf-8")).hexdigest()
        == record.get("process_command_sha256")
        and isinstance(nonce, str) and HEX32.fullmatch(nonce) is not None
        and "doctor_v5_pass_b_queue.py run" in command
        and f"--nonce {nonce}" in command
    )


def _thermal() -> dict[str, Any]:
    try:
        scheduler = importlib.import_module("ram_scheduler")
        checker: Callable[[int, str], bool] = getattr(scheduler, "thermal_output_ok")
        result = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                                text=True, timeout=5, check=False)
        output = (result.stdout + result.stderr).strip()
        return {"ok": checker(result.returncode, output),
                "returncode": result.returncode, "output": output[-1000:]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _resource_gate(extra_scratch_gb: float = 0.0) -> dict[str, Any]:
    blockers: list[str] = []
    if (isinstance(extra_scratch_gb, bool) or not isinstance(extra_scratch_gb, (int, float))
            or not math.isfinite(extra_scratch_gb) or extra_scratch_gb < 0):
        extra_scratch_gb = float("nan")
    try:
        scheduler = importlib.import_module("ram_scheduler")
        snapshot = scheduler.resource_snapshot()
    except Exception as exc:
        snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    thermal = _thermal()
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    pressure = snapshot.get("pressure_level", memory.get("pressure_level"))
    swap = snapshot.get("swap_used_mb", memory.get("swap_used_mb"))
    free = snapshot.get("disk_free_gb")
    power = str(snapshot.get("power_source", ""))
    admitted_scratch_gb = max(MIN_PROCESSING_SCRATCH_GB, extra_scratch_gb) \
        if math.isfinite(extra_scratch_gb) else float("nan")
    required = MIN_DISK_FREE_GB + CONTROL_RESERVE_GB + admitted_scratch_gb
    if isinstance(pressure, bool) or not isinstance(pressure, int) or pressure != 1:
        blockers.append("memory pressure is not normal")
    if (isinstance(swap, bool) or not isinstance(swap, (int, float))
            or not math.isfinite(swap) or swap != 0):
        blockers.append("swap is nonzero or unavailable")
    if "AC Power" not in power:
        blockers.append("AC power is not confirmed")
    if not thermal.get("ok"):
        blockers.append("thermal state is not green")
    if (isinstance(free, bool) or not isinstance(free, (int, float))
            or not math.isfinite(free) or not math.isfinite(required) or free < required):
        blockers.append(f"disk free is below {required:.3f} GB")
    return {
        "schema": "hawking.doctor_v5_pass_b_resource_gate.v1",
        "sampled_at": _now(), "ok": not blockers, "blockers": blockers,
        "required_free_gb": round(required, 6) if math.isfinite(required) else None,
        "extra_scratch_gb": round(extra_scratch_gb, 6)
        if math.isfinite(extra_scratch_gb) else None,
        "admitted_scratch_gb": round(admitted_scratch_gb, 6)
        if math.isfinite(admitted_scratch_gb) else None,
        "resources": snapshot, "thermal": thermal,
    }


def _read_injections_unlocked(identity_sha: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    prior = "0" * 64
    if not INJECTIONS.exists():
        return rows, errors
    try:
        lines = INJECTIONS.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"cannot read injection ledger: {exc}"]
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line, object_pairs_hook=_no_duplicate_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(
                                 ValueError(value)))
        except (json.JSONDecodeError, ValueError):
            errors.append(f"injection line {line_number} is invalid JSON")
            continue
        required = {
            "schema", "sequence", "created_at", "action", "payload",
            "queue_identity_sha256", "prior_sha256", "injection_sha256",
        }
        if not isinstance(row, dict) or set(row) != required:
            errors.append(f"injection line {line_number} keys are not exact")
            continue
        if row.get("schema") != INJECTION_SCHEMA:
            errors.append(f"injection line {line_number} schema mismatch")
        sequence = row.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) \
                or sequence != line_number:
            errors.append(f"injection line {line_number} sequence mismatch")
        if row.get("queue_identity_sha256") != identity_sha:
            errors.append(f"injection line {line_number} targets another queue")
        if row.get("prior_sha256") != prior:
            errors.append(f"injection line {line_number} prior hash mismatch")
        expected = _hash_value({key: value for key, value in row.items()
                                if key != "injection_sha256"})
        if row.get("injection_sha256") != expected:
            errors.append(f"injection line {line_number} self hash mismatch")
        action, payload = row.get("action"), row.get("payload")
        if action not in ALLOWED_ACTIONS or not isinstance(payload, dict):
            errors.append(f"injection line {line_number} action/payload is invalid")
        elif action in {"pause", "resume", "drain"} and payload:
            errors.append(f"injection line {line_number} control payload must be empty")
        elif action == "pilot":
            expected_keys = PILOT_KEYS - {"injection_sha256"}
            if set(payload) != expected_keys:
                errors.append(f"injection line {line_number} pilot payload keys are not exact")
            elif (payload.get("label") not in LABELS
                  or not isinstance(payload.get("adapter_id"), str)
                  or not payload.get("adapter_id")
                  or not isinstance(payload.get("command_id"), str)
                  or not payload.get("command_id")
                  or not HEX64.fullmatch(str(payload.get("registry_file_sha256", "")))
                  or not HEX64.fullmatch(str(payload.get(
                      "parameter_manifest_file_sha256", "")))
                  or not HEX64.fullmatch(str(payload.get(
                      "pilot_spec_file_sha256", "")))):
                errors.append(f"injection line {line_number} pilot payload is invalid")
        prior = row.get("injection_sha256") if isinstance(row.get("injection_sha256"), str) else prior
        rows.append(row)
    return rows, errors


def _read_injections(identity_sha: str) -> tuple[list[dict[str, Any]], list[str]]:
    INJECTION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(INJECTION_LOCK, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        result = _read_injections_unlocked(identity_sha)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return result


def _append_injection(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    identity = _identity()
    INJECTION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(INJECTION_LOCK, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows, errors = _read_injections_unlocked(identity["queue_identity_sha256"])
        if errors:
            raise QueueError("existing injection ledger is invalid: " + "; ".join(errors))
        row = {
            "schema": INJECTION_SCHEMA, "sequence": len(rows) + 1,
            "created_at": _now(), "action": action, "payload": payload,
            "queue_identity_sha256": identity["queue_identity_sha256"],
            "prior_sha256": rows[-1]["injection_sha256"] if rows else "0" * 64,
        }
        row["injection_sha256"] = _hash_value(row)
        _append_durable(INJECTIONS, row)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return row


def _apply_injections(state: dict[str, Any]) -> str:
    rows, errors = _read_injections(state["queue_identity_sha256"])
    if errors:
        raise QueueError("injection ledger invalid: " + "; ".join(errors))
    consumed = int(state["last_consumed_injection_seq"])
    control = state["control_mode"]
    for row in rows:
        if row["sequence"] <= consumed:
            continue
        action = row["action"]
        if action == "pause":
            control = "pause"
        elif action == "resume":
            control = "run"
        elif action == "drain":
            control = "drain"
        elif action == "pilot":
            label = row["payload"]["label"]
            if state["items"][label]["status"] != "complete":
                state["pilot_requests"][label] = {
                    **row["payload"], "injection_sha256": row["injection_sha256"],
                }
                if state["items"][label]["status"] == "blocked":
                    state["items"][label]["status"] = "pending"
                    state["items"][label]["error"] = None
        consumed = row["sequence"]
    state["last_consumed_injection_seq"] = consumed
    state["control_mode"] = control
    _save_state(state)
    return control


def _parameter_manifest_path(label: str) -> Path:
    return PARAMETER_MANIFESTS / f"{label}.json"


def _pilot_spec_path(label: str) -> Path:
    return PILOT_SPECS / f"{label}-q4-control.json"


def _contract_snapshot(label: str, pilot: dict[str, Any] | None = None,
                       verify_files: bool = True) -> dict[str, Any]:
    blockers: list[str] = []
    adapter_module, error = _lazy_contract("doctor_v5_adapter_abi", ADAPTER_MODULE_PATH)
    if error:
        blockers.append(error)
    parameter_module, error = _lazy_contract("doctor_v5_parameter_manifest",
                                             PARAMETER_MODULE_PATH)
    if error:
        blockers.append(error)
    registry = _read_json(ADAPTER_REGISTRY)
    manifest_path = _parameter_manifest_path(label)
    manifest = _read_json(manifest_path)
    pilot_spec_path = _pilot_spec_path(label)
    pilot_spec = _read_json(pilot_spec_path)
    if not isinstance(registry, dict):
        blockers.append("reviewed adapter registry is missing or unreadable")
    if ADAPTER_REGISTRY.is_symlink():
        blockers.append("reviewed adapter registry may not be a symlink")
    if not isinstance(manifest, dict):
        blockers.append(f"exact parameter manifest is missing or unreadable: {label}")
    if manifest_path.is_symlink():
        blockers.append(f"exact parameter manifest may not be a symlink: {label}")
    if not isinstance(pilot_spec, dict):
        blockers.append(f"reviewed typed pilot spec is missing or unreadable: {label}")
    if pilot_spec_path.is_symlink():
        blockers.append(f"reviewed typed pilot spec may not be a symlink: {label}")
    if adapter_module is not None and isinstance(registry, dict):
        blockers.extend(_call_validator(adapter_module, "validate_registry", registry,
                                        verify_files=verify_files, base_dir=ROOT))
    if parameter_module is not None and isinstance(manifest, dict):
        blockers.extend(_call_validator(parameter_module, "validate_manifest", manifest,
                                        verify_files=verify_files))
    registry_file_sha = _sha256_file(ADAPTER_REGISTRY) \
        if ADAPTER_REGISTRY.is_file() and not ADAPTER_REGISTRY.is_symlink() else None
    manifest_file_sha = _sha256_file(manifest_path) \
        if manifest_path.is_file() and not manifest_path.is_symlink() else None
    pilot_spec_file_sha = _sha256_file(pilot_spec_path) \
        if pilot_spec_path.is_file() and not pilot_spec_path.is_symlink() else None
    if isinstance(pilot_spec, dict):
        required_spec = {
            "schema", "program_spec_sha256", "resource_admission_sha256",
            "model_family", "backend", "seed", "inputs", "scratch_budget_bytes",
            "disk_reserve_bytes",
        }
        if not required_spec.issubset(pilot_spec):
            blockers.append("typed pilot spec lacks required request-binding fields")
        if label == "0.5B" and pilot_spec.get("schema") \
                != "hawking.doctor_v5_pass_b_strand_control_spec.v1":
            blockers.append("0.5B typed pilot spec schema is invalid")
        if (pilot_spec.get("label") != label
                or pilot_spec.get("source_deletion_permitted") is not False):
            blockers.append("typed pilot spec label/source-deletion boundary is invalid")
        if label == "0.5B" and (
                pilot_spec.get("adapter_id") != "doctor-v5-strand-q4-control"
                or pilot_spec.get("operation") != "condense_pilot"
                or pilot_spec.get("profile") != "strand-scalar-quality-rhtcols-v1"
                or pilot_spec.get("bits") != 4):
            blockers.append("0.5B typed pilot spec is not the frozen Q4 control")
        for field in ("program_spec_sha256", "resource_admission_sha256"):
            if not HEX64.fullmatch(str(pilot_spec.get(field, ""))):
                blockers.append(f"typed pilot spec {field} is invalid")
        if (not isinstance(pilot_spec.get("model_family"), str)
                or not pilot_spec.get("model_family")
                or not isinstance(pilot_spec.get("backend"), str)
                or not pilot_spec.get("backend")
                or isinstance(pilot_spec.get("seed"), bool)
                or not isinstance(pilot_spec.get("seed"), int)
                or pilot_spec.get("seed", -1) < 0):
            blockers.append("typed pilot spec model/backend/seed fields are invalid")
        scratch_bytes = pilot_spec.get("scratch_budget_bytes")
        if (isinstance(scratch_bytes, bool) or not isinstance(scratch_bytes, int)
                or scratch_bytes < 12_000_000_000):
            blockers.append("typed pilot spec scratch admission is below 12 GB")
        reserve_bytes = pilot_spec.get("disk_reserve_bytes")
        if (isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int)
                or reserve_bytes < int(MIN_DISK_FREE_GB * 1e9)):
            blockers.append("typed pilot spec disk reserve is below 150 GB")
        inputs = pilot_spec.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            blockers.append("typed pilot spec inputs are empty or invalid")
        else:
            for position, row in enumerate(inputs):
                if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
                    blockers.append(f"typed pilot input {position} keys are invalid")
                    continue
                relative = _safe_relative(row.get("path"))
                size = row.get("bytes")
                if (not isinstance(row.get("role"), str) or not row.get("role")
                        or relative is None
                        or not HEX64.fullmatch(str(row.get("sha256", "")))
                        or isinstance(size, bool) or not isinstance(size, int) or size < 0):
                    blockers.append(f"typed pilot input {position} identity is invalid")
                    continue
                path = ROOT / relative
                if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
                    blockers.append(f"typed pilot input {position} file/size is invalid")
                elif verify_files and _sha256_file(path) != row["sha256"]:
                    blockers.append(f"typed pilot input {position} hash mismatch")
    if pilot is not None:
        if pilot.get("registry_file_sha256") != registry_file_sha:
            blockers.append("pilot is not bound to the current reviewed adapter registry")
        if pilot.get("parameter_manifest_file_sha256") != manifest_file_sha:
            blockers.append("pilot is not bound to the current exact parameter manifest")
        if pilot.get("pilot_spec_file_sha256") != pilot_spec_file_sha:
            blockers.append("pilot is not bound to the current reviewed typed spec")
        if isinstance(pilot_spec, dict) and (
                pilot.get("adapter_id") != pilot_spec.get("adapter_id")
                or pilot.get("command_id") != pilot_spec.get("operation")):
            blockers.append("pilot adapter/operation differs from the reviewed typed spec")
    census_path = PASS_A_ROOT / label / "census.json"
    census = _read_json(census_path)
    if isinstance(manifest, dict) and isinstance(census, dict):
        source = census.get("source") if isinstance(census.get("source"), dict) else {}
        # These identity fields are required independent of the external validator.
        if manifest.get("label") != label:
            blockers.append("parameter manifest label differs from the ladder rung")
        if manifest.get("hf_id") != COHORT_BY_LABEL[label]["hf_id"]:
            blockers.append("parameter manifest model ID differs from Pass A")
        census_hash = census.get("report_sha256")
        source_hash = source.get("source_manifest_sha256")
        candidate_values = set()
        for key in ("census_report_sha256", "pass_a_report_sha256"):
            value = manifest.get(key)
            if value is not None:
                candidate_values.add(value)
        if census_hash not in candidate_values:
            blockers.append("parameter manifest is not bound to the Pass-A census report")
        candidate_sources = set()
        for key in ("source_manifest_sha256", "pass_a_source_manifest_sha256"):
            value = manifest.get(key)
            if value is not None:
                candidate_sources.add(value)
        if source_hash not in candidate_sources:
            blockers.append("parameter manifest is not bound to the Pass-A source manifest")
    return {
        "blockers": blockers, "adapter_module": adapter_module,
        "parameter_module": parameter_module, "registry": registry,
        "manifest": manifest, "manifest_path": manifest_path,
        "pilot_spec": pilot_spec, "pilot_spec_path": pilot_spec_path,
        "registry_file_sha256": registry_file_sha,
        "parameter_manifest_file_sha256": manifest_file_sha,
        "pilot_spec_file_sha256": pilot_spec_file_sha,
    }


def _builder(module: Any, candidates: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in candidates:
        value = getattr(module, name, None)
        if callable(value):
            return value
    return None


def _request_binds_pilot(request: Any, label: str, pilot: dict[str, Any]) -> bool:
    if not isinstance(request, dict):
        return False
    model = request.get("model") if isinstance(request.get("model"), dict) else {}
    observed_label = request.get("label", model.get("label"))
    authorization = request.get("authorization") \
        if isinstance(request.get("authorization"), dict) else {}
    observed_greenlight = authorization.get(
        "operator_greenlight_sha256", request.get("pilot_injection_sha256")
    )
    return observed_label == label and observed_greenlight == pilot["injection_sha256"]


def _prepare_execution(label: str, pilot: dict[str, Any], *, write: bool) -> dict[str, Any]:
    snapshot = _contract_snapshot(label, pilot=pilot, verify_files=True)
    blockers = list(snapshot["blockers"])
    adapter_module = snapshot["adapter_module"]
    registry = snapshot["registry"]
    manifest = snapshot["manifest"]
    pilot_spec = snapshot["pilot_spec"]
    attempt_root = RESULTS_ROOT / label
    request_path = attempt_root / "request.json"
    result_path = attempt_root / "result.json"
    receipt_path = attempt_root / "execution_receipt.json"
    checkpoint_path = attempt_root / "checkpoint.json"
    census_path = PASS_A_ROOT / label / "census.json"
    request = _read_json(request_path)
    if not _request_binds_pilot(request, label, pilot):
        request = None
    command = None
    scratch_bytes = pilot_spec.get("scratch_budget_bytes", 0) \
        if isinstance(pilot_spec, dict) else 0
    scratch_gb = scratch_bytes / 1e9 \
        if isinstance(scratch_bytes, int) and not isinstance(scratch_bytes, bool) \
        and scratch_bytes >= 0 else 0.0
    if (adapter_module is not None and isinstance(registry, dict)
            and isinstance(manifest, dict) and isinstance(pilot_spec, dict)):
        if request is None:
            request_builder = _builder(adapter_module, ("build_request", "make_request"))
            if request_builder is None:
                blockers.append("adapter ABI does not expose build_request/make_request")
            else:
                try:
                    census = _read_json(census_path, {})
                    normalized_inputs = [
                        {**row, "path": str((ROOT / row["path"]).resolve(strict=True))}
                        for row in pilot_spec["inputs"]
                    ]
                    request = request_builder(
                        registry=registry, adapter_id=pilot["adapter_id"],
                        operation=pilot["command_id"],
                        program_spec_sha256=pilot_spec["program_spec_sha256"],
                        parameter_manifest_path=str(snapshot["manifest_path"]),
                        parameter_manifest_sha256=snapshot[
                            "parameter_manifest_file_sha256"
                        ],
                        source_census_sha256=census.get("report_sha256"),
                        model_label=label, model_family=pilot_spec["model_family"],
                        backend=pilot_spec["backend"], seed=pilot_spec["seed"],
                        inputs=normalized_inputs,
                        pilot_spec_path=str(snapshot["pilot_spec_path"]),
                        pilot_spec_sha256=snapshot["pilot_spec_file_sha256"],
                        pilot_spec_schema=pilot_spec["schema"],
                        request_path=str(request_path), output_dir=str(attempt_root),
                        checkpoint_path=str(checkpoint_path), result_path=str(result_path),
                        execution_receipt_path=str(receipt_path),
                        operator_greenlight_sha256=pilot["injection_sha256"],
                        resource_admission_sha256=pilot_spec[
                            "resource_admission_sha256"
                        ],
                    )
                except Exception as exc:
                    blockers.append(f"adapter request builder failed: {type(exc).__name__}: {exc}")
        if isinstance(request, dict):
            blockers.extend(_call_validator(adapter_module, "validate_request", request,
                                            registry, verify_files=True))
            resolver = _builder(adapter_module, ("resolve_command", "command_argv"))
            if resolver is None:
                blockers.append("adapter ABI does not expose resolve_command/command_argv")
            else:
                try:
                    command = resolver(request, registry, request_path=str(request_path))
                except Exception as exc:
                    blockers.append(f"adapter command resolver failed: {type(exc).__name__}: {exc}")
        elif request is not None:
            blockers.append("adapter request builder did not return an object")
    if command is not None:
        if (not isinstance(command, list) or not command
                or any(not isinstance(token, str) or not token or "\x00" in token
                       for token in command)):
            blockers.append("adapter command resolver did not return typed argv")
        else:
            raw_executable = Path(command[0])
            located = shutil.which(command[0]) if len(raw_executable.parts) == 1 else None
            executable = Path(located) if located else raw_executable
            if not executable.is_absolute():
                executable = ROOT / executable
            resolved = executable.resolve()
            system_python = resolved == Path(sys.executable).resolve()
            if not system_python:
                try:
                    resolved.relative_to(ROOT.resolve())
                except ValueError:
                    blockers.append("adapter executable resolves outside the workspace")
            if not resolved.is_file() or (executable.is_symlink() and not system_python):
                blockers.append("adapter executable is missing or is an unreviewed symlink")
    if write and not blockers and isinstance(request, dict):
        attempt_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(request_path, request)
    return {
        **snapshot, "blockers": blockers, "request": request, "command": command,
        "scratch_gb": scratch_gb, "attempt_root": attempt_root,
        "request_path": request_path, "result_path": result_path,
        "receipt_path": receipt_path, "checkpoint_path": checkpoint_path,
    }


def _validate_outputs(execution: dict[str, Any]) -> tuple[dict[str, Any] | None,
                                                          dict[str, Any] | None,
                                                          list[str]]:
    errors: list[str] = []
    module = execution["adapter_module"]
    request = execution["request"]
    registry = execution["registry"]
    result = _read_json(execution["result_path"])
    receipt = _read_json(execution["receipt_path"])
    checkpoint = _read_json(execution["checkpoint_path"])
    if not isinstance(result, dict):
        errors.append("adapter result is missing or unreadable")
    else:
        errors.extend(_call_validator(module, "validate_result", result, request, registry,
                                      verify_files=True))
    if not isinstance(checkpoint, dict):
        errors.append("adapter exact-resume checkpoint is missing or unreadable")
    else:
        errors.extend(_call_validator(module, "validate_checkpoint", checkpoint,
                                      request, registry, verify_files=True))
    if not isinstance(receipt, dict):
        errors.append("adapter execution receipt is missing or unreadable")
    elif isinstance(result, dict):
        errors.extend(_call_validator(module, "validate_execution_receipt", receipt,
                                      request, result, registry,
                                      checkpoint=checkpoint if isinstance(checkpoint, dict)
                                      else None,
                                      command_argv=execution.get("command")))
    return result if isinstance(result, dict) else None, \
        receipt if isinstance(receipt, dict) else None, errors


def _acquire_lease() -> Any | None:
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = open(HEAVY_LOCK, "a+")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    return lease


def _release_lease(lease: Any | None) -> None:
    if lease is not None:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()


def _terminate_child(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _run_external(state: dict[str, Any], label: str, execution: dict[str, Any]) -> int:
    lease = None
    while lease is None:
        control = _apply_injections(state)
        if _STOP or DRAIN_REQUEST.exists() or control == "drain":
            return 130
        if control == "pause":
            return PAUSE_RC
        lease = _acquire_lease()
        if lease is None:
            _save_state(state, "waiting-heavy-lease", active_label=label)
            time.sleep(POLL_SECONDS)
    try:
        if execution["result_path"].is_file() and execution["receipt_path"].is_file():
            result, receipt, errors = _validate_outputs(execution)
            if not errors and isinstance(result, dict) and isinstance(receipt, dict):
                return ADOPT_RC
        gate = _resource_gate(execution["scratch_gb"])
        state["last_resource_gate"] = gate
        if not gate["ok"]:
            _save_state(state, "waiting-resources", active_label=label)
            return 75
        command = execution["command"]
        if not isinstance(command, list):
            raise QueueError("internal typed-command invariant failed")
        log_path = execution["attempt_root"] / "pilot.log"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HAWKING_HEAVY_LEASE_FD"] = str(lease.fileno())
        with open(log_path, "ab", buffering=0) as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, env=environment,
                pass_fds=(lease.fileno(),), shell=False,
            )
            state["active_child"] = {
                "label": label, "pid": process.pid, "pgid": process.pid,
                "command_sha256": _hash_value(command),
                "request_sha256": _hash_value(execution["request"]),
                "started_at": _now(),
            }
            _save_state(state, "running-pilot", active_label=label)
            while process.poll() is None:
                time.sleep(POLL_SECONDS)
                control = _apply_injections(state)
                if _STOP or DRAIN_REQUEST.exists() or control == "drain":
                    _terminate_child(process)
                    return 130
                if control == "pause":
                    _terminate_child(process)
                    return PAUSE_RC
                gate = _resource_gate(execution["scratch_gb"])
                state["last_resource_gate"] = gate
                if not gate["ok"]:
                    _terminate_child(process)
                    return 75
            return int(process.returncode)
    finally:
        state["active_child"] = None
        _save_state(state)
        _release_lease(lease)


def _wait_control(state: dict[str, Any]) -> bool:
    while True:
        control = _apply_injections(state)
        if _STOP or DRAIN_REQUEST.exists() or control == "drain":
            return False
        if control == "run":
            return True
        _save_state(state, "paused-control", active_label=None, active_child=None)
        time.sleep(POLL_SECONDS)


def _first_incomplete(state: dict[str, Any]) -> str | None:
    for label in LABELS:
        if state["items"][label]["status"] != "complete":
            return label
    return None


def _commit_complete(state: dict[str, Any], label: str, request: dict[str, Any],
                     result: dict[str, Any], receipt: dict[str, Any]) -> None:
    row = state["items"][label]
    row["status"] = "complete"
    row["completed_at"] = _now()
    row["request_sha256"] = _hash_value(request)
    row["result_sha256"] = _hash_value(result)
    row["execution_receipt_sha256"] = _hash_value(receipt)
    row["error"] = None
    state["pending_order"] = list(LABELS[LABELS.index(label) + 1:])
    _save_state(state, "running", active_label=None, active_child=None)


def run_queue(nonce: str) -> int:
    if not HEX32.fullmatch(nonce):
        raise QueueError("ownership nonce is invalid")
    identity = _identity()
    if identity["pass_a_validation_errors"]:
        raise QueueError("Pass-A validation failed: "
                         + "; ".join(identity["pass_a_validation_errors"]))
    QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_LOCK, "a+") as singleton:
        try:
            fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueError("another Pass-B supervisor holds the singleton lease") from exc
        record = _pid_record(identity, nonce)
        _atomic_json(PID_FILE, record)
        state = _load_state(identity)
        state["supervisor_pid"] = os.getpid()
        state["error"] = None
        _save_state(state, "running")
        while True:
            live_identity = _identity()
            if live_identity["queue_identity_sha256"] != state["queue_identity_sha256"]:
                raise QueueError("queue, contracts, or Pass-A identity changed during execution")
            if not _wait_control(state):
                state["control_mode"] = "drain"
                _save_state(state, "drained", supervisor_pid=None, active_label=None,
                            active_child=None, drained_at=_now())
                return 0
            label = _first_incomplete(state)
            if label is None:
                state["pending_order"] = []
                _save_state(state, "complete", supervisor_pid=None,
                            active_label=None, active_child=None, completed_at=_now())
                return 0
            pilot = state["pilot_requests"].get(label)
            if not isinstance(pilot, dict):
                _save_state(state, "waiting-pilot", active_label=label,
                            last_prerequisite_blockers=[
                                f"typed reviewed pilot injection required for {label}"
                            ])
                time.sleep(POLL_SECONDS)
                continue
            execution = _prepare_execution(label, pilot, write=True)
            blockers = execution["blockers"]
            if blockers:
                _save_state(state, "waiting-prerequisites", active_label=label,
                            last_prerequisite_blockers=blockers)
                time.sleep(POLL_SECONDS)
                continue
            # A worker inherits the heavy lease.  If its supervisor died after the
            # worker committed valid artifacts, a restarted supervisor adopts that
            # completed result instead of launching the typed command twice.
            if execution["result_path"].is_file() and execution["receipt_path"].is_file():
                result, receipt, output_errors = _validate_outputs(execution)
                if not output_errors and isinstance(result, dict) and isinstance(receipt, dict):
                    _commit_complete(state, label, execution["request"], result, receipt)
                    continue
            gate = _resource_gate(execution["scratch_gb"])
            state["last_resource_gate"] = gate
            if not gate["ok"]:
                _save_state(state, "waiting-resources", active_label=label)
                time.sleep(POLL_SECONDS)
                continue
            row = state["items"][label]
            row["status"] = "running"
            row["attempts"] += 1
            row["started_at"] = _now()
            row["request_sha256"] = _hash_value(execution["request"])
            row["error"] = None
            state["last_prerequisite_blockers"] = []
            _save_state(state, "running-pilot", active_label=label)
            rc = _run_external(state, label, execution)
            row["last_exit_code"] = rc
            if rc == ADOPT_RC:
                result, receipt, output_errors = _validate_outputs(execution)
                if output_errors or not isinstance(result, dict) or not isinstance(receipt, dict):
                    raise QueueError("adopted worker artifacts changed after validation")
                _commit_complete(state, label, execution["request"], result, receipt)
                continue
            if rc == PAUSE_RC:
                row["status"] = "pending"
                _save_state(state, "paused-control")
                continue
            if rc == 130:
                row["status"] = "pending"
                state["control_mode"] = "drain"
                _save_state(state, "drained", supervisor_pid=None, active_label=None,
                            drained_at=_now())
                return 0
            if rc == 75:
                row["status"] = "pending"
                _save_state(state, "waiting-resources")
                time.sleep(POLL_SECONDS)
                continue
            if rc != 0:
                row["status"] = "blocked"
                row["error"] = f"reviewed pilot exited with status {rc}"
                _save_state(state, "blocked-execution", supervisor_pid=None,
                            active_label=label, error=row["error"])
                return rc if 0 < rc < 126 else 2
            result, receipt, output_errors = _validate_outputs(execution)
            if output_errors:
                row["status"] = "blocked"
                row["error"] = "invalid execution artifacts: " + "; ".join(output_errors)
                _save_state(state, "blocked-execution", supervisor_pid=None,
                            active_label=label, error=row["error"])
                return 2
            if not isinstance(result, dict) or not isinstance(receipt, dict):
                raise QueueError("output validator returned no errors but omitted artifacts")
            _commit_complete(state, label, execution["request"], result, receipt)


def start_queue() -> int:
    identity = _identity()
    if identity["pass_a_validation_errors"]:
        raise QueueError("refusing to start: Pass-A validation failed: "
                         + "; ".join(identity["pass_a_validation_errors"]))
    old = _read_json(PID_FILE, {})
    if _owner_alive(old, identity):
        print(f"[doctor-v5-pass-b] already active pid={old['pid']}")
        return 0
    nonce = secrets.token_hex(16)
    command = [sys.executable, str(QUEUE_SCRIPT), "run", "--nonce", nonce]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True, shell=False)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = _read_json(PID_FILE, {})
        if record.get("ownership_nonce") == nonce and _owner_alive(record, identity):
            print(f"[doctor-v5-pass-b] detached pid={record['pid']} log={LOG_FILE}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise QueueError("detached ownership handshake failed")


def _pilot_payload(label: str, adapter_id: str, command_id: str) -> dict[str, Any]:
    if label not in LABELS:
        raise QueueError("pilot label is not in the fixed ladder")
    if not adapter_id or not command_id:
        raise QueueError("pilot adapter-id and command-id must be nonempty")
    snapshot = _contract_snapshot(label, verify_files=True)
    if snapshot["blockers"]:
        raise ContractBlocked("pilot prerequisites invalid: "
                              + "; ".join(snapshot["blockers"]))
    payload = {
        "label": label, "adapter_id": adapter_id, "command_id": command_id,
        "registry_file_sha256": snapshot["registry_file_sha256"],
        "parameter_manifest_file_sha256": snapshot["parameter_manifest_file_sha256"],
        "pilot_spec_file_sha256": snapshot["pilot_spec_file_sha256"],
    }
    provisional = {**payload, "injection_sha256": "0" * 64}
    # Prove that this adapter/command pair resolves before it enters the ledger.
    execution = _prepare_execution(label, provisional, write=False)
    filtered = [row for row in execution["blockers"]
                if "pilot is not bound" not in row]
    if filtered:
        raise ContractBlocked("pilot command is not executable through the reviewed ABI: "
                              + "; ".join(filtered))
    return payload


def inject(args: argparse.Namespace) -> int:
    if args.action in {"pause", "resume", "drain"}:
        payload: dict[str, Any] = {}
    elif args.action == "pilot":
        payload = _pilot_payload(args.label, args.adapter_id, args.command_id)
    else:
        raise QueueError("unsupported injection action")
    row = _append_injection(args.action, payload)
    if args.action == "drain":
        _atomic_json(DRAIN_REQUEST, {
            "schema": "hawking.doctor_v5_pass_b_drain_request.v1",
            "requested_at": _now(), "injection_sha256": row["injection_sha256"],
        })
        owner = _read_json(PID_FILE, {})
        if _owner_alive(owner):
            try:
                os.kill(int(owner["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
    elif args.action == "resume" and DRAIN_REQUEST.exists():
        DRAIN_REQUEST.unlink()
        _fsync_dir(DRAIN_REQUEST.parent)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def drain() -> int:
    args = argparse.Namespace(action="drain", label=None, adapter_id=None, command_id=None)
    return inject(args)


def resume() -> int:
    args = argparse.Namespace(action="resume", label=None, adapter_id=None, command_id=None)
    inject(args)
    return start_queue()


def _plan_payload(include_state: bool) -> tuple[dict[str, Any], int]:
    identity = _identity()
    state_error = None
    try:
        state = _load_state(identity)
    except Exception as exc:
        state = _read_json(STATE, {})
        state_error = f"{type(exc).__name__}: {exc}"
    current = _first_incomplete(state) if isinstance(state, dict) \
        and isinstance(state.get("items"), dict) else LABELS[0]
    pilot = state.get("pilot_requests", {}).get(current) \
        if current and isinstance(state, dict) else None
    contracts = _contract_snapshot(current, pilot=pilot, verify_files=False) \
        if current else {"blockers": []}
    payload = {
        "schema": STATUS_SCHEMA if include_state else "hawking.doctor_v5_pass_b_plan.v1",
        "generated_at": _now(), "queue_identity": identity,
        "fixed_ladder": list(LABELS), "current_label": current,
        "pass_a_valid": not identity["pass_a_validation_errors"],
        "pass_a_validation_errors": identity["pass_a_validation_errors"],
        "prerequisite_blockers": contracts["blockers"],
        "resource_gate": _resource_gate(),
        "source_deletion_permitted": False,
        "typed_command_policy": {
            "shell": False, "arbitrary_argv_injection": False,
            "selection_fields": ["label", "adapter_id", "command_id"],
            "registry_validation_required": True,
            "parameter_manifest_validation_required": True,
        },
        "restart_command": "python3.12 tools/condense/doctor_v5_pass_b_queue.py resume",
    }
    if include_state:
        identity_min = identity if not identity["pass_a_validation_errors"] else None
        owner = _read_json(PID_FILE, {})
        rows, injection_errors = _read_injections(identity["queue_identity_sha256"])
        payload.update({
            "active": _owner_alive(owner, identity_min) if identity_min else False,
            "pid": owner.get("pid") if isinstance(owner, dict) else None,
            "state": state, "state_error": state_error,
            "drain_requested": DRAIN_REQUEST.exists(),
            "injection_count": len(rows), "injection_errors": injection_errors,
        })
        return payload, 0 if state_error is None and not injection_errors \
            and not identity["pass_a_validation_errors"] else 1
    return payload, 0 if not identity["pass_a_validation_errors"] else 1


def plan() -> int:
    payload, code = _plan_payload(False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


def status() -> int:
    payload, code = _plan_payload(True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


def selftest() -> int:
    global PASS_B_ROOT, STATE, PID_FILE, QUEUE_LOCK, INJECTION_LOCK, INJECTIONS
    global DRAIN_REQUEST, LOG_FILE, ADAPTER_REGISTRY, PARAMETER_MANIFESTS, PILOT_SPECS
    global RESULTS_ROOT
    _, pass_a_errors = _pass_a_index()
    assert not pass_a_errors, pass_a_errors
    (ROOT / "scratch").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as raw:
        old = (PASS_B_ROOT, STATE, PID_FILE, QUEUE_LOCK, INJECTION_LOCK, INJECTIONS,
               DRAIN_REQUEST, LOG_FILE, ADAPTER_REGISTRY, PARAMETER_MANIFESTS, PILOT_SPECS,
               RESULTS_ROOT)
        PASS_B_ROOT = Path(raw)
        STATE = PASS_B_ROOT / "state.json"
        PID_FILE = PASS_B_ROOT / "pid.json"
        QUEUE_LOCK = PASS_B_ROOT / "queue.lock"
        INJECTION_LOCK = PASS_B_ROOT / "injections.lock"
        INJECTIONS = PASS_B_ROOT / "injections.jsonl"
        DRAIN_REQUEST = PASS_B_ROOT / "drain.json"
        LOG_FILE = PASS_B_ROOT / "queue.log"
        ADAPTER_REGISTRY = PASS_B_ROOT / "adapter_registry.json"
        PARAMETER_MANIFESTS = PASS_B_ROOT / "parameter_manifests"
        PILOT_SPECS = PASS_B_ROOT / "pilot_specs"
        RESULTS_ROOT = PASS_B_ROOT / "results"
        identity = _identity()
        state = _base_state(identity)
        _save_state(state)
        pause = _append_injection("pause", {})
        assert _apply_injections(state) == "pause"
        resume_row = _append_injection("resume", {})
        assert _apply_injections(state) == "run"
        assert resume_row["prior_sha256"] == pause["injection_sha256"]
        assert not _read_injections(identity["queue_identity_sha256"])[1]
        damaged = _read_json(STATE)
        damaged["source_deletion_permitted"] = True
        damaged["state_sha256"] = _hash_value({key: value for key, value in damaged.items()
                                                if key != "state_sha256"})
        _atomic_json(STATE, damaged)
        try:
            _load_state(identity)
        except QueueError:
            pass
        else:
            raise AssertionError("source-deletion state tamper was accepted")
        lease = _acquire_lease()
        # The real shared heavy lease may be held by another workflow; lease behavior
        # is tested only when this selftest acquires it without disturbing that owner.
        if lease is not None:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.2)"],
                                     pass_fds=(lease.fileno(),))
            lease.close()
            assert _acquire_lease() is None
            child.wait(timeout=3)
            reacquired = _acquire_lease()
            assert reacquired is not None
            _release_lease(reacquired)
        assert _safe_relative("a/b") == Path("a/b")
        assert _safe_relative("../a") is None and _safe_relative("/tmp/a") is None
        gate = _resource_gate()
        assert gate["required_free_gb"] == (MIN_DISK_FREE_GB + CONTROL_RESERVE_GB
                                             + MIN_PROCESSING_SCRATCH_GB)
        (PASS_B_ROOT, STATE, PID_FILE, QUEUE_LOCK, INJECTION_LOCK, INJECTIONS,
         DRAIN_REQUEST, LOG_FILE, ADAPTER_REGISTRY, PARAMETER_MANIFESTS, PILOT_SPECS,
         RESULTS_ROOT) = old
    print("doctor_v5_pass_b_queue.py selftest OK")
    return 0


def _request_stop(_sig: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("status")
    commands.add_parser("ping")
    commands.add_parser("start")
    run = commands.add_parser("run")
    run.add_argument("--nonce", required=True)
    commands.add_parser("drain")
    commands.add_parser("resume")
    injection = commands.add_parser("inject")
    injection.add_argument("action", choices=sorted(ALLOWED_ACTIONS))
    injection.add_argument("--label", choices=LABELS)
    injection.add_argument("--adapter-id")
    injection.add_argument("--command-id")
    commands.add_parser("selftest")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        if args.command == "plan":
            return plan()
        if args.command in {"status", "ping"}:
            return status()
        if args.command == "start":
            return start_queue()
        if args.command == "run":
            return run_queue(args.nonce)
        if args.command == "drain":
            return drain()
        if args.command == "resume":
            return resume()
        if args.command == "inject":
            return inject(args)
        if args.command == "selftest":
            return selftest()
    except Exception as exc:
        print(f"[doctor-v5-pass-b] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
