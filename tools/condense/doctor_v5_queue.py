#!/usr/bin/env python3.12
"""Detached two-pass control plane for the Doctor-v5 Studio scale chain.

Pass A produces one bootstrap source census for each fixed scale rung and may
download the official 120B MXFP4 subset after conservative disk admission.
Pass B is deliberately fail-closed until independently reviewed executable
Doctor-v5 adapters and all role-separated receipts are injected.  This queue
does not execute legacy Doctor code or relabel legacy observations as v5.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

import doctor_v5_audit
import doctor_v5_census
import procure
from ram_scheduler import resource_snapshot, thermal_output_ok


SCHEMA = "hawking.doctor_v5_scale_queue.v1"
INDEX_SCHEMA = "hawking.doctor_v5_scale_index.v1"
INJECTION_SCHEMA = "hawking.doctor_v5_scale_injection.v1"
QUEUE_VERSION = "2026-07-12.1"
SCALE_ROOT = ROOT / "reports" / "condense" / "doctor_v5_scale"
STATE = SCALE_ROOT / "queue_state.json"
PID_FILE = SCALE_ROOT / "queue.pid.json"
LOCK_FILE = SCALE_ROOT / "queue.lock"
INJECTION_LOCK = SCALE_ROOT / "injections.lock"
INJECTIONS = SCALE_ROOT / "injections.jsonl"
DRAIN_REQUEST = SCALE_ROOT / "drain.request"
LOG_FILE = SCALE_ROOT / "queue.log"
INDEX_FILE = SCALE_ROOT / "index.json"
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
DOWNLOAD_LOCK = ROOT / "reports" / "condense" / "download_state" / "120B.lock"
CENSUS_SCRIPT = ROOT / "tools" / "condense" / "doctor_v5_census.py"
QUEUE_SCRIPT = Path(__file__).resolve()

DISK_IMMUTABLE_FLOOR_GB = 150.0
PROCESSING_SCRATCH_GB = 64.0
HF_CACHE_RESERVE_GB = 32.0
CONTROL_OVERHEAD_GB = 2.0
DOWNLOAD_FLOOR_GB = DISK_IMMUTABLE_FLOOR_GB + PROCESSING_SCRATCH_GB + HF_CACHE_RESERVE_GB
POLL_SECONDS = 10.0
PAUSE_RC = 131

ROOT_MANIFEST = ROOT / "reports" / "condense" / "doctor_v5_root.json"
CAMPAIGN = ROOT / "reports" / "condense" / "doctor_v5_campaign.json"
LADDER = ROOT / "reports" / "condense" / "training_ladder_v5.json"
BATTERY = ROOT / "reports" / "condense" / "quality_battery_v5.json"
AUDIT = ROOT / "reports" / "condense" / "doctor_v5_audit.json"
REMOTE_120B_MANIFEST = ROOT / "reports" / "condense" / "doctor_v5_120b_remote_manifest.json"

COHORT = (
    {"label": "0.5B", "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
     "model_dir": "scratch/qwen-05b", "download_marker": None},
    {"label": "1.5B", "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
     "model_dir": "scratch/qwen-15b", "download_marker": None},
    {"label": "7B", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
     "model_dir": "scratch/qwen-7b", "download_marker": None},
    {"label": "14B", "hf_id": "Qwen/Qwen2.5-14B-Instruct",
     "model_dir": "scratch/staging/qwen-14b.partial",
     "download_marker": "reports/condense/download_state/14B.verified.json"},
    {"label": "32B", "hf_id": "Qwen/Qwen2.5-32B-Instruct",
     "model_dir": "scratch/staging/qwen-32b.partial",
     "download_marker": "reports/condense/download_state/32B.verified.json"},
    {"label": "72B", "hf_id": "Qwen/Qwen2.5-72B-Instruct",
     "model_dir": "scratch/staging/qwen-72b.partial",
     "download_marker": "reports/condense/download_state/72B.verified.json"},
    {"label": "120B", "hf_id": "openai/gpt-oss-120b",
     "model_dir": "scratch/staging/gpt-oss-120b.partial",
     "download_marker": "reports/condense/download_state/120B.verified.json"},
)

ALLOWED_ACTIONS = {
    "pause", "resume", "drain", "reprioritize_pending", "add_future_candidate",
}
_STOP = False
STATE_KEYS = {
    "schema", "version", "queue_identity_sha256", "created_at", "updated_at",
    "status", "active_pass", "pending_order", "items", "last_consumed_injection_seq",
    "control_mode", "future_candidates", "pass_b", "source_release",
    "reboot_autostart", "restart_command", "supervisor_pid", "preflight_errors",
    "queue_identity", "active_label", "active_child", "last_resource_gate",
    "index_sha256", "pass_a_completed_at", "error", "blocked_at", "drained_at",
    "state_sha256",
}


class QueueError(RuntimeError):
    pass


class ResourcePause(RuntimeError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
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
                          object_pairs_hook=_no_duplicate_object)
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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
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


def _identity_payload() -> dict[str, Any]:
    reports = {}
    for name, path, identity_field in (
        ("root", ROOT_MANIFEST, "root_manifest_sha256"),
        ("campaign", CAMPAIGN, "campaign_sha256"),
        ("ladder", LADDER, "ladder_sha256"),
        ("battery", BATTERY, "manifest_sha256"),
        ("audit", AUDIT, "audit_sha256"),
    ):
        document = _read_json(path, {})
        reports[name] = {"path": str(path.relative_to(ROOT)),
                         "file_sha256": _sha256_file(path),
                         "identity": document.get(identity_field)}
    return {
        "schema": "hawking.doctor_v5_scale_queue_identity.v1",
        "version": QUEUE_VERSION,
        "cohort": [dict(item) for item in COHORT],
        "sources": {
            "queue": _sha256_file(QUEUE_SCRIPT),
            "census": _sha256_file(CENSUS_SCRIPT),
            "procure": _sha256_file(Path(procure.__file__).resolve()),
            "ram_scheduler": _sha256_file(
                ROOT / "tools" / "condense" / "ram_scheduler.py"
            ),
            "doctor_v5_audit": _sha256_file(Path(doctor_v5_audit.__file__).resolve()),
        },
        "canonical_v5": reports,
        "remote_120b_manifest": {
            "path": str(REMOTE_120B_MANIFEST.relative_to(ROOT)),
            "file_sha256": _sha256_file(REMOTE_120B_MANIFEST),
            "manifest_sha256": _remote_120b_manifest()["manifest_sha256"],
            "revision": _remote_120b_manifest()["revision"],
        },
        "pass_policy": {
            "pass_a": "bootstrap_source_census_only",
            "pass_b": "hard_block_until_audited_executable_programs_and_receipts",
            "legacy_results_are_v5_evidence": False,
            "speed_campaign_enabled": False,
        },
    }


def _queue_identity() -> dict[str, Any]:
    payload = _identity_payload()
    return {**payload, "queue_identity_sha256": _hash_value(payload)}


def _base_state(identity: dict[str, Any]) -> dict[str, Any]:
    state = {
        "schema": SCHEMA,
        "version": QUEUE_VERSION,
        "queue_identity_sha256": identity["queue_identity_sha256"],
        "created_at": _now(),
        "updated_at": _now(),
        "status": "new",
        "active_pass": "pass_a_census",
        "pending_order": [item["label"] for item in COHORT],
        "items": {item["label"]: {"status": "pending", "attempts": 0,
                                  "download_attempts": 0, "started_at": None,
                                  "completed_at": None, "duration_seconds": None}
                  for item in COHORT},
        "last_consumed_injection_seq": 0,
        "control_mode": "run",
        "future_candidates": [],
        "pass_b": {
            "status": "waiting-treatment-adapters",
            "launch_permitted": False,
            "all_ceiling_retention_permitted": False,
            "blockers": [
                "no source-hashed reviewed Doctor-v5 execution adapter",
                "no role-separated greenlight/allowlist/admission/teacher/data receipts",
                "no exact-resume replay receipt",
                "no per-ceiling whole-artifact/transient/checkpoint/resident-memory admission",
                "120B source fit does not imply any 120B treatment ceiling fits",
            ],
        },
        "source_release": {
            "automatic_deletion": False,
            "currently_authorized": False,
            "operator_authorized_after_two_quality_result_bundles": True,
            "required_bundles": ["sub_120B_aggregate_quality", "120B_quality"],
            "census_is_not_a_release_result": True,
        },
        "reboot_autostart": False,
        "restart_command": "python3.12 tools/condense/doctor_v5_queue.py resume",
        "supervisor_pid": None,
        "preflight_errors": [],
        "queue_identity": None,
        "active_label": None,
        "active_child": None,
        "last_resource_gate": None,
        "index_sha256": None,
        "pass_a_completed_at": None,
        "error": None,
        "blocked_at": None,
        "drained_at": None,
    }
    state["state_sha256"] = _hash_value(state)
    return state


def _validate_state(state: Any, identity: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(state, dict):
        return ["state is not an object"]
    if set(state) != STATE_KEYS:
        errors.append("state keys are not exact")
        return errors
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    if state.get("state_sha256") != _hash_value(payload):
        errors.append("state hash mismatch")
    if state.get("schema") != SCHEMA or state.get("version") != QUEUE_VERSION \
            or state.get("queue_identity_sha256") != identity["queue_identity_sha256"]:
        errors.append("state identity mismatch")
    def finite_json(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(finite_json(item) for item in value)
        if isinstance(value, dict):
            return all(isinstance(key, str) and finite_json(item)
                       for key, item in value.items())
        return True
    if not finite_json(state):
        errors.append("state contains a non-finite number")
    allowed_statuses = {
        "new", "running-pass-a", "running-census", "running-download",
        "waiting-heavy-lease", "waiting-resources", "waiting-concurrent-120b-download",
        "paused-control", "drained", "blocked", "blocked-preflight",
        "waiting-treatment-adapters",
    }
    if state.get("status") not in allowed_statuses:
        errors.append("queue status is invalid")
    if state.get("active_pass") not in {"pass_a_census", "pass_b_treatment"}:
        errors.append("active pass is invalid")
    labels = [item["label"] for item in COHORT]
    pending = state.get("pending_order")
    if (not isinstance(pending, list) or len(pending) != len(set(pending))
            or any(label not in labels for label in pending)):
        errors.append("pending order is invalid")
    items = state.get("items")
    item_keys = {"status", "attempts", "download_attempts", "started_at",
                 "completed_at", "duration_seconds"}
    if not isinstance(items, dict) or set(items) != set(labels):
        errors.append("state item registry is invalid")
    else:
        noncomplete: set[str] = set()
        for label, row in items.items():
            if not isinstance(row, dict) or set(row) != item_keys:
                errors.append(f"state item {label} keys are invalid")
                continue
            if row.get("status") not in {"pending", "running", "complete", "blocked"}:
                errors.append(f"state item {label} status is invalid")
            if row.get("status") != "complete":
                noncomplete.add(label)
            else:
                report = _read_json(SCALE_ROOT / label / "census.json")
                cohort_item = next(item for item in COHORT if item["label"] == label)
                if (not isinstance(report, dict) or report.get("label") != label
                        or report.get("hf_id") != cohort_item["hf_id"]
                        or doctor_v5_census.validate_report(report)):
                    errors.append(f"state item {label} is complete without a valid report")
            for key in ("attempts", "download_attempts"):
                if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) \
                        or row[key] < 0:
                    errors.append(f"state item {label} {key} is invalid")
        if isinstance(pending, list) and not noncomplete.issubset(set(pending)):
            errors.append("pending order omits incomplete items")
        if state.get("active_pass") == "pass_b_treatment" and noncomplete:
            errors.append("Pass B selected before all census items completed")
    if state.get("control_mode") not in {"run", "pause", "drain"}:
        errors.append("control mode is invalid")
    sequence = state.get("last_consumed_injection_seq")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        errors.append("last consumed injection sequence is invalid")
    pass_b = state.get("pass_b")
    required_pass_b = {
        "status", "launch_permitted", "all_ceiling_retention_permitted", "blockers",
    }
    if (not isinstance(pass_b, dict) or set(pass_b) != required_pass_b
            or pass_b.get("status") != "waiting-treatment-adapters"
            or pass_b.get("launch_permitted") is not False
            or pass_b.get("all_ceiling_retention_permitted") is not False):
        errors.append("Pass-B fail-closed state is invalid")
    release = state.get("source_release")
    required_release = {
        "automatic_deletion", "currently_authorized",
        "operator_authorized_after_two_quality_result_bundles", "required_bundles",
        "census_is_not_a_release_result",
    }
    if (not isinstance(release, dict) or set(release) != required_release
            or release.get("automatic_deletion") is not False
            or release.get("currently_authorized") is not False
            or release.get("operator_authorized_after_two_quality_result_bundles") is not True
            or release.get("required_bundles") !=
            ["sub_120B_aggregate_quality", "120B_quality"]
            or release.get("census_is_not_a_release_result") is not True):
        errors.append("source-release boundary is invalid")
    if state.get("reboot_autostart") is not False:
        errors.append("reboot-autostart claim is invalid")
    return errors


def _load_state(identity: dict[str, Any] | None = None) -> dict[str, Any]:
    identity = identity or _queue_identity()
    state = _read_json(STATE)
    if state is None:
        if STATE.exists():
            raise QueueError("queue state exists but is corrupt/unreadable")
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
    errors = _validate_state(state, {"queue_identity_sha256": state["queue_identity_sha256"]})
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


def _owner_alive(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("schema") != "hawking.doctor_v5_scale_queue_pid.v1":
        return False
    try:
        if record.get("queue_identity_sha256") != _queue_identity()["queue_identity_sha256"]:
            return False
    except Exception:
        return False
    observed = _ps_identity(record.get("pid"))
    if observed is None:
        return False
    start, command = observed
    nonce = record.get("ownership_nonce")
    return (
        start == record.get("process_start")
        and isinstance(nonce, str) and len(nonce) == 32
        and "doctor_v5_queue.py run" in command
        and f"--nonce {nonce}" in command
    )


def _thermal() -> dict[str, Any]:
    try:
        result = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                                text=True, timeout=5, check=False)
        output = (result.stdout + result.stderr).strip()
        return {"ok": thermal_output_ok(result.returncode, output),
                "returncode": result.returncode, "output": output[-1000:]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _remote_120b_manifest() -> dict[str, Any]:
    document = _read_json(REMOTE_120B_MANIFEST)
    required = {"schema", "repo_id", "revision", "include_patterns", "files",
                "total_bytes", "manifest_sha256"}
    if not isinstance(document, dict) or set(document) != required:
        raise QueueError("120B remote manifest keys are invalid")
    payload = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("manifest_sha256") != _hash_value(payload):
        raise QueueError("120B remote manifest hash is invalid")
    if document.get("schema") != "hawking.doctor_v5_remote_source_manifest.v1" \
            or document.get("repo_id") != "openai/gpt-oss-120b" \
            or document.get("include_patterns") != ["original/*"] \
            or not re.fullmatch(r"[0-9a-f]{40}", str(document.get("revision", ""))):
        raise QueueError("120B remote manifest identity is invalid")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise QueueError("120B remote file manifest is empty")
    total = 0
    names = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "size", "blob_id", "lfs_sha256"}:
            raise QueueError("120B remote file row keys are invalid")
        path, size = row.get("path"), row.get("size")
        if not isinstance(path, str) or not path.startswith("original/") \
                or Path(path).is_absolute() or ".." in Path(path).parts:
            raise QueueError("120B remote file path is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise QueueError("120B remote file size is invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row.get("blob_id", ""))):
            raise QueueError("120B remote blob identity is invalid")
        lfs = row.get("lfs_sha256")
        if lfs is not None and not re.fullmatch(r"[0-9a-f]{64}", str(lfs)):
            raise QueueError("120B remote LFS identity is invalid")
        names.append(path)
        total += size
    if names != sorted(names) or len(names) != len(set(names)) or total != document.get("total_bytes"):
        raise QueueError("120B remote file ordering/total is invalid")
    return document


def _remaining_120b_gb() -> float:
    manifest = _remote_120b_manifest()
    root = ROOT / next(item["model_dir"] for item in COHORT if item["label"] == "120B")
    present = 0
    for row in manifest["files"]:
        path = root / row["path"]
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
            if size > row["size"]:
                raise QueueError(f"120B local file exceeds pinned manifest: {row['path']}")
            present += size
    return max(0, manifest["total_bytes"] - present) / 1e9


def _verify_local_120b_manifest() -> list[str]:
    errors: list[str] = []
    manifest = _remote_120b_manifest()
    root = ROOT / next(item["model_dir"] for item in COHORT if item["label"] == "120B")
    expected = {row["path"] for row in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    } if root.is_dir() else set()
    if actual != expected:
        return ["120B local inventory does not match the pinned original/* manifest"]
    for row in manifest["files"]:
        path = root / row["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size"]:
            errors.append(f"120B size/type mismatch: {row['path']}")
            continue
        raw_hash = _sha256_file(path)
        if row["lfs_sha256"] is not None:
            if raw_hash != row["lfs_sha256"]:
                errors.append(f"120B LFS hash mismatch: {row['path']}")
        else:
            header = f"blob {row['size']}\0".encode("ascii")
            git_blob = hashlib.sha1(header + path.read_bytes()).hexdigest()
            if git_blob != row["blob_id"]:
                errors.append(f"120B Git blob mismatch: {row['path']}")
    return errors


def _resource_gate(*, remaining_download_gb: float = 0.0) -> dict[str, Any]:
    if (isinstance(remaining_download_gb, bool)
            or not isinstance(remaining_download_gb, (int, float))
            or not math.isfinite(remaining_download_gb) or remaining_download_gb < 0):
        remaining_download_gb = float("nan")
    snapshot = resource_snapshot()
    thermal = _thermal()
    memory = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
    pressure = snapshot.get("pressure_level", memory.get("pressure_level"))
    swap = snapshot.get("swap_used_mb", memory.get("swap_used_mb"))
    free = snapshot.get("disk_free_gb")
    power = str(snapshot.get("power_source", ""))
    required = (DOWNLOAD_FLOOR_GB + remaining_download_gb + CONTROL_OVERHEAD_GB
                if math.isfinite(remaining_download_gb) else float("nan"))
    blockers = []
    if isinstance(pressure, bool) or not isinstance(pressure, int) or pressure != 1:
        blockers.append("memory pressure is not normal")
    if (not isinstance(swap, (int, float)) or isinstance(swap, bool)
            or not math.isfinite(swap) or swap < 0 or swap > 0.0):
        blockers.append("swap is nonzero or unavailable")
    if "AC Power" not in power:
        blockers.append("AC power is not confirmed")
    if not thermal.get("ok"):
        blockers.append("thermal state is not green")
    if (not isinstance(free, (int, float)) or isinstance(free, bool)
            or not math.isfinite(free) or not math.isfinite(required) or free < required):
        blockers.append(f"disk free is below {required:.3f} GB")
    return {"schema": "hawking.doctor_v5_scale_resource_gate.v1",
            "sampled_at": _now(), "ok": not blockers, "blockers": blockers,
            "remaining_download_gb": round(remaining_download_gb, 6),
            "required_free_gb": round(required, 6), "resources": snapshot,
            "thermal": thermal}


def _preflight(identity: dict[str, Any]) -> list[str]:
    errors = []
    commands = (
        [sys.executable, "tools/condense/doctor_v5.py", "validate", str(CAMPAIGN)],
        [sys.executable, "tools/condense/training_ladder_v5.py", "validate", str(LADDER)],
        [sys.executable, "tools/condense/quality_battery_v5.py", "validate", str(BATTERY)],
        [sys.executable, "tools/condense/doctor_v5_root.py", "validate", str(ROOT_MANIFEST)],
    )
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                timeout=180, check=False)
        if result.returncode:
            errors.append(f"validator failed ({' '.join(command[1:3])}): "
                          f"{(result.stderr or result.stdout).strip()[-500:]}")
    audit = _read_json(AUDIT, {})
    audit_errors = doctor_v5_audit.validate_audit(audit)
    if audit_errors:
        errors.append("Doctor-v5 static audit receipt invalid: " + "; ".join(audit_errors[:5]))
    if audit.get("static_integrity_pass") is not True \
            or audit.get("design_package_complete") is not True \
            or audit.get("execution_authorized") is not False:
        errors.append("Doctor-v5 static audit boundary is not the expected fail-closed state")
    if identity != _queue_identity():
        errors.append("queue identity changed during preflight")
    return errors


def _read_injections_unlocked(identity_sha: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    prior = "0" * 64
    if not INJECTIONS.exists():
        return rows, errors
    for line_number, line in enumerate(INJECTIONS.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=_no_duplicate_object)
        except (json.JSONDecodeError, ValueError):
            errors.append(f"injection line {line_number} is invalid JSON")
            continue
        required = {"schema", "sequence", "created_at", "action", "payload",
                    "queue_identity_sha256", "prior_sha256", "injection_sha256"}
        if not isinstance(row, dict) or set(row) != required:
            errors.append(f"injection line {line_number} keys are not exact")
            continue
        if row.get("schema") != INJECTION_SCHEMA:
            errors.append(f"injection line {line_number} schema is invalid")
        sequence = row.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            errors.append(f"injection line {line_number} sequence type is invalid")
        if not isinstance(row.get("created_at"), str) or not row.get("created_at"):
            errors.append(f"injection line {line_number} timestamp is invalid")
        for field in ("queue_identity_sha256", "prior_sha256", "injection_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, ""))):
                errors.append(f"injection line {line_number} {field} is invalid")
        expected = _hash_value({key: value for key, value in row.items()
                                if key != "injection_sha256"})
        if row.get("injection_sha256") != expected or row.get("prior_sha256") != prior:
            errors.append(f"injection line {line_number} hash chain is invalid")
        if row.get("sequence") != line_number:
            errors.append(f"injection line {line_number} sequence is invalid")
        if row.get("queue_identity_sha256") != identity_sha:
            errors.append(f"injection line {line_number} targets another queue identity")
        action, payload = row.get("action"), row.get("payload")
        if action not in ALLOWED_ACTIONS or not isinstance(payload, dict):
            errors.append(f"injection line {line_number} action/payload is invalid")
        else:
            expected_payload = {
                "pause": set(), "resume": set(), "drain": set(),
                "reprioritize_pending": {"label"},
                "add_future_candidate": {"label", "hf_id", "model_dir", "spec_sha256"},
            }[action]
            if set(payload) != expected_payload:
                errors.append(f"injection line {line_number} payload keys are not exact")
            for key, value in payload.items():
                if not isinstance(value, str) or not value:
                    errors.append(f"injection line {line_number} payload {key} is invalid")
            if action == "add_future_candidate" and payload.get("spec_sha256") != _hash_value({
                    key: payload.get(key) for key in ("label", "hf_id", "model_dir")}):
                errors.append(f"injection line {line_number} future spec hash is invalid")
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
    identity = _queue_identity()
    INJECTION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(INJECTION_LOCK, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows, errors = _read_injections_unlocked(identity["queue_identity_sha256"])
        if errors:
            raise QueueError("existing injection ledger is invalid: " + "; ".join(errors))
        prior = rows[-1]["injection_sha256"] if rows else "0" * 64
        row = {
            "schema": INJECTION_SCHEMA,
            "sequence": len(rows) + 1,
            "created_at": _now(),
            "action": action,
            "payload": payload,
            "queue_identity_sha256": identity["queue_identity_sha256"],
            "prior_sha256": prior,
        }
        row["injection_sha256"] = _hash_value(row)
        _append_durable(INJECTIONS, row)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return row


def _apply_injections(state: dict[str, Any], identity_sha: str) -> str:
    rows, errors = _read_injections(identity_sha)
    if errors:
        raise QueueError("injection ledger invalid: " + "; ".join(errors))
    control = state.get("control_mode", "run")
    consumed = int(state.get("last_consumed_injection_seq", 0))
    for row in rows:
        if row["sequence"] <= consumed:
            continue
        action, payload = row["action"], row["payload"]
        if action == "pause":
            control = "pause"
        elif action == "resume":
            control = "run"
        elif action == "drain":
            control = "drain"
        elif action == "reprioritize_pending":
            label = payload["label"]
            if state.get("items", {}).get(label, {}).get("status") == "pending":
                order = [item for item in state["pending_order"] if item != label]
                state["pending_order"] = [label, *order]
        elif action == "add_future_candidate":
            if payload not in state["future_candidates"]:
                state["future_candidates"].append({**payload, "status": "held-new-generation-required"})
        consumed = row["sequence"]
        state["last_consumed_injection_seq"] = consumed
        state["control_mode"] = control
    _save_state(state)
    return control


def _wait_boundary_control(state: dict[str, Any], identity_sha: str) -> bool:
    while True:
        control = _apply_injections(state, identity_sha)
        if _STOP or DRAIN_REQUEST.exists() or control == "drain":
            return False
        if control != "pause":
            return True
        _save_state(state, "paused-control")
        time.sleep(POLL_SECONDS)


def _write_index(state: dict[str, Any], identity: dict[str, Any], *,
                 verify_files: bool = False) -> dict[str, Any]:
    reports = []
    for item in COHORT:
        path = SCALE_ROOT / item["label"] / "census.json"
        document = _read_json(path)
        if (isinstance(document, dict)
                and not doctor_v5_census.validate_report(document, verify_files=verify_files)):
            reports.append({"label": item["label"], "path": str(path.relative_to(ROOT)),
                            "report_sha256": document["report_sha256"],
                            "source_manifest_sha256": document["source"]["source_manifest_sha256"]})
    sub_reports = [row for row in reports if row["label"] != "120B"]
    report_120b = next((row for row in reports if row["label"] == "120B"), None)
    payload = {
        "schema": INDEX_SCHEMA,
        "queue_identity_sha256": identity["queue_identity_sha256"],
        "cohort_labels": [item["label"] for item in COHORT],
        "reports": reports,
        "report_count": len(reports),
        "pass_a_complete": len(reports) == len(COHORT),
        "pass_b_launch_permitted": False,
        "report_bundles": {
            "sub_120B_aggregate": {
                "labels": [item["label"] for item in COHORT if item["label"] != "120B"],
                "reports": sub_reports,
                "complete": len(sub_reports) == len(COHORT) - 1,
                "bundle_sha256": _hash_value(sub_reports),
                "quality_result": False,
            },
            "120B": {
                "report": report_120b,
                "complete": report_120b is not None,
                "quality_result": False,
            },
        },
        "all_at_once_source_cleanup": {
            "operator_requested": True,
            "eligible": False,
            "reason": "both bundles are census-only; verified Doctor-v5 quality results are required",
        },
        "generated_at": _now(),
    }
    payload["index_sha256"] = _hash_value(payload)
    _atomic_json(INDEX_FILE, payload)
    state["index_sha256"] = payload["index_sha256"]
    return payload


def _acquire_lease(path: Path) -> Any | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = open(path, "a+")
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


def _terminate_child(process: subprocess.Popen[Any], reason: str) -> None:
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


def _run_child(state: dict[str, Any], item: dict[str, Any], command: list[str], *,
               role: str, heavy: bool, remaining_download: bool = False) -> int:
    lease = None
    lease_path = HEAVY_LOCK if heavy else DOWNLOAD_LOCK
    while lease is None:
        control = _apply_injections(state, state["queue_identity_sha256"])
        if _STOP or DRAIN_REQUEST.exists() or control == "drain":
            return 130
        if control == "pause":
            return PAUSE_RC
        lease = _acquire_lease(lease_path)
        if lease is None:
            _save_state(state, "waiting-heavy-lease", active_label=item["label"])
            time.sleep(POLL_SECONDS)
    try:
        if role == "download" and _verified_marker(item):
            # An orphaned predecessor may have completed while this supervisor
            # waited on its inherited singleton lease.
            return 0
        remaining = _remaining_120b_gb() if remaining_download else 0.0
        gate = _resource_gate(remaining_download_gb=remaining)
        if not gate["ok"]:
            state["last_resource_gate"] = gate
            _save_state(state, "waiting-resources", active_label=item["label"])
            raise ResourcePause("; ".join(gate["blockers"]))
        log_path = SCALE_ROOT / item["label"] / f"{role}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        if role == "download":
            # hf_xet currently returns fatal HTTP 416 for this reconstruction;
            # use the resumable LFS/hf_transfer path for the pinned source.
            child_env["HF_HUB_DISABLE_XET"] = "1"
            child_env["HF_MAX_WORKERS"] = "4"
        with open(log_path, "ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=child_env,
                                       pass_fds=(lease.fileno(),))
            state["active_child"] = {"label": item["label"], "role": role,
                                     "pid": process.pid, "pgid": process.pid,
                                     "command_sha256": _hash_value(command),
                                     "started_at": _now(), "log": str(log_path.relative_to(ROOT))}
            _save_state(state, f"running-{role}", active_label=item["label"])
            try:
                while process.poll() is None:
                    time.sleep(POLL_SECONDS)
                    control = _apply_injections(state, state["queue_identity_sha256"])
                    if _STOP or DRAIN_REQUEST.exists() or control == "drain":
                        _terminate_child(process, "drain")
                        return 130
                    if control == "pause":
                        _terminate_child(process, "pause")
                        return PAUSE_RC
                    remaining = _remaining_120b_gb() if remaining_download else 0.0
                    gate = _resource_gate(remaining_download_gb=remaining)
                    state["last_resource_gate"] = gate
                    _save_state(state)
                    if not gate["ok"]:
                        _terminate_child(process, "resource gate")
                        raise ResourcePause("; ".join(gate["blockers"]))
                return int(process.returncode or 0)
            except BaseException:
                if process.poll() is None:
                    _terminate_child(process, "supervisor exception")
                raise
    finally:
        state["active_child"] = None
        _save_state(state)
        _release_lease(lease)


def _verified_marker(item: dict[str, Any]) -> bool:
    marker_name = item.get("download_marker")
    if not marker_name:
        return True
    document = _read_json(ROOT / marker_name, {})
    marker_kwargs: dict[str, Any] = {}
    if item["label"] == "120B":
        marker_kwargs = {
            "revision": _remote_120b_manifest()["revision"],
            "include_patterns": ("original/*",),
        }
    marker_ok = procure._verified_marker_valid(
        document, label=item["label"], hf_id=item["hf_id"],
        local_dir=item["model_dir"], require_verify=True, **marker_kwargs,
    )
    if marker_ok and item["label"] == "120B":
        return not _verify_local_120b_manifest()
    return marker_ok


def _external_120b_download_alive() -> bool:
    record = _read_json(ROOT / "reports" / "condense" / "download_state" / "120B.pid.json", {})
    if not isinstance(record, dict) or record.get("schema") != "hawking.frontier_download_pid.v1":
        return False
    observed = _ps_identity(record.get("pid"))
    if observed is None:
        return False
    start, command = observed
    revision = _remote_120b_manifest()["revision"]
    recorded_start = record.get("process_start")
    return (record.get("label") == "120B"
            and (not recorded_start or recorded_start == start)
            and "procure.py 120B" in command
            and "scratch/staging/gpt-oss-120b.partial" in command
            and "--verify" in command
            and f"--revision {revision}" in command)


def _report_complete(item: dict[str, Any]) -> bool:
    report = _read_json(SCALE_ROOT / item["label"] / "census.json")
    return (isinstance(report, dict) and report.get("label") == item["label"]
            and report.get("hf_id") == item["hf_id"]
            and isinstance(report.get("producer"), dict)
            and report["producer"].get("sha256") == _sha256_file(CENSUS_SCRIPT)
            and Path(str(report.get("model_dir", ""))).resolve() == (ROOT / item["model_dir"]).resolve()
            and not doctor_v5_census.validate_report(report, verify_files=True))


def _census_command(item: dict[str, Any]) -> list[str]:
    directory = SCALE_ROOT / item["label"]
    command = [sys.executable, str(CENSUS_SCRIPT), "run", "--label", item["label"],
               "--hf-id", item["hf_id"], "--model-dir", item["model_dir"],
               "--output", str(directory / "census.json"),
               "--checkpoint", str(directory / "census.checkpoint.json")]
    if item.get("download_marker"):
        command += ["--expected-download-marker", item["download_marker"]]
    return command


def _download_command(item: dict[str, Any]) -> list[str]:
    return [sys.executable, str(ROOT / "tools" / "condense" / "procure.py"), "120B",
            "--dir", item["model_dir"], "--verify", "--retries", "4",
            "--progress-interval-s", "60", "--stall-timeout-s", "1800",
            "--disk-free-floor-gb", f"{DOWNLOAD_FLOOR_GB:.1f}",
            "--revision", _remote_120b_manifest()["revision"]]


def _ensure_concurrent_120b_download(state: dict[str, Any]) -> bool:
    """Start the pinned 120B transfer early without serializing census work."""
    item = next(row for row in COHORT if row["label"] == "120B")
    if _verified_marker(item) or _external_120b_download_alive():
        return True
    lease = _acquire_lease(DOWNLOAD_LOCK)
    if lease is None:
        # An inherited lease can outlive a killed queue supervisor. Waiting for
        # it prevents a replacement queue from duplicating the orphaned child.
        return False
    child_inherited_lease = False
    try:
        if _verified_marker(item) or _external_120b_download_alive():
            return True
        remaining = _remaining_120b_gb()
        gate = _resource_gate(remaining_download_gb=remaining)
        state["last_resource_gate"] = gate
        _save_state(state)
        if not gate["ok"]:
            return False
        command = _download_command(item)
        if shutil.which("caffeinate"):
            command = ["caffeinate", "-dimsu", *command]
        child_env = os.environ.copy()
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["HF_MAX_WORKERS"] = "4"
        log_path = ROOT / "reports" / "condense" / "download_120B.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab", buffering=0) as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=child_env, pass_fds=(lease.fileno(),),
            )
        child_inherited_lease = True
        observed = _ps_identity(process.pid)
        if observed is None:
            _terminate_child(process, "concurrent download ownership handshake")
            raise QueueError("concurrent 120B download ownership handshake failed")
        process_start, observed_command = observed
        required_revision = _remote_120b_manifest()["revision"]
        if ("procure.py 120B" not in observed_command
                or f"--revision {required_revision}" not in observed_command):
            _terminate_child(process, "concurrent download identity mismatch")
            raise QueueError("concurrent 120B download command identity mismatch")
        _atomic_json(
            ROOT / "reports" / "condense" / "download_state" / "120B.pid.json",
            {
                "schema": "hawking.frontier_download_pid.v1",
                "label": "120B",
                "pid": process.pid,
                "process_start": process_start,
                "started_at": _now(),
                "log_path": str(log_path),
                "cmd": command,
                "queue_identity_sha256": state["queue_identity_sha256"],
            },
        )
        return True
    finally:
        # The child inherited this descriptor. Closing the parent's duplicate
        # leaves the lease held until the detached procurement supervisor exits.
        if child_inherited_lease:
            lease.close()
        else:
            _release_lease(lease)


def _run_item(state: dict[str, Any], identity: dict[str, Any], item: dict[str, Any]) -> bool:
    label = item["label"]
    if state["items"][label].get("started_at") is None:
        state["items"][label]["started_at"] = _now()
    state["items"][label]["status"] = "running"
    _save_state(state)
    if _report_complete(item):
        state["items"][label]["status"] = "complete"
        _write_index(state, identity)
        _save_state(state)
        return True
    if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
        return False
    if label == "120B" and not _verified_marker(item):
        while _external_120b_download_alive() and not _verified_marker(item):
            if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
                return False
            _save_state(state, "waiting-concurrent-120b-download", active_label=label,
                        last_resource_gate=_resource_gate(
                            remaining_download_gb=_remaining_120b_gb()))
            time.sleep(POLL_SECONDS)
        if not _verified_marker(item):
            while True:
                try:
                    state["items"][label]["download_attempts"] = \
                        int(state["items"][label].get("download_attempts", 0)) + 1
                    rc = _run_child(state, item, _download_command(item), role="download",
                                    heavy=False, remaining_download=True)
                    if rc == 130:
                        return False
                    if rc == PAUSE_RC:
                        if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
                            return False
                        continue
                    if rc != 0:
                        raise QueueError(f"120B download failed rc={rc}")
                    if not _verified_marker(item):
                        raise QueueError("120B download exited successfully without a valid marker")
                    break
                except ResourcePause:
                    time.sleep(30)
                    if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
                        return False
    # An independently running download is allowed to continue during a pause,
    # but the fixed ladder never crosses the download-to-census boundary until
    # an explicit resume has been consumed.
    if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
        return False
    if not (ROOT / item["model_dir"]).is_dir():
        raise QueueError(f"source directory is absent for {label}: {item['model_dir']}")
    if not _verified_marker(item):
        raise QueueError(f"verified source marker is absent/invalid for {label}")
    while True:
        try:
            state["items"][label]["attempts"] = int(state["items"][label].get("attempts", 0)) + 1
            rc = _run_child(state, item, _census_command(item), role="census", heavy=True)
            if rc == 130:
                return False
            if rc == PAUSE_RC:
                if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
                    return False
                continue
            if rc != 0:
                raise QueueError(f"census failed for {label} rc={rc}")
            break
        except ResourcePause:
            time.sleep(30)
            if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
                return False
    if not _report_complete(item):
        raise QueueError(f"census report failed validation for {label}")
    state["items"][label]["status"] = "complete"
    state["items"][label]["completed_at"] = _now()
    try:
        started = _dt.datetime.fromisoformat(state["items"][label]["started_at"])
        ended = _dt.datetime.fromisoformat(state["items"][label]["completed_at"])
        state["items"][label]["duration_seconds"] = round((ended - started).total_seconds(), 3)
    except Exception:
        state["items"][label]["duration_seconds"] = None
    _write_index(state, identity)
    _save_state(state)
    return True


def run_queue(nonce: str) -> int:
    identity = _queue_identity()
    SCALE_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[doctor-v5-queue] another owner holds the singleton lock", file=sys.stderr)
        return 2
    observed = _ps_identity(os.getpid())
    if observed is None:
        return 2
    start, command = observed
    pid_record = {"schema": "hawking.doctor_v5_scale_queue_pid.v1", "pid": os.getpid(),
                  "process_start": start, "command": command, "ownership_nonce": nonce,
                  "queue_identity_sha256": identity["queue_identity_sha256"],
                  "started_at": _now(), "log": str(LOG_FILE)}
    _atomic_json(PID_FILE, pid_record)
    state = _load_state(identity)
    errors = _preflight(identity)
    if errors:
        _save_state(state, "blocked-preflight", preflight_errors=errors)
        return 3
    _save_state(state, "running-pass-a", supervisor_pid=os.getpid(),
                preflight_errors=[], queue_identity=identity)
    try:
        while state.get("pending_order"):
            if not _wait_boundary_control(state, identity["queue_identity_sha256"]):
                _save_state(state, "drained", active_label=None, drained_at=_now())
                return 130
            _ensure_concurrent_120b_download(state)
            label = state["pending_order"][0]
            item = next((row for row in COHORT if row["label"] == label), None)
            if item is None:
                raise QueueError(f"pending order contains unknown label {label}")
            if not _run_item(state, identity, item):
                _save_state(state, "drained", active_label=None, drained_at=_now())
                return 130
            # Injections may have reprioritized another pending item while this
            # child was active. Remove the completed label by value so the new
            # front of the queue is never accidentally discarded.
            state["pending_order"] = [pending for pending in state["pending_order"]
                                      if pending != label]
            _save_state(state, "running-pass-a", active_label=None)
        index = _write_index(state, identity, verify_files=True)
        if not index["pass_a_complete"]:
            raise QueueError("pass A ended without all seven reports")
        state["active_pass"] = "pass_b_treatment"
        _save_state(state, "waiting-treatment-adapters", active_label=None,
                    pass_a_completed_at=_now())
        return 0
    except Exception as exc:
        _save_state(state, "blocked", active_label=None,
                    error=f"{type(exc).__name__}: {exc}", blocked_at=_now())
        print(f"[doctor-v5-queue] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    finally:
        record = _read_json(PID_FILE, {})
        if isinstance(record, dict) and record.get("ownership_nonce") == nonce:
            try:
                PID_FILE.unlink()
                _fsync_dir(PID_FILE.parent)
            except FileNotFoundError:
                pass
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def start_queue() -> int:
    if DRAIN_REQUEST.exists():
        print(f"[doctor-v5-queue] drain active; use resume: {DRAIN_REQUEST}", file=sys.stderr)
        return 130
    old = _read_json(PID_FILE, {})
    if _owner_alive(old):
        print(f"[doctor-v5-queue] already active pid={old['pid']}")
        return 0
    identity = _queue_identity()
    state = _read_json(STATE)
    if isinstance(state, dict) and state.get("queue_identity_sha256") != identity["queue_identity_sha256"]:
        print("[doctor-v5-queue] existing state belongs to another queue identity", file=sys.stderr)
        return 3
    nonce = os.urandom(16).hex()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(QUEUE_SCRIPT), "run", "--nonce", nonce]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    with open(LOG_FILE, "ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=os.environ.copy())
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = _read_json(PID_FILE, {})
        if record.get("ownership_nonce") == nonce and _owner_alive(record):
            print(f"[doctor-v5-queue] detached pid={record['pid']} log={LOG_FILE}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.2)
    # The parent owns this newly created process group.  A failed nonce handshake
    # must not leave unacknowledged work behind.
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    record = _read_json(PID_FILE, {})
    if isinstance(record, dict) and record.get("ownership_nonce") == nonce:
        try:
            PID_FILE.unlink()
            _fsync_dir(PID_FILE.parent)
        except FileNotFoundError:
            pass
    print("[doctor-v5-queue] detached ownership handshake failed", file=sys.stderr)
    return 2


def _timing_snapshot(state: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"census": {}, "download_120B": {}}
    if isinstance(state, dict) and isinstance(state.get("items"), dict):
        for item in COHORT:
            label = item["label"]
            row = state["items"].get(label, {})
            checkpoint = _read_json(SCALE_ROOT / label / "census.checkpoint.json", {})
            completed_bytes = 0
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("shards"), dict):
                for shard in checkpoint["shards"].values():
                    if isinstance(shard, dict) and isinstance(shard.get("bytes"), int):
                        completed_bytes += shard["bytes"]
            total_bytes = sum(
                path.stat().st_size for path in (ROOT / item["model_dir"]).rglob("*.safetensors")
                if path.is_file() and not path.is_symlink()
            ) if (ROOT / item["model_dir"]).is_dir() else None
            elapsed = None
            try:
                elapsed = max(0.0, (_dt.datetime.now(_dt.timezone.utc)
                                    - _dt.datetime.fromisoformat(row["started_at"])).total_seconds())
            except Exception:
                pass
            rate = completed_bytes / elapsed if elapsed and completed_bytes else None
            eta = ((total_bytes - completed_bytes) / rate
                   if rate and total_bytes is not None and total_bytes >= completed_bytes else None)
            result["census"][label] = {
                "status": row.get("status"), "completed_bytes": completed_bytes,
                "total_weight_file_bytes": total_bytes,
                "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
                "observed_bytes_per_second": round(rate, 3) if rate else None,
                "eta_seconds": round(eta, 3) if eta is not None else None,
                "eta_basis": "completed-shard hashing; unavailable until first shard boundary",
            }
    download = _read_json(ROOT / "reports" / "condense" / "download_state" / "120B.state.json", {})
    progress = download.get("progress") if isinstance(download, dict) else None
    remaining_gb = None
    estimated_remaining_gb = None
    eta = None
    rate_mbs = None
    try:
        remaining_gb = _remaining_120b_gb()
        rate_mbs = progress.get("window_mb_s") if isinstance(progress, dict) else None
        tracked_local_gb = progress.get("local_dir_gb") if isinstance(progress, dict) else None
        if (isinstance(tracked_local_gb, (int, float))
                and not isinstance(tracked_local_gb, bool) and math.isfinite(tracked_local_gb)):
            estimated_remaining_gb = max(
                _remote_120b_manifest()["total_bytes"] / 1e9 - tracked_local_gb, 0.0
            )
        if isinstance(rate_mbs, (int, float)) and not isinstance(rate_mbs, bool) \
                and math.isfinite(rate_mbs) and rate_mbs > 0:
            eta = (estimated_remaining_gb if estimated_remaining_gb is not None
                   else remaining_gb) * 1000 / rate_mbs
    except Exception:
        rate_mbs = None
    result["download_120B"] = {
        "active": _external_120b_download_alive(),
        "status": download.get("status") if isinstance(download, dict) else None,
        "attempt": download.get("attempt") if isinstance(download, dict) else None,
        "remaining_gb_conservative_completed_files_basis": round(remaining_gb, 6)
        if remaining_gb is not None else None,
        "estimated_remaining_gb_progress_basis": round(estimated_remaining_gb, 6)
        if estimated_remaining_gb is not None else None,
        "window_mb_s": rate_mbs,
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "updated_at": download.get("updated_at") if isinstance(download, dict) else None,
    }
    return result


def status() -> int:
    identity = _queue_identity()
    state_error = None
    try:
        state = _load_state(identity)
    except Exception as exc:
        state, state_error = _read_json(STATE, {}), f"{type(exc).__name__}: {exc}"
    record = _read_json(PID_FILE, {})
    rows, injection_errors = _read_injections(identity["queue_identity_sha256"])
    payload = {
        "schema": "hawking.doctor_v5_scale_queue_status.v1",
        "generated_at": _now(),
        "active": _owner_alive(record),
        "pid": record.get("pid") if isinstance(record, dict) else None,
        "queue_identity_sha256": identity["queue_identity_sha256"],
        "state": state,
        "state_error": state_error,
        "drain_requested": DRAIN_REQUEST.exists(),
        "injection_count": len(rows),
        "injection_errors": injection_errors,
        "resources": _resource_gate(),
        "timing": _timing_snapshot(state),
        "reboot_autostart": False,
        "restart_command": "python3.12 tools/condense/doctor_v5_queue.py resume",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if state_error is None and not injection_errors else 1


def drain() -> int:
    _atomic_json(DRAIN_REQUEST, {"schema": "hawking.doctor_v5_scale_drain.v1",
                                 "requested_at": _now()})
    download_record = _read_json(
        ROOT / "reports" / "condense" / "download_state" / "120B.pid.json", {}
    )
    try:
        owned_download = (
            isinstance(download_record, dict)
            and download_record.get("queue_identity_sha256")
            == _queue_identity()["queue_identity_sha256"]
            and _external_120b_download_alive()
        )
    except Exception:
        owned_download = False
    if owned_download:
        download_pid = int(download_record["pid"])
        try:
            # Signal only the procurement supervisor. It checkpoints the signal,
            # terminates its HF child, and disables retry for an interrupted run.
            os.kill(download_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 45
        while _ps_identity(download_pid) is not None and time.monotonic() < deadline:
            time.sleep(0.25)
        if _ps_identity(download_pid) is not None:
            try:
                os.killpg(download_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    record = _read_json(PID_FILE, {})
    if _owner_alive(record):
        os.kill(int(record["pid"]), signal.SIGTERM)
    print(f"[doctor-v5-queue] drain requested: {DRAIN_REQUEST}")
    return 0


def resume() -> int:
    # A durable resume row supersedes a drain row even when SIGTERM reached the
    # supervisor before it could consume that earlier ledger entry.
    _append_injection("resume", {})
    if DRAIN_REQUEST.exists():
        DRAIN_REQUEST.unlink()
        _fsync_dir(DRAIN_REQUEST.parent)
    return start_queue()


def inject(args: argparse.Namespace) -> int:
    action = args.action
    if action in {"pause", "resume", "drain"}:
        payload: dict[str, Any] = {}
    elif action == "reprioritize_pending":
        if args.label not in {item["label"] for item in COHORT}:
            raise QueueError("reprioritize label is not in the fixed cohort")
        payload = {"label": args.label}
    elif action == "add_future_candidate":
        spec = {"label": args.label, "hf_id": args.hf_id, "model_dir": args.model_dir}
        if not all(isinstance(value, str) and value for value in spec.values()):
            raise QueueError("future candidate fields must be nonempty")
        payload = {**spec, "spec_sha256": _hash_value(spec)}
    else:
        raise QueueError("unsupported injection action")
    row = _append_injection(action, payload)
    if action == "drain":
        drain()
    elif action == "resume" and DRAIN_REQUEST.exists():
        DRAIN_REQUEST.unlink()
        _fsync_dir(DRAIN_REQUEST.parent)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def plan() -> int:
    identity = _queue_identity()
    print(json.dumps({"queue_identity": identity, "state": _base_state(identity),
                      "resource_gate": _resource_gate(),
                      "note": "plan is bootstrap-only; Pass B remains unlaunchable"},
                     indent=2, sort_keys=True))
    return 0


def selftest() -> int:
    global SCALE_ROOT, STATE, PID_FILE, LOCK_FILE, INJECTION_LOCK, INJECTIONS, DRAIN_REQUEST
    global LOG_FILE, INDEX_FILE
    with tempfile.TemporaryDirectory() as raw:
        old = (SCALE_ROOT, STATE, PID_FILE, LOCK_FILE, INJECTION_LOCK, INJECTIONS,
               DRAIN_REQUEST, LOG_FILE, INDEX_FILE)
        SCALE_ROOT = Path(raw)
        STATE, PID_FILE = SCALE_ROOT / "state.json", SCALE_ROOT / "pid.json"
        LOCK_FILE, INJECTION_LOCK = SCALE_ROOT / "queue.lock", SCALE_ROOT / "inject.lock"
        INJECTIONS, DRAIN_REQUEST = SCALE_ROOT / "inject.jsonl", SCALE_ROOT / "drain.json"
        LOG_FILE, INDEX_FILE = SCALE_ROOT / "queue.log", SCALE_ROOT / "index.json"
        identity = _queue_identity()
        state = _base_state(identity)
        _save_state(state)
        first = _append_injection("pause", {})
        assert _apply_injections(state, identity["queue_identity_sha256"]) == "pause"
        assert _apply_injections(state, identity["queue_identity_sha256"]) == "pause"
        second = _append_injection("resume", {})
        assert _apply_injections(state, identity["queue_identity_sha256"]) == "run"
        assert first["sequence"] == 1 and second["prior_sha256"] == first["injection_sha256"]
        _append_injection("reprioritize_pending", {"label": "1.5B"})
        assert _apply_injections(state, identity["queue_identity_sha256"]) == "run"
        assert state["pending_order"][0] == "1.5B"
        pending_after_current = [label for label in state["pending_order"] if label != "0.5B"]
        assert pending_after_current[0] == "1.5B" and "0.5B" not in pending_after_current
        rows, errors = _read_injections(identity["queue_identity_sha256"])
        assert len(rows) == 3 and not errors
        lease_path = SCALE_ROOT / "inherited.lock"
        lease = _acquire_lease(lease_path)
        assert lease is not None
        sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.5)"],
            pass_fds=(lease.fileno(),),
        )
        lease.close()
        assert _acquire_lease(lease_path) is None
        sleeper.wait(timeout=3)
        reacquired = _acquire_lease(lease_path)
        assert reacquired is not None
        _release_lease(reacquired)
        lines = INJECTIONS.read_text().splitlines()
        damaged = json.loads(lines[0])
        damaged["schema"] = "evil.schema"
        damaged["injection_sha256"] = _hash_value({key: value for key, value in damaged.items()
                                                    if key != "injection_sha256"})
        lines[0] = json.dumps(damaged)
        INJECTIONS.write_text("\n".join(lines) + "\n")
        assert _read_injections(identity["queue_identity_sha256"])[1]
        state_doc = _read_json(STATE)
        state_doc["pass_b"]["launch_permitted"] = True
        state_doc["state_sha256"] = _hash_value({key: value for key, value in state_doc.items()
                                                  if key != "state_sha256"})
        _atomic_json(STATE, state_doc)
        try:
            _load_state(identity)
        except QueueError:
            pass
        else:
            raise AssertionError("semantic state tamper was accepted")
        manifest_gb = _remote_120b_manifest()["total_bytes"] / 1e9
        gate = _resource_gate(remaining_download_gb=manifest_gb)
        assert gate["required_free_gb"] == round(DOWNLOAD_FLOOR_GB + manifest_gb
                                                  + CONTROL_OVERHEAD_GB, 6)
        (SCALE_ROOT, STATE, PID_FILE, LOCK_FILE, INJECTION_LOCK, INJECTIONS,
         DRAIN_REQUEST, LOG_FILE, INDEX_FILE) = old
    print("doctor_v5_queue.py selftest OK")
    return 0


def _request_stop(_sig: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("start")
    run = commands.add_parser("run")
    run.add_argument("--nonce", required=True)
    commands.add_parser("status")
    commands.add_parser("ping")
    commands.add_parser("drain")
    commands.add_parser("resume")
    injection = commands.add_parser("inject")
    injection.add_argument("action", choices=sorted(ALLOWED_ACTIONS))
    injection.add_argument("--label")
    injection.add_argument("--hf-id")
    injection.add_argument("--model-dir")
    commands.add_parser("selftest")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        if args.command == "plan":
            return plan()
        if args.command == "start":
            return start_queue()
        if args.command == "run":
            return run_queue(args.nonce)
        if args.command in {"status", "ping"}:
            return status()
        if args.command == "drain":
            return drain()
        if args.command == "resume":
            return resume()
        if args.command == "inject":
            return inject(args)
        if args.command == "selftest":
            return selftest()
    except Exception as exc:
        print(f"[doctor-v5-queue] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
