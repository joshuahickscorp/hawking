#!/usr/bin/env python3.12
"""Durable, detached, pressure-aware Studio download queue.

The unattended ladder extension is deliberately finite: reconcile the existing
32B download, then fetch and verify 72B and the official 120B MXFP4 checkpoint,
then expose two architecture-gated frontier installs. 120B is not admitted until
verified 14B quantization processing completes. DeepSeek V4 Flash (284B) is the
smaller architecture bring-up target; Kimi K2.6 (1.1T repository / 1T declared
model, native INT4) is the largest parameter-count source that can still fit the
1 TB Studio lifecycle after guarded source release. Each requires an explicit
architecture-ready receipt and live disk admission; absent either, it remains
visibly held. All downloads remain in isolated staging directories. The queue
shares Studio's drain request, runs one download at a time, and never deletes
source models.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

import procure
import processing_queue
from ram_scheduler import resource_snapshot, thermal_output_ok
from studio_manifest import DEFAULT_HARDWARE

STATE = ROOT / "reports/condense/download_queue_state.json"
PID_FILE = ROOT / "reports/condense/download_queue.pid.json"
LOCK_FILE = ROOT / "reports/condense/download_queue.lock"
LOG_FILE = ROOT / "reports/condense/download_queue.log"
SHARED_DRAIN = ROOT / "reports/cron/studio_drain.request"
STUDIO_RUN_PID = ROOT / "reports/cron/studio_run.pid"
STUDIO_WAIT_PID = ROOT / "reports/cron/studio_wait.pid"
DOWNLOAD_STATE = ROOT / "reports/condense/download_state"
V4_ARCHITECTURE_GATE = ROOT / "reports/condense/DeepSeek-V4-Flash.architecture_ready.json"
KIMI_ARCHITECTURE_GATE = ROOT / "reports/condense/Kimi-K2.6.architecture_ready.json"
POLL_S = float(os.environ.get("HAWKING_DOWNLOAD_QUEUE_POLL_S", "30"))
MAX_ATTEMPTS = int(os.environ.get("HAWKING_DOWNLOAD_QUEUE_MAX_ATTEMPTS", "12"))
STUDIO_MAX_RUNNABLE_PEAK_GB = 65.0
DOWNLOAD_OVERLAP_RESERVATION_GB = 10.0
DOWNLOAD_OVERLAP_MARGIN_GB = 2.0
OVERLAP_PAUSE_RC = 76

PLAN = (
    {"label": "32B", "local_dir": "scratch/staging/qwen-32b.partial"},
    {"label": "72B", "local_dir": "scratch/staging/qwen-72b.partial"},
    {"label": "120B", "local_dir": "scratch/staging/gpt-oss-120b.partial"},
    {
        "label": "DeepSeek-V4-Flash",
        "local_dir": "scratch/staging/deepseek-v4-flash-dspark.partial",
        "architecture_gate": str(V4_ARCHITECTURE_GATE.relative_to(ROOT)),
    },
    {
        "label": "Kimi-K2.6",
        "local_dir": "scratch/staging/kimi-k2.6.partial",
        "terminal": True,
        "architecture_gate": str(KIMI_ARCHITECTURE_GATE.relative_to(ROOT)),
    },
)

_stop_requested = False


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _base_state():
    return {
        "schema": "hawking.download_queue.v1",
        "created_at": _now(),
        "updated_at": _now(),
        "status": "new",
        "plan": [dict(item) for item in PLAN],
        "items": {item["label"]: {"status": "pending", "attempts": 0} for item in PLAN},
        "source_release": {
            "status": "blocked",
            "automatic_deletion": False,
            "reason": "durable artifact inventory and artifact-bound verification are required",
        },
    }


def _load_state():
    state = _read_json(STATE, _base_state())
    if state.get("schema") != "hawking.download_queue.v1":
        state = _base_state()
    state.setdefault("items", {})
    for item in PLAN:
        state["items"].setdefault(item["label"], {"status": "pending", "attempts": 0})
    state["plan"] = [dict(item) for item in PLAN]
    state["source_release"] = _base_state()["source_release"]
    return state


def _update_state(status=None, active_label=None, **extra):
    state = _load_state()
    if status is not None:
        state["status"] = status
    if active_label is not None:
        state["active_label"] = active_label
    state.update(extra)
    state["updated_at"] = _now()
    _atomic_json(STATE, state)
    return state


def _update_item(label, **updates):
    state = _load_state()
    row = dict(state["items"].get(label, {}))
    row.update(updates)
    row["updated_at"] = _now()
    state["items"][label] = row
    state["active_label"] = label
    state["updated_at"] = row["updated_at"]
    _atomic_json(STATE, state)
    return row


def _thermal_snapshot():
    try:
        r = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                           timeout=5, check=False)
        text = (r.stdout + r.stderr).strip()
        return {"ok": thermal_output_ok(r.returncode, text),
                "returncode": r.returncode, "detail": text[-1000:]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _evaluate_safety(snapshot, thermal, *, disk_free_gb, remaining_gb):
    blockers = []
    if not snapshot.get("ok"):
        blockers.append(f"resource snapshot unavailable: {snapshot.get('error', 'unknown')}")
    pressure = snapshot.get("pressure_level")
    raw_swap = snapshot.get("swap_used_mb")
    if pressure != 1:
        blockers.append(f"memory pressure is not normal (level={pressure!r})")
    if isinstance(raw_swap, bool) or not isinstance(raw_swap, (int, float)) \
            or not math.isfinite(float(raw_swap)):
        blockers.append("swap measurement unavailable")
    elif float(raw_swap) >= 2048.0:
        blockers.append(f"swap {float(raw_swap):.0f}MB >= 2048MB")
    if "AC Power" not in str(snapshot.get("power_source", "")):
        blockers.append("AC power not confirmed")
    if not thermal.get("ok"):
        blockers.append("thermal/performance state is not green")
    # This daemon is download-only and never admits the model to processing. Keep the hard untouched
    # floor plus a transient HF-cache margin; processing applies its separate scratch admission later.
    operational_reserve = DEFAULT_HARDWARE.cache_reserve_gb
    required = remaining_gb + DEFAULT_HARDWARE.disk_reserve_gb + operational_reserve
    if disk_free_gb < required:
        blockers.append(
            f"disk {disk_free_gb:.1f}GB < remaining {remaining_gb:.1f}GB + "
            f"{DEFAULT_HARDWARE.disk_reserve_gb:.0f}GB reserve + "
            f"{operational_reserve:.0f}GB operational scratch/cache"
        )
    return {
        "ok": not blockers,
        "blockers": blockers,
        "required_free_gb": round(required, 3),
        "disk_free_gb": round(disk_free_gb, 3),
        "remaining_gb": round(remaining_gb, 3),
        "resources": snapshot,
        "thermal": thermal,
    }


def _studio_activity():
    """Fail closed on a live Studio PID whose mode cannot be identified."""
    last_stale = {"pid": None, "mode": None}
    for activity_path in (STUDIO_RUN_PID, STUDIO_WAIT_PID):
        if not activity_path.exists():
            continue
        mode = None
        try:
            info = json.loads(activity_path.read_text())
            if not isinstance(info, dict):
                raise ValueError("Studio PID record must be a JSON object")
            pid = int(info.get("pid"))
            if pid <= 0:
                raise ValueError("Studio PID must be positive")
        except Exception as exc:
            return {"ok": False, "active": None, "pid": None, "mode": None,
                    "error": f"{activity_path}: {type(exc).__name__}: {exc}"}
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            last_stale = {"pid": pid, "mode": mode}
            continue
        except OSError as exc:
            return {"ok": False, "active": None, "pid": pid, "mode": mode,
                    "error": f"{activity_path}: {type(exc).__name__}: {exc}"}
        try:
            mode = info.get("mode")
            if activity_path == STUDIO_WAIT_PID and (
                info.get("schema") != "hawking.studio_wait_pid.v1"
                or mode != "waiting-admission"
            ):
                raise ValueError("Studio waiter PID schema/mode mismatch")
            if mode is None and info.get("role") == "processing-queue":
                mode = "processing-queue"
            if mode not in {"running", "waiting-admission", "processing-queue"}:
                raise ValueError(f"unrecognized live Studio mode {mode!r}")
        except Exception as exc:
            return {"ok": False, "active": None, "pid": pid, "mode": None,
                    "error": f"{activity_path}: {type(exc).__name__}: {exc}"}
        return {"ok": True, "active": True, "pid": pid, "mode": mode,
                "source": str(activity_path), "error": None}
    return {"ok": True, "active": False, **last_stale, "error": None}


def _download_tree_rss():
    """Measure the complete queue/procure/HF process groups; never guess zero on probe failure."""
    state = _load_state()
    groups = {os.getpgrp()}
    child_pid = state.get("child_pid")
    child_pgid = state.get("child_pgid")
    child_live = _pid_alive(child_pid)
    if child_live:
        try:
            groups.add(int(child_pgid))
        except (TypeError, ValueError):
            return {"ok": False, "rss_gib": None, "rows": [],
                    "error": "live download child has no valid process-group identity"}
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,pgid=,rss=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except Exception as exc:
        return {"ok": False, "rss_gib": None, "rows": [],
                "error": f"{type(exc).__name__}: {exc}"}
    rows = []
    child_observed = not child_live
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid, pgid, rss_kib = int(fields[0]), int(fields[1]), int(fields[2])
        except ValueError:
            continue
        if pgid not in groups:
            continue
        child_observed = child_observed or pid == int(child_pid)
        rows.append({
            "pid": pid, "pgid": pgid,
            "rss_gib": round(rss_kib / (1024 * 1024), 6),
            "command": fields[3][:300],
        })
    if not child_observed:
        return {"ok": False, "rss_gib": None, "rows": rows,
                "error": "live download child was absent from the RSS inventory"}
    return {
        "ok": True,
        "rss_gib": round(sum(row["rss_gib"] for row in rows), 6),
        "rows": rows,
        "error": None,
    }


def _overlap_safety(studio, download_rss):
    blockers = []
    if not studio.get("ok"):
        blockers.append(f"Studio activity probe unavailable: {studio.get('error', 'unknown')}")
    if not download_rss.get("ok"):
        blockers.append(
            f"download-tree RSS probe unavailable: {download_rss.get('error', 'unknown')}"
        )
    forecast = None
    charged_download_gb = None
    if studio.get("ok") and studio.get("active") and download_rss.get("ok"):
        charged_download_gb = max(
            float(download_rss.get("rss_gib", 0.0)), DOWNLOAD_OVERLAP_RESERVATION_GB
        )
        forecast = (STUDIO_MAX_RUNNABLE_PEAK_GB + charged_download_gb
                    + DOWNLOAD_OVERLAP_MARGIN_GB)
        process_budget = float(getattr(
            DEFAULT_HARDWARE, "process_budget_gb", DEFAULT_HARDWARE.weight_budget_gb
        ))
        if forecast > process_budget:
            blockers.append(
                f"Studio/download overlap forecast {forecast:.1f}GB exceeds "
                f"{process_budget:.1f}GB process budget"
            )
    return {
        "ok": not blockers,
        "blockers": blockers,
        "studio": studio,
        "download_tree": download_rss,
        "studio_peak_gb": STUDIO_MAX_RUNNABLE_PEAK_GB,
        "download_charged_gb": charged_download_gb,
        "margin_gb": DOWNLOAD_OVERLAP_MARGIN_GB,
        "forecast_gb": forecast,
    }


def _safety(item):
    spec = procure._resolve(item["label"])
    local_size = procure._path_size_gb(item["local_dir"])
    remaining = max(float(spec.download_gb) - local_size, 0.0)
    disk_free = procure._disk_free_gb(item["local_dir"])
    gate = _evaluate_safety(resource_snapshot(), _thermal_snapshot(),
                            disk_free_gb=disk_free, remaining_gb=remaining)
    overlap = _overlap_safety(_studio_activity(), _download_tree_rss())
    gate["overlap"] = overlap
    if not overlap["ok"]:
        gate["blockers"].extend(overlap["blockers"])
        gate["ok"] = False
    return gate


def _marker_status(item):
    spec = procure._resolve(item["label"])
    _, marker_path = procure._checkpoint_paths(item["label"])
    marker = procure._read_json(marker_path, {})
    ok = procure._verified_marker_valid(
        marker,
        label=spec.label,
        hf_id=spec.hf_id,
        local_dir=item["local_dir"],
        require_verify=True,
    )
    return {"ok": ok, "path": marker_path, "marker": marker if ok else None}


def _processing_barrier_status():
    """Fail closed: 120B waits for both 14B lane receipts, not a process exit code."""
    try:
        status = processing_queue.completion_status()
    except Exception as exc:
        status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    pid_info = _read_json(processing_queue.PID_FILE, {})
    status["processor_active"] = _pid_alive(pid_info.get("pid"))
    status["processor_pid"] = pid_info.get("pid")
    return status


def _validate_architecture_gate(doc, item):
    spec = procure._resolve(item["label"])
    return bool(
        isinstance(doc, dict)
        and doc.get("schema") == "hawking.download_architecture_ready.v1"
        and doc.get("status") == "pass"
        and doc.get("architecture_ready") is True
        and doc.get("label") == spec.label
        and doc.get("hf_id") == spec.hf_id
        and (doc.get("completed_at") or doc.get("generated_at") or doc.get("timestamp"))
    )


def _architecture_gate_status(item):
    path = ROOT / item.get("architecture_gate", "")
    doc = _read_json(path, {})
    return {
        "ok": _validate_architecture_gate(doc, item),
        "path": str(path),
        "document": doc if _validate_architecture_gate(doc, item) else None,
        "required_schema": "hawking.download_architecture_ready.v1",
        "required_status": "pass",
    }


def _wait_processing_barrier(item):
    while True:
        if _stop_requested or SHARED_DRAIN.exists():
            _update_item(item["label"], status="paused-drain", child_pid=None)
            return 130
        barrier = _processing_barrier_status()
        if barrier.get("ok"):
            _update_item(item["label"], status="processing-barrier-pass",
                         processing_barrier=barrier, child_pid=None)
            return 0
        _update_state(status="waiting-14B-processing", active_label=item["label"])
        _update_item(item["label"], status="waiting-14B-processing",
                     processing_barrier=barrier, child_pid=None)
        if not _sleep_interruptible(POLL_S):
            continue


def _wait_largest_install_admission(item):
    """Every frontier install needs both explicit architecture and live disk gates."""
    while True:
        if _stop_requested or SHARED_DRAIN.exists():
            _update_item(item["label"], status="paused-drain", child_pid=None)
            return 130
        architecture = _architecture_gate_status(item)
        safety = _safety(item)
        if architecture["ok"] and safety["ok"]:
            _update_item(item["label"], status="largest-install-admission-pass",
                         architecture_gate=architecture, safety=safety, child_pid=None)
            return 0
        if not architecture["ok"]:
            status = "planned-blocked-architecture"
        else:
            status = "planned-blocked-disk"
        _update_state(status=status, active_label=item["label"])
        _update_item(
            item["label"], status=status, architecture_gate=architecture,
            safety=safety, child_pid=None,
            source_release={
                "status": "blocked",
                "automatic_deletion": False,
                "reason": "durable artifact-bound verification is required before any source release",
            },
        )
        if not _sleep_interruptible(POLL_S):
            continue


def _live_downloads():
    rows = []
    DOWNLOAD_STATE.mkdir(parents=True, exist_ok=True)
    for path in DOWNLOAD_STATE.glob("*.pid.json"):
        row = _read_json(path, {})
        pid = row.get("pid")
        if _pid_alive(pid):
            rows.append({"label": row.get("label"), "pid": int(pid), "source": str(path)})
    state = _load_state()
    child = state.get("child_pid")
    if _pid_alive(child) and all(int(child) != row["pid"] for row in rows):
        rows.append({"label": state.get("active_label"), "pid": int(child), "source": "queue-state"})
    return rows


def _terminate_group(pid, reason):
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            return
    _update_state(last_termination={"at": _now(), "pid": int(pid), "reason": reason})


def _sleep_interruptible(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _stop_requested or SHARED_DRAIN.exists():
            return False
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    return True


def _retry_wait_with_heartbeat(label, returncode, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _stop_requested or SHARED_DRAIN.exists():
            return False
        remaining = max(0.0, deadline - time.monotonic())
        _update_item(
            label, status="retry-wait", returncode=returncode, child_pid=None,
            retry_after_s=round(seconds, 3), retry_remaining_s=round(remaining, 3),
            retry_heartbeat_at=_now(),
        )
        if not _sleep_interruptible(min(30.0, remaining)):
            return False
    return True


def _child_command(item):
    return [
        sys.executable, str(ROOT / "tools/condense/procure.py"), item["label"],
        "--dir", item["local_dir"], "--verify", "--retries", "2",
        "--progress-interval-s", "60", "--stall-timeout-s", "900",
        "--disk-free-floor-gb", str(DEFAULT_HARDWARE.disk_reserve_gb),
    ]


def _monitor_child(proc, item):
    while proc.poll() is None:
        heartbeat_path, _ = procure._checkpoint_paths(item["label"])
        heartbeat = procure._read_json(heartbeat_path, {})
        gate = _safety(item)
        _update_item(item["label"], status="downloading", child_pid=proc.pid,
                     heartbeat=heartbeat, safety=gate)
        if _stop_requested or SHARED_DRAIN.exists():
            _terminate_group(proc.pid, "Studio/queue drain requested")
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return 130
        if not gate["ok"]:
            _terminate_group(proc.pid, "; ".join(gate["blockers"]))
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return OVERLAP_PAUSE_RC if not gate.get("overlap", {}).get("ok", True) else 75
        studio_live = gate.get("overlap", {}).get("studio", {}).get("active") is True
        if not _sleep_interruptible(min(POLL_S, 5.0) if studio_live else POLL_S):
            continue
    return int(proc.returncode or 0)


def _wait_for_studio_overlap_clear(label):
    """After a live overlap cap/probe failure, do not churn resumable HF children."""
    while True:
        if _stop_requested or SHARED_DRAIN.exists():
            return 130
        studio = _studio_activity()
        _update_item(label, status="waiting-studio-overlap", child_pid=None,
                     studio_activity=studio)
        if studio.get("ok") and studio.get("active") is False:
            return 0
        if not _sleep_interruptible(POLL_S):
            continue


def _controlled_resume_accounting(prior):
    if prior.get("status") == "paused-drain" and prior.get("returncode") == 130:
        return {
            "status": "pending-resume",
            "attempts": max(0, int(prior.get("attempts", 0)) - 1),
            "controlled_interruptions": int(prior.get("controlled_interruptions", 0)) + 1,
            "returncode": None,
        }
    return None


def _run_item(item):
    label = item["label"]
    # A clean supervisor drain is a checkpoint boundary, not a failed network attempt. Refund the
    # interrupted launch once on resume so moving/unplugging the Studio cannot exhaust the bounded
    # retry budget. Genuine nonzero/stall exits remain charged.
    prior = _load_state()["items"].get(label, {})
    resume_accounting = _controlled_resume_accounting(prior)
    if resume_accounting is not None:
        _update_item(label, **resume_accounting)
    while True:
        marker = _marker_status(item)
        if marker["ok"]:
            _update_item(label, status="verified", verified_at=_now(),
                         verified_marker=marker["path"], child_pid=None)
            return 0
        if _stop_requested or SHARED_DRAIN.exists():
            for row in _live_downloads():
                if row["pid"] != os.getpid():
                    _terminate_group(row["pid"], "Studio/queue drain requested")
            _update_item(label, status="paused-drain", child_pid=None)
            return 130

        live = [row for row in _live_downloads() if row["pid"] != os.getpid()]
        if live:
            gate = _safety(item)
            _update_item(label, status="waiting-existing-download", live_downloads=live, safety=gate)
            if not gate["ok"]:
                for row in live:
                    _terminate_group(row["pid"], "; ".join(gate["blockers"]))
            if not _sleep_interruptible(POLL_S):
                continue
            continue

        gate = _safety(item)
        if not gate["ok"]:
            _update_item(label, status="waiting-resources", safety=gate, child_pid=None)
            if not _sleep_interruptible(POLL_S):
                continue
            continue

        state = _load_state()
        attempts = int(state["items"].get(label, {}).get("attempts", 0))
        if attempts >= MAX_ATTEMPTS:
            _update_item(label, status="blocked-retries", child_pid=None,
                         error=f"{attempts} queue attempts exhausted")
            return 1

        log_path = ROOT / f"reports/condense/download_queue_{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab", buffering=0)
        env = os.environ.copy()
        env.setdefault("HF_MAX_WORKERS", "4")
        env.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
        proc = subprocess.Popen(
            _child_command(item), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        attempts += 1
        _update_state(status="running", active_label=label, child_pid=proc.pid,
                      child_pgid=proc.pid, child_log=str(log_path))
        _update_item(label, status="downloading", attempts=attempts, child_pid=proc.pid,
                     started_at=_now(), safety=gate)
        rc = _monitor_child(proc, item)
        _update_state(child_pid=None, child_pgid=None)

        marker = _marker_status(item)
        if rc == 0 and marker["ok"]:
            _update_item(label, status="verified", returncode=0, child_pid=None,
                         verified_at=_now(), verified_marker=marker["path"])
            return 0
        if rc == 130:
            _update_item(label, status="paused-drain", returncode=rc, child_pid=None)
            return rc
        if rc == OVERLAP_PAUSE_RC:
            _update_item(label, status="paused-studio-overlap", returncode=rc, child_pid=None)
            wait_rc = _wait_for_studio_overlap_clear(label)
            if wait_rc != 0:
                return wait_rc
            continue
        if rc == 75:
            _update_item(label, status="paused-resources", returncode=rc, child_pid=None)
            continue
        backoff = min(900.0, 60.0 * (2 ** min(attempts - 1, 4)))
        if not _retry_wait_with_heartbeat(label, rc, backoff):
            continue


def run_queue():
    global _stop_requested
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[download-queue] another queue supervisor holds the singleton lock", file=sys.stderr)
        return 2

    def request_stop(_sig, _frame):
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _atomic_json(PID_FILE, {"schema": "hawking.download_queue_pid.v1", "pid": os.getpid(),
                            "started_at": _now(), "log": str(LOG_FILE)})
    _update_state(status="running", supervisor_pid=os.getpid(), plan=[dict(x) for x in PLAN])
    rc = 0
    try:
        for item in PLAN:
            # Existing 32B/72B semantics are unchanged. Before admitting 120B, require
            # path-bound completion + coverage for both 14B processing lanes.
            if item["label"] == "120B" and not _marker_status(item)["ok"]:
                rc = _wait_processing_barrier(item)
                if rc != 0:
                    break
            # Architecture-gated frontier entries are scheduled targets, not permission to consume
            # the remaining SSD. Neither can reach procure.py without both hard gates.
            if item.get("architecture_gate") and not _marker_status(item)["ok"]:
                rc = _wait_largest_install_admission(item)
                if rc != 0:
                    break
            rc = _run_item(item)
            if rc != 0:
                break
        if rc == 0:
            _update_state(status="complete", active_label=None, completed_at=_now(),
                          terminal_reason=(
                              "32B, 72B, 120B, and architecture/disk-admitted frontier "
                              "installs verified; source release remains artifact-gated"
                          ))
        elif rc == 130:
            _update_state(status="paused-drain", paused_at=_now())
        else:
            _update_state(status="blocked", returncode=rc, blocked_at=_now())
        return rc
    finally:
        info = _read_json(PID_FILE, {})
        if info.get("pid") == os.getpid():
            try:
                PID_FILE.unlink()
                _fsync_dir(PID_FILE.parent)
            except FileNotFoundError:
                pass
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def start_queue():
    if SHARED_DRAIN.exists():
        print(f"[download-queue] Studio drain is active at {SHARED_DRAIN}; resume Studio first",
              file=sys.stderr)
        return 130
    info = _read_json(PID_FILE, {})
    if _pid_alive(info.get("pid")):
        print(f"[download-queue] already active pid={info['pid']}", file=sys.stderr)
        return 0
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab", buffering=0)
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "run"]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-dimsu", *cmd]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    _atomic_json(PID_FILE, {"schema": "hawking.download_queue_pid.v1", "pid": proc.pid,
                            "started_at": _now(), "log": str(LOG_FILE), "cmd": cmd})
    print(f"[download-queue] detached pid={proc.pid}; log={LOG_FILE}", file=sys.stderr)
    return 0


def status():
    info = _read_json(PID_FILE, {})
    largest = PLAN[-1]
    payload = {
        "schema": "hawking.download_queue_status.v1",
        "generated_at": _now(),
        "active": _pid_alive(info.get("pid")),
        "pid": info.get("pid"),
        "state": _load_state(),
        "live_downloads": _live_downloads(),
        "resources": resource_snapshot(),
        "thermal": _thermal_snapshot(),
        "drain_requested": SHARED_DRAIN.exists(),
        "markers": {item["label"]: _marker_status(item)["ok"] for item in PLAN},
        "pre_120b_processing_barrier": _processing_barrier_status(),
        "largest_install_admission": {
            "architecture_gate": _architecture_gate_status(largest),
            "disk_gate": _safety(largest),
        },
        "source_release": {
            "status": "blocked",
            "automatic_deletion": False,
            "reason": "durable artifact inventory and artifact-bound verification are required",
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def selftest():
    global STUDIO_RUN_PID, STUDIO_WAIT_PID, STATE
    green = {"ok": True, "pressure_level": 1, "swap_used_mb": 0.0,
             "power_source": "Now drawing from 'AC Power'"}
    thermal = {"ok": True}
    assert _evaluate_safety(green, thermal, disk_free_gb=400, remaining_gb=100)["ok"]
    assert not _evaluate_safety({**green, "pressure_level": 2}, thermal,
                                disk_free_gb=400, remaining_gb=100)["ok"]
    assert not _evaluate_safety({**green, "pressure_level": None}, thermal,
                                disk_free_gb=400, remaining_gb=100)["ok"]
    assert not _evaluate_safety({**green, "swap_used_mb": None}, thermal,
                                disk_free_gb=400, remaining_gb=100)["ok"]
    assert not _evaluate_safety(green, thermal, disk_free_gb=200, remaining_gb=100)["ok"]
    assert not _evaluate_safety(green, {"ok": False}, disk_free_gb=400, remaining_gb=100)["ok"]
    inactive = {"ok": True, "active": False, "pid": None, "mode": None}
    waiting = {"ok": True, "active": True, "pid": 7, "mode": "waiting-admission"}
    malformed = {"ok": False, "active": None, "error": "synthetic probe failure"}
    for rss_gb, expected in ((10.0, True), (11.0, True), (12.0, False)):
        overlap = _overlap_safety(
            waiting, {"ok": True, "rss_gib": rss_gb, "rows": [], "error": None}
        )
        assert overlap["ok"] is expected, (rss_gb, overlap)
    assert _overlap_safety(
        inactive, {"ok": True, "rss_gib": 100.0, "rows": [], "error": None}
    )["ok"], "download RSS is not charged against a Studio peak when Studio is absent"
    assert not _overlap_safety(
        malformed, {"ok": True, "rss_gib": 0.0, "rows": [], "error": None}
    )["ok"]
    assert not _overlap_safety(
        waiting, {"ok": False, "rss_gib": None, "rows": [], "error": "ps failed"}
    )["ok"]
    assert OVERLAP_PAUSE_RC == 76
    original_paths = (STUDIO_RUN_PID, STUDIO_WAIT_PID, STATE)
    with tempfile.TemporaryDirectory(prefix="download_queue_safety_") as td:
        root = pathlib.Path(td)
        STUDIO_RUN_PID = root / "studio_run.pid"
        STUDIO_WAIT_PID = root / "studio_wait.pid"
        STATE = root / "queue_state.json"
        assert _studio_activity()["active"] is False
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        try:
            _atomic_json(STUDIO_WAIT_PID, {
                "schema": "hawking.studio_wait_pid.v1", "pid": sleeper.pid,
                "mode": "waiting-admission",
            })
            assert _studio_activity()["active"] is True
            _atomic_json(STUDIO_WAIT_PID, {
                "schema": "wrong", "pid": sleeper.pid, "mode": "waiting-admission",
            })
            assert not _studio_activity()["ok"]
            STUDIO_WAIT_PID.unlink()
            _atomic_json(STATE, _base_state())
            rss_probe = _download_tree_rss()
            assert rss_probe["ok"] and rss_probe["rss_gib"] > 0, rss_probe
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)
        _atomic_json(STUDIO_RUN_PID, {"pid": sleeper.pid, "started_at": _now()})
        assert _studio_activity()["active"] is False, (
            "a dead legacy PID record must be ignored before live-mode validation"
        )
    STUDIO_RUN_PID, STUDIO_WAIT_PID, STATE = original_paths
    assert [item["label"] for item in PLAN] == [
        "32B", "72B", "120B", "DeepSeek-V4-Flash", "Kimi-K2.6"
    ]
    assert _controlled_resume_accounting({
        "status": "paused-drain", "returncode": 130, "attempts": 4,
        "controlled_interruptions": 2,
    }) == {
        "status": "pending-resume", "returncode": None, "attempts": 3,
        "controlled_interruptions": 3,
    }
    assert _controlled_resume_accounting({
        "status": "retry-wait", "returncode": 1, "attempts": 4,
    }) is None
    largest = PLAN[-1]
    spec = procure._resolve(largest["label"])
    architecture = {
        "schema": "hawking.download_architecture_ready.v1",
        "status": "pass",
        "architecture_ready": True,
        "label": spec.label,
        "hf_id": spec.hf_id,
        "completed_at": _now(),
    }
    assert _validate_architecture_gate(architecture, largest)
    assert not _validate_architecture_gate({**architecture, "architecture_ready": False}, largest)
    print("download_queue.py selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "start":
        raise SystemExit(start_queue())
    if command == "run":
        raise SystemExit(run_queue())
    if command == "status":
        raise SystemExit(status())
    if command == "--selftest":
        raise SystemExit(selftest())
    print("usage: download_queue.py start|run|status|--selftest", file=sys.stderr)
    raise SystemExit(2)
