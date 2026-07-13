#!/usr/bin/env python3.12
"""Detached, observation-only supervisor for the Doctor-v2 frontier campaign.

The queue validates and pins the campaign, observes the existing Studio owner,
heavy-work lease, and machine safety gates, and asks ``doctor_frontier`` for the
next component-launchable candidate.  It is deliberately incapable of running
that candidate: no worker command, HealerProgram, or campaign-provided argv is
ever invoked.  Its only child process is a fixed invocation of this file's own
``run`` command when ``start`` detaches the observer.

The durable state is an atomic checkpoint.  A power loss or unplug therefore
preserves the pinned campaign and selection history; ``start`` resumes it only
when the on-disk campaign still has exactly the same validated identity.
"""
from __future__ import annotations

import argparse
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
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Required selector.  doctor_frontier and healer_abi are themselves stdlib-only.
import doctor_frontier  # noqa: E402


CAMPAIGN = ROOT / "reports/condense/doctor_v2_frontier_campaign.json"
OBSERVATIONS = ROOT / "reports/condense/doctor_v2_frontier_observations.jsonl"
STATE = ROOT / "reports/condense/doctor_frontier_queue_state.json"
PID_FILE = ROOT / "reports/condense/doctor_frontier_queue.pid.json"
LOCK_FILE = ROOT / "reports/condense/doctor_frontier_queue.lock"
STOP_FILE = ROOT / "reports/condense/doctor_frontier_queue.stop.json"
LOG_FILE = ROOT / "reports/condense/doctor_frontier_queue.log"
STUDIO_RUN_PID = ROOT / "reports/cron/studio_run.pid"
STUDIO_WAIT_PID = ROOT / "reports/cron/studio_wait.pid"
HEAVY_LOCK = ROOT / "reports/cron/studio_heavy.lock"

STATE_SCHEMA = "hawking.doctor_frontier_queue.v1"
PID_SCHEMA = "hawking.doctor_frontier_queue_pid.v1"
STATUS_SCHEMA = "hawking.doctor_frontier_queue_status.v1"
POLL_SECONDS = 120
DISK_FREE_FLOOR_GB = 150.0
COMPONENT_FIELDS = ("diagnostic", "base", "transform", "correction", "state", "runtime")
TERMINAL_OBSERVATION_STATES = {"pass", "fail", "complete-negative", "blocked"}

# This is both a machine-readable contract and a guard used in every state update.
INVOKE_WORKERS = False
WORKER_LAUNCHES_TOTAL = 0

# The supervisor may run only these fixed, read-only operating-system probes.
PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    "pressure": ("/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"),
    "swap": ("/usr/sbin/sysctl", "-n", "vm.swapusage"),
    "power": ("/usr/bin/pmset", "-g", "batt"),
    "thermal": ("/usr/bin/pmset", "-g", "therm"),
}

_terminate_requested = False


class QueueError(ValueError):
    """Fail-closed validation or observer error."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some filesystems do not implement directory fsync.  File fsync and
        # atomic replace still protect against partial JSON in that case.
        pass


def _atomic_json(path: Path, value: Any) -> None:
    """Write a restart-safe JSON checkpoint without exposing a partial file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
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


def _read_json(path: Path, *, optional: bool = False) -> dict[str, Any] | None:
    if optional and not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise QueueError(f"required file is absent: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"cannot read valid JSON from {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueError(f"JSON root must be an object: {path}")
    return value


def _positive_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _candidate_identity(candidate: dict[str, Any]) -> str:
    compact = {
        key: candidate.get(key)
        for key in (
            "model", "params_b", "active_b", "moe", "target_bpw", "diagnostic",
            "base", "transform", "correction", "state", "runtime", "objective",
            "calibration", "seed", "campaign_version",
        )
    }
    return _hash_value(compact)


def _component_launchability(
    campaign: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Independently verify the selector's component-level launch permission."""
    problems: list[str] = []
    catalog = campaign.get("operator_catalog")
    if not isinstance(catalog, dict):
        return False, ["operator_catalog is not an object"]
    if candidate.get("launchable") is not True:
        problems.append("candidate launchable flag is not true")
    for field in COMPONENT_FIELDS:
        name = candidate.get(field)
        row = catalog.get(name) if isinstance(name, str) else None
        if not isinstance(row, dict):
            problems.append(f"{field} component is absent from operator_catalog")
            continue
        if row.get("executor_wired") is not True:
            problems.append(f"{field} component {name!r} is not executor-wired")
        if row.get("implementation_status") in {"research", "unimplemented", "runtime_gated"}:
            problems.append(
                f"{field} component {name!r} is {row.get('implementation_status')}"
            )
    params_b = candidate.get("params_b")
    if not _positive_finite(params_b):
        problems.append("candidate params_b is not positive and finite")
    elif float(params_b) >= 32.0:
        problems.append("candidate still requires the streamed model/evaluation path")
    return not problems, problems


def validate_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    """Validate schema, canonical hash, experiment identities, and launch flags."""
    problems: list[str] = []
    if campaign.get("schema") != doctor_frontier.CAMPAIGN_SCHEMA:
        problems.append(
            f"schema must be {doctor_frontier.CAMPAIGN_SCHEMA!r}, got {campaign.get('schema')!r}"
        )
    if campaign.get("mode") != "plan_only_no_execution":
        problems.append("mode must remain plan_only_no_execution")
    if not isinstance(campaign.get("campaign_version"), str) or not campaign["campaign_version"]:
        problems.append("campaign_version must be a non-empty string")
    required_objects = (
        "operator_catalog", "axes", "search_size", "scheduler", "regime_policy",
        "transfer_policy", "backend_policy",
    )
    for key in required_objects:
        if not isinstance(campaign.get(key), dict):
            problems.append(f"{key} must be an object")
    if not isinstance(campaign.get("models"), list) or not campaign.get("models"):
        problems.append("models must be a non-empty array")
    if not isinstance(campaign.get("fidelity_ladder"), list) or not campaign.get("fidelity_ladder"):
        problems.append("fidelity_ladder must be a non-empty array")

    claimed = campaign.get("campaign_sha256")
    if not _is_sha256(claimed):
        problems.append("campaign_sha256 is not a lowercase SHA-256")
    projected = {
        key: value
        for key, value in campaign.items()
        if key not in {"campaign_sha256", "generated_at"}
    }
    computed = _hash_value(projected)
    if claimed != computed:
        problems.append(f"campaign hash mismatch: claimed={claimed!r} computed={computed}")

    experiments = campaign.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        problems.append("experiments must be a non-empty array")
        experiments = []
    search_size = campaign.get("search_size")
    if isinstance(search_size, dict) and search_size.get("explicit_candidates") != len(experiments):
        problems.append("search_size.explicit_candidates does not match experiments length")

    seen: set[str] = set()
    component_launchable = 0
    for index, candidate in enumerate(experiments):
        prefix = f"experiments[{index}]"
        if not isinstance(candidate, dict):
            problems.append(f"{prefix} must be an object")
            continue
        experiment_id = candidate.get("experiment_id")
        if not isinstance(experiment_id, str) or not re.fullmatch(r"dr2-[0-9a-f]{16}", experiment_id):
            problems.append(f"{prefix}.experiment_id is malformed")
        elif experiment_id in seen:
            problems.append(f"duplicate experiment_id {experiment_id}")
        else:
            seen.add(experiment_id)
        identity = candidate.get("identity_sha256")
        computed_identity = _candidate_identity(candidate)
        if not _is_sha256(identity) or identity != computed_identity:
            problems.append(f"{prefix}.identity_sha256 does not bind the candidate")
        if isinstance(experiment_id, str) and experiment_id != f"dr2-{computed_identity[:16]}":
            problems.append(f"{prefix}.experiment_id does not bind identity_sha256")
        if candidate.get("campaign_version") != campaign.get("campaign_version"):
            problems.append(f"{prefix}.campaign_version differs from campaign")
        if not _positive_finite(candidate.get("params_b")):
            problems.append(f"{prefix}.params_b must be positive and finite")
        if not _positive_finite(candidate.get("target_bpw")):
            problems.append(f"{prefix}.target_bpw must be positive and finite")
        if not isinstance(candidate.get("launchable"), bool):
            problems.append(f"{prefix}.launchable must be boolean")
        if not isinstance(candidate.get("blockers"), list):
            problems.append(f"{prefix}.blockers must be an array")
        launchable, launch_problems = _component_launchability(campaign, candidate)
        if candidate.get("launchable") is True:
            if launchable:
                component_launchable += 1
            else:
                problems.append(f"{prefix} falsely claims launchable: {'; '.join(launch_problems)}")

    scheduler = campaign.get("scheduler")
    if isinstance(scheduler, dict):
        if scheduler.get("one_heavy_lease") is not True:
            problems.append("scheduler.one_heavy_lease must be true")
        safety = str(scheduler.get("safety", "")).lower()
        for required in ("normal pressure", "zero swap", "ac", "thermal", "disk"):
            if required not in safety:
                problems.append(f"scheduler safety contract omits {required!r}")

    if problems:
        preview = "; ".join(problems[:12])
        if len(problems) > 12:
            preview += f"; ... {len(problems) - 12} more"
        raise QueueError(preview)
    return {
        "ok": True,
        "schema": campaign["schema"],
        "campaign_version": campaign["campaign_version"],
        "campaign_sha256": claimed,
        "experiment_count": len(experiments),
        "component_launchable_count": component_launchable,
        "mode": campaign["mode"],
    }


def load_campaign(path: Path = CAMPAIGN) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = _read_json(path)
    assert campaign is not None
    return campaign, validate_campaign(campaign)


def load_observations(path: Path = OBSERVATIONS) -> dict[str, dict[str, Any]]:
    """Strictly read optional JSONL so corrupt completion history cannot be ignored."""
    if not path.exists():
        return {}
    observations: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QueueError(f"cannot read observations {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QueueError(f"invalid observation JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("experiment_id"), str):
            raise QueueError(f"observation at {path}:{line_number} lacks experiment_id")
        observations[row["experiment_id"]] = row
    return observations


def select_component_candidate(
    campaign: dict[str, Any], observations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Delegate ranking to doctor_frontier, never enabling unwired candidates."""
    selection = doctor_frontier.select_next(campaign, observations, allow_unwired=False)
    if selection.get("schema") != doctor_frontier.SELECTION_SCHEMA:
        raise QueueError("doctor_frontier.select_next returned an unexpected schema")
    if selection.get("campaign_sha256") != campaign.get("campaign_sha256"):
        raise QueueError("selection is not bound to the validated campaign")
    if selection.get("allow_unwired") is not False:
        raise QueueError("selector attempted to enable unwired candidates")
    selected = selection.get("selected")
    if selected is not None:
        if not isinstance(selected, dict):
            raise QueueError("selected candidate is not an object")
        launchable, problems = _component_launchability(campaign, selected)
        if not launchable:
            raise QueueError("selector returned a non-launchable candidate: " + "; ".join(problems))
    return selection


def _pid_alive(pid: Any) -> bool:
    try:
        numeric = int(pid)
        if numeric <= 0:
            return False
        os.kill(numeric, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _observe_pid_record(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        return {"ok": True, "role": role, "path": str(path), "active": False, "pid": None}
    try:
        info = _read_json(path)
        assert info is not None
        pid = int(info.get("pid"))
        if pid <= 0:
            raise ValueError("pid must be positive")
        return {
            "ok": True,
            "role": role,
            "path": str(path),
            "active": _pid_alive(pid),
            "pid": pid,
            "record": info,
        }
    except (QueueError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "role": role,
            "path": str(path),
            "active": None,
            "pid": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def observe_studio() -> dict[str, Any]:
    rows = [
        _observe_pid_record(STUDIO_RUN_PID, "studio-run"),
        _observe_pid_record(STUDIO_WAIT_PID, "studio-wait"),
    ]
    return {
        "ok": all(row.get("ok") is True for row in rows),
        "active": any(row.get("active") is True for row in rows),
        "owners": rows,
    }


def observe_heavy_lock(path: Path = HEAVY_LOCK) -> dict[str, Any]:
    """Probe the existing lease without creating, truncating, or retaining it."""
    if not path.exists():
        return {"ok": True, "path": str(path), "exists": False, "held": False}
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            return {"ok": True, "path": str(path), "exists": True, "held": False}
        except BlockingIOError:
            return {"ok": True, "path": str(path), "exists": True, "held": True}
    except OSError as exc:
        return {
            "ok": False,
            "path": str(path),
            "exists": path.exists(),
            "held": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if descriptor is not None:
            if acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)


def _run_fixed_probe(name: str) -> dict[str, Any]:
    """Run one compile-time allowlisted diagnostic; callers cannot supply argv."""
    if name not in PROBE_COMMANDS:
        raise QueueError(f"unknown fixed resource probe: {name}")
    argv = PROBE_COMMANDS[name]
    if not Path(argv[0]).is_file():
        return {"ok": False, "name": name, "error": f"probe executable absent: {argv[0]}"}
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "name": name, "error": f"{type(exc).__name__}: {exc}"}
    detail = (result.stdout + result.stderr).strip()
    return {
        "ok": result.returncode == 0,
        "name": name,
        "returncode": result.returncode,
        "detail": detail[-1000:],
    }


def _parse_swap_mb(text: str) -> float | None:
    match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * {"M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}[match.group(2)]


def _thermal_green(returncode: int, text: str) -> bool:
    if returncode != 0 or not text.strip():
        return False
    lowered = text.lower()
    explicit = (
        "no thermal warning level has been recorded" in lowered
        and "no performance warning level has been recorded" in lowered
    )
    numeric = {
        key.lower(): int(value)
        for key, value in re.findall(r"([A-Za-z_]+)\s*[:=]\s*(\d+)", text)
    }
    numeric_green = bool(
        {"cpu_speed_limit", "scheduler_limit", "available_cpus"}.issubset(numeric)
        and numeric["cpu_speed_limit"] >= 100
        and numeric["scheduler_limit"] >= 100
        and numeric["available_cpus"] > 0
    )
    return explicit or numeric_green


def observe_resources(root: Path = ROOT) -> dict[str, Any]:
    pressure_probe = _run_fixed_probe("pressure")
    swap_probe = _run_fixed_probe("swap")
    power_probe = _run_fixed_probe("power")
    thermal_probe = _run_fixed_probe("thermal")

    pressure_level: int | None = None
    if pressure_probe.get("ok"):
        try:
            pressure_level = int(str(pressure_probe.get("detail", "")).strip())
        except ValueError:
            pressure_level = None
    swap_used_mb = (
        _parse_swap_mb(str(swap_probe.get("detail", ""))) if swap_probe.get("ok") else None
    )
    power_detail = str(power_probe.get("detail", ""))
    thermal_detail = str(thermal_probe.get("detail", ""))
    thermal_ok = _thermal_green(int(thermal_probe.get("returncode", -1)), thermal_detail)
    try:
        usage = shutil.disk_usage(root)
        disk_free_gb = usage.free / 1e9
        disk_probe_ok = True
        disk_error = None
    except OSError as exc:
        disk_free_gb = None
        disk_probe_ok = False
        disk_error = f"{type(exc).__name__}: {exc}"

    blockers: list[str] = []
    if pressure_level != 1:
        blockers.append(f"memory pressure is not confirmed normal (level={pressure_level!r})")
    if swap_used_mb is None or swap_used_mb != 0.0:
        blockers.append(f"swap is not confirmed zero (used_mb={swap_used_mb!r})")
    if not power_probe.get("ok") or "AC Power" not in power_detail:
        blockers.append("AC power is not confirmed")
    if not thermal_ok:
        blockers.append("thermal/performance state is not confirmed green")
    if disk_free_gb is None or disk_free_gb < DISK_FREE_FLOOR_GB:
        rendered = "unknown" if disk_free_gb is None else f"{disk_free_gb:.3f}"
        blockers.append(
            f"disk free {rendered}GB is below the {DISK_FREE_FLOOR_GB:.1f}GB hard floor"
        )
    return {
        "schema": "hawking.doctor_frontier_resource_gate.v1",
        "sampled_at": _now(),
        "ok": not blockers,
        "blockers": blockers,
        "requirements": {
            "pressure_level": 1,
            "pressure_name": "normal",
            "swap_used_mb": 0.0,
            "power_source": "AC Power",
            "thermal": "green",
            "disk_free_floor_gb": DISK_FREE_FLOOR_GB,
        },
        "observed": {
            "pressure_level": pressure_level,
            "pressure_name": {1: "normal", 2: "warning", 4: "critical"}.get(
                pressure_level, "unknown"
            ),
            "swap_used_mb": swap_used_mb,
            "power_source": power_detail,
            "thermal_green": thermal_ok,
            "thermal_detail": thermal_detail,
            "disk_free_gb": round(disk_free_gb, 3) if disk_free_gb is not None else None,
            "disk_probe_ok": disk_probe_ok,
            "disk_error": disk_error,
        },
        "probes": {
            "pressure": pressure_probe,
            "swap": swap_probe,
            "power": power_probe,
            "thermal": thermal_probe,
        },
    }


def _initial_state(campaign_report: dict[str, Any]) -> dict[str, Any]:
    timestamp = _now()
    return {
        "schema": STATE_SCHEMA,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "new",
        "observation_sequence": 0,
        "poll_seconds": POLL_SECONDS,
        "campaign": {
            **campaign_report,
            "path": str(CAMPAIGN),
            "observations_path": str(OBSERVATIONS),
        },
        "plan_checkpoint": {
            "campaign_sha256": campaign_report["campaign_sha256"],
            "campaign_version": campaign_report["campaign_version"],
            "experiment_count": campaign_report["experiment_count"],
            "component_launchable_count": campaign_report["component_launchable_count"],
            "preserved_across_restart": True,
        },
        "invoke_workers": False,
        "worker_launches_total": 0,
        "execution_policy": {
            "observer_only": True,
            "invoke_workers": False,
            "arbitrary_commands": False,
            "campaign_argv_trusted_or_executed": False,
            "heavy_lease_acquired": False,
        },
    }


def _load_state(path: Path = STATE) -> dict[str, Any] | None:
    state = _read_json(path, optional=True)
    if state is None:
        return None
    if state.get("schema") != STATE_SCHEMA:
        raise QueueError(f"state schema mismatch at {path}")
    if state.get("invoke_workers") is not False or state.get("worker_launches_total") != 0:
        raise QueueError("state violates the observer-only execution contract")
    return state


def _bind_state(
    previous: dict[str, Any] | None, campaign_report: dict[str, Any]
) -> dict[str, Any]:
    if previous is None:
        return _initial_state(campaign_report)
    pinned = previous.get("plan_checkpoint")
    if not isinstance(pinned, dict):
        raise QueueError("state lacks a plan_checkpoint")
    if pinned.get("campaign_sha256") != campaign_report.get("campaign_sha256"):
        raise QueueError(
            "campaign identity differs from the durable plan checkpoint; refusing implicit replacement"
        )
    return copy.deepcopy(previous)


def evaluate_once(
    campaign: dict[str, Any],
    campaign_report: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    previous: dict[str, Any] | None,
    *,
    studio: dict[str, Any],
    heavy_lock: dict[str, Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    """Build one durable observer checkpoint; this function has no execution path."""
    state = _bind_state(previous, campaign_report)
    selection = select_component_candidate(campaign, observations)
    selected = selection.get("selected")

    telemetry_ok = studio.get("ok") is True and heavy_lock.get("ok") is True
    current_studio = studio.get("active") is True or heavy_lock.get("held") is True
    if current_studio:
        status = "waiting-current-studio"
        reason = "the current Studio PID and/or heavy-work lease still owns the machine"
    elif not telemetry_ok:
        status = "waiting-observer-telemetry"
        reason = "Studio PID or heavy-lock state could not be confirmed"
    elif resources.get("ok") is not True:
        status = "waiting-resource-gate"
        reason = "zero-swap/normal-pressure/AC/thermal/disk admission is not green"
    elif selected is None:
        status = "waiting-no-component-launchable"
        reason = "doctor_frontier.select_next found no component-launchable candidate"
    else:
        status = "candidate-ready-observer-only"
        reason = "a candidate is advisory-ready; worker invocation remains disabled"

    timestamp = _now()
    state.update(
        {
            "updated_at": timestamp,
            "status": status,
            "reason": reason,
            "observation_sequence": int(state.get("observation_sequence", 0)) + 1,
            "last_observed_at": timestamp,
            "poll_seconds": POLL_SECONDS,
            "campaign": {
                **campaign_report,
                "path": str(CAMPAIGN),
                "observations_path": str(OBSERVATIONS),
            },
            "selection": selection,
            "selected_experiment_id": (
                selected.get("experiment_id") if isinstance(selected, dict) else None
            ),
            "observations_loaded": len(observations),
            "terminal_observations": sum(
                1 for row in observations.values() if row.get("status") in TERMINAL_OBSERVATION_STATES
            ),
            "studio": studio,
            "heavy_lock": heavy_lock,
            "resources": resources,
            "invoke_workers": False,
            "worker_launches_total": 0,
            "next_poll_not_before": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=POLL_SECONDS)
            ).isoformat(timespec="seconds"),
        }
    )
    # Reassert these values rather than inheriting mutable state from disk.
    state["execution_policy"] = {
        "observer_only": True,
        "invoke_workers": False,
        "arbitrary_commands": False,
        "campaign_argv_trusted_or_executed": False,
        "heavy_lease_acquired": False,
    }
    return state


def _remove_if_ours(path: Path, pid: int) -> None:
    try:
        value = _read_json(path, optional=True)
        if isinstance(value, dict) and value.get("pid") == pid:
            path.unlink()
            _fsync_dir(path.parent)
    except (OSError, QueueError):
        pass


def _stop_requested() -> bool:
    return _terminate_requested or STOP_FILE.exists()


def _sleep_until_poll(seconds: int) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _stop_requested():
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return True


def run_supervisor() -> int:
    """Run the foreground observer loop.  No campaign or worker is executed."""
    global _terminate_requested
    try:
        campaign, campaign_report = load_campaign()
        previous = _bind_state(_load_state(), campaign_report)
    except QueueError as exc:
        print(f"[doctor-frontier-queue] launch validation failed: {exc}", file=sys.stderr)
        return 2

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    singleton = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        singleton.close()
        print("[doctor-frontier-queue] another supervisor holds the singleton lock", file=sys.stderr)
        return 2

    def request_stop(_signal: int, _frame: Any) -> None:
        global _terminate_requested
        _terminate_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    pid_record = {
        "schema": PID_SCHEMA,
        "pid": os.getpid(),
        "started_at": _now(),
        "role": "observer-only",
        "poll_seconds": POLL_SECONDS,
        "campaign_sha256": campaign_report["campaign_sha256"],
        "invoke_workers": False,
        "log": str(LOG_FILE),
    }
    _atomic_json(PID_FILE, pid_record)

    try:
        while not _stop_requested():
            try:
                campaign, current_report = load_campaign()
                previous = _bind_state(_load_state() or previous, current_report)
                observations = load_observations()
                previous = evaluate_once(
                    campaign,
                    current_report,
                    observations,
                    previous,
                    studio=observe_studio(),
                    heavy_lock=observe_heavy_lock(),
                    resources=observe_resources(),
                )
            except Exception as exc:  # persist and retry, but never broaden authority
                timestamp = _now()
                previous = copy.deepcopy(previous)
                previous.update(
                    {
                        "updated_at": timestamp,
                        "status": "blocked-observer-error",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "last_observed_at": timestamp,
                        "invoke_workers": False,
                        "worker_launches_total": 0,
                    }
                )
            _atomic_json(STATE, previous)
            if not _sleep_until_poll(POLL_SECONDS):
                break

        timestamp = _now()
        previous = copy.deepcopy(previous)
        previous.update(
            {
                "updated_at": timestamp,
                "status": "stopped",
                "reason": "cooperative stop checkpoint observed",
                "stopped_at": timestamp,
                "invoke_workers": False,
                "worker_launches_total": 0,
            }
        )
        _atomic_json(STATE, previous)
        return 0
    finally:
        _remove_if_ours(PID_FILE, os.getpid())
        fcntl.flock(singleton.fileno(), fcntl.LOCK_UN)
        singleton.close()


def start_supervisor() -> int:
    """Detach only this observer's fixed ``run`` entrypoint."""
    try:
        _campaign, campaign_report = load_campaign()
        _bind_state(_load_state(), campaign_report)
    except QueueError as exc:
        print(f"[doctor-frontier-queue] start refused: {exc}", file=sys.stderr)
        return 2

    pid_record = _read_json(PID_FILE, optional=True)
    if isinstance(pid_record, dict) and _pid_alive(pid_record.get("pid")):
        print(f"[doctor-frontier-queue] already active pid={pid_record['pid']}", file=sys.stderr)
        return 0
    if STOP_FILE.exists():
        STOP_FILE.unlink()
        _fsync_dir(STOP_FILE.parent)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab", buffering=0)
    fixed_argv = (sys.executable, str(Path(__file__).resolve()), "run")
    try:
        process = subprocess.Popen(
            fixed_argv,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    _atomic_json(
        PID_FILE,
        {
            "schema": PID_SCHEMA,
            "pid": process.pid,
            "started_at": _now(),
            "role": "observer-only",
            "poll_seconds": POLL_SECONDS,
            "campaign_sha256": campaign_report["campaign_sha256"],
            "invoke_workers": False,
            "log": str(LOG_FILE),
            "fixed_self_argv": list(fixed_argv),
        },
    )
    print(
        f"[doctor-frontier-queue] detached observer pid={process.pid}; log={LOG_FILE}",
        file=sys.stderr,
    )
    return 0


def stop_supervisor() -> int:
    """Request a cooperative stop; never signal a Studio or queue process."""
    pid_record = _read_json(PID_FILE, optional=True)
    active = isinstance(pid_record, dict) and _pid_alive(pid_record.get("pid"))
    _atomic_json(
        STOP_FILE,
        {
            "schema": "hawking.doctor_frontier_queue_stop.v1",
            "requested_at": _now(),
            "target_pid": pid_record.get("pid") if isinstance(pid_record, dict) else None,
            "cooperative_only": True,
            "signals_sent": False,
            "invoke_workers": False,
        },
    )
    if active:
        print(
            f"[doctor-frontier-queue] cooperative stop requested for pid={pid_record['pid']}",
            file=sys.stderr,
        )
    else:
        print("[doctor-frontier-queue] stop checkpoint recorded; observer is not active", file=sys.stderr)
    return 0


def status_payload() -> dict[str, Any]:
    """Produce a live read-only view without creating queue state."""
    pid_error = None
    state_error = None
    campaign_error = None
    try:
        pid_record = _read_json(PID_FILE, optional=True)
    except QueueError as exc:
        pid_record = None
        pid_error = str(exc)
    try:
        saved_state = _load_state()
    except QueueError as exc:
        saved_state = None
        state_error = str(exc)

    studio = observe_studio()
    heavy_lock = observe_heavy_lock()
    resources = observe_resources()
    projected_state = None
    campaign_report = None
    try:
        campaign, campaign_report = load_campaign()
        observations = load_observations()
        projected_state = evaluate_once(
            campaign,
            campaign_report,
            observations,
            saved_state,
            studio=studio,
            heavy_lock=heavy_lock,
            resources=resources,
        )
    except QueueError as exc:
        campaign_error = str(exc)

    return {
        "schema": STATUS_SCHEMA,
        "generated_at": _now(),
        "active": isinstance(pid_record, dict) and _pid_alive(pid_record.get("pid")),
        "pid": pid_record.get("pid") if isinstance(pid_record, dict) else None,
        "pid_record": pid_record,
        "pid_error": pid_error,
        "state_path": str(STATE),
        "saved_state": saved_state,
        "state_error": state_error,
        "campaign_path": str(CAMPAIGN),
        "campaign": campaign_report,
        "campaign_error": campaign_error,
        "projected_status": projected_state.get("status") if projected_state else None,
        "projected_reason": projected_state.get("reason") if projected_state else None,
        "selection": projected_state.get("selection") if projected_state else None,
        "studio": studio,
        "heavy_lock": heavy_lock,
        "resources": resources,
        "stop_requested": STOP_FILE.exists(),
        "poll_seconds": POLL_SECONDS,
        "invoke_workers": False,
        "worker_launches_total": 0,
    }


def print_status() -> int:
    print(json.dumps(status_payload(), indent=2, sort_keys=True))
    return 0


def _selftest_campaign() -> dict[str, Any]:
    version = "selftest.1"
    components = {
        "diag": {"executor_wired": True, "implementation_status": "measured"},
        "codec": {"executor_wired": True, "implementation_status": "measured"},
        "transform": {"executor_wired": True, "implementation_status": "measured"},
        "zero_correction": {"executor_wired": True, "implementation_status": "measured"},
        "kv": {"executor_wired": True, "implementation_status": "measured"},
        "runtime": {"executor_wired": True, "implementation_status": "measured"},
    }
    candidate: dict[str, Any] = {
        "model": "synthetic-7B",
        "params_b": 7.0,
        "active_b": None,
        "moe": False,
        "target_bpw": 2.0,
        "diagnostic": "diag",
        "base": "codec",
        "transform": "transform",
        "correction": "zero_correction",
        "state": "kv",
        "runtime": "runtime",
        "objective": "capability_composite",
        "calibration": "multidomain",
        "seed": 17,
        "campaign_version": version,
        "implementation_status": "measured",
        "launchable": True,
        "blockers": [],
        "resource_class": "resident",
        "dynamic_cost_required": False,
        "fidelity": "F0",
        "status": "pending",
        "proof_state": "planned",
    }
    identity = _candidate_identity(candidate)
    candidate["identity_sha256"] = identity
    candidate["experiment_id"] = f"dr2-{identity[:16]}"
    campaign: dict[str, Any] = {
        "schema": doctor_frontier.CAMPAIGN_SCHEMA,
        "campaign_version": version,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "mode": "plan_only_no_execution",
        "models": [{"label": "synthetic-7B"}],
        "operator_catalog": components,
        "axes": {},
        "search_size": {"explicit_candidates": 1},
        "experiments": [candidate],
        "fidelity_ladder": [{"id": "F0"}],
        "scheduler": {
            "one_heavy_lease": True,
            "safety": "existing heavy lease; normal pressure; zero swap; AC/thermal/disk gates",
        },
        "regime_policy": {},
        "transfer_policy": {},
        "backend_policy": {},
        "campaign_sha256": None,
    }
    campaign["campaign_sha256"] = _hash_value(
        {key: value for key, value in campaign.items() if key not in {"campaign_sha256", "generated_at"}}
    )
    return campaign


def selftest() -> int:
    assert INVOKE_WORKERS is False
    assert WORKER_LAUNCHES_TOTAL == 0
    assert POLL_SECONDS == 120
    campaign = _selftest_campaign()
    report = validate_campaign(campaign)
    assert report["component_launchable_count"] == 1
    selection = select_component_candidate(campaign, {})
    assert selection["selected"]["experiment_id"] == campaign["experiments"][0]["experiment_id"]
    assert selection["allow_unwired"] is False

    green = {
        "ok": True,
        "blockers": [],
        "observed": {
            "pressure_level": 1,
            "pressure_name": "normal",
            "swap_used_mb": 0.0,
            "power_source": "AC Power",
            "thermal_green": True,
            "disk_free_gb": 300.0,
        },
    }
    idle = {"ok": True, "active": False, "owners": []}
    unlocked = {"ok": True, "held": False, "exists": True}
    busy = {"ok": True, "active": True, "owners": [{"pid": 42, "active": True}]}
    held = {"ok": True, "held": True, "exists": True}

    waiting = evaluate_once(
        campaign, report, {}, None, studio=busy, heavy_lock=held, resources=green
    )
    assert waiting["status"] == "waiting-current-studio"
    assert waiting["invoke_workers"] is False
    assert waiting["worker_launches_total"] == 0

    ready = evaluate_once(
        campaign, report, {}, waiting, studio=idle, heavy_lock=unlocked, resources=green
    )
    assert ready["status"] == "candidate-ready-observer-only"
    assert ready["selected_experiment_id"] == selection["selected"]["experiment_id"]
    assert ready["plan_checkpoint"] == waiting["plan_checkpoint"]

    swap_blocked = copy.deepcopy(green)
    swap_blocked["ok"] = False
    swap_blocked["blockers"] = ["swap is not confirmed zero (used_mb=1.0)"]
    swap_blocked["observed"]["swap_used_mb"] = 1.0
    gated = evaluate_once(
        campaign, report, {}, ready, studio=idle, heavy_lock=unlocked, resources=swap_blocked
    )
    assert gated["status"] == "waiting-resource-gate"

    tampered = copy.deepcopy(campaign)
    tampered["experiments"][0]["target_bpw"] = 1.0
    try:
        validate_campaign(tampered)
    except QueueError as exc:
        assert "hash mismatch" in str(exc) or "identity_sha256" in str(exc)
    else:
        raise AssertionError("campaign tamper was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        state_path = Path(temporary) / "state.json"
        _atomic_json(state_path, gated)
        restored = _read_json(state_path)
        assert restored == gated
        assert restored["plan_checkpoint"]["campaign_sha256"] == campaign["campaign_sha256"]

    print("doctor_frontier_queue.py selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start", help="detach the observation-only supervisor")
    subparsers.add_parser("run", help="run the observation-only supervisor in foreground")
    subparsers.add_parser("status", help="print a live read-only JSON status")
    subparsers.add_parser("stop", help="write a cooperative stop checkpoint; send no signals")
    subparsers.add_parser("selftest", help="run synthetic tests without touching live queue state")
    args = parser.parse_args(argv)
    if args.command == "start":
        return start_supervisor()
    if args.command == "run":
        return run_supervisor()
    if args.command == "status":
        return print_status()
    if args.command == "stop":
        return stop_supervisor()
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
