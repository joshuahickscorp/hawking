#!/usr/bin/env python3.12
"""Sealed 25 GB disk and measured 14B parallel-admission successor.

This successor deliberately leaves the campaign plan, runtime specifications,
completed evidence, results, and the controlled-swap v2 authority unchanged.
It validates that predecessor chain, then changes only the in-memory parent
queue disk gate and the launch-time reservation of an exact 14B accelerated
lane.  Existing 150 GB admission records stay truthful historical inputs; a
non-applicable phase-aware relaxation falls back to this explicit 25 GB parent
gate.  The reduced reservation is deliberately narrow: it makes room for one
smaller companion under the unchanged 78 GB aggregate hard stop while the
existing CPU, pressure, swap, thermal, and checkpoint guards remain in force.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
from typing import Any

import doctor_v5_controlled_swap_activation as activation
import doctor_v5_controlled_swap_successor as predecessor


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
ULTRA = ROOT / "reports/condense/doctor_v5_ultra"
STAGE = ULTRA / "staged_acceleration/controlled_swap_v3_disk25"
POLICY = STAGE / "policy.json"
STAGED_MARKER = STAGE / "staged_marker.json"
ACTIVE_MARKER = STAGE / "active_marker.json"
SERVICE_CANDIDATE = STAGE / "service.plist"
SERVICE_BACKUP = STAGE / "predecessor_service.plist"
LAUNCH_AGENT = Path.home() / (
    "Library/LaunchAgents/com.hawking.doctorv5ultra.autoresume.plist"
)
SERVICE_LABEL = "com.hawking.doctorv5ultra.autoresume"
RESERVE_BYTES = 25_000_000_000
PREDECESSOR_RESERVE_BYTES = 150_000_000_000
PARALLEL_14B_RESERVATION_BYTES = 26_000_000_000
PROCESS_BUDGET_BYTES = 78_000_000_000
MIN_CALIBRATION_SAMPLES = 12
CALIBRATION_CELL = "qwen2-5-14b__4bpw__doctor-static"
CALIBRATION_MAX_BYTES = 20_000_000_000
VERSION = "2026-07-15.2"
POLICY_SCHEMA = "hawking.doctor_v5_disk25_successor_policy.v1"
MARKER_SCHEMA = "hawking.doctor_v5_disk25_successor_marker.v1"
MAX_JSON_BYTES = 16 * 1024 * 1024


class Disk25Error(RuntimeError):
    """The reserve successor is stale, mixed, or unsafe to activate."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _read_json(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_size > MAX_JSON_BYTES:
        raise Disk25Error(f"unsafe JSON artifact: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Disk25Error(f"JSON root is not an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_size > MAX_JSON_BYTES:
        raise Disk25Error(f"unsafe bound artifact: {path}")
    raw = path.read_bytes()
    try:
        name = str(path.relative_to(ROOT.resolve()))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _matches(reference: Any, path: Path) -> bool:
    try:
        return isinstance(reference, dict) and reference == _artifact(path)
    except (OSError, Disk25Error):
        return False


def _atomic_bytes(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False).encode() + b"\n")


def _predecessor_chain() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return activation.validate_active_marker(verify_service=False)
    except (activation.ActivationError, OSError, ValueError, KeyError) as exc:
        raise Disk25Error(f"controlled-swap predecessor is invalid: {exc}") from exc


def _service_document() -> bytes:
    document = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [sys.executable, str(SCRIPT), "autoresume"],
        "WorkingDirectory": str(ROOT), "RunAtLoad": True,
        "StartInterval": 30, "ProcessType": "Interactive",
        "StandardOutPath": str(STAGE / "autoresume.log"),
        "StandardErrorPath": str(STAGE / "autoresume.log"),
    }
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True)


def _calibration(state: dict[str, Any]) -> dict[str, Any]:
    """Freeze the latest exact-attempt 14B RSS evidence into the policy."""
    path = ULTRA / "child_resources.jsonl"
    if not path.is_file() or path.is_symlink():
        raise Disk25Error("14B child-resource calibration log is unavailable")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            root_pid, request_sha = row.get("root_pid"), row.get("request_sha256")
            if row.get("cell_id") != CALIBRATION_CELL \
                    or row.get("plan_sha256") != state.get("plan_sha256") \
                    or row.get("process_budget_bytes") != PROCESS_BUDGET_BYTES \
                    or isinstance(root_pid, bool) or not isinstance(root_pid, int) \
                    or not isinstance(request_sha, str) or len(request_sha) != 64 \
                    or isinstance(row.get("tree_rss_bytes"), bool) \
                    or not isinstance(row.get("tree_rss_bytes"), int):
                continue
            groups.setdefault((root_pid, request_sha), []).append(row)
    if not groups:
        raise Disk25Error("no exact 14B attempt can calibrate parallel admission")
    key, rows = max(groups.items(), key=lambda item: item[1][-1]["sampled_at"])
    peak = max(row["tree_rss_bytes"] for row in rows)
    if len(rows) < MIN_CALIBRATION_SAMPLES or not 0 < peak <= CALIBRATION_MAX_BYTES:
        raise Disk25Error("latest 14B attempt is not a bounded measured calibration")
    return {
        "cell_id": CALIBRATION_CELL, "root_pid": key[0],
        "request_sha256": key[1], "sample_count": len(rows),
        "first_sampled_at": rows[0]["sampled_at"],
        "last_sampled_at": rows[-1]["sampled_at"],
        "observed_peak_tree_rss_bytes": peak,
        "reservation_bytes": PARALLEL_14B_RESERVATION_BYTES,
        "headroom_bytes": PARALLEL_14B_RESERVATION_BYTES - peak,
        "aggregate_hard_stop_bytes": PROCESS_BUDGET_BYTES,
    }


def stage() -> dict[str, Any]:
    marker, packet = _predecessor_chain()
    state = _read_json(ULTRA / "queue_state.json")
    if state.get("active_cells") or state.get("active_children"):
        raise Disk25Error("cannot stage while a Doctor child is active")
    if state.get("status") != "drained":
        raise Disk25Error("parallel admission must be staged at a drained checkpoint")
    calibration = _calibration(state)
    document: dict[str, Any] = {
        "schema": POLICY_SCHEMA, "version": VERSION, "created_at": _now(),
        "mode": "pending_execution_parent_gate_supersession",
        "disk_reserve_bytes": RESERVE_BYTES,
        "predecessor_disk_reserve_bytes": PREDECESSOR_RESERVE_BYTES,
        "parallel_14b_reservation_bytes": PARALLEL_14B_RESERVATION_BYTES,
        "calibration": calibration,
        "predecessor_marker": _artifact(activation.production_paths().active_marker),
        "predecessor_marker_sha256": marker["marker_sha256"],
        "predecessor_packet_sha256": packet["packet_sha256"],
        "plan_sha256": state["plan_sha256"],
        "sources": {
            "successor": _artifact(SCRIPT),
            "predecessor_successor": _artifact(activation.production_paths().successor_queue),
            "predecessor_activation": _artifact(Path(activation.__file__).resolve()),
        },
        "mutation_boundary": {
            "plan": False, "runtime_specs": False, "state_history": False,
            "campaign_results": False, "evidence": False,
            "source_deletion": False, "pending_parent_gate_only": True,
            "pending_14b_launch_reservation_only": True,
        },
    }
    document["policy_sha256"] = _hash_value(document)
    STAGE.mkdir(parents=True, exist_ok=True)
    _atomic_json(POLICY, document)
    _atomic_bytes(SERVICE_CANDIDATE, _service_document())
    staged: dict[str, Any] = {
        "schema": MARKER_SCHEMA, "version": VERSION, "prepared_at": _now(),
        "policy": _artifact(POLICY), "policy_sha256": document["policy_sha256"],
        "service": _artifact(SERVICE_CANDIDATE),
        "predecessor_marker_sha256": marker["marker_sha256"],
        "reserve_bytes": RESERVE_BYTES, "marker_sha256": "",
    }
    staged["marker_sha256"] = _hash_value(_without(staged, "marker_sha256"))
    _atomic_json(STAGED_MARKER, staged)
    errors = validate(staged_only=True)
    if errors:
        raise Disk25Error("staged disk25 authority is invalid: " + "; ".join(errors))
    return {"staged": True, "reserve_bytes": RESERVE_BYTES,
            "parallel_14b_reservation_bytes": PARALLEL_14B_RESERVATION_BYTES,
            "calibration_peak_bytes": calibration["observed_peak_tree_rss_bytes"],
            "policy_sha256": document["policy_sha256"],
            "marker_sha256": staged["marker_sha256"]}


def validate(*, staged_only: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        prior_marker, prior_packet = _predecessor_chain()
        policy = _read_json(POLICY)
        marker_path = STAGED_MARKER if staged_only else ACTIVE_MARKER
        marker = _read_json(marker_path)
    except (OSError, ValueError, KeyError, Disk25Error) as exc:
        return [str(exc)]
    if set(policy) != {
        "schema", "version", "created_at", "mode", "disk_reserve_bytes",
        "predecessor_disk_reserve_bytes", "predecessor_marker",
        "predecessor_marker_sha256", "predecessor_packet_sha256", "plan_sha256",
        "parallel_14b_reservation_bytes", "calibration",
        "sources", "mutation_boundary", "policy_sha256",
    } or policy.get("schema") != POLICY_SCHEMA \
            or policy.get("version") != VERSION \
            or policy.get("policy_sha256") != _hash_value(_without(policy, "policy_sha256")):
        errors.append("disk25 policy identity is invalid")
    if policy.get("disk_reserve_bytes") != RESERVE_BYTES \
            or policy.get("predecessor_disk_reserve_bytes") \
            != PREDECESSOR_RESERVE_BYTES:
        errors.append("disk reserve transition is not exactly 150 GB to 25 GB")
    calibration = policy.get("calibration")
    if policy.get("parallel_14b_reservation_bytes") \
            != PARALLEL_14B_RESERVATION_BYTES \
            or not isinstance(calibration, dict) \
            or calibration.get("cell_id") != CALIBRATION_CELL \
            or calibration.get("reservation_bytes") \
            != PARALLEL_14B_RESERVATION_BYTES \
            or calibration.get("aggregate_hard_stop_bytes") \
            != PROCESS_BUDGET_BYTES \
            or not isinstance(calibration.get("sample_count"), int) \
            or calibration.get("sample_count", 0) < MIN_CALIBRATION_SAMPLES \
            or not isinstance(calibration.get("observed_peak_tree_rss_bytes"), int) \
            or not 0 < calibration.get("observed_peak_tree_rss_bytes", 0) \
            <= CALIBRATION_MAX_BYTES \
            or calibration.get("headroom_bytes") \
            != PARALLEL_14B_RESERVATION_BYTES \
                - calibration.get("observed_peak_tree_rss_bytes", 0):
        errors.append("parallel 14B calibration is invalid")
    if policy.get("predecessor_marker_sha256") != prior_marker.get("marker_sha256") \
            or policy.get("predecessor_packet_sha256") \
            != prior_packet.get("packet_sha256") \
            or not _matches(policy.get("predecessor_marker"),
                            activation.production_paths().active_marker):
        errors.append("disk25 predecessor chain changed")
    sources = policy.get("sources")
    if not isinstance(sources, dict) \
            or not _matches(sources.get("successor"), SCRIPT) \
            or not _matches(sources.get("predecessor_successor"),
                            activation.production_paths().successor_queue) \
            or not _matches(sources.get("predecessor_activation"),
                            Path(activation.__file__).resolve()):
        errors.append("disk25 source binding changed")
    boundary = policy.get("mutation_boundary")
    if boundary != {"plan": False, "runtime_specs": False,
                    "state_history": False, "campaign_results": False,
                    "evidence": False, "source_deletion": False,
                    "pending_parent_gate_only": True,
                    "pending_14b_launch_reservation_only": True}:
        errors.append("disk25 mutation boundary changed")
    expected_marker_keys = {"schema", "version", "prepared_at", "policy",
                            "policy_sha256", "service",
                            "predecessor_marker_sha256", "reserve_bytes",
                            "marker_sha256"}
    if set(marker) != expected_marker_keys or marker.get("schema") != MARKER_SCHEMA \
            or marker.get("marker_sha256") \
            != _hash_value(_without(marker, "marker_sha256")) \
            or not _matches(marker.get("policy"), POLICY) \
            or marker.get("policy_sha256") != policy.get("policy_sha256") \
            or marker.get("predecessor_marker_sha256") \
            != prior_marker.get("marker_sha256") \
            or marker.get("reserve_bytes") != RESERVE_BYTES \
            or not _matches(marker.get("service"), SERVICE_CANDIDATE):
        errors.append("disk25 marker identity is invalid")
    if not staged_only:
        try:
            if ACTIVE_MARKER.read_bytes() != STAGED_MARKER.read_bytes():
                errors.append("active disk25 marker differs from staged marker")
            if LAUNCH_AGENT.read_bytes() != SERVICE_CANDIDATE.read_bytes():
                errors.append("installed disk25 service differs from candidate")
        except OSError as exc:
            errors.append(f"disk25 active service is unavailable: {exc}")
    return errors


def install(*, expected_policy_sha256: str) -> dict[str, Any]:
    errors = validate(staged_only=True)
    policy = _read_json(POLICY)
    if errors or policy.get("policy_sha256") != expected_policy_sha256:
        raise Disk25Error("activation key or staged authority is invalid: "
                          + "; ".join(errors))
    state = _read_json(ULTRA / "queue_state.json")
    if state.get("active_cells") or state.get("active_children") \
            or state.get("status") != "drained":
        raise Disk25Error("queue must be cleanly drained before disk25 activation")
    pid = _read_json(ULTRA / "queue.pid.json").get("pid")
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise Disk25Error("predecessor queue owner is still alive")
    if LAUNCH_AGENT.exists():
        _atomic_bytes(SERVICE_BACKUP, LAUNCH_AGENT.read_bytes())
    domain = f"gui/{os.getuid()}"
    subprocess.run(["/bin/launchctl", "bootout", domain, str(LAUNCH_AGENT)],
                   capture_output=True, check=False)
    _atomic_bytes(LAUNCH_AGENT, SERVICE_CANDIDATE.read_bytes())
    _atomic_bytes(ACTIVE_MARKER, STAGED_MARKER.read_bytes())
    # The base control writer is the existing sealed authority for this field.
    predecessor._BASE.set_control("run")
    result = subprocess.run(["/bin/launchctl", "bootstrap", domain,
                             str(LAUNCH_AGENT)], text=True,
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise Disk25Error("launchd refused the disk25 successor")
    subprocess.run(["/bin/launchctl", "kickstart", "-k",
                    f"{domain}/{SERVICE_LABEL}"], capture_output=True, check=False)
    return {"installed": True, "reserve_bytes": RESERVE_BYTES,
            "policy_sha256": policy["policy_sha256"],
            "service": SERVICE_LABEL}


def _delegate() -> int:
    errors = validate(staged_only=False)
    if errors:
        raise Disk25Error("active disk25 authority is invalid: " + "; ".join(errors))
    paths = activation.production_paths()
    prior_marker = _read_json(paths.active_marker)
    prior_policy = _read_json(paths.successor_policy)
    os.environ[predecessor.ENV_MARKER] = str(paths.active_marker.resolve())
    os.environ[predecessor.ENV_MARKER_SHA256] = prior_marker["marker_sha256"]
    os.environ[predecessor.ENV_POLICY] = str(paths.successor_policy.resolve())
    os.environ[predecessor.ENV_POLICY_SHA256] = prior_policy["policy_sha256"]
    original_configure = predecessor.configure
    original_generation_errors = predecessor._successor_generation_errors

    def generation_errors(marker: dict[str, Any], packet: dict[str, Any],
                          policy: dict[str, Any], policy_path: Path, *,
                          verify_installed_service: bool = True) -> list[str]:
        # V3 validates its own installed plist byte-for-byte.  V2 remains the
        # immutable semantic predecessor, but its superseded plist is retained
        # as an artifact rather than remaining installed.
        return original_generation_errors(
            marker, packet, policy, policy_path,
            verify_installed_service=False,
        )

    def configure(overlay: dict[str, Any], policy: dict[str, Any], *,
                  policy_path: Path) -> None:
        original_configure(overlay, policy, policy_path=policy_path)
        if predecessor._BASE.DISK_RESERVE_BYTES != PREDECESSOR_RESERVE_BYTES:
            raise Disk25Error("predecessor reserve changed before supersession")
        predecessor._BASE.DISK_RESERVE_BYTES = RESERVE_BYTES
        original_reservation = predecessor._BASE._cell_reservation

        def measured_reservation(cell: dict[str, Any]) -> int:
            value = original_reservation(cell)
            if cell.get("model_label") == "14B" \
                    and cell.get("model_family") == "qwen2.5-dense" \
                    and cell.get("backend") == "apple-cpu-strand":
                return min(value, PARALLEL_14B_RESERVATION_BYTES)
            return value

        predecessor._BASE._cell_reservation = measured_reservation
        # Successor ownership must bind the actual OS entrypoint.
        predecessor.SCRIPT = SCRIPT

    predecessor._successor_generation_errors = generation_errors
    predecessor.configure = configure
    return int(predecessor.main())


def autoresume() -> int:
    try:
        control = _read_json(ULTRA / "control.json")
        state = _read_json(ULTRA / "queue_state.json")
        if control.get("mode") != "run" or state.get("status") == "complete":
            return 0
        result = subprocess.run([sys.executable, str(SCRIPT), "start"],
                                cwd=str(ROOT), check=False)
        return int(result.returncode)
    except (OSError, ValueError, KeyError, Disk25Error):
        return 2


def status() -> dict[str, Any]:
    errors = validate(staged_only=not ACTIVE_MARKER.exists())
    state = _read_json(ULTRA / "queue_state.json")
    return {"valid": not errors, "errors": errors,
            "reserve_bytes": RESERVE_BYTES,
            "parallel_14b_reservation_bytes": PARALLEL_14B_RESERVATION_BYTES,
            "process_type": "Interactive",
            "active": ACTIVE_MARKER.exists(), "queue_status": state.get("status"),
            "active_cells": state.get("active_cells", [])}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["stage"]:
            result: Any = stage()
        elif len(args) == 2 and args[0] == "install":
            result = install(expected_policy_sha256=args[1])
        elif args == ["autoresume"]:
            return autoresume()
        elif args == ["status"]:
            result = status()
        elif args and args[0] in {"start", "run", "pause", "resume", "drain",
                                "readiness"}:
            return _delegate()
        else:
            raise Disk25Error("usage: stage | install POLICY_SHA256 | status | "
                              "autoresume | start | run --nonce NONCE")
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except (Disk25Error, predecessor.SuccessorError, OSError, ValueError,
            KeyError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
