#!/usr/bin/env python3.12
"""Crash-safe activation for the unbound Doctor V5 controlled-swap successor.

The successor changes policy/service authority only.  Plan, queue state,
campaign, control, PID records, runtime specs, evidence, and results are
compare-and-swap inputs and are never write targets.  The commit point is a
separate successor marker; failures before that point restore service artifacts
only, while failures after it recover forward.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Protocol


ROOT = Path(__file__).resolve().parents[2]
VERSION = "2026-07-15.1"
PACKET_SCHEMA = "hawking.doctor_v5_controlled_swap_successor_packet.v1"
MARKER_SCHEMA = "hawking.doctor_v5_controlled_swap_successor_marker.v1"
JOURNAL_SCHEMA = "hawking.doctor_v5_controlled_swap_activation_journal.v1"
WAL_SCHEMA = "hawking.doctor_v5_controlled_swap_activation_wal.v1"
PHASE_GATE_SCHEMA = "hawking.doctor_v5_controlled_swap_phase_gate.v1"
PHASE_API_SCHEMA = "hawking.doctor_v5_phase_gate_api.v1"
SERVICE_LABEL = "com.hawking.doctorv5ultra.autoresume"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_HASHED_RESULT_JSON_BYTES = 32 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")

POLICY_SCHEMA = "hawking.doctor_v5_controlled_swap_successor_policy.v1"
POLICY_MODE = "pending_only_operational_supersession"
POLICY: dict[str, Any] = {
    "swap_used_mb_max": 512.0,
    "swap_boundary": "absolute_inclusive",
    "required_pressure": "normal",
    "pressure_level_required": 1,
    "ram_capacity_credit_bytes": 0,
    "preserve_ac_power_gate": True,
    "preserve_thermal_gate": True,
}
POLICY_SHA256 = hashlib.sha256(json.dumps(
    POLICY, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode()).hexdigest()

HARD_CUT_LABELS = (
    "after:journal-prepared",
    "after:old-service-bootout",
    "after:new-service-installed",
    "after:marker-commit",
    "after:new-service-bootstrap",
    "after:new-service-kickstart",
    "after:journal-active",
)
MARKER_KEYS = frozenset({
    "schema", "version", "generation_id", "prepared_at", "packet",
    "successor_policy", "policy_sha256", "predecessor_marker",
    "predecessor_marker_sha256", "predecessor_overlay",
    "predecessor_overlay_sha256", "activation_source", "successor_queue",
    "successor_autoresume",
    "phase_gate", "phase_gate_declaration", "phase_aware_disk_gate",
    "phase_receipt_root", "service_candidate", "activation_snapshot_sha256",
    "result_mutation_permitted", "evidence_mutation_permitted", "marker_sha256",
})


class ActivationError(RuntimeError):
    """The successor transaction is stale, unsafe, or ambiguous."""


def _fault(_label: str) -> None:
    """Production no-op; patched by isolated hard-cut tests."""


@dataclass(frozen=True)
class Paths:
    root: Path
    ultra: Path
    old_marker: Path
    overlay: Path
    accelerated_queue: Path
    stacked_admission: Path
    forward_journal: Path
    pending_runtime_packet: Path
    plan: Path
    state: Path
    campaign: Path
    control: Path
    pid_record: Path
    results: Path
    queue_lock: Path
    heavy_lock: Path
    successor_queue: Path
    successor_autoresume: Path
    activation_source: Path
    stage_root: Path
    phase_gate: Path
    phase_receipt_root: Path
    successor_policy: Path
    packet: Path
    staged_marker: Path
    active_marker: Path
    staged_service: Path
    service_backup: Path
    journal: Path
    wal_root: Path
    transaction_lock: Path
    launch_agent: Path


def production_paths(root: Path = ROOT, *, launch_agent: Path | None = None) -> Paths:
    root = root.resolve()
    ultra = root / "reports/condense/doctor_v5_ultra"
    stage = ultra / "staged_acceleration/controlled_swap_v2"
    return Paths(
        root=root, ultra=ultra,
        old_marker=ultra / "staged_acceleration/active_stack.json",
        overlay=ultra / "staged_acceleration/stacked_admission_overlay.json",
        accelerated_queue=root / "tools/condense/doctor_v5_ultra_accelerated_queue.py",
        stacked_admission=root / "tools/condense/doctor_v5_stacked_admission.py",
        forward_journal=ultra / (
            "staged_acceleration/forward_recovery_v1/activation_journal.json"
        ),
        pending_runtime_packet=ultra / "staged_acceleration/pending_runtime_packet.json",
        plan=ultra / "campaign_plan.json", state=ultra / "queue_state.json",
        campaign=ultra / "campaign.json", control=ultra / "control.json",
        pid_record=ultra / "queue.pid.json", results=ultra / "results",
        queue_lock=ultra / "queue.lock",
        heavy_lock=root / "reports/cron/studio_heavy.lock",
        successor_queue=root / "tools/condense/doctor_v5_controlled_swap_successor.py",
        successor_autoresume=root / "tools/condense/doctor_v5_controlled_swap_autoresume.py",
        activation_source=Path(__file__).resolve(),
        stage_root=stage, phase_gate=stage / "phase_gate.json",
        phase_receipt_root=stage / "phase_receipts",
        successor_policy=stage / "successor_policy.json",
        packet=stage / "pending_generation.json",
        staged_marker=stage / "staged_marker.json",
        active_marker=stage / "active_marker.json",
        staged_service=stage / "successor_service.plist",
        service_backup=stage / "service_backup.plist",
        journal=stage / "activation_journal.json", wal_root=stage / "wal",
        transaction_lock=stage / "transaction.lock",
        launch_agent=launch_agent or Path.home() / (
            "Library/LaunchAgents/com.hawking.doctorv5ultra.autoresume.plist"
        ),
    )


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _stable_bytes(path: Path, *, maximum: int | None = None) -> bytes:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ActivationError(f"bound path is not a regular file: {path}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ActivationError(f"cannot open bound file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if maximum is not None and before.st_size > maximum:
            raise ActivationError(f"bound file is too large: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        if identity(before) != identity(after):
            raise ActivationError(f"bound file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_stable_bytes(path, maximum=MAX_JSON_BYTES))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ActivationError(f"cannot decode JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActivationError(f"JSON root is not an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    raw = _stable_bytes(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    value = _artifact(path)
    value["path"] = str(path.resolve().relative_to(root.resolve()))
    return value


def _artifact_matches(row: Any, expected: Path) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    try:
        return Path(row["path"]).resolve(strict=True) == expected.resolve(strict=True) \
            and _artifact(expected) == row
    except (OSError, KeyError, TypeError, ValueError, ActivationError):
        return False


def _sealed(document: dict[str, Any], field: str) -> bool:
    return document.get(field) == _hash_value(_without(document, field))


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, raw: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                  ensure_ascii=False).encode() + b"\n")


def _structural_rows(root: Path) -> list[tuple[Any, ...]]:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ActivationError(f"result tree unavailable: {exc}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ActivationError("result tree root is not a real directory")
    rows: list[tuple[Any, ...]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort(); files.sort()
        for name in directories:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ActivationError(f"unsafe result-tree directory: {path}")
            rel = path.relative_to(root).as_posix()
            rows.append(("d", rel, info.st_dev, info.st_ino, info.st_mtime_ns))
        for name in files:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ActivationError(f"unsafe result-tree file: {path}")
            rel = path.relative_to(root).as_posix()
            content_sha: str | None = None
            evidence_name = rel.lower()
            if path.suffix in {".json", ".jsonl"} \
                    and any(token in evidence_name for token in ("receipt", "checkpoint")) \
                    and info.st_size <= MAX_HASHED_RESULT_JSON_BYTES:
                content_sha = hashlib.sha256(_stable_bytes(path)).hexdigest()
            rows.append(("f", rel, info.st_dev, info.st_ino, info.st_size,
                         info.st_mtime_ns, content_sha))
    return rows


def result_tree_identity(root: Path) -> dict[str, Any]:
    """Cheap stable inventory; never reads large model/reconstruction payloads."""
    before = _structural_rows(root)
    after = _structural_rows(root)
    if before != after:
        raise ActivationError("result tree changed while inventorying")
    return {
        "path": str(root.resolve()),
        "tree_sha256": _hash_value(before),
        "file_count": sum(row[0] == "f" for row in before),
        "directory_count": sum(row[0] == "d" for row in before),
        "total_file_bytes": sum(row[4] for row in before if row[0] == "f"),
        "large_payload_content_hashed": False,
        "json_receipt_content_hashed": True,
    }


def _tree_matches(row: Any, root: Path) -> bool:
    try:
        return isinstance(row, dict) and row == result_tree_identity(root)
    except ActivationError:
        return False


def _phase_declaration(queue_path: Path) -> dict[str, Any]:
    """Call the successor's reviewed declaration API; do not invent a parallel one."""
    name = "_doctor_v5_controlled_swap_successor_activation_bound"
    specification = importlib.util.spec_from_file_location(name, queue_path)
    if specification is None or specification.loader is None:
        raise ActivationError("cannot load successor queue declaration API")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
        factory = getattr(module, "phase_gate_declaration")
        phase_path = getattr(module, "PHASE_GATE_PATH")
        declaration = factory(phase_path)
    except Exception as exc:
        raise ActivationError(f"successor phase-gate declaration failed: {exc}") from exc
    if not isinstance(declaration, dict) or set(declaration) != {
            "api_schema", "module", "install_callable"} \
            or declaration.get("api_schema") != PHASE_API_SCHEMA \
            or declaration.get("install_callable") != "install_phase_gate":
        raise ActivationError("successor phase-gate declaration is invalid")
    return declaration


def _queue_policy_errors(queue_path: Path, policy: dict[str, Any],
                         policy_path: Path, *, deep_predecessor: bool) -> list[str]:
    name = "_doctor_v5_controlled_swap_successor_policy_bound"
    specification = importlib.util.spec_from_file_location(name, queue_path)
    if specification is None or specification.loader is None:
        return ["cannot load successor queue policy validator"]
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
        errors = module.validate_policy(policy, policy_path=policy_path,
                                        deep_predecessor=deep_predecessor)
    except Exception as exc:
        return [f"successor queue policy validation failed: {exc}"]
    return list(errors) if isinstance(errors, list) else [
        "successor queue policy validator returned a non-list"]


def _resource_errors(sample: Any) -> list[str]:
    if not isinstance(sample, dict):
        return ["resource sample is absent"]
    swap = sample.get("swap_used_mb")
    errors: list[str] = []
    if sample.get("pressure_level") != 1 or sample.get("pressure_name") != "normal":
        errors.append("memory pressure is not exact numeric level 1/normal")
    if isinstance(swap, bool) or not isinstance(swap, (int, float)) \
            or not math.isfinite(float(swap)) or not 0 <= float(swap) <= 512.0:
        errors.append("swap is outside the absolute inclusive 512 MB cap")
    if sample.get("power_source") != "AC":
        errors.append("host is not on AC power")
    if sample.get("thermal_green") is not True:
        errors.append("thermal state is not green")
    return errors


class ResourceProbe(Protocol):
    def sample(self) -> dict[str, Any]: ...


class HostResourceProbe:
    def _run(self, argv: list[str]) -> str:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=8,
                                check=False)
        if result.returncode != 0:
            raise ActivationError(f"resource probe failed: {' '.join(argv)}")
        return (result.stdout + result.stderr).strip()

    def sample(self) -> dict[str, Any]:
        pressure_raw = self._run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
        swap_raw = self._run(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
        power = self._run(["/usr/bin/pmset", "-g", "batt"])
        thermal = self._run(["/usr/bin/pmset", "-g", "therm"])
        match = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)M", swap_raw)
        if match is None:
            raise ActivationError("cannot parse swap probe")
        warning = re.search(
            r"(?:thermal warning level|performance warning level|cpu power status)"
            r"\s*:\s*[1-9]", thermal, re.IGNORECASE,
        )
        try:
            pressure = int(pressure_raw.strip())
        except ValueError as exc:
            raise ActivationError("cannot parse numeric memory-pressure level") from exc
        return {"pressure_level": pressure,
                "pressure_name": "normal" if pressure == 1 else "non-normal",
                "swap_used_mb": float(match.group(1)),
                "power_source": "AC" if "AC Power" in power else "battery",
                "thermal_green": warning is None}


class ServiceController(Protocol):
    def is_loaded(self, label: str) -> bool: ...
    def bootout(self, label: str) -> None: ...
    def bootstrap(self, plist: Path) -> None: ...
    def kickstart(self, label: str) -> None: ...


class LaunchctlService:
    @property
    def domain(self) -> str:
        return f"gui/{os.getuid()}"

    def _run(self, argv: list[str], *, allow_missing: bool = False) -> None:
        row = subprocess.run(argv, capture_output=True, text=True, check=False)
        if row.returncode and not (allow_missing and row.returncode in {3, 113}):
            raise ActivationError(
                f"service command failed ({row.returncode}): {' '.join(argv)}: "
                f"{(row.stdout + row.stderr).strip()[-1000:]}"
            )

    def is_loaded(self, label: str) -> bool:
        row = subprocess.run(["/bin/launchctl", "print", f"{self.domain}/{label}"],
                             capture_output=True, text=True, check=False)
        return row.returncode == 0

    def bootout(self, label: str) -> None:
        self._run(["/bin/launchctl", "bootout", f"{self.domain}/{label}"],
                  allow_missing=True)

    def bootstrap(self, plist: Path) -> None:
        self._run(["/bin/launchctl", "bootstrap", self.domain, str(plist)])

    def kickstart(self, label: str) -> None:
        self._run(["/bin/launchctl", "kickstart", "-k",
                   f"{self.domain}/{label}"])


def _forward_wal_head(paths: Paths, journal: dict[str, Any]) -> dict[str, Any]:
    if journal.get("schema") != "hawking.doctor_v5_forward_recovery_journal.v2" \
            or not _sealed(journal, "journal_sha256") \
            or journal.get("status") != "active" or journal.get("phase") != "active":
        raise ActivationError("forward recovery is not at its active terminal head")
    index = journal.get("wal_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1 \
            or not _valid_sha(journal.get("wal_entry_sha256")):
        raise ActivationError("forward journal WAL pointer is invalid")
    backup = Path(journal.get("backup_root", "")).resolve(strict=True)
    backup.relative_to((paths.ultra / "staged_acceleration/forward_recovery_v1").resolve())
    head_path = backup / "wal" / f"{index:08d}.json"
    head = _read_json(head_path)
    if head.get("schema") != "hawking.doctor_v5_forward_recovery_wal_entry.v1" \
            or head.get("index") != index \
            or head.get("entry_sha256") != journal["wal_entry_sha256"] \
            or not _sealed(head, "entry_sha256") \
            or head.get("forward_packet_sha256") != journal.get("forward_packet_sha256"):
        raise ActivationError("forward journal WAL head is invalid")
    return _artifact(head_path)


def _document_binding(path: Path, schema: str, hash_field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _read_json(path)
    if document.get("schema") != schema or not _sealed(document, hash_field):
        raise ActivationError(f"invalid source document identity: {path}")
    return document, _artifact(path)


def _snapshot(paths: Paths) -> dict[str, Any]:
    plan, plan_ref = _document_binding(
        paths.plan, "hawking.doctor_v5_ultra_campaign_plan.v1", "plan_sha256")
    state, state_ref = _document_binding(
        paths.state, "hawking.doctor_v5_ultra_queue_state.v1", "state_sha256")
    campaign, campaign_ref = _document_binding(
        paths.campaign, "hawking.doctor_v5_ultra_campaign.v1", "campaign_sha256")
    control, control_ref = _document_binding(
        paths.control, "hawking.doctor_v5_ultra_control.v1", "control_sha256")
    pid, pid_ref = _document_binding(
        paths.pid_record, "hawking.doctor_v5_ultra_queue_pid.v1", "pid_record_sha256")
    plan_sha = plan["plan_sha256"]
    if any(row.get("plan_sha256") != plan_sha for row in (state, campaign, control, pid)):
        raise ActivationError("current plan/state/campaign/control/PID generations differ")
    if state.get("active_cells") != [] or state.get("active_children") != {}:
        raise ActivationError("successor transition is not at a quiescent state")

    overlay, overlay_ref = _document_binding(
        paths.overlay, "hawking.doctor_v5_stacked_admission_overlay.v1",
        "overlay_sha256")
    old_marker, old_marker_ref = _document_binding(
        paths.old_marker, "hawking.doctor_v5_acceleration_active_marker.v1",
        "marker_sha256")
    accelerated_ref = _artifact(paths.accelerated_queue)
    if old_marker.get("overlay_sha256") != overlay["overlay_sha256"] \
            or Path(old_marker.get("overlay_path", "")).resolve(strict=True) \
            != paths.overlay.resolve(strict=True) \
            or old_marker.get("accelerated_queue") != accelerated_ref:
        raise ActivationError("old marker/overlay/accelerated source generation is mixed")

    pending, pending_ref = _document_binding(
        paths.pending_runtime_packet,
        "hawking.doctor_v5_acceleration_pending_runtime.v1", "packet_sha256")
    if pending.get("plan_sha256") != plan_sha \
            or old_marker.get("pending_runtime_generation_sha256") \
            != pending.get("packet_sha256"):
        raise ActivationError("pending runtime packet differs from active generation")
    forward, forward_ref = _document_binding(
        paths.forward_journal,
        "hawking.doctor_v5_forward_recovery_journal.v2", "journal_sha256")
    if forward.get("plan_sha256") != plan_sha:
        raise ActivationError("forward journal plan differs")
    wal_head = _forward_wal_head(paths, forward)
    return {
        "plan": plan_ref, "plan_sha256": plan_sha,
        "state": state_ref, "state_sha256": state["state_sha256"],
        "campaign": campaign_ref, "campaign_sha256": campaign["campaign_sha256"],
        "control": control_ref, "control_sha256": control["control_sha256"],
        "pid_record": pid_ref, "pid_record_sha256": pid["pid_record_sha256"],
        "results": result_tree_identity(paths.results),
        "old_marker": old_marker_ref, "old_marker_sha256": old_marker["marker_sha256"],
        "overlay": overlay_ref, "overlay_sha256": overlay["overlay_sha256"],
        "accelerated_queue": accelerated_ref,
        "forward_journal": forward_ref,
        "forward_journal_sha256": forward["journal_sha256"],
        "forward_wal_head": wal_head,
        "forward_wal_entry_sha256": forward["wal_entry_sha256"],
        "pending_runtime_packet": pending_ref,
        "pending_runtime_packet_sha256": pending["packet_sha256"],
    }


def _snapshot_matches(snapshot: dict[str, Any], paths: Paths) -> bool:
    try:
        current = _snapshot(paths)
    except (ActivationError, OSError, KeyError, TypeError, ValueError):
        return False
    return current == snapshot


def _phase_gate_errors(gate: Any, snapshot: dict[str, Any],
                       declaration: dict[str, Any]) -> list[str]:
    required = {
        "schema", "version", "ready", "quiescent", "active_child_count",
        "heavy_owner_count", "plan_sha256", "state_sha256", "campaign_sha256",
        "old_marker_sha256", "forward_wal_entry_sha256",
        "pending_runtime_packet_sha256", "policy", "policy_sha256",
        "phase_gate", "phase_aware_disk_gate", "resource_sample", "gate_sha256",
    }
    if not isinstance(gate, dict) or set(gate) != required:
        return ["phase gate keys are invalid"]
    errors: list[str] = []
    if gate.get("schema") != PHASE_GATE_SCHEMA or gate.get("version") != VERSION \
            or not _sealed(gate, "gate_sha256"):
        errors.append("phase gate identity is invalid")
    expected = {
        "plan_sha256": snapshot["plan_sha256"],
        "state_sha256": snapshot["state_sha256"],
        "campaign_sha256": snapshot["campaign_sha256"],
        "old_marker_sha256": snapshot["old_marker_sha256"],
        "forward_wal_entry_sha256": snapshot["forward_wal_entry_sha256"],
        "pending_runtime_packet_sha256": snapshot["pending_runtime_packet_sha256"],
    }
    if any(gate.get(name) != value for name, value in expected.items()):
        errors.append("phase gate is replayed across a source generation")
    if gate.get("ready") is not True or gate.get("quiescent") is not True \
            or gate.get("active_child_count") != 0 or gate.get("heavy_owner_count") != 0:
        errors.append("phase gate is not owner-free and quiescent")
    if gate.get("policy") != POLICY or gate.get("policy_sha256") != POLICY_SHA256:
        errors.append("phase gate policy is not the exact controlled-swap bridge")
    if gate.get("phase_gate") != declaration:
        errors.append("phase gate API declaration differs from the successor queue")
    if not isinstance(gate.get("phase_aware_disk_gate"), dict):
        errors.append("phase-aware disk policy config is absent")
    errors.extend(_resource_errors(gate.get("resource_sample")))
    return errors


def phase_gate_document(snapshot: dict[str, Any], declaration: dict[str, Any],
                        resource_sample: dict[str, Any],
                        phase_aware_disk_gate: dict[str, Any]) -> dict[str, Any]:
    """API for a separate cheap phase-gate producer; this function does not write."""
    value: dict[str, Any] = {
        "schema": PHASE_GATE_SCHEMA, "version": VERSION,
        "ready": True, "quiescent": True, "active_child_count": 0,
        "heavy_owner_count": 0, "plan_sha256": snapshot["plan_sha256"],
        "state_sha256": snapshot["state_sha256"],
        "campaign_sha256": snapshot["campaign_sha256"],
        "old_marker_sha256": snapshot["old_marker_sha256"],
        "forward_wal_entry_sha256": snapshot["forward_wal_entry_sha256"],
        "pending_runtime_packet_sha256": snapshot["pending_runtime_packet_sha256"],
        "policy": copy.deepcopy(POLICY), "policy_sha256": POLICY_SHA256,
        "phase_gate": copy.deepcopy(declaration),
        "phase_aware_disk_gate": copy.deepcopy(phase_aware_disk_gate),
        "resource_sample": copy.deepcopy(resource_sample),
    }
    value["gate_sha256"] = _hash_value(value)
    return value


def _phase_policy_bindings(paths: Paths, snapshot: dict[str, Any],
                           declaration: dict[str, Any]) -> dict[str, Any]:
    plan = _read_json(paths.plan); state = _read_json(paths.state)
    cells = plan.get("cells"); state_rows = state.get("cells")
    if not isinstance(cells, list) or not isinstance(state_rows, dict):
        raise ActivationError("plan/state cannot issue phase bindings")
    plan_file_sha = snapshot["plan"]["sha256"]
    rows: list[dict[str, Any]] = []
    terminal = {"complete", "negative", "unsupported"}
    for cell in sorted((row for row in cells if isinstance(row, dict)),
                       key=lambda row: str(row.get("cell_id", ""))):
        cell_id = cell.get("cell_id"); state_row = state_rows.get(cell_id)
        if not isinstance(cell_id, str) or not isinstance(state_row, dict) \
                or state_row.get("status") in terminal \
                or cell.get("runtime_spec_schema") \
                != "hawking.doctor_v5_strand_ladder_spec.v1" \
                or cell.get("adapter_id") \
                != "doctor-v5-strand-ladder-qwen25-dense" \
                or cell.get("command") != "condense_control" \
                or cell.get("branch") != "codec_control":
            continue
        raw_spec = cell.get("runtime_spec_path")
        if not isinstance(raw_spec, str) or not raw_spec:
            continue
        spec_path = Path(raw_spec)
        if not spec_path.is_absolute(): spec_path = paths.root / spec_path
        if not spec_path.is_file() or spec_path.is_symlink():
            continue
        spec = _read_json(spec_path)
        if spec.get("schema") != "hawking.doctor_v5_strand_ladder_spec.v1" \
                or spec.get("adapter_id") \
                != "doctor-v5-strand-ladder-qwen25-dense" \
                or spec.get("operation") != "condense_control":
            continue
        admission = cell.get("admission")
        if not isinstance(admission, dict):
            continue
        scratch = admission.get("recommended_scratch_bytes")
        reserve = admission.get("disk_reserve_bytes")
        projection = cell.get("projected_output_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (scratch, reserve, projection)) \
                or not _valid_sha(cell.get("cell_identity_sha256")) \
                or not _valid_sha(spec.get("program_spec_sha256")):
            continue
        rows.append({
            "plan_path": str(paths.plan.resolve()),
            "plan_file_sha256": plan_file_sha,
            "plan_sha256": snapshot["plan_sha256"],
            "plan_cell_sha256": _hash_value(cell), "cell_id": cell_id,
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "runtime_spec_path": str(spec_path.resolve()),
            "runtime_spec_file_sha256": _artifact(spec_path)["sha256"],
            "program_spec_sha256": spec["program_spec_sha256"],
            "execution_output_root": str((paths.results / cell_id).resolve()),
            "disk_reserve_bytes": reserve, "declared_scratch_bytes": scratch,
            "frozen_projected_output_bytes": projection,
        })
    if not rows:
        raise ActivationError("no pending exact runtime cells can receive a phase gate")
    module = declaration["module"]
    ledger_path = paths.root / "tools/condense/doctor_v5_remaining_scratch_ledger.py"
    return {
        "schema": PHASE_API_SCHEMA, "enabled": True,
        "module_sha256": module["sha256"],
        "ledger_module_sha256": _artifact(ledger_path)["sha256"],
        "ram_credit_bytes": 0, "bindings": rows,
    }


def issue_phase_gate(*, paths: Paths | None = None,
                     probe: ResourceProbe | None = None,
                     _leases_held: bool = False) -> dict[str, Any]:
    """Issue the reproducible gate only while queue/heavy/transaction leases are free."""
    paths = paths or production_paths(); probe = probe or HostResourceProbe()
    if not _leases_held:
        with _exclusive_leases(paths):
            return issue_phase_gate(paths=paths, probe=probe, _leases_held=True)
    if paths.packet.exists() or paths.journal.exists() or paths.active_marker.exists():
        raise ActivationError("phase gate cannot replace existing transaction authority")
    snapshot = _snapshot(paths)
    declaration = _phase_declaration(paths.successor_queue)
    sample = probe.sample(); errors = _resource_errors(sample)
    if errors:
        raise ActivationError("phase gate resource probe failed: " + "; ".join(errors))
    config = _phase_policy_bindings(paths, snapshot, declaration)
    document = phase_gate_document(snapshot, declaration, sample, config)
    paths.stage_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths.phase_gate, document)
    if _read_json(paths.phase_gate) != document:
        raise ActivationError("phase gate differs after durable write")
    return document


def _service_document(paths: Paths, policy_sha256: str) -> bytes:
    document = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [sys.executable, str(paths.successor_autoresume.resolve())],
        "WorkingDirectory": str(paths.root), "RunAtLoad": True,
        "StartInterval": 30,
        "StandardOutPath": str(paths.stage_root / "autoresume.log"),
        "StandardErrorPath": str(paths.stage_root / "autoresume.log"),
        "EnvironmentVariables": {
            "DOCTOR_V5_CONTROLLED_SWAP_MARKER": str(paths.active_marker),
            "DOCTOR_V5_CONTROLLED_SWAP_POLICY": str(paths.successor_policy),
            "DOCTOR_V5_CONTROLLED_SWAP_POLICY_SHA256": policy_sha256,
        },
    }
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True)


def _packet_errors(packet: Any, paths: Paths, *, live: bool,
                   probe: ResourceProbe | None = None) -> list[str]:
    required = {
        "schema", "version", "created_at", "generation_id", "snapshot",
        "policy", "policy_sha256", "successor_policy", "sources", "phase_gate",
        "phase_receipt_root", "service", "mutation_boundary", "packet_sha256",
    }
    if not isinstance(packet, dict) or set(packet) != required:
        return ["successor packet keys are invalid"]
    errors: list[str] = []
    if packet.get("schema") != PACKET_SCHEMA or packet.get("version") != VERSION \
            or not isinstance(packet.get("generation_id"), str) \
            or not packet["generation_id"] or not _sealed(packet, "packet_sha256"):
        errors.append("successor packet identity is invalid")
    if packet.get("policy") != POLICY:
        errors.append("successor packet controlled-swap policy changed")
    if not _artifact_matches(packet.get("successor_policy"), paths.successor_policy):
        errors.append("successor policy artifact changed")
    else:
        try:
            policy_document = _read_json(paths.successor_policy)
            if policy_document.get("schema") != POLICY_SCHEMA \
                    or not _sealed(policy_document, "policy_sha256") \
                    or policy_document.get("policy") != POLICY \
                    or packet.get("policy_sha256") != policy_document.get("policy_sha256"):
                errors.append("successor policy semantic identity changed")
            errors.extend(_queue_policy_errors(paths.successor_queue,
                                                policy_document,
                                                paths.successor_policy,
                                                deep_predecessor=live))
        except ActivationError as exc:
            errors.append(str(exc))
    boundary = packet.get("mutation_boundary")
    if boundary != {
        "policy_only": True, "plan_mutation_permitted": False,
        "state_mutation_permitted": False, "campaign_mutation_permitted": False,
        "control_mutation_permitted": False, "pid_mutation_permitted": False,
        "result_mutation_permitted": False, "evidence_mutation_permitted": False,
        "runtime_spec_mutation_permitted": False, "source_deletion_permitted": False,
        "old_marker_mutation_permitted": False,
    }:
        errors.append("successor mutation boundary is not policy/service only")
    sources = packet.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
            "activation_source", "successor_queue", "successor_autoresume",
            "phase_gate_declaration"}:
        errors.append("successor source bindings are invalid")
    else:
        if not _artifact_matches(sources["activation_source"],
                                 paths.activation_source):
            errors.append("successor activation source changed")
        if not _artifact_matches(sources["successor_queue"], paths.successor_queue):
            errors.append("successor queue source changed")
        if not _artifact_matches(sources["successor_autoresume"],
                                 paths.successor_autoresume):
            errors.append("successor autoresume source changed")
        try:
            if _phase_declaration(paths.successor_queue) \
                    != sources["phase_gate_declaration"]:
                errors.append("successor phase-gate declaration changed")
        except ActivationError as exc:
            errors.append(str(exc))
    gate_ref = packet.get("phase_gate")
    if not _artifact_matches(gate_ref, paths.phase_gate):
        errors.append("phase-gate receipt changed")
    else:
        try:
            gate = _read_json(paths.phase_gate)
            errors.extend(_phase_gate_errors(
                gate, packet.get("snapshot", {}),
                sources.get("phase_gate_declaration", {}) if isinstance(sources, dict) else {},
            ))
        except ActivationError as exc:
            errors.append(str(exc))
    try:
        receipt_root = Path(packet.get("phase_receipt_root", "")).resolve(strict=True)
        receipt_root.relative_to(paths.stage_root.resolve(strict=True))
        if receipt_root != paths.phase_receipt_root.resolve(strict=True) \
                or not receipt_root.is_dir() or receipt_root.is_symlink():
            errors.append("phase receipt root is invalid")
    except (OSError, TypeError, ValueError):
        errors.append("phase receipt root is invalid")
    service = packet.get("service")
    if not isinstance(service, dict) or set(service) != {
            "label", "target", "candidate", "preexisting", "was_loaded"} \
            or service.get("label") != SERVICE_LABEL \
            or service.get("target") != str(paths.launch_agent.resolve()) \
            or not _artifact_matches(service.get("candidate"), paths.staged_service) \
            or not isinstance(service.get("preexisting"), (dict, type(None))) \
            or not isinstance(service.get("was_loaded"), bool):
        errors.append("successor service binding is invalid")
    elif live and service["preexisting"] is not None \
            and not _artifact_matches(service["preexisting"], paths.launch_agent):
        errors.append("predecessor service artifact changed before activation")
    snapshot = packet.get("snapshot")
    if not isinstance(snapshot, dict):
        errors.append("activation snapshot is absent")
    elif live and not _snapshot_matches(snapshot, paths):
        errors.append("plan/state/campaign/control/PID/results or predecessor chain changed")
    if live and probe is not None:
        try:
            errors.extend(_resource_errors(probe.sample()))
        except (ActivationError, OSError, ValueError) as exc:
            errors.append(f"fresh resource preflight failed: {exc}")
    return errors


def stage(*, paths: Paths | None = None, service: ServiceController | None = None,
          probe: ResourceProbe | None = None,
          _leases_held: bool = False) -> dict[str, Any]:
    paths = paths or production_paths(); service = service or LaunchctlService()
    probe = probe or HostResourceProbe()
    if not _leases_held:
        with _exclusive_leases(paths):
            return stage(paths=paths, service=service, probe=probe,
                         _leases_held=True)
    if paths.active_marker.exists() or paths.journal.exists() or paths.packet.exists():
        raise ActivationError("successor stage already contains transaction authority")
    snapshot = _snapshot(paths)
    declaration = _phase_declaration(paths.successor_queue)
    gate = _read_json(paths.phase_gate)
    gate_errors = _phase_gate_errors(gate, snapshot, declaration)
    sample = probe.sample(); gate_errors.extend(_resource_errors(sample))
    if gate_errors:
        raise ActivationError("phase/resource gate failed: " + "; ".join(gate_errors))
    paths.stage_root.mkdir(parents=True, exist_ok=True)
    paths.phase_receipt_root.mkdir(parents=True, exist_ok=True)
    _fsync_dir(paths.stage_root)
    predecessor = {
        "accelerated_queue": _relative_artifact(paths.accelerated_queue, paths.root),
        "stacked_admission": _relative_artifact(paths.stacked_admission, paths.root),
        "active_marker": _relative_artifact(paths.old_marker, paths.root),
        "admission_overlay": _relative_artifact(paths.overlay, paths.root),
        "marker_sha256": snapshot["old_marker_sha256"],
        "overlay_sha256": snapshot["overlay_sha256"],
    }
    policy_document: dict[str, Any] = {
        "schema": POLICY_SCHEMA, "version": VERSION, "created_at": _now(),
        "mode": POLICY_MODE,
        "operational_root": str(paths.stage_root.resolve().relative_to(paths.root)),
        "predecessor": predecessor, "policy": copy.deepcopy(POLICY),
        "phase_gate": declaration,
        "phase_aware_disk_gate": copy.deepcopy(gate["phase_aware_disk_gate"]),
        "promotion": {
            "automatic_activation_permitted": False,
            "completed_evidence_mutation_permitted": False,
            "runtime_spec_mutation_permitted": False,
            "result_mutation_permitted": False, "pending_cells_only": True,
        },
    }
    policy_document["policy_sha256"] = _hash_value(policy_document)
    _atomic_json(paths.successor_policy, policy_document)
    policy_errors = _queue_policy_errors(paths.successor_queue, policy_document,
                                         paths.successor_policy,
                                         deep_predecessor=True)
    if policy_errors:
        raise ActivationError("queue rejected successor policy: "
                              + "; ".join(policy_errors))
    _atomic_bytes(paths.staged_service,
                  _service_document(paths, policy_document["policy_sha256"]),
                  mode=0o600)
    preexisting = _artifact(paths.launch_agent) if paths.launch_agent.exists() else None
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA, "version": VERSION, "created_at": _now(),
        "generation_id": hashlib.sha256(_canonical({
            "snapshot": snapshot, "policy_sha256": POLICY_SHA256,
            "gate": gate["gate_sha256"], "queue": _artifact(paths.successor_queue),
        })).hexdigest(),
        "snapshot": snapshot, "policy": copy.deepcopy(POLICY),
        "policy_sha256": policy_document["policy_sha256"],
        "sources": {
            "activation_source": _artifact(paths.activation_source),
            "successor_queue": _artifact(paths.successor_queue),
            "successor_autoresume": _artifact(paths.successor_autoresume),
            "phase_gate_declaration": declaration,
        },
        "successor_policy": _artifact(paths.successor_policy),
        "phase_gate": _artifact(paths.phase_gate),
        "phase_receipt_root": str(paths.phase_receipt_root.resolve()),
        "service": {"label": SERVICE_LABEL,
                    "target": str(paths.launch_agent.resolve()),
                    "candidate": _artifact(paths.staged_service),
                    "preexisting": preexisting,
                    "was_loaded": service.is_loaded(SERVICE_LABEL)},
        "mutation_boundary": {
            "policy_only": True, "plan_mutation_permitted": False,
            "state_mutation_permitted": False, "campaign_mutation_permitted": False,
            "control_mutation_permitted": False, "pid_mutation_permitted": False,
            "result_mutation_permitted": False, "evidence_mutation_permitted": False,
            "runtime_spec_mutation_permitted": False,
            "source_deletion_permitted": False,
            "old_marker_mutation_permitted": False,
        },
    }
    packet["packet_sha256"] = _hash_value(packet)
    _atomic_json(paths.packet, packet)
    marker: dict[str, Any] = {
        "schema": MARKER_SCHEMA, "version": VERSION,
        "generation_id": packet["generation_id"], "prepared_at": _now(),
        "packet": _artifact(paths.packet),
        "policy_sha256": policy_document["policy_sha256"],
        "predecessor_marker": snapshot["old_marker"],
        "predecessor_marker_sha256": snapshot["old_marker_sha256"],
        "successor_policy": packet["successor_policy"],
        "activation_source": packet["sources"]["activation_source"],
        "successor_queue": packet["sources"]["successor_queue"],
        "successor_autoresume": packet["sources"]["successor_autoresume"],
        "phase_gate": packet["phase_gate"],
        "phase_gate_declaration": declaration,
        "phase_aware_disk_gate": copy.deepcopy(gate["phase_aware_disk_gate"]),
        "predecessor_overlay": snapshot["overlay"],
        "predecessor_overlay_sha256": snapshot["overlay_sha256"],
        "phase_receipt_root": packet["phase_receipt_root"],
        "service_candidate": packet["service"]["candidate"],
        "activation_snapshot_sha256": _hash_value(snapshot),
        "result_mutation_permitted": False,
        "evidence_mutation_permitted": False,
    }
    marker["marker_sha256"] = _hash_value(marker)
    _atomic_json(paths.staged_marker, marker)
    errors = _packet_errors(packet, paths, live=True, probe=probe)
    if errors:
        raise ActivationError("staged successor failed self-audit: " + "; ".join(errors))
    return packet


def _wal_entries(paths: Paths, generation_id: str) -> list[dict[str, Any]]:
    if not paths.wal_root.exists():
        return []
    if paths.wal_root.is_symlink() or not paths.wal_root.is_dir():
        raise ActivationError("successor WAL root is unsafe")
    files = sorted(paths.wal_root.glob("*.json"))
    if any(path.name != f"{index:08d}.json"
           for index, path in enumerate(files, 1)):
        raise ActivationError("successor WAL has a gap or extra entry")
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for index, path in enumerate(files, 1):
        row = _read_json(path)
        if set(row) != {"schema", "version", "created_at", "generation_id",
                       "index", "phase", "operation", "previous_entry_sha256",
                       "details", "entry_sha256"} \
                or row.get("schema") != WAL_SCHEMA or row.get("version") != VERSION \
                or row.get("generation_id") != generation_id \
                or row.get("index") != index \
                or row.get("previous_entry_sha256") != previous \
                or not _sealed(row, "entry_sha256"):
            raise ActivationError(f"invalid successor WAL entry: {path}")
        previous = row["entry_sha256"]; rows.append(row)
    return rows


def _seal_journal(journal: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(journal); value["updated_at"] = _now()
    value["journal_sha256"] = _hash_value(_without(value, "journal_sha256"))
    return value


def _append_step(paths: Paths, journal: dict[str, Any], phase: str,
                 operation: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    paths.wal_root.mkdir(parents=True, exist_ok=True)
    rows = _wal_entries(paths, journal["generation_id"])
    row: dict[str, Any] = {
        "schema": WAL_SCHEMA, "version": VERSION, "created_at": _now(),
        "generation_id": journal["generation_id"], "index": len(rows) + 1,
        "phase": phase, "operation": operation,
        "previous_entry_sha256": rows[-1]["entry_sha256"] if rows else None,
        "details": details or {},
    }
    row["entry_sha256"] = _hash_value(row)
    _atomic_json(paths.wal_root / f"{row['index']:08d}.json", row)
    value = copy.deepcopy(journal)
    value.update({"phase": phase, "operation": operation,
                  "wal_index": row["index"],
                  "wal_entry_sha256": row["entry_sha256"]})
    if phase == "rolled-back": value["status"] = "rolled-back"
    elif phase in {"marker-committed", "service-ready", "active"}:
        value["status"] = "committed" if phase != "active" else "active"
    value = _seal_journal(value); _atomic_json(paths.journal, value)
    return value


def _journal_document(paths: Paths, packet: dict[str, Any],
                      service: ServiceController) -> dict[str, Any]:
    live_existed = paths.launch_agent.exists()
    if live_existed:
        if paths.service_backup.exists():
            raise ActivationError("service backup already exists without a journal")
        _atomic_bytes(paths.service_backup, _stable_bytes(paths.launch_agent), mode=0o600)
        backup: dict[str, Any] | None = _artifact(paths.service_backup)
    else:
        backup = None
    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA, "version": VERSION, "created_at": _now(),
        "updated_at": _now(), "generation_id": packet["generation_id"],
        "status": "prepared", "phase": "prepared", "operation": "prepare",
        "wal_index": 0, "wal_entry_sha256": None,
        "packet": _artifact(paths.packet), "staged_marker": _artifact(paths.staged_marker),
        "service_target": str(paths.launch_agent.resolve()),
        "service_candidate": packet["service"]["candidate"],
        "service_preexisted": live_existed, "service_backup": backup,
        "service_was_loaded": service.is_loaded(SERVICE_LABEL),
        "live_marker": None, "source_deletion_permitted": False,
        "evidence_mutation_permitted": False, "result_mutation_permitted": False,
        "journal_sha256": "",
    }
    journal = _seal_journal(journal); _atomic_json(paths.journal, journal)
    return _append_step(paths, journal, "prepared", "transaction-prepared")


def _journal_errors(journal: Any, packet: dict[str, Any], paths: Paths) -> list[str]:
    required = {"schema", "version", "created_at", "updated_at", "generation_id",
                "status", "phase", "operation", "wal_index", "wal_entry_sha256",
                "packet", "staged_marker", "service_target", "service_candidate",
                "service_preexisted", "service_backup", "service_was_loaded",
                "live_marker", "source_deletion_permitted",
                "evidence_mutation_permitted", "result_mutation_permitted",
                "journal_sha256"}
    if not isinstance(journal, dict) or set(journal) != required:
        return ["activation journal keys are invalid"]
    errors: list[str] = []
    if journal.get("schema") != JOURNAL_SCHEMA or journal.get("version") != VERSION \
            or journal.get("generation_id") != packet.get("generation_id") \
            or not _sealed(journal, "journal_sha256"):
        errors.append("activation journal identity is invalid")
    if not _artifact_matches(journal.get("packet"), paths.packet) \
            or not _artifact_matches(journal.get("staged_marker"), paths.staged_marker) \
            or journal.get("service_candidate") != packet.get("service", {}).get("candidate"):
        errors.append("activation journal bundle is mixed")
    if journal.get("source_deletion_permitted") is not False \
            or journal.get("evidence_mutation_permitted") is not False \
            or journal.get("result_mutation_permitted") is not False:
        errors.append("activation journal permits forbidden mutation")
    try:
        rows = _wal_entries(paths, packet["generation_id"])
        index = journal.get("wal_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1 \
                or index > len(rows) \
                or journal.get("wal_entry_sha256") != rows[index - 1]["entry_sha256"]:
            errors.append("activation journal WAL pointer is invalid")
    except ActivationError as exc:
        errors.append(str(exc))
    return errors


@contextlib.contextmanager
def _exclusive_leases(paths: Paths):
    handles = []
    try:
        for path in (paths.transaction_lock, paths.queue_lock, paths.heavy_lock):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise ActivationError(f"activation lease is busy: {path}") from exc
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()


def _marker_state(paths: Paths) -> str:
    if not paths.active_marker.exists():
        return "absent"
    if _stable_bytes(paths.active_marker) == _stable_bytes(paths.staged_marker):
        return "committed"
    return "foreign"


def _rollback_precommit(paths: Paths, packet: dict[str, Any],
                        journal: dict[str, Any], service: ServiceController) -> dict[str, Any]:
    if _marker_state(paths) != "absent":
        raise ActivationError("rollback is forbidden after marker commit")
    service.bootout(SERVICE_LABEL)
    if journal["service_preexisted"]:
        backup = journal.get("service_backup")
        if not _artifact_matches(backup, paths.service_backup):
            raise ActivationError("predecessor service backup changed")
        _atomic_bytes(paths.launch_agent, _stable_bytes(paths.service_backup), mode=0o600)
        if journal["service_was_loaded"]:
            service.bootstrap(paths.launch_agent)
    elif paths.launch_agent.exists():
        paths.launch_agent.unlink(); _fsync_dir(paths.launch_agent.parent)
    return _append_step(paths, journal, "rolled-back", "restore-service-only")


def _forward_recover(paths: Paths, packet: dict[str, Any], journal: dict[str, Any],
                     service: ServiceController) -> dict[str, Any]:
    if _marker_state(paths) != "committed":
        raise ActivationError("forward recovery requires the exact committed marker")
    service.bootout(SERVICE_LABEL)
    _atomic_bytes(paths.launch_agent, _stable_bytes(paths.staged_service), mode=0o600)
    service.bootstrap(paths.launch_agent)
    _fault("after:new-service-bootstrap")
    journal = _append_step(paths, journal, "service-ready", "bootstrap-successor")
    service.kickstart(SERVICE_LABEL)
    _fault("after:new-service-kickstart")
    journal = _append_step(paths, journal, "active", "successor-service-active",
                           {"marker_sha256": _read_json(paths.active_marker)["marker_sha256"]})
    _fault("after:journal-active")
    return journal


def activate(*, packet_sha256: str, generation_id: str,
             paths: Paths | None = None, service: ServiceController | None = None,
             probe: ResourceProbe | None = None) -> dict[str, Any]:
    paths = paths or production_paths(); service = service or LaunchctlService()
    probe = probe or HostResourceProbe()
    with _exclusive_leases(paths):
        packet = _read_json(paths.packet)
        if packet.get("packet_sha256") != packet_sha256 \
                or packet.get("generation_id") != generation_id:
            raise ActivationError("explicit activation keys differ from packet")
        errors = _packet_errors(packet, paths, live=True, probe=probe)
        if errors:
            raise ActivationError("activation preflight failed: " + "; ".join(errors))
        if paths.journal.exists():
            raise ActivationError("activation journal already exists; use recover")
        journal = _journal_document(paths, packet, service)
        _fault("after:journal-prepared")
        try:
            service.bootout(SERVICE_LABEL)
            _fault("after:old-service-bootout")
            journal = _append_step(paths, journal, "service-stopped", "bootout-predecessor")
            _atomic_bytes(paths.launch_agent, _stable_bytes(paths.staged_service), mode=0o600)
            _fault("after:new-service-installed")
            journal = _append_step(paths, journal, "service-installed", "install-successor")
            if not _snapshot_matches(packet["snapshot"], paths):
                raise ActivationError("CAS changed during service transition")
            _atomic_bytes(paths.active_marker, _stable_bytes(paths.staged_marker), mode=0o600)
            _fault("after:marker-commit")
            journal = _append_step(paths, journal, "marker-committed", "commit-marker",
                                   {"marker": _artifact(paths.active_marker)})
            return _forward_recover(paths, packet, journal, service)
        except Exception:
            if _marker_state(paths) == "absent":
                _rollback_precommit(paths, packet, _read_json(paths.journal), service)
            raise


def recover(*, paths: Paths | None = None,
            service: ServiceController | None = None) -> dict[str, Any]:
    paths = paths or production_paths(); service = service or LaunchctlService()
    with _exclusive_leases(paths):
        packet = _read_json(paths.packet); journal = _read_json(paths.journal)
        errors = _packet_errors(packet, paths, live=False)
        errors.extend(_journal_errors(journal, packet, paths))
        if errors:
            raise ActivationError("recovery authority invalid: " + "; ".join(errors))
        marker = _marker_state(paths)
        if marker == "foreign":
            raise ActivationError("foreign successor marker blocks recovery")
        if marker == "absent":
            if journal.get("status") == "rolled-back":
                return journal
            return _rollback_precommit(paths, packet, journal, service)
        return _forward_recover(paths, packet, journal, service)


def validate_active_marker(*, paths: Paths | None = None,
                           verify_service: bool = True) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    paths = paths or production_paths()
    marker = _read_json(paths.active_marker); staged = _read_json(paths.staged_marker)
    if marker != staged or set(marker) != MARKER_KEYS \
            or marker.get("schema") != MARKER_SCHEMA \
            or not _sealed(marker, "marker_sha256"):
        raise ActivationError("successor marker is absent, foreign, or invalid")
    packet = _read_json(paths.packet)
    errors = _packet_errors(packet, paths, live=False)
    if marker.get("packet") != _artifact(paths.packet) \
            or marker.get("successor_policy") != packet.get("successor_policy") \
            or marker.get("policy_sha256") != packet.get("policy_sha256"):
        errors.append("successor marker/packet/policy mixture is invalid")
    journal = _read_json(paths.journal)
    errors.extend(_journal_errors(journal, packet, paths))
    if journal.get("status") not in {"committed", "active"}:
        errors.append("successor journal is not committed")
    if verify_service and _stable_bytes(paths.launch_agent) != _stable_bytes(paths.staged_service):
        errors.append("installed successor service differs")
    if errors:
        raise ActivationError("active successor invalid: " + "; ".join(errors))
    return marker, packet


def audit(*, paths: Paths | None = None, probe: ResourceProbe | None = None) \
        -> dict[str, Any]:
    paths = paths or production_paths(); probe = probe or HostResourceProbe()
    packet = _read_json(paths.packet)
    errors = _packet_errors(packet, paths, live=True, probe=probe)
    return {"ready": not errors, "errors": errors,
            "packet_sha256": packet.get("packet_sha256"),
            "generation_id": packet.get("generation_id"),
            "policy_sha256": packet.get("policy_sha256"),
            "activation_executed": False}


def status(*, paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or production_paths()
    packet = _read_json(paths.packet) if paths.packet.exists() else None
    journal = _read_json(paths.journal) if paths.journal.exists() else None
    return {"staged": packet is not None, "marker_state": _marker_state(paths),
            "journal_status": journal.get("status") if journal else None,
            "journal_phase": journal.get("phase") if journal else None,
            "packet_sha256": packet.get("packet_sha256") if packet else None,
            "generation_id": packet.get("generation_id") if packet else None,
            "policy": copy.deepcopy(POLICY),
            "activation_permitted_automatically": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("issue-phase-gate", "stage", "audit",
                                            "status", "activate", "recover"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--launch-agent", type=Path)
    parser.add_argument("--packet-sha256")
    parser.add_argument("--generation-id")
    parser.add_argument("--confirm-policy-only-successor", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = production_paths(args.root, launch_agent=args.launch_agent)
    try:
        if args.command == "issue-phase-gate": result = issue_phase_gate(paths=paths)
        elif args.command == "stage": result = stage(paths=paths)
        elif args.command == "audit": result = audit(paths=paths)
        elif args.command == "status": result = status(paths=paths)
        elif args.command == "recover": result = recover(paths=paths)
        else:
            if not args.confirm_policy_only_successor \
                    or not args.packet_sha256 or not args.generation_id:
                raise ActivationError("activation requires confirmation and both exact keys")
            result = activate(packet_sha256=args.packet_sha256,
                              generation_id=args.generation_id, paths=paths)
    except (ActivationError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
