#!/usr/bin/env python3.12
"""Hash-bound controlled-swap successor for Doctor V5 Ultra.

This is an entrypoint shim, not an activation tool.  It cannot stage, promote,
or edit campaign artifacts.  A separate two-key policy must bind the already
active accelerated generation.  Only after that predecessor validates does the
shim retain all v1 scheduling behavior and replace its resource observation
with an absolute, inclusive 512 MB swap ceiling.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from typing import Any, Callable
import re

import doctor_v5_accel_loader as accel_loader
import doctor_v5_stacked_admission as stacked
import doctor_v5_ultra_accelerated_queue as v1


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = Path(__file__).resolve()
V1_PATH = HERE / "doctor_v5_ultra_accelerated_queue.py"
V1_SHA256 = "1e5a61582b398117ca197483a1ec0c59870288bb032bee5118d942087d131365"
STACKED_PATH = HERE / "doctor_v5_stacked_admission.py"
STACKED_SHA256 = "2227c4ebb32039b87d3d9d40f24e8d984a8f96cef4113ed29a48f125b809aaa5"

STAGE_ROOT = ROOT / (
    "reports/condense/doctor_v5_ultra/staged_acceleration/controlled_swap_v2"
)
DEFAULT_POLICY = STAGE_ROOT / "successor_policy.json"
ACTIVE_MARKER = STAGE_ROOT / "active_marker.json"
STAGED_MARKER = STAGE_ROOT / "staged_marker.json"
GENERATION_PACKET = STAGE_ROOT / "pending_generation.json"
PHASE_GATE_RECEIPT = STAGE_ROOT / "phase_gate.json"
PHASE_RECEIPT_ROOT = STAGE_ROOT / "phase_receipts"
SERVICE_CANDIDATE = STAGE_ROOT / "successor_service.plist"
SUCCESSOR_AUTORESUME = HERE / "doctor_v5_controlled_swap_autoresume.py"
ACTIVATION_SOURCE = HERE / "doctor_v5_controlled_swap_activation.py"
LAUNCH_AGENT = Path.home() / (
    "Library/LaunchAgents/com.hawking.doctorv5ultra.autoresume.plist"
)
POLICY_SCHEMA = "hawking.doctor_v5_controlled_swap_successor_policy.v1"
POLICY_VERSION = "2026-07-15.1"
POLICY_MODE = "pending_only_operational_supersession"
MARKER_SCHEMA = "hawking.doctor_v5_controlled_swap_successor_marker.v1"
PACKET_SCHEMA = "hawking.doctor_v5_controlled_swap_successor_packet.v1"
PHASE_GATE_API_SCHEMA = "hawking.doctor_v5_phase_gate_api.v1"
PHASE_GATE_PATH = HERE / "doctor_v5_phase_aware_disk_gate.py"
LEDGER_PATH = HERE / "doctor_v5_remaining_scratch_ledger.py"
SWAP_USED_MB_MAX = 512.0
PRESSURE_LEVEL_NORMAL = 1
ENV_POLICY = "DOCTOR_V5_CONTROLLED_SWAP_POLICY"
ENV_POLICY_SHA256 = "DOCTOR_V5_CONTROLLED_SWAP_POLICY_SHA256"
ENV_MARKER = "DOCTOR_V5_CONTROLLED_SWAP_MARKER"
ENV_MARKER_SHA256 = "DOCTOR_V5_CONTROLLED_SWAP_MARKER_SHA256"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_PHASE_MODULE_BYTES = 2 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")

_BASE = v1._BASE
_ORIGINAL_RESOURCE_SNAPSHOT = _BASE.ram_scheduler.resource_snapshot
_PREDECESSOR_EXECUTION_GATE = _BASE._execution_resource_gate
_POLICY: dict[str, Any] | None = None
_POLICY_PATH: Path | None = None
_PREDECESSOR_OVERLAY: dict[str, Any] | None = None
_SUCCESSOR_MARKER: dict[str, Any] | None = None
_SUCCESSOR_MARKER_PATH: Path | None = None


class SuccessorError(RuntimeError):
    """The successor policy or its predecessor cannot be trusted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _regular_bytes(path: Path, *, ceiling: int) -> bytes:
    resolved = path.resolve(strict=True)
    info = resolved.lstat()
    if resolved.is_symlink() or not resolved.is_file() \
            or not 0 < info.st_size <= ceiling:
        raise SuccessorError(f"unsafe bound artifact: {resolved}")
    before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
              info.st_ctime_ns)
    raw = resolved.read_bytes()
    after = resolved.lstat()
    if before != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                  after.st_ctime_ns) or len(raw) != info.st_size:
        raise SuccessorError(f"bound artifact changed while reading: {resolved}")
    return raw


def _artifact(path: Path, *, ceiling: int = MAX_JSON_BYTES) -> dict[str, Any]:
    path = path.resolve(strict=True)
    raw = _regular_bytes(path, ceiling=ceiling)
    try:
        name = str(path.relative_to(ROOT.resolve()))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _artifact_errors(reference: Any, expected_path: Path, *,
                     ceiling: int = MAX_JSON_BYTES) -> list[str]:
    if not isinstance(reference, dict) \
            or set(reference) != {"path", "sha256", "bytes"}:
        return ["artifact reference shape is invalid"]
    try:
        path = (ROOT / reference["path"]).resolve(strict=True)
        if path != expected_path.resolve(strict=True):
            return ["artifact path identity differs"]
        observed = _artifact(path, ceiling=ceiling)
    except (OSError, TypeError, ValueError, SuccessorError) as exc:
        return [f"artifact is unreadable: {exc}"]
    return [] if observed == reference else ["artifact content identity differs"]


def _bound_artifact_errors(reference: Any, expected_path: Path, *,
                           ceiling: int = MAX_JSON_BYTES) -> list[str]:
    """Validate activation artifacts whose sealed paths are absolute."""
    if not isinstance(reference, dict) \
            or set(reference) != {"path", "sha256", "bytes"}:
        return ["artifact reference shape is invalid"]
    try:
        raw_path = Path(reference["path"])
        path = (raw_path if raw_path.is_absolute() else ROOT / raw_path).resolve(strict=True)
        if path != expected_path.resolve(strict=True):
            return ["artifact path identity differs"]
        raw = _regular_bytes(path, ceiling=ceiling)
    except (OSError, TypeError, ValueError, SuccessorError) as exc:
        return [f"artifact is unreadable: {exc}"]
    if len(raw) != reference.get("bytes") \
            or hashlib.sha256(raw).hexdigest() != reference.get("sha256"):
        return ["artifact content identity differs"]
    return []


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = _regular_bytes(path, ceiling=MAX_JSON_BYTES)
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, SuccessorError) as exc:
        raise SuccessorError(f"cannot read successor JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SuccessorError(f"successor JSON root is not an object: {path}")
    return value


def _verify_static_bindings() -> None:
    for path, expected, label in (
        (V1_PATH, V1_SHA256, "accelerated v1 queue"),
        (STACKED_PATH, STACKED_SHA256, "stacked admission source"),
    ):
        observed, _ = accel_loader.hash_file(path)
        if observed != expected:
            raise SuccessorError(
                f"{label} drifted: expected={expected} observed={observed}"
            )
    v1._verify_static_bindings()


def phase_gate_declaration(module_path: Path) -> dict[str, Any]:
    """Return the only accepted optional phase-gate declaration shape."""
    try:
        if module_path.resolve(strict=True) != PHASE_GATE_PATH.resolve(strict=True):
            raise SuccessorError("phase gate is not the reviewed exact module path")
    except OSError as exc:
        raise SuccessorError("reviewed phase gate module is absent") from exc
    return {
        "api_schema": PHASE_GATE_API_SCHEMA,
        "module": _artifact(module_path, ceiling=MAX_PHASE_MODULE_BYTES),
        "install_callable": "install_phase_gate",
    }


def _policy_root(policy_path: Path) -> Path:
    path = policy_path.resolve(strict=True)
    root = path.parent
    try:
        root.relative_to(STAGE_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SuccessorError("successor policy escapes the sealed stage root") from exc
    if root.is_symlink() or not root.is_dir():
        raise SuccessorError("successor policy root is unsafe")
    return root


def _predecessor_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    marker = _read_json(v1.MARKER)
    try:
        overlay_path = Path(marker["overlay_path"]).resolve(strict=True)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SuccessorError("predecessor marker overlay path is invalid") from exc
    if overlay_path != stacked.DEFAULT_OVERLAY.resolve(strict=True):
        raise SuccessorError("predecessor marker selects an unexpected overlay")
    return marker, _read_json(overlay_path)


def _predecessor_errors(policy: dict[str, Any], marker: dict[str, Any],
                        overlay: dict[str, Any], *, deep: bool = True) -> list[str]:
    errors: list[str] = []
    predecessor = policy.get("predecessor")
    if not isinstance(predecessor, dict) or set(predecessor) != {
        "accelerated_queue", "stacked_admission", "active_marker",
        "admission_overlay", "marker_sha256", "overlay_sha256",
    }:
        return ["successor predecessor binding shape is invalid"]
    for name, path in (
        ("accelerated_queue", V1_PATH), ("stacked_admission", STACKED_PATH),
        ("active_marker", v1.MARKER),
        ("admission_overlay", stacked.DEFAULT_OVERLAY),
    ):
        errors.extend(f"predecessor {name}: {row}" for row in
                      _artifact_errors(predecessor.get(name), path))
    if marker.get("schema") != v1.MARKER_SCHEMA \
            or marker.get("marker_sha256") != _hash_value(
                _without(marker, "marker_sha256")
            ) \
            or predecessor.get("marker_sha256") != marker.get("marker_sha256"):
        errors.append("predecessor marker semantic identity is invalid")
    if marker.get("overlay_sha256") != overlay.get("overlay_sha256") \
            or predecessor.get("overlay_sha256") != overlay.get("overlay_sha256"):
        errors.append("predecessor marker/overlay mixture is invalid")
    overlay_errors = stacked.validate_overlay(overlay)
    errors.extend(f"predecessor overlay: {row}" for row in overlay_errors)
    if deep and not errors:
        # Existing v1 validation hashes only its reviewed sealed metadata/tools;
        # it does not recursively hash model or result payloads.
        errors.extend(v1._active_generation_errors(overlay))
    return errors


def validate_policy(policy: dict[str, Any], *, policy_path: Path | None = None,
                    deep_predecessor: bool = False) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "schema", "version", "created_at", "mode", "operational_root",
        "predecessor", "policy", "phase_gate", "phase_aware_disk_gate",
        "promotion", "policy_sha256",
    }
    if set(policy) != expected_keys or policy.get("schema") != POLICY_SCHEMA \
            or policy.get("version") != POLICY_VERSION:
        errors.append("successor policy keys/schema/version are invalid")
    if policy.get("mode") != POLICY_MODE:
        errors.append("successor policy is not pending-only operational supersession")
    if policy.get("policy_sha256") != _hash_value(_without(policy, "policy_sha256")):
        errors.append("successor policy canonical hash is invalid")
    guard = policy.get("policy")
    expected_guard = {
        "swap_used_mb_max": SWAP_USED_MB_MAX,
        "swap_boundary": "absolute_inclusive",
        "required_pressure": "normal",
        "pressure_level_required": PRESSURE_LEVEL_NORMAL,
        "ram_capacity_credit_bytes": 0,
        "preserve_ac_power_gate": True,
        "preserve_thermal_gate": True,
    }
    if not isinstance(guard, dict) or guard != expected_guard \
            or type(guard.get("swap_used_mb_max")) is not float \
            or type(guard.get("pressure_level_required")) is not int:
        errors.append("successor controlled-swap guard is not exact")
    promotion = policy.get("promotion")
    if not isinstance(promotion, dict) or promotion != {
        "automatic_activation_permitted": False,
        "completed_evidence_mutation_permitted": False,
        "runtime_spec_mutation_permitted": False,
        "result_mutation_permitted": False,
        "pending_cells_only": True,
    }:
        errors.append("successor promotion boundary is invalid")
    phase = policy.get("phase_gate")
    if phase is not None:
        if not isinstance(phase, dict) or set(phase) != {
            "api_schema", "module", "install_callable"
        } or phase.get("api_schema") != PHASE_GATE_API_SCHEMA \
                or phase.get("install_callable") != "install_phase_gate":
            errors.append("optional phase-gate declaration is invalid")
        else:
            try:
                module_path = (ROOT / phase["module"]["path"]).resolve(strict=True)
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("optional phase-gate module path is invalid")
            else:
                try:
                    expected_phase_path = PHASE_GATE_PATH.resolve(strict=True)
                except OSError:
                    errors.append("reviewed exact phase-gate module is absent")
                    expected_phase_path = None
                if module_path != expected_phase_path:
                    errors.append("optional phase gate is not the reviewed exact module")
                errors.extend(f"optional phase gate: {row}" for row in
                              _artifact_errors(phase.get("module"),
                                               PHASE_GATE_PATH,
                                               ceiling=MAX_PHASE_MODULE_BYTES))
    phase_config = policy.get("phase_aware_disk_gate")
    if phase is None:
        if phase_config is not None:
            errors.append("phase-aware disk config exists without its module")
    else:
        errors.extend(_phase_config_errors(phase_config, phase))
    if policy_path is not None:
        try:
            root = _policy_root(policy_path)
            declared = Path(policy.get("operational_root", ""))
            declared = (ROOT / declared).resolve(strict=True)
            if declared != root:
                errors.append("successor operational root differs from policy location")
        except (OSError, TypeError, ValueError, SuccessorError) as exc:
            errors.append(str(exc))
    if deep_predecessor:
        try:
            marker, overlay = _predecessor_documents()
            errors.extend(_predecessor_errors(policy, marker, overlay, deep=True))
        except SuccessorError as exc:
            errors.append(str(exc))
    return errors


def _phase_config_errors(config: Any, declaration: dict[str, Any]) -> list[str]:
    """Validate the installer's exact, zero-RAM, multi-cell policy bindings."""
    errors: list[str] = []
    if not isinstance(config, dict) or set(config) != {
        "schema", "enabled", "module_sha256", "ledger_module_sha256",
        "ram_credit_bytes", "bindings",
    }:
        return ["phase-aware disk policy keys are not exact"]
    try:
        module_sha = _artifact(
            PHASE_GATE_PATH, ceiling=MAX_PHASE_MODULE_BYTES
        )["sha256"]
        ledger_sha = _artifact(
            LEDGER_PATH, ceiling=MAX_PHASE_MODULE_BYTES
        )["sha256"]
    except (OSError, SuccessorError) as exc:
        return [f"phase-aware disk source binding is unavailable: {exc}"]
    if config.get("schema") != PHASE_GATE_API_SCHEMA \
            or config.get("enabled") is not True \
            or config.get("ram_credit_bytes") != 0 \
            or config.get("module_sha256") != module_sha \
            or config.get("module_sha256") \
            != declaration.get("module", {}).get("sha256") \
            or config.get("ledger_module_sha256") != ledger_sha:
        errors.append("phase-aware disk module/ledger/zero-RAM binding is invalid")
    rows = config.get("bindings")
    expected = {
        "plan_path", "plan_file_sha256", "plan_sha256", "plan_cell_sha256",
        "cell_id", "cell_identity_sha256", "runtime_spec_path",
        "runtime_spec_file_sha256", "program_spec_sha256",
        "execution_output_root", "disk_reserve_bytes",
        "declared_scratch_bytes", "frozen_projected_output_bytes",
    }
    if not isinstance(rows, list) or not rows:
        return errors + ["phase-aware disk binding list is empty"]
    sha_fields = {
        "plan_file_sha256", "plan_sha256", "plan_cell_sha256",
        "cell_identity_sha256", "runtime_spec_file_sha256",
        "program_spec_sha256",
    }
    path_fields = {
        "plan_path", "runtime_spec_path", "execution_output_root",
    }
    seen: set[str] = set()
    for index, bindings in enumerate(rows):
        if not isinstance(bindings, dict) or set(bindings) != expected:
            errors.append(f"phase-aware disk binding row[{index}] is not exact")
            continue
        if any(not _valid_sha(bindings.get(name)) for name in sha_fields):
            errors.append(f"phase-aware disk binding row[{index}] has invalid SHA-256")
        for name in path_fields:
            raw = bindings.get(name)
            try:
                path = Path(raw)
                if not isinstance(raw, str) or not path.is_absolute() \
                        or "\x00" in raw:
                    raise ValueError
                path.resolve(strict=False).relative_to(ROOT.resolve())
            except (TypeError, ValueError):
                errors.append(
                    f"phase-aware disk binding row[{index}] path is unsafe: {name}"
                )
        cell_id = bindings.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id \
                or Path(cell_id).name != cell_id or "\x00" in cell_id \
                or cell_id in seen:
            errors.append(f"phase-aware disk binding row[{index}] cell is unsafe/duplicate")
        else:
            seen.add(cell_id)
        for name in (
            "disk_reserve_bytes", "declared_scratch_bytes",
            "frozen_projected_output_bytes",
        ):
            value = bindings.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(
                    f"phase-aware disk binding row[{index}] byte field is invalid: {name}"
                )
        if bindings.get("disk_reserve_bytes") != _BASE.DISK_RESERVE_BYTES \
                or not isinstance(bindings.get("declared_scratch_bytes"), int) \
                or bindings.get("declared_scratch_bytes", 0) < _BASE.MIN_SCRATCH_BYTES:
            errors.append(
                f"phase-aware disk binding row[{index}] resource envelope is invalid"
            )
    return errors


def load_successor_policy(env: dict[str, str] | None = None) \
        -> tuple[dict[str, Any], Path]:
    source = os.environ if env is None else env
    raw_path, expected = source.get(ENV_POLICY), source.get(ENV_POLICY_SHA256)
    if not raw_path or not expected:
        raise SuccessorError("both controlled-swap successor keys are required")
    path = Path(raw_path).resolve(strict=True)
    _policy_root(path)
    policy = _read_json(path)
    errors = validate_policy(policy, policy_path=path)
    if expected != policy.get("policy_sha256"):
        errors.append("successor activation SHA differs from policy")
    if errors:
        raise SuccessorError("invalid controlled-swap successor policy: "
                             + "; ".join(errors))
    return policy, path


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
PACKET_KEYS = frozenset({
    "schema", "version", "created_at", "generation_id", "snapshot",
    "policy", "policy_sha256", "successor_policy", "sources", "phase_gate",
    "phase_receipt_root", "service", "mutation_boundary", "packet_sha256",
})
MUTATION_BOUNDARY = {
    "policy_only": True, "plan_mutation_permitted": False,
    "state_mutation_permitted": False, "campaign_mutation_permitted": False,
    "control_mutation_permitted": False, "pid_mutation_permitted": False,
    "result_mutation_permitted": False, "evidence_mutation_permitted": False,
    "runtime_spec_mutation_permitted": False,
    "source_deletion_permitted": False, "old_marker_mutation_permitted": False,
}


def _successor_generation_errors(marker: dict[str, Any], packet: dict[str, Any],
                                 policy: dict[str, Any], policy_path: Path, *,
                                 verify_installed_service: bool = True) -> list[str]:
    """Validate one exact marker/packet/policy/service generation."""
    errors: list[str] = []
    if set(marker) != MARKER_KEYS or marker.get("schema") != MARKER_SCHEMA \
            or marker.get("version") != POLICY_VERSION \
            or marker.get("marker_sha256") != _hash_value(
                _without(marker, "marker_sha256")
            ) \
            or not _valid_sha(marker.get("generation_id")) \
            or not _valid_sha(marker.get("activation_snapshot_sha256")):
        errors.append("successor marker keys/schema/hash are invalid")
    if set(packet) != PACKET_KEYS or packet.get("schema") != PACKET_SCHEMA \
            or packet.get("version") != POLICY_VERSION \
            or packet.get("packet_sha256") != _hash_value(
                _without(packet, "packet_sha256")
            ) \
            or packet.get("generation_id") != marker.get("generation_id"):
        errors.append("successor packet keys/schema/hash/generation are invalid")
    for label, row, path, ceiling in (
        ("packet", marker.get("packet"), GENERATION_PACKET, MAX_JSON_BYTES),
        ("policy", marker.get("successor_policy"), policy_path, MAX_JSON_BYTES),
        ("queue", marker.get("successor_queue"), SCRIPT, MAX_PHASE_MODULE_BYTES),
        ("activation", marker.get("activation_source"), ACTIVATION_SOURCE,
         MAX_PHASE_MODULE_BYTES),
        ("autoresume", marker.get("successor_autoresume"),
         SUCCESSOR_AUTORESUME, MAX_PHASE_MODULE_BYTES),
        ("phase gate", marker.get("phase_gate"), PHASE_GATE_RECEIPT,
         MAX_JSON_BYTES),
        ("predecessor marker", marker.get("predecessor_marker"), v1.MARKER,
         MAX_JSON_BYTES),
        ("predecessor overlay", marker.get("predecessor_overlay"),
         stacked.DEFAULT_OVERLAY, MAX_JSON_BYTES),
        ("service candidate", marker.get("service_candidate"),
         SERVICE_CANDIDATE, MAX_JSON_BYTES),
    ):
        errors.extend(f"successor marker {label}: {item}" for item in
                      _bound_artifact_errors(row, path, ceiling=ceiling))
    try:
        staged = _read_json(STAGED_MARKER)
        if staged != marker:
            errors.append("active and staged successor markers differ")
    except SuccessorError as exc:
        errors.append(str(exc))
    if marker.get("policy_sha256") != policy.get("policy_sha256") \
            or marker.get("phase_gate_declaration") != policy.get("phase_gate") \
            or marker.get("phase_aware_disk_gate") \
            != policy.get("phase_aware_disk_gate") \
            or marker.get("result_mutation_permitted") is not False \
            or marker.get("evidence_mutation_permitted") is not False:
        errors.append("successor marker policy/mutation boundary is mixed")
    try:
        old_marker = _read_json(v1.MARKER)
        old_overlay = _read_json(stacked.DEFAULT_OVERLAY)
        if marker.get("predecessor_marker_sha256") \
                != old_marker.get("marker_sha256") \
                or marker.get("predecessor_overlay_sha256") \
                != old_overlay.get("overlay_sha256"):
            errors.append("successor marker predecessor semantic identity is mixed")
    except SuccessorError as exc:
        errors.append(str(exc))
    try:
        receipt_root = Path(marker.get("phase_receipt_root", "")).resolve(strict=True)
        if receipt_root != PHASE_RECEIPT_ROOT.resolve(strict=True) \
                or receipt_root.is_symlink() or not receipt_root.is_dir():
            errors.append("successor phase receipt root is invalid")
    except (OSError, TypeError, ValueError):
        errors.append("successor phase receipt root is invalid")

    if packet.get("policy") != policy.get("policy") \
            or packet.get("policy_sha256") != policy.get("policy_sha256") \
            or packet.get("successor_policy") != marker.get("successor_policy") \
            or packet.get("phase_gate") != marker.get("phase_gate") \
            or packet.get("phase_receipt_root") != marker.get("phase_receipt_root") \
            or packet.get("mutation_boundary") != MUTATION_BOUNDARY:
        errors.append("successor packet policy/phase/mutation boundary is mixed")
    sources = packet.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "activation_source", "successor_queue", "successor_autoresume",
        "phase_gate_declaration"
    } or sources.get("successor_queue") != marker.get("successor_queue") \
            or sources.get("activation_source") != marker.get("activation_source") \
            or sources.get("successor_autoresume") \
            != marker.get("successor_autoresume") \
            or sources.get("phase_gate_declaration") \
            != marker.get("phase_gate_declaration"):
        errors.append("successor packet source generation is mixed")
    snapshot = packet.get("snapshot")
    if not isinstance(snapshot, dict) \
            or _hash_value(snapshot) != marker.get("activation_snapshot_sha256") \
            or snapshot.get("old_marker_sha256") \
            != marker.get("predecessor_marker_sha256") \
            or snapshot.get("overlay_sha256") \
            != marker.get("predecessor_overlay_sha256"):
        errors.append("successor activation snapshot binding is mixed")
    service = packet.get("service")
    try:
        service_target = Path(service.get("target", "")).resolve(strict=False) \
            if isinstance(service, dict) else None
    except (TypeError, ValueError):
        service_target = None
    if not isinstance(service, dict) or set(service) != {
        "label", "target", "candidate", "preexisting", "was_loaded"
    } or service.get("candidate") != marker.get("service_candidate") \
            or service.get("label") != "com.hawking.doctorv5ultra.autoresume" \
            or service_target != LAUNCH_AGENT.resolve(strict=False):
        errors.append("successor service generation is mixed")
    if verify_installed_service:
        try:
            if _regular_bytes(LAUNCH_AGENT, ceiling=MAX_JSON_BYTES) \
                    != _regular_bytes(SERVICE_CANDIDATE, ceiling=MAX_JSON_BYTES):
                errors.append("installed successor service differs from candidate")
        except (OSError, SuccessorError) as exc:
            errors.append(f"installed successor service is invalid: {exc}")
    return errors


def load_successor_marker(policy: dict[str, Any], policy_path: Path,
                          env: dict[str, str] | None = None) \
        -> tuple[dict[str, Any], Path]:
    source = os.environ if env is None else env
    raw_path, expected = source.get(ENV_MARKER), source.get(ENV_MARKER_SHA256)
    if not raw_path or not expected:
        raise SuccessorError("both controlled-swap marker keys are required")
    try:
        path = Path(raw_path).resolve(strict=True)
        expected_path = ACTIVE_MARKER.resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise SuccessorError("controlled-swap active marker is unavailable") from exc
    if path != expected_path:
        raise SuccessorError("controlled-swap marker path is not the active marker")
    marker = _read_json(path)
    packet = _read_json(GENERATION_PACKET)
    errors = _successor_generation_errors(marker, packet, policy, policy_path)
    if expected != marker.get("marker_sha256"):
        errors.append("successor marker activation SHA differs")
    if errors:
        raise SuccessorError("invalid controlled-swap successor generation: "
                             + "; ".join(errors))
    return marker, path


def resource_health(snapshot: dict[str, Any], thermal: dict[str, Any]) -> dict[str, Any]:
    """Pure, fail-closed absolute 512 MB health contract."""
    blockers: list[str] = []
    pressure = snapshot.get("pressure_level")
    if type(pressure) is not int or pressure != PRESSURE_LEVEL_NORMAL:
        blockers.append("memory pressure is not exactly normal or is unavailable")
    swap = snapshot.get("swap_used_mb")
    if isinstance(swap, bool) or not isinstance(swap, (int, float)) \
            or not math.isfinite(float(swap)) or float(swap) < 0 \
            or float(swap) > SWAP_USED_MB_MAX:
        blockers.append("swap exceeds the absolute 512 MB ceiling or is unavailable")
    if "AC Power" not in str(snapshot.get("power_source", "")):
        blockers.append("AC power is not confirmed")
    if thermal.get("ok") is not True:
        blockers.append("thermal state is not explicitly green")
    return {
        "ok": not blockers, "blockers": blockers,
        "swap_used_mb_max": SWAP_USED_MB_MAX,
        "swap_boundary": "absolute_inclusive", "ram_capacity_credit_bytes": 0,
        "snapshot": snapshot, "thermal": thermal,
    }


def _strict_resource_snapshot(root: str) -> dict[str, Any]:
    """Make invalid probe types fail closed in unchanged predecessor guards."""
    snapshot = _ORIGINAL_RESOURCE_SNAPSHOT(root)
    if not isinstance(snapshot, dict):
        return {"error": "resource snapshot is not an object"}
    strict = dict(snapshot)
    pressure = strict.get("pressure_level")
    if type(pressure) is not int or pressure != PRESSURE_LEVEL_NORMAL:
        strict["successor_observed_pressure_level"] = pressure
        strict["pressure_level"] = None
    swap = strict.get("swap_used_mb")
    if isinstance(swap, bool) or not isinstance(swap, (int, float)) \
            or not math.isfinite(float(swap)) or float(swap) < 0:
        strict["successor_observed_swap_used_mb"] = swap
        strict["swap_used_mb"] = None
    return strict


def _controlled_execution_gate(plan: dict[str, Any], state: dict[str, Any],
                               execution: dict[str, Any]) -> dict[str, Any]:
    gate = _PREDECESSOR_EXECUTION_GATE(plan, state, execution)
    resources = gate.get("resources")
    thermal = gate.get("thermal")
    health = resource_health(
        resources if isinstance(resources, dict) else {},
        thermal if isinstance(thermal, dict) else {},
    )
    for blocker in health["blockers"]:
        if blocker not in gate["blockers"]:
            gate["blockers"].append(blocker)
    gate["ok"] = not gate["blockers"]
    gate["controlled_swap_successor"] = {
        "swap_used_mb_max": SWAP_USED_MB_MAX,
        "swap_boundary": "absolute_inclusive", "ram_capacity_credit_bytes": 0,
        "pressure_level_required": PRESSURE_LEVEL_NORMAL,
    }
    return gate


def _load_phase_module(declaration: dict[str, Any]) -> types.ModuleType:
    reference = declaration["module"]
    path = (ROOT / reference["path"]).resolve(strict=True)
    errors = _artifact_errors(reference, path, ceiling=MAX_PHASE_MODULE_BYTES)
    if errors:
        raise SuccessorError("phase gate binding failed: " + "; ".join(errors))
    raw = _regular_bytes(path, ceiling=MAX_PHASE_MODULE_BYTES)
    module_name = (
        "doctor_v5_controlled_swap_bound_phase_gate_"
        + reference["sha256"][:16]
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    prior = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        if prior is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior
        raise
    if module.__dict__.get("PHASE_GATE_API_SCHEMA") != PHASE_GATE_API_SCHEMA:
        raise SuccessorError("phase gate API schema differs")
    installer = module.__dict__.get(declaration["install_callable"])
    if not callable(installer):
        raise SuccessorError("phase gate installer is absent")
    signature = inspect.signature(installer)
    if list(signature.parameters) != [
        "base_module", "predecessor_gate", "successor_policy", "policy_root"
    ]:
        raise SuccessorError("phase gate installer signature differs")
    return module


def _install_phase_gate(policy: dict[str, Any], policy_root: Path,
                        predecessor_gate: Callable[..., dict[str, Any]]) \
        -> Callable[..., dict[str, Any]]:
    declaration = policy.get("phase_gate")
    if declaration is None:
        return predecessor_gate
    module = _load_phase_module(declaration)
    installer = getattr(module, declaration["install_callable"])
    installed = installer(
        base_module=_BASE, predecessor_gate=predecessor_gate,
        successor_policy=policy, policy_root=policy_root,
    )
    if not callable(installed):
        raise SuccessorError("phase gate installer did not return a callable")
    return installed


def _resume_preflight(policy: dict[str, Any], marker: dict[str, Any],
                      overlay: dict[str, Any]) -> dict[str, Any]:
    """Compose v1 structural checks while intentionally replacing old health."""
    blockers = validate_policy(policy)
    blockers.extend(_predecessor_errors(policy, marker, overlay, deep=False))
    plan = stacked._read_json(stacked.PLAN)
    campaign = stacked._read_json(stacked.CAMPAIGN)
    state = stacked._read_json(stacked.QUEUE_STATE)
    try:
        stacked._validate_live_documents(plan, campaign, state)
    except stacked.OverlayError as exc:
        blockers.append(str(exc))
    if overlay.get("source_bindings", {}).get("plan_sha256") != plan.get("plan_sha256"):
        blockers.append("live plan differs from predecessor overlay")
    if not stacked._reference_matches(
            overlay.get("source_bindings", {}).get("observer_source")):
        blockers.append("reviewed observer source differs from predecessor overlay")
    blockers.extend(stacked._observer_structure_errors())
    if not any(row.startswith("live observer") for row in blockers):
        observer = stacked._read_json(stacked.OBSERVER_STATE)
        if overlay.get("simulation", {}).get("gpt_oss_120b_execution_ready") is True \
                and observer.get("gpt_oss_120b_execution_ready") is not True:
            blockers.append("live observer regressed from staged 120B readiness")
    blockers.extend(v1._terminal_subset_errors(overlay, campaign))
    blockers.extend(v1._active_generation_errors(overlay))
    try:
        snapshot = _ORIGINAL_RESOURCE_SNAPSHOT(str(ROOT))
    except Exception as exc:
        snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    health = resource_health(snapshot, stacked._thermal_probe())
    if health.get("ok") is not True:
        blockers.extend(f"resource: {row}" for row in health["blockers"])
    return {"ready": not blockers, "blockers": blockers,
            "mode": "controlled-swap-resume-safe", "resource_health": health}


def configure(overlay: dict[str, Any], policy: dict[str, Any], *,
              policy_path: Path) -> None:
    """Retain v1 behavior, then install only reviewed successor hooks."""
    global _POLICY, _POLICY_PATH, _PREDECESSOR_OVERLAY
    errors = validate_policy(policy, policy_path=policy_path)
    if errors:
        raise SuccessorError("successor configure refused: " + "; ".join(errors))
    # v1 validates and configures its exact old overlay first.
    v1.configure(overlay)
    _BASE.SWAP_TOLERANCE_MB = SWAP_USED_MB_MAX
    _BASE.ram_scheduler.resource_snapshot = _strict_resource_snapshot
    stacked.resource_health = resource_health
    controlled_gate = _install_phase_gate(
        policy, _policy_root(policy_path), _controlled_execution_gate
    )
    _BASE._execution_resource_gate = controlled_gate
    _BASE._owner_alive = _owner_alive
    _BASE.start_queue = _start_queue
    _POLICY, _POLICY_PATH, _PREDECESSOR_OVERLAY = policy, policy_path, overlay


def _owner_alive(record: Any, plan: dict[str, Any]) -> bool:
    if not isinstance(record, dict) or record.get("schema") != _BASE.PID_SCHEMA \
            or record.get("version") != _BASE.VERSION \
            or record.get("plan_sha256") != plan.get("plan_sha256") \
            or record.get("pid_record_sha256") != _BASE._hash_value(
                _BASE._without(record, "pid_record_sha256")
            ):
        return False
    nonce = record.get("ownership_nonce")
    identity = _BASE._process_identity(record.get("pid"))
    if identity is None or not isinstance(nonce, str) \
            or _BASE.NONCE_RE.fullmatch(nonce) is None:
        return False
    command, started = identity
    try:
        tokens = shlex.split(command)
        entrypoints = [index for index, token in enumerate(tokens)
                       if Path(token).is_absolute()
                       and Path(token).resolve(strict=False) == SCRIPT]
    except ValueError:
        return False
    return (started == record.get("process_started")
            and hashlib.sha256(command.encode("utf-8")).hexdigest()
            == record.get("process_command_sha256")
            and len(entrypoints) == 1
            and entrypoints[0] + 1 < len(tokens)
            and tokens[entrypoints[0] + 1] == "run"
            and tokens.count("--nonce") == 1
            and f"--nonce {nonce}" in command)


def _start_queue() -> int:
    if _POLICY is None or _POLICY_PATH is None or _PREDECESSOR_OVERLAY is None \
            or _SUCCESSOR_MARKER is None or _SUCCESSOR_MARKER_PATH is None:
        raise SuccessorError("successor is not configured")
    plan = _BASE._load_plan()
    owner = _BASE._read_json(_BASE.PID_FILE, {})
    if _owner_alive(owner, plan):
        print(f"[doctor-v5-controlled-swap] already active pid={owner['pid']}")
        return 0
    if v1._owner_alive(owner, plan):
        raise SuccessorError("predecessor Hawking queue PID is still active")
    control = _BASE._load_control(plan)
    if control["mode"] != "run":
        _BASE.set_control("run")
    nonce = secrets.token_hex(16)
    command = [sys.executable, str(SCRIPT), "run", "--nonce", nonce]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    env = os.environ.copy()
    env[stacked.ENV_OVERLAY] = str(stacked.DEFAULT_OVERLAY.resolve())
    env[stacked.ENV_OVERLAY_SHA256] = _PREDECESSOR_OVERLAY["overlay_sha256"]
    env[ENV_POLICY] = str(_POLICY_PATH.resolve())
    env[ENV_POLICY_SHA256] = _POLICY["policy_sha256"]
    env[ENV_MARKER] = str(_SUCCESSOR_MARKER_PATH.resolve())
    env[ENV_MARKER_SHA256] = _SUCCESSOR_MARKER["marker_sha256"]
    _BASE.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _BASE.LOG_FILE.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=_BASE.ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True, shell=False,
        )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = _BASE._read_json(_BASE.PID_FILE, {})
        if record.get("ownership_nonce") == nonce and _owner_alive(record, plan):
            print(f"[doctor-v5-controlled-swap] detached pid={record['pid']} "
                  f"log={_BASE.LOG_FILE}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise SuccessorError("controlled-swap detached ownership handshake failed")


def main() -> int:
    global _SUCCESSOR_MARKER, _SUCCESSOR_MARKER_PATH
    _verify_static_bindings()
    policy, policy_path = load_successor_policy()
    successor_marker, marker_path = load_successor_marker(policy, policy_path)
    predecessor_marker, overlay = _predecessor_documents()
    preflight = _resume_preflight(policy, predecessor_marker, overlay)
    if preflight.get("ready") is not True:
        raise SuccessorError("controlled-swap successor refused: "
                             + "; ".join(preflight["blockers"]))
    _SUCCESSOR_MARKER, _SUCCESSOR_MARKER_PATH = successor_marker, marker_path
    configure(overlay, policy, policy_path=policy_path)
    return int(_BASE.main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SuccessorError, stacked.OverlayError,
            accel_loader.AccelerationBindingError) as exc:
        print(f"doctor_v5_controlled_swap_successor: {exc}", file=sys.stderr)
        raise SystemExit(2)
