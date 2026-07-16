#!/usr/bin/env python3.12
"""Hash-bound aggressive-v2 Doctor V5 supervisor.

This is a separate, default-off live consumer for a future quiescent generation.
It loads the exact frozen Ultra queue and replaces only admission/resource hooks.
The consumer never writes runtime specifications, the adapter registry, completed
evidence, or defaults.  Those files must already have been atomically promoted by
an external transaction whose immutable packet and rollback manifest are bound by
the active marker.

Production activation requires exact vendor selections for 8/12/16/20 threads.
There is deliberately no nominal-tier fallback.  RAM reservations, the 24-core
pack, and bounded swap transitions come from
``doctor_v5_aggressive_admission_policy``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from fractions import Fraction
from typing import Any, Iterable

import doctor_v5_accel_loader as accel_loader
import doctor_v5_accelerated_resource_policy as resource_policy
import doctor_v5_aggressive_admission_policy as aggressive


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
STAGE_ROOT = ULTRA_ROOT / "staged_acceleration/aggressive_v2"
BASE_PATH = HERE / "doctor_v5_ultra_queue.py"
MARKER = STAGE_ROOT / "active_aggressive_stack.json"
AGGRESSIVE_AUTORESUME = HERE / "doctor_v5_ultra_aggressive_autoresume.py"

BASE_SHA256 = "45ef2de60d690985d37560f77988c76df298300e47d034e0cf67029227ca74ed"
ACCEL_LOADER_SHA256 = "81adc24e3f6a50cdbd31aa368ed60ab42cf0942b1477075455735984587038f3"
AGGRESSIVE_POLICY_SHA256 = "d9e11003f630c4dbbf0ae68ea4395ae9160b0240d0ba936e01c66ba0995de745"
RESOURCE_POLICY_SHA256 = "8ddd44ee84e8c995bdfe9fbcae74d5207f2beb3b1833599e6f1690ec3e0eee47"

ENV_OVERLAY = "DOCTOR_V5_AGGRESSIVE_ADMISSION_OVERLAY"
ENV_OVERLAY_SHA256 = "DOCTOR_V5_AGGRESSIVE_ADMISSION_SHA256"
ENV_MARKER_SHA256 = "DOCTOR_V5_AGGRESSIVE_MARKER_SHA256"

MARKER_SCHEMA = "hawking.doctor_v5_aggressive_active_marker.v1"
PACKET_SCHEMA = "hawking.doctor_v5_aggressive_pending_runtime.v1"
ROLLBACK_SCHEMA = "hawking.doctor_v5_aggressive_rollback_manifest.v1"
SOURCE_BOUND_CONTRACT_SCHEMA = (
    "hawking.doctor_v5_reviewed_source_bound_execution_thread_contract.v1"
)
CPU_STATE_SCHEMA = "hawking.doctor_v5_global_cpu_launch_state.v1"
VERSION = "2026-07-15.1"
MAX_JSON_BYTES = 64 * 1024 * 1024
SWAP_DECISION_MAX_AGE_SECONDS = 2.0
CHECKPOINT_SHED_GRACE_SECONDS = 60.0
CPU_DECISION_MAX_AGE_SECONDS = 2.0
CPU_TARGET_CORES = 24.0
CPU_SATURATED_HEADROOM_CORES = 0.5
CPU_RECOVERY_HEADROOM_CORES = 2.0
CPU_RECOVERY_SAMPLES = 3
QWEN_FAMILY = "qwen2.5-dense"
GPTOSS_FAMILY = "gpt-oss-moe"
ALLOWED_MODEL_FAMILIES = frozenset({QWEN_FAMILY, GPTOSS_FAMILY})
CPU_USAGE_RE = re.compile(
    r"CPU usage:\s*([0-9]+(?:\.[0-9]+)?)% user,\s*"
    r"([0-9]+(?:\.[0-9]+)?)% sys,.*?"
    r"([0-9]+(?:\.[0-9]+)?)% idle"
)

_BASE = accel_loader.load_frozen(
    "doctor_v5_ultra_queue_aggressive_frozen", BASE_PATH, BASE_SHA256
)
_ORIGINAL_SCAN_HEADS = _BASE._scan_runnable_heads
_ORIGINAL_RESOURCE_GATE = _BASE._resource_gate
_ORIGINAL_ENFORCE_POOL_BUDGET = _BASE._enforce_pool_budget

_OVERLAY: dict[str, Any] | None = None
_MARKER: dict[str, Any] | None = None
_PROFILE_BY_CELL: dict[str, dict[str, Any]] = {}
_SWAP_STATE_PATH: Path | None = None
_CPU_STATE_PATH: Path | None = None
_LAST_SWAP_DECISION: dict[str, Any] | None = None
_LAST_SWAP_DECISION_EPOCH = 0.0
_LAST_CPU_DECISION: dict[str, Any] | None = None
_LAST_CPU_DECISION_EPOCH = 0.0


class AggressiveQueueError(RuntimeError):
    """The aggressive live generation is absent, stale, or unsafe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and aggressive.SHA256_RE.fullmatch(value) is not None


def _verify_static_bindings() -> None:
    for path, expected, label in (
        (Path(accel_loader.__file__), ACCEL_LOADER_SHA256, "acceleration loader"),
        (Path(resource_policy.__file__), RESOURCE_POLICY_SHA256,
         "accelerated resource policy"),
        (Path(aggressive.__file__), AGGRESSIVE_POLICY_SHA256,
         "aggressive admission policy"),
    ):
        observed, _ = accel_loader.hash_file(path)
        if observed != expected:
            raise AggressiveQueueError(
                f"{label} source drifted: expected={expected} observed={observed}"
            )


def _confined(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise AggressiveQueueError(f"path escapes its activation root: {resolved}") from exc
    return resolved


def _read_json(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    resolved = _confined(path, root)
    info = resolved.lstat()
    if resolved.is_symlink() or not resolved.is_file() or info.st_size > MAX_JSON_BYTES:
        raise AggressiveQueueError(f"unsafe activation JSON: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggressiveQueueError(f"cannot read activation JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggressiveQueueError(f"activation JSON root is not an object: {resolved}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = accel_loader.hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _artifact_matches(row: Any, *, expected_path: Path | None = None,
                      cache: dict[str, tuple[str, int]] | None = None) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    try:
        path = Path(row["path"]).resolve(strict=True)
        path.relative_to(ROOT.resolve())
        if expected_path is not None and path != expected_path.resolve(strict=True):
            return False
        observed = cache.get(str(path)) if cache is not None else None
        if observed is None:
            observed = accel_loader.hash_file(path)
            if cache is not None:
                cache[str(path)] = observed
    except (OSError, KeyError, TypeError, ValueError,
            accel_loader.AccelerationBindingError):
        return False
    return observed == (row["sha256"], row["bytes"])


def _rollback_errors(document: Any, *, generation_id: str) -> list[str]:
    required = {
        "schema", "version", "generation_id", "predecessor_marker_sha256",
        "activation_cas", "restore_inventory", "preserved_result_directories",
        "completed_evidence_mutation_permitted",
        "result_directory_deletion_permitted", "source_deletion_permitted",
        "rollback_manifest_sha256",
    }
    if not isinstance(document, dict) or set(document) != required:
        return ["rollback manifest keys are invalid"]
    errors: list[str] = []
    if document.get("schema") != ROLLBACK_SCHEMA or document.get("version") != VERSION \
            or document.get("generation_id") != generation_id:
        errors.append("rollback manifest identity is invalid")
    if document.get("rollback_manifest_sha256") != _hash_value(
            _without(document, "rollback_manifest_sha256")):
        errors.append("rollback manifest hash mismatch")
    predecessor = document.get("predecessor_marker_sha256")
    if predecessor is not None and not _valid_sha(predecessor):
        errors.append("rollback predecessor marker hash is invalid")
    cas = document.get("activation_cas")
    if not isinstance(cas, dict) or set(cas) != {
            "plan_sha256", "state_sha256", "campaign_sha256", "registry_sha256"} \
            or any(not _valid_sha(value) for value in cas.values()):
        errors.append("rollback activation CAS is invalid")
    inventory = document.get("restore_inventory")
    if not isinstance(inventory, list) or not inventory \
            or any(not isinstance(row, dict) or set(row) != {
                "role", "target", "backup", "sha256", "bytes"} for row in inventory):
        errors.append("rollback restore inventory is invalid")
    preserved = document.get("preserved_result_directories")
    if not isinstance(preserved, list) or any(
            not isinstance(row, dict) or set(row) != {
                "cell_id", "live_path", "preserved_path"} for row in preserved):
        errors.append("rollback preserved-result inventory is invalid")
    if document.get("completed_evidence_mutation_permitted") is not False \
            or document.get("result_directory_deletion_permitted") is not False \
            or document.get("source_deletion_permitted") is not False:
        errors.append("rollback mutation boundary is not fail-closed")
    return errors


def _marker_errors(marker: Any, *, marker_path: Path = MARKER,
                   verify_files: bool = True) -> list[str]:
    required = {
        "schema", "version", "activated_at", "generation_id",
        "activation_state_sha256", "overlay", "overlay_sha256", "generation",
        "generation_sha256", "aggressive_queue", "aggressive_autoresume",
        "aggressive_policy", "accelerated_resource_policy", "rollback_manifest",
        "swap_state", "cpu_state",
        "completed_evidence_mutation_permitted", "runtime_defaults_mutation_permitted",
        "marker_sha256",
    }
    if not isinstance(marker, dict) or set(marker) != required:
        return ["aggressive marker keys are invalid"]
    errors: list[str] = []
    if marker.get("schema") != MARKER_SCHEMA or marker.get("version") != VERSION \
            or not isinstance(marker.get("generation_id"), str) \
            or not marker["generation_id"]:
        errors.append("aggressive marker identity is invalid")
    if marker.get("marker_sha256") != _hash_value(_without(marker, "marker_sha256")):
        errors.append("aggressive marker canonical hash mismatch")
    if not _valid_sha(marker.get("activation_state_sha256")) \
            or not _valid_sha(marker.get("overlay_sha256")) \
            or not _valid_sha(marker.get("generation_sha256")):
        errors.append("aggressive marker hash fields are invalid")
    if marker.get("completed_evidence_mutation_permitted") is not False \
            or marker.get("runtime_defaults_mutation_permitted") is not False:
        errors.append("aggressive marker permits forbidden mutation")
    swap = marker.get("swap_state")
    if not isinstance(swap, dict) or set(swap) != {
            "path", "sealed_baseline_swap_mb", "initial_state_sha256"} \
            or isinstance(swap.get("sealed_baseline_swap_mb"), bool) \
            or not isinstance(swap.get("sealed_baseline_swap_mb"), (int, float)) \
            or not math.isfinite(float(swap["sealed_baseline_swap_mb"])) \
            or float(swap["sealed_baseline_swap_mb"]) < 0 \
            or not _valid_sha(swap.get("initial_state_sha256")):
        errors.append("aggressive marker swap-state binding is invalid")
    cpu = marker.get("cpu_state")
    if not isinstance(cpu, dict) or set(cpu) != {
            "path", "target_cores", "initial_state_sha256"} \
            or cpu.get("target_cores") != CPU_TARGET_CORES \
            or not _valid_sha(cpu.get("initial_state_sha256")):
        errors.append("aggressive marker global-CPU-state binding is invalid")
    if not verify_files:
        return errors
    cache: dict[str, tuple[str, int]] = {}
    bindings = (
        (marker.get("aggressive_queue"), Path(__file__), "queue"),
        (marker.get("aggressive_autoresume"), AGGRESSIVE_AUTORESUME, "autoresume"),
        (marker.get("aggressive_policy"), Path(aggressive.__file__), "policy"),
        (marker.get("accelerated_resource_policy"), Path(resource_policy.__file__),
         "accelerated resource policy"),
    )
    for row, expected, label in bindings:
        if not _artifact_matches(row, expected_path=expected, cache=cache):
            errors.append(f"aggressive marker {label} source binding changed")
    try:
        overlay_path = _confined(Path(marker["overlay"]["path"]), STAGE_ROOT)
        packet_path = _confined(Path(marker["generation"]["path"]), STAGE_ROOT)
        rollback_path = _confined(Path(marker["rollback_manifest"]["path"]), STAGE_ROOT)
        swap_path = Path(swap["path"]).resolve(strict=True)
        swap_path.relative_to(STAGE_ROOT.resolve(strict=True))
        cpu_path = Path(cpu["path"]).resolve(strict=True)
        cpu_path.relative_to(STAGE_ROOT.resolve(strict=True))
    except (OSError, KeyError, TypeError, ValueError, AggressiveQueueError):
        errors.append("aggressive marker activation paths are invalid")
        return errors
    if not _artifact_matches(marker.get("overlay"), expected_path=overlay_path, cache=cache):
        errors.append("aggressive overlay artifact changed")
    if not _artifact_matches(marker.get("generation"), expected_path=packet_path, cache=cache):
        errors.append("aggressive pending generation artifact changed")
    if not _artifact_matches(marker.get("rollback_manifest"),
                             expected_path=rollback_path, cache=cache):
        errors.append("aggressive rollback manifest artifact changed")
    try:
        overlay = _read_json(overlay_path, root=STAGE_ROOT)
        packet = _read_json(packet_path, root=STAGE_ROOT)
        rollback = _read_json(rollback_path, root=STAGE_ROOT)
        swap_document = _read_json(swap_path, root=STAGE_ROOT)
        cpu_document = _read_json(cpu_path, root=STAGE_ROOT)
    except AggressiveQueueError as exc:
        errors.append(str(exc))
        return errors
    if overlay.get("overlay_sha256") != marker.get("overlay_sha256"):
        errors.append("aggressive marker overlay canonical binding changed")
    if packet.get("packet_sha256") != marker.get("generation_sha256") \
            or packet.get("generation_id") != marker.get("generation_id"):
        errors.append("aggressive marker generation canonical binding changed")
    errors.extend(_rollback_errors(rollback, generation_id=marker["generation_id"]))
    baseline = swap["sealed_baseline_swap_mb"]
    if aggressive.validate_swap_state(swap_document,
                                      sealed_baseline_swap_mb=float(baseline)):
        errors.append("aggressive live swap state is invalid")
    if _validate_cpu_state(cpu_document):
        errors.append("aggressive live global-CPU state is invalid")
    elif cpu_document.get("initial_state_sha256") != cpu.get("initial_state_sha256"):
        errors.append("aggressive global-CPU activation seed changed")
    return errors


def _load_activated_marker(*, marker_path: Path = MARKER,
                           environ: dict[str, str] | os._Environ[str] = os.environ) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        marker = _read_json(marker_path, root=STAGE_ROOT)
    except AggressiveQueueError as exc:
        raise AggressiveQueueError(f"aggressive marker unavailable: {exc}") from exc
    marker_key = environ.get(ENV_MARKER_SHA256)
    overlay_key = environ.get(ENV_OVERLAY_SHA256)
    overlay_raw = environ.get(ENV_OVERLAY)
    if marker_key != marker.get("marker_sha256") \
            or overlay_key != marker.get("overlay_sha256") \
            or not isinstance(overlay_raw, str):
        raise AggressiveQueueError("all aggressive activation keys are required")
    errors = _marker_errors(marker, marker_path=marker_path, verify_files=True)
    if errors:
        raise AggressiveQueueError("invalid aggressive marker: " + "; ".join(errors))
    overlay_path = Path(marker["overlay"]["path"]).resolve(strict=True)
    if Path(overlay_raw).resolve(strict=True) != overlay_path:
        raise AggressiveQueueError("overlay path activation key differs from the marker")
    overlay = _read_json(overlay_path, root=STAGE_ROOT)
    return marker, overlay


def _strict_overlay_errors(overlay: dict[str, Any]) -> list[str]:
    errors = list(aggressive.validate_overlay(overlay))
    policy = overlay.get("resource_policy", {})
    if policy.get("cpu_budget_cores") != 24 \
            or policy.get("admission_ceiling_bytes") != 66_000_000_000:
        errors.append("aggressive 24-core/66GB envelope changed")
    pending = overlay.get("pending_profiles")
    if not isinstance(pending, list) or not pending:
        errors.append("aggressive overlay has no pending profile rows")
        pending = []
    qwen_rows = [row for row in pending if isinstance(row, dict)
                 and row.get("model_family") == QWEN_FAMILY]
    qualification = overlay.get("thread_profile_qualification")
    if not isinstance(qualification, dict):
        qualification = {}
    if qwen_rows:
        if qualification.get("status") != "qualified":
            errors.append("qualified exact 8/12/16/20 Qwen vendor profile is absent")
        if qualification.get("required_threads") \
                != list(aggressive.REQUIRED_THREAD_PARITY):
            errors.append("Qwen thread-profile candidate set is not exact 8/12/16/20")
        for name in ("contract", "profile", "binary"):
            if not aggressive._reference_matches(qualification.get(name)):
                errors.append(f"qualified Qwen thread-profile {name} artifact changed")
    selections = qualification.get("selections", {})
    seen: set[str] = set()
    for row in pending:
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            errors.append(f"invalid/duplicate aggressive pending cell: {cell_id}")
            continue
        seen.add(cell_id)
        family = row.get("model_family") if isinstance(row, dict) else None
        reserve = row.get("reservation_bytes") if isinstance(row, dict) else None
        if family not in ALLOWED_MODEL_FAMILIES:
            errors.append(f"pending cell has unknown execution family: {cell_id}")
            continue
        if isinstance(reserve, bool) or not isinstance(reserve, int) \
                or not 1 <= reserve <= aggressive.ADMISSION_CEILING_BYTES:
            errors.append(f"pending cell has invalid RAM reservation: {cell_id}")
        if family == GPTOSS_FAMILY:
            # GPT-OSS is not a Qwen nominal-tier fallback. Its reviewed runtime
            # packet supplies an independent source-bound thread contract.
            if row.get("threads") is not None \
                    or row.get("thread_selection_sha256") is not None \
                    or row.get("selection_source") is not None \
                    or row.get("exact_parity_approved") is not False \
                    or row.get("all_four_candidates_eligible") is not False:
                errors.append(f"GPT-OSS row invents a Qwen vendor profile: {cell_id}")
            continue
        threads = row.get("threads")
        selection_sha = row.get("thread_selection_sha256")
        key = json.dumps([row.get("model_label"), row.get("rate_id")],
                         separators=(",", ":"), ensure_ascii=False)
        selection = selections.get(key) if isinstance(selections, dict) else None
        if threads not in aggressive.REQUIRED_THREAD_PARITY \
                or row.get("exact_parity_approved") is not True \
                or row.get("all_four_candidates_eligible") is not True \
                or not _valid_sha(selection_sha) \
                or not isinstance(selection, dict) \
                or selection.get("selection_sha256") != selection_sha \
                or selection.get("selected_threads") != threads:
            errors.append(
                f"Qwen cell lacks exact qualified thread selection: {cell_id}"
            )
    return errors


def _runtime_profile_errors(cell: dict[str, Any], spec: dict[str, Any],
                            profile: dict[str, Any],
                            qualification: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if cell.get("cell_id") != profile.get("cell_id"):
        errors.append("runtime/profile cell identity differs")
    if cell.get("model_family") != QWEN_FAMILY \
            or profile.get("model_family") != QWEN_FAMILY \
            or spec.get("model_family") != QWEN_FAMILY:
        errors.append("Qwen runtime/profile family differs")
    threads = spec.get("resources", {}).get("threads")
    if threads != profile.get("threads") \
            or threads not in aggressive.REQUIRED_THREAD_PARITY:
        errors.append("runtime does not use its exact selected thread count")
    if profile.get("exact_parity_approved") is not True \
            or profile.get("all_four_candidates_eligible") is not True \
            or not _valid_sha(profile.get("thread_selection_sha256")):
        errors.append("runtime profile lacks exact all-candidate parity")
    binary = qualification.get("binary")
    quantizers = [row for row in spec.get("inputs", [])
                  if isinstance(row, dict) and row.get("role") == "quantizer"]
    if not isinstance(binary, dict) or len(quantizers) != 1 \
            or quantizers[0].get("sha256") != binary.get("sha256") \
            or quantizers[0].get("bytes") != binary.get("bytes"):
        errors.append("runtime quantizer differs from the qualified profile binary")
    return errors


def _gptoss_contract_profile(cell: dict[str, Any], spec: dict[str, Any],
                             runtime_artifact: dict[str, Any],
                             contract_artifact: Any, *,
                             cache: dict[str, tuple[str, int]]) \
        -> tuple[dict[str, Any] | None, list[str]]:
    """Validate GPT-OSS's independent reviewed source/thread contract.

    The contract is deliberately not represented as a Qwen vendor selection.
    It binds the promoted runtime spec, every runtime input reference, the exact
    thread declaration, and at least one physical exact-output review receipt.
    """
    errors: list[str] = []
    if cell.get("model_family") != GPTOSS_FAMILY \
            or spec.get("model_family") != GPTOSS_FAMILY:
        return None, ["GPT-OSS runtime/campaign family differs"]
    if not _artifact_matches(contract_artifact, cache=cache):
        return None, ["GPT-OSS reviewed execution/thread contract changed"]
    try:
        contract = _read_json(Path(contract_artifact["path"]))
    except (AggressiveQueueError, KeyError, TypeError) as exc:
        return None, [f"cannot read GPT-OSS execution/thread contract: {exc}"]
    required = {
        "schema", "version", "cell_id", "model_family", "runtime_spec_sha256",
        "runtime_inputs_sha256", "selected_threads", "projected_wall_seconds",
        "exclusive_cpu", "review_status", "exact_output_receipts",
        "contract_sha256",
    }
    if set(contract) != required \
            or contract.get("schema") != SOURCE_BOUND_CONTRACT_SCHEMA \
            or contract.get("version") != VERSION \
            or contract.get("contract_sha256") != _hash_value(
                _without(contract, "contract_sha256")):
        return None, ["GPT-OSS execution/thread contract identity is invalid"]
    threads = contract.get("selected_threads")
    wall = contract.get("projected_wall_seconds")
    if contract.get("cell_id") != cell.get("cell_id") \
            or contract.get("model_family") != GPTOSS_FAMILY \
            or contract.get("runtime_spec_sha256") != runtime_artifact.get("sha256") \
            or contract.get("runtime_inputs_sha256") \
            != _hash_value(spec.get("inputs", [])):
        errors.append("GPT-OSS contract/runtime/source-input binding differs")
    if isinstance(threads, bool) or not isinstance(threads, int) \
            or not 1 <= threads <= int(CPU_TARGET_CORES) \
            or spec.get("resources", {}).get("threads") != threads:
        errors.append("GPT-OSS runtime differs from its reviewed thread declaration")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) <= 0:
        errors.append("GPT-OSS reviewed projected wall time is invalid")
    if contract.get("exclusive_cpu") not in {True, False} \
            or contract.get("exclusive_cpu") is not (threads >= 20):
        errors.append("GPT-OSS reviewed CPU exclusivity is invalid")
    if contract.get("review_status") != "approved-source-bound-exact-output":
        errors.append("GPT-OSS source-bound execution/thread contract is unreviewed")
    receipts = contract.get("exact_output_receipts")
    if not isinstance(receipts, list) or not receipts \
            or any(not _artifact_matches(row, cache=cache) for row in receipts) \
            or len({row.get("sha256") for row in receipts
                    if isinstance(row, dict)}) != len(receipts):
        errors.append("GPT-OSS exact-output review receipts are absent or changed")
    inputs = spec.get("inputs")
    roles = {row.get("role") for row in inputs if isinstance(row, dict)} \
        if isinstance(inputs, list) else set()
    if not any(isinstance(role, str) and (
            role.startswith("source_") or role.startswith("source:")
            or role.startswith("source_shard:") or role in {
                "parameter_manifest", "source_census", "source_seal",
                "source_inventory", "source_work_plan"}) for role in roles):
        errors.append("GPT-OSS runtime has no source-bound input authority")
    if errors:
        return None, errors
    assert isinstance(threads, int) and isinstance(wall, (int, float))
    return {
        **profile_identity(cell),
        "reservation_bytes": None,
        "threads": threads,
        "projected_wall_seconds": float(wall),
        "exclusive_cpu_profile": contract["exclusive_cpu"],
        "selection_source": "reviewed-gptoss-source-bound-thread-contract",
        "source_bound_contract_sha256": contract["contract_sha256"],
    }, []


def profile_identity(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": cell.get("cell_id"), "priority": cell.get("priority"),
        "model_family": cell.get("model_family"),
    }


def _terminal_seal_errors(seal: Any, plan: dict[str, Any],
                          state: dict[str, Any], *,
                          cache: dict[str, tuple[str, int]]) -> list[str]:
    rows = seal.get("rows") if isinstance(seal, dict) else None
    if not isinstance(rows, list) or seal.get("rows_sha256") != _hash_value(rows):
        return ["aggressive terminal seal is invalid"]
    cells = {row["cell_id"]: row for row in plan["cells"]}
    errors: list[str] = []
    for row in rows:
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        if cell_id not in cells or state["cells"][cell_id]["status"] != row.get("status") \
                or cells[cell_id]["cell_identity_sha256"] \
                != row.get("cell_identity_sha256"):
            errors.append(f"sealed terminal cell changed: {cell_id}")
            continue
        if any(not _artifact_matches(artifact, cache=cache)
               for artifact in row.get("artifacts", [])):
            errors.append(f"sealed terminal evidence changed: {cell_id}")
    return errors


def _active_generation_errors(marker: dict[str, Any], overlay: dict[str, Any], *,
                              plan: dict[str, Any] | None = None,
                              state: dict[str, Any] | None = None) -> list[str]:
    errors = _strict_overlay_errors(overlay)
    try:
        packet = _read_json(Path(marker["generation"]["path"]), root=STAGE_ROOT)
    except (AggressiveQueueError, KeyError, TypeError) as exc:
        return errors + [f"cannot load aggressive generation: {exc}"]
    required = {
        "schema", "version", "created_at", "generation_id", "plan_sha256",
        "overlay_sha256", "thread_profile_qualification_sha256", "registry",
        "runtime_specs", "terminal_seal", "rollback_manifest",
        "completed_evidence_mutation_permitted", "runtime_defaults_mutation_permitted",
        "source_deletion_permitted", "packet_sha256",
    }
    if set(packet) != required or packet.get("schema") != PACKET_SCHEMA \
            or packet.get("version") != VERSION \
            or packet.get("packet_sha256") != _hash_value(
                _without(packet, "packet_sha256")):
        return errors + ["aggressive pending generation identity is invalid"]
    if packet.get("generation_id") != marker.get("generation_id") \
            or packet.get("packet_sha256") != marker.get("generation_sha256") \
            or packet.get("overlay_sha256") != overlay.get("overlay_sha256") \
            or packet.get("thread_profile_qualification_sha256") \
            != overlay.get("thread_profile_qualification", {}).get(
                "qualification_sha256"):
        errors.append("aggressive packet/marker/overlay bindings differ")
    if packet.get("completed_evidence_mutation_permitted") is not False \
            or packet.get("runtime_defaults_mutation_permitted") is not False \
            or packet.get("source_deletion_permitted") is not False:
        errors.append("aggressive packet permits forbidden mutation")
    if packet.get("rollback_manifest") != marker.get("rollback_manifest"):
        errors.append("aggressive packet rollback binding differs from marker")
    plan = plan or _BASE._load_plan()
    state = state or _BASE._load_state(plan)
    if packet.get("plan_sha256") != plan.get("plan_sha256") \
            or overlay.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("live plan differs from aggressive generation")
    cache: dict[str, tuple[str, int]] = {}
    if not _artifact_matches(packet.get("registry"),
                             expected_path=_BASE.REGISTRY_PATH, cache=cache):
        errors.append("aggressive promoted registry changed")
    cells = {row["cell_id"]: row for row in plan["cells"]}
    profiles = {row["cell_id"]: row for row in overlay.get("pending_profiles", [])
                if isinstance(row, dict) and isinstance(row.get("cell_id"), str)}
    rows = packet.get("runtime_specs")
    if not isinstance(rows, list) or not rows:
        errors.append("aggressive runtime generation is empty")
        rows = []
    seen: set[str] = set()
    qualification = overlay.get("thread_profile_qualification", {})
    for row in rows:
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        cell, profile = cells.get(cell_id), profiles.get(cell_id)
        family = cell.get("model_family") if isinstance(cell, dict) else None
        qwen_keys = {"cell_id", "model_family", "runtime_spec", "selected_threads",
                     "thread_selection_sha256"}
        gptoss_keys = {"cell_id", "model_family", "runtime_spec",
                       "source_bound_execution_contract"}
        expected_keys = qwen_keys if family == QWEN_FAMILY else gptoss_keys
        if cell is None or profile is None or cell_id in seen \
                or family not in ALLOWED_MODEL_FAMILIES \
                or profile.get("model_family") != family \
                or not isinstance(row, dict) or set(row) != expected_keys \
                or row.get("model_family") != family:
            errors.append(f"aggressive runtime row is invalid: {cell_id}")
            continue
        seen.add(cell_id)
        target = (ROOT / cell["runtime_spec_path"]).resolve(strict=True)
        if not _artifact_matches(row.get("runtime_spec"),
                                 expected_path=target, cache=cache):
            errors.append(f"aggressive runtime spec changed: {cell_id}")
            continue
        try:
            spec = _read_json(target)
        except AggressiveQueueError as exc:
            errors.append(str(exc)); continue
        runtime, _, runtime_errors = _BASE._validate_runtime_spec(
            cell, spec, target, verify_inputs=False
        )
        if runtime is None or runtime_errors:
            errors.append(f"aggressive runtime spec is semantically invalid: {cell_id}")
        if family == QWEN_FAMILY:
            if row.get("selected_threads") != profile.get("threads") \
                    or row.get("thread_selection_sha256") \
                    != profile.get("thread_selection_sha256"):
                errors.append(f"Qwen runtime selection differs from overlay: {cell_id}")
            errors.extend(f"{cell_id}: {item}" for item in
                          _runtime_profile_errors(cell, spec, profile, qualification))
        else:
            _derived, contract_errors = _gptoss_contract_profile(
                cell, spec, row["runtime_spec"],
                row.get("source_bound_execution_contract"), cache=cache,
            )
            errors.extend(f"{cell_id}: {item}" for item in contract_errors)
        for input_row in spec.get("inputs", []):
            role = input_row.get("role") if isinstance(input_row, dict) else None
            if isinstance(role, str) and role.startswith("source_shard:"):
                continue
            artifact = ({name: input_row.get(name)
                         for name in ("path", "sha256", "bytes")}
                        if isinstance(input_row, dict) else None)
            if not _artifact_matches(artifact, cache=cache):
                errors.append(f"aggressive runtime input changed: {cell_id}:{role}")
    if seen != set(profiles):
        errors.append("aggressive runtime packet does not cover the exact profile set")
    errors.extend(_terminal_seal_errors(packet.get("terminal_seal"), plan, state,
                                        cache=cache))
    return errors


def _bound_pack_profiles(marker: dict[str, Any], overlay: dict[str, Any],
                         plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Materialize only already-validated runtime admission claims in memory."""
    packet = _read_json(Path(marker["generation"]["path"]), root=STAGE_ROOT)
    cells = {row["cell_id"]: row for row in plan["cells"]}
    overlay_rows = {row["cell_id"]: row for row in overlay["pending_profiles"]}
    profiles: dict[str, dict[str, Any]] = {}
    cache: dict[str, tuple[str, int]] = {}
    for packet_row in packet["runtime_specs"]:
        cell_id = packet_row["cell_id"]
        cell, staged = cells[cell_id], overlay_rows[cell_id]
        if cell["model_family"] == QWEN_FAMILY:
            profiles[cell_id] = dict(staged)
            continue
        target = (ROOT / cell["runtime_spec_path"]).resolve(strict=True)
        spec = _read_json(target)
        derived, errors = _gptoss_contract_profile(
            cell, spec, packet_row["runtime_spec"],
            packet_row["source_bound_execution_contract"], cache=cache,
        )
        if derived is None or errors:
            raise AggressiveQueueError(
                f"validated GPT-OSS profile could not be reconstructed: {cell_id}: "
                + "; ".join(errors)
            )
        profiles[cell_id] = {
            **staged, **derived,
            "reservation_bytes": staged["reservation_bytes"],
            "exact_parity_approved": False,
            "all_four_candidates_eligible": False,
        }
    return profiles


def _mixed_profile_errors(row: Any) -> list[str]:
    if not isinstance(row, dict):
        return ["mixed-family pack candidate is not an object"]
    errors: list[str] = []
    cell_id, family = row.get("cell_id"), row.get("model_family")
    reserve, threads, wall = (row.get("reservation_bytes"), row.get("threads"),
                              row.get("projected_wall_seconds"))
    if not isinstance(cell_id, str) or not cell_id:
        errors.append("pack candidate has no cell identity")
    if family not in ALLOWED_MODEL_FAMILIES:
        errors.append(f"pack candidate family is invalid: {cell_id}")
    if isinstance(row.get("priority"), bool) \
            or not isinstance(row.get("priority"), int):
        errors.append(f"pack candidate priority is invalid: {cell_id}")
    if isinstance(reserve, bool) or not isinstance(reserve, int) \
            or not 1 <= reserve <= aggressive.ADMISSION_CEILING_BYTES:
        errors.append(f"pack candidate RAM claim is invalid: {cell_id}")
    if isinstance(threads, bool) or not isinstance(threads, int) \
            or not 1 <= threads <= int(CPU_TARGET_CORES):
        errors.append(f"pack candidate thread claim is invalid: {cell_id}")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) <= 0:
        errors.append(f"pack candidate wall-time claim is invalid: {cell_id}")
    if row.get("exclusive_cpu_profile") is not (isinstance(threads, int)
                                                  and threads >= 20):
        errors.append(f"pack candidate exclusivity is invalid: {cell_id}")
    if family == QWEN_FAMILY and (
            threads not in aggressive.REQUIRED_THREAD_PARITY
            or row.get("exact_parity_approved") is not True
            or row.get("all_four_candidates_eligible") is not True
            or row.get("selection_source")
            != "qualified-vendor-thread-profile-contract"):
        errors.append(f"Qwen pack candidate is not exact-profile bound: {cell_id}")
    if family == GPTOSS_FAMILY and (
            row.get("selection_source")
            != "reviewed-gptoss-source-bound-thread-contract"
            or not _valid_sha(row.get("source_bound_contract_sha256"))):
        errors.append(f"GPT-OSS pack candidate is not source-contract bound: {cell_id}")
    return errors


def _choose_mixed_pack(candidates: Iterable[dict[str, Any]], *,
                       active_reserved_bytes: int, active_threads: int,
                       active_lanes: int, launch_limit: int) -> dict[str, Any]:
    """Pack Qwen exact-profile and GPT-OSS source-contract rows together."""
    scalars = (active_reserved_bytes, active_threads, active_lanes, launch_limit)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in scalars):
        raise AggressiveQueueError("mixed-family pack envelope is invalid")
    ram_free = aggressive.ADMISSION_CEILING_BYTES - active_reserved_bytes
    cpu_free = int(CPU_TARGET_CORES) - active_threads
    lane_free = min(aggressive.MAX_LANES - active_lanes, launch_limit)
    if ram_free <= 0 or cpu_free <= 0 or lane_free <= 0:
        return {"selected_cell_ids": [], "reason": "no launch capacity"}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        errors = _mixed_profile_errors(candidate)
        cell_id = candidate.get("cell_id") if isinstance(candidate, dict) else None
        if errors or cell_id in seen:
            raise AggressiveQueueError("; ".join(errors or [
                f"duplicate mixed-family pack candidate: {cell_id}"
            ]))
        seen.add(cell_id)
        rows.append(dict(candidate))
    rows.sort(key=lambda row: (row["priority"], row["cell_id"]))
    rows = rows[:aggressive.MAX_PACK_CANDIDATES]
    best: tuple[tuple[Any, ...], tuple[dict[str, Any], ...]] | None = None
    for count in range(1, min(lane_free, len(rows)) + 1):
        for combination in itertools.combinations(rows, count):
            ram = sum(row["reservation_bytes"] for row in combination)
            cpu = sum(row["threads"] for row in combination)
            if ram > ram_free or cpu > cpu_free:
                continue
            if any(row["exclusive_cpu_profile"] for row in combination) \
                    and (active_lanes != 0 or len(combination) != 1):
                continue
            throughput = sum(
                Fraction(1, 1) / Fraction(str(row["projected_wall_seconds"]))
                for row in combination
            )
            wave = max(Fraction(str(row["projected_wall_seconds"]))
                       for row in combination)
            score = (throughput, -wave, cpu, ram, count,
                     -sum(row["priority"] for row in combination))
            ids = tuple(row["cell_id"] for row in combination)
            if best is None or score > best[0][0] \
                    or (score == best[0][0] and ids < best[0][1]):
                best = ((score, ids), combination)
    selected = [] if best is None else sorted(
        best[1], key=lambda row: (row["priority"], row["cell_id"])
    )
    return {
        "selected_cell_ids": [row["cell_id"] for row in selected],
        "ram_free_before": ram_free, "cpu_free_before": cpu_free,
        "reserved_after_bytes": active_reserved_bytes
            + sum(row["reservation_bytes"] for row in selected),
        "threads_after": active_threads + sum(row["threads"] for row in selected),
        "selection_basis": "mixed-reviewed-sum-inverse-wall-seconds",
        "reason": "best mixed-family reviewed subset" if selected
                  else "no candidate fits",
    }


def _cpu_state_payload(*, samples: list[float], logical_cores: int,
                       mode: str, last_sample_epoch: float,
                       last_transition_epoch: float,
                       initial_state_sha256: str,
                       recovered_from_invalid: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": CPU_STATE_SCHEMA, "version": VERSION,
        "target_cores": CPU_TARGET_CORES,
        "logical_cores": logical_cores,
        "samples": [round(float(row), 3) for row in samples],
        "mode": mode, "last_sample_epoch": float(last_sample_epoch),
        "last_transition_epoch": float(last_transition_epoch),
        "initial_state_sha256": initial_state_sha256,
        "recovered_from_invalid": recovered_from_invalid,
    }
    value["state_sha256"] = _hash_value(value)
    return value


def _validate_cpu_state(state: Any) -> list[str]:
    required = {
        "schema", "version", "target_cores", "logical_cores", "samples",
        "mode", "last_sample_epoch", "last_transition_epoch",
        "initial_state_sha256", "recovered_from_invalid", "state_sha256",
    }
    if not isinstance(state, dict) or set(state) != required:
        return ["global-CPU state keys are invalid"]
    errors: list[str] = []
    if state.get("schema") != CPU_STATE_SCHEMA or state.get("version") != VERSION \
            or state.get("target_cores") != CPU_TARGET_CORES:
        errors.append("global-CPU state identity is invalid")
    if state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        errors.append("global-CPU state hash mismatch")
    logical = state.get("logical_cores")
    if isinstance(logical, bool) or not isinstance(logical, int) or logical <= 0:
        errors.append("global-CPU logical-core count is invalid")
    samples = state.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= CPU_RECOVERY_SAMPLES \
            or any(isinstance(row, bool) or not isinstance(row, (int, float))
                   or not math.isfinite(float(row)) or float(row) < 0
                   for row in samples):
        errors.append("global-CPU occupancy window is invalid")
    if state.get("mode") not in {"green", "saturated", "unknown"}:
        errors.append("global-CPU launch mode is invalid")
    if any(isinstance(state.get(key), bool)
           or not isinstance(state.get(key), (int, float))
           or not math.isfinite(float(state[key])) or float(state[key]) < 0
           for key in ("last_sample_epoch", "last_transition_epoch")):
        errors.append("global-CPU state timestamps are invalid")
    if not _valid_sha(state.get("initial_state_sha256")):
        errors.append("global-CPU activation seed is invalid")
    if state.get("recovered_from_invalid") not in {True, False}:
        errors.append("global-CPU recovery marker is invalid")
    return errors


def _initial_cpu_state(snapshot: dict[str, Any], *, now_epoch: float) -> dict[str, Any]:
    logical = snapshot.get("logical_cores")
    occupied = snapshot.get("occupied_cores")
    if snapshot.get("ok") is not True \
            or isinstance(logical, bool) or not isinstance(logical, int) or logical <= 0 \
            or isinstance(occupied, bool) or not isinstance(occupied, (int, float)) \
            or not math.isfinite(float(occupied)) or float(occupied) < 0:
        raise AggressiveQueueError(
            "activation requires a valid total-host CPU occupancy sample"
        )
    seed = _hash_value({
        "target_cores": CPU_TARGET_CORES, "logical_cores": logical,
        "occupied_cores": round(float(occupied), 3),
        "sampled_at_epoch": float(now_epoch),
    })
    mode = ("saturated" if float(occupied)
            >= CPU_TARGET_CORES - CPU_SATURATED_HEADROOM_CORES else "green")
    return _cpu_state_payload(
        samples=[float(occupied)], logical_cores=logical, mode=mode,
        last_sample_epoch=now_epoch, last_transition_epoch=now_epoch,
        initial_state_sha256=seed, recovered_from_invalid=False,
    )


def _global_cpu_snapshot() -> dict[str, Any]:
    """Read total-host CPU, including owners that hold no Doctor lease."""
    try:
        completed = subprocess.run(
            ["top", "-l", "2", "-s", "0.2", "-n", "0"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        matches = list(CPU_USAGE_RE.finditer(completed.stdout))
        logical = os.cpu_count()
        if not matches or isinstance(logical, bool) \
                or not isinstance(logical, int) or logical <= 0:
            raise ValueError("macOS top did not expose aggregate CPU usage")
        user, system, idle = (float(value) for value in matches[-1].groups())
        if any(not math.isfinite(value) or not 0 <= value <= 100
               for value in (user, system, idle)) \
                or user + system > 100.5:
            raise ValueError("aggregate CPU percentages are outside the host envelope")
        used_percent = max(0.0, min(100.0, 100.0 - idle))
        return {
            "ok": True, "source": "macos-top-two-sample-total-host",
            "logical_cores": logical, "used_percent": round(used_percent, 3),
            "occupied_cores": round(logical * used_percent / 100.0, 3),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _advance_cpu(snapshot: dict[str, Any] | None = None, *,
                 now_epoch: float | None = None,
                 force: bool = False) -> dict[str, Any]:
    global _LAST_CPU_DECISION, _LAST_CPU_DECISION_EPOCH
    if _CPU_STATE_PATH is None or _MARKER is None:
        raise AggressiveQueueError("global-CPU launch controller is not configured")
    now = float(time.time() if now_epoch is None else now_epoch)
    if not force and _LAST_CPU_DECISION is not None \
            and now - _LAST_CPU_DECISION_EPOCH <= CPU_DECISION_MAX_AGE_SECONDS:
        return _LAST_CPU_DECISION
    try:
        state = _read_json(_CPU_STATE_PATH, root=STAGE_ROOT)
    except AggressiveQueueError:
        state = {}
    observed = snapshot or _global_cpu_snapshot()
    valid_state = not _validate_cpu_state(state)
    logical = observed.get("logical_cores")
    occupied = observed.get("occupied_cores")
    probe_valid = observed.get("ok") is True \
        and not isinstance(logical, bool) and isinstance(logical, int) and logical > 0 \
        and not isinstance(occupied, bool) and isinstance(occupied, (int, float)) \
        and math.isfinite(float(occupied)) and float(occupied) >= 0
    seed = _MARKER["cpu_state"]["initial_state_sha256"]
    if not valid_state:
        prior_samples = [CPU_TARGET_CORES]
        prior_logical = int(logical) if probe_valid else int(CPU_TARGET_CORES)
        previous_mode = "unknown"
        recovered = True
    else:
        prior_samples = [float(row) for row in state["samples"]]
        prior_logical = state["logical_cores"]
        previous_mode = state["mode"]
        recovered = state["recovered_from_invalid"]
        if state["initial_state_sha256"] != seed:
            probe_valid = False
            recovered = True
    if probe_valid:
        assert isinstance(logical, int) and isinstance(occupied, (int, float))
        samples = (prior_samples + [float(occupied)])[-CPU_RECOVERY_SAMPLES:]
        charged = max(samples)
        target = min(CPU_TARGET_CORES, float(logical))
        guard = max(0.0, float(logical) - target)
        try:
            launch = resource_policy.fixed_thread_cpu_launch_decision(
                samples, logical_cores=logical, guard_cores=guard,
                launch_threads=1.0,
            )
        except resource_policy.ResourcePolicyError:
            probe_valid = False
        else:
            available = launch["available_cpu_cores"]
            mode = ("saturated" if charged
                    >= target - CPU_SATURATED_HEADROOM_CORES else "green")
            allow_launch = mode == "green" and available >= 1.0
    if not probe_valid:
        samples = prior_samples[-CPU_RECOVERY_SAMPLES:]
        logical = prior_logical
        charged, available = max(samples), 0.0
        mode, allow_launch = "unknown", False
    transition = now if mode != previous_mode else (
        state.get("last_transition_epoch", now) if isinstance(state, dict) else now
    )
    updated = _cpu_state_payload(
        samples=samples, logical_cores=int(logical), mode=mode,
        last_sample_epoch=now, last_transition_epoch=float(transition),
        initial_state_sha256=seed, recovered_from_invalid=recovered,
    )
    aggressive._atomic_json(_CPU_STATE_PATH, updated)
    decision = {
        "mode": mode, "allow_launch": allow_launch,
        "probe_valid": probe_valid, "charged_global_cpu_cores": round(charged, 3),
        "available_cpu_cores": round(float(available), 3),
        "target_cores": CPU_TARGET_CORES, "samples": updated["samples"],
        "launch_only": True, "shed_or_stop_authority": False,
        "state_sha256": updated["state_sha256"],
    }
    decision["decision_sha256"] = _hash_value(decision)
    _LAST_CPU_DECISION, _LAST_CPU_DECISION_EPOCH = decision, now
    return decision


def _swap_snapshot() -> dict[str, Any]:
    try:
        return _BASE.ram_scheduler.resource_snapshot(str(ROOT))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _advance_swap(snapshot: dict[str, Any] | None = None, *,
                  now_epoch: float | None = None,
                  force: bool = False) -> dict[str, Any]:
    global _LAST_SWAP_DECISION, _LAST_SWAP_DECISION_EPOCH
    if _OVERLAY is None or _SWAP_STATE_PATH is None:
        raise AggressiveQueueError("aggressive swap controller is not configured")
    now = float(time.time() if now_epoch is None else now_epoch)
    if not force and _LAST_SWAP_DECISION is not None \
            and now - _LAST_SWAP_DECISION_EPOCH <= SWAP_DECISION_MAX_AGE_SECONDS:
        return _LAST_SWAP_DECISION
    baseline = float(_OVERLAY["resource_policy"]["sealed_swap_baseline_mb"])
    try:
        state = _read_json(_SWAP_STATE_PATH, root=STAGE_ROOT)
    except AggressiveQueueError:
        # The policy accepts malformed state and self-heals from the separately
        # sealed baseline. An absent file is represented as malformed, never as
        # permission to manufacture a new baseline.
        state = {}
    updated, decision = aggressive.advance_swap_state(
        state, snapshot or _swap_snapshot(), now_epoch=now,
        sealed_baseline_swap_mb=baseline,
    )
    aggressive._atomic_json(_SWAP_STATE_PATH, updated)
    decision = {**decision, "state_sha256": updated["state_sha256"]}
    decision["decision_sha256"] = _hash_value(decision)
    _LAST_SWAP_DECISION = decision
    _LAST_SWAP_DECISION_EPOCH = now
    return decision


def _profile_for_cell(cell: dict[str, Any]) -> dict[str, Any]:
    profile = _PROFILE_BY_CELL.get(cell.get("cell_id"))
    if not isinstance(profile, dict):
        raise AggressiveQueueError(f"cell has no aggressive profile: {cell.get('cell_id')}")
    return profile


def _aggressive_reservation(cell: dict[str, Any]) -> int:
    value = _profile_for_cell(cell).get("reservation_bytes")
    if isinstance(value, bool) or not isinstance(value, int) \
            or not 1 <= value <= aggressive.ADMISSION_CEILING_BYTES:
        raise AggressiveQueueError("aggressive cell reservation is invalid")
    return value


def _active_pack_claims(state: dict[str, Any]) -> tuple[int, int, int, bool]:
    active = state.get("active_children")
    if not isinstance(active, dict):
        raise AggressiveQueueError("active-child inventory is invalid")
    reserved = 0
    threads = 0
    exclusive = False
    for cell_id, child in active.items():
        profile = _PROFILE_BY_CELL.get(cell_id)
        reserve = child.get("reserved_bytes") if isinstance(child, dict) else None
        if profile is None or isinstance(reserve, bool) or not isinstance(reserve, int) \
                or reserve < 0:
            raise AggressiveQueueError("active child lacks an exact aggressive claim")
        reserved += max(reserve, int(profile["reservation_bytes"]))
        threads += int(profile["threads"])
        exclusive = exclusive or profile.get("exclusive_cpu_profile") is True
    return reserved, threads, len(active), exclusive


def _aggressive_scan_heads(plan: dict[str, Any], state: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    heads, blockers = _ORIGINAL_SCAN_HEADS(plan, state)
    swap_decision = _advance_swap()
    cpu_decision = _advance_cpu()
    if swap_decision.get("allow_launch") is not True \
            or cpu_decision.get("allow_launch") is not True:
        return [], blockers
    try:
        reserved, threads, lanes, active_exclusive = _active_pack_claims(state)
        if active_exclusive:
            return [], blockers
        candidates = []
        execution_by_id = {}
        for execution in heads:
            cell = execution["cell"]
            profile = _profile_for_cell(cell)
            candidates.append(profile)
            execution_by_id[cell["cell_id"]] = execution
        effective_threads = max(
            threads, int(math.ceil(cpu_decision["charged_global_cpu_cores"]))
        )
        pack = _choose_mixed_pack(
            candidates, active_reserved_bytes=reserved,
            active_threads=effective_threads, active_lanes=lanes,
            launch_limit=int(swap_decision["launch_limit"]),
        )
    except (AggressiveQueueError, KeyError, TypeError, ValueError):
        return [], blockers
    return [execution_by_id[cell_id] for cell_id in pack["selected_cell_ids"]], blockers


def _aggressive_resource_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    gate = _ORIGINAL_RESOURCE_GATE(*args, **kwargs)
    decision = _advance_swap()
    cpu_decision = _advance_cpu()
    # Swap/pressure launch authority belongs to the persistent aggressive
    # controller. The base gate retains disk, AC, and thermal authority.
    memory_blockers = {
        "memory pressure is not normal", "swap exceeds tolerance or is unavailable"
    }
    blockers = [row for row in gate.get("blockers", []) if row not in memory_blockers]
    if cpu_decision.get("allow_launch") is not True:
        blockers.append("total-host CPU admission is saturated or unavailable")
    gate["blockers"] = blockers
    gate["ok"] = not blockers
    gate["aggressive_swap_decision"] = decision
    gate["aggressive_global_cpu_decision"] = cpu_decision
    return gate


def _checkpoint_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest, size = _BASE._sha_file(path)
    return {"path": _BASE._relative(path), "sha256": digest, "bytes": size}


def _request_checkpoint_exit(live: Any) -> None:
    if live.process.poll() is not None:
        return
    if live.process_pgid != live.process.pid \
            or _BASE._process_identity(live.process.pid) != live.process_identity:
        raise AggressiveQueueError("checkpoint shed victim identity changed")
    try:
        os.killpg(live.process_pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + CHECKPOINT_SHED_GRACE_SECONDS
    while live.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    _BASE._terminate(live.process, expected_pgid=live.process_pgid,
                     expected_identity=live.process_identity)


def _shed_one_checkpoint_lane(plan: dict[str, Any], state: dict[str, Any],
                              live_cells: dict[str, Any],
                              samples_by_cell: dict[str, dict[str, Any]],
                              decision: dict[str, Any]) -> str | None:
    candidates = [cell_id for cell_id in samples_by_cell if cell_id in live_cells]
    if not candidates:
        return None
    victim_id = max(candidates, key=lambda cell_id: (
        samples_by_cell[cell_id]["tree_rss_bytes"], cell_id
    ))
    live = live_cells[victim_id]
    checkpoint_before = _checkpoint_artifact(live.execution["checkpoint_path"])
    intent = {
        "schema": "hawking.doctor_v5_aggressive_swap_shed_intent.v1",
        "version": VERSION, "plan_sha256": plan["plan_sha256"],
        "cell_id": victim_id,
        "request_sha256": live.execution["request"]["request_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "swap_state_sha256": decision["state_sha256"],
        "checkpoint_before": checkpoint_before,
        "completed_evidence_mutation_permitted": False,
        "source_deletion_permitted": False, "recorded_at": _BASE._now(),
    }
    intent["intent_sha256"] = _hash_value(intent)
    intent_path = live.execution["output_dir"] / "aggressive_swap_shed_intent.json"
    _BASE._atomic_json(intent_path, intent)
    _request_checkpoint_exit(live)
    checkpoint_after = _checkpoint_artifact(live.execution["checkpoint_path"])
    sample = samples_by_cell[victim_id]
    stop = _BASE._resource_stop_receipt(
        plan, live.execution, reason="aggressive_swap_emergency_checkpoint_shed",
        sample=sample, max_rss_bytes=live.max_tree_rss_bytes,
        process_identity=live.process_identity,
    )
    intent_digest, intent_size = _BASE._sha_file(intent_path)
    shed_receipt = {
        "schema": "hawking.doctor_v5_aggressive_swap_shed_receipt.v1",
        "version": VERSION, "plan_sha256": plan["plan_sha256"],
        "cell_id": victim_id,
        "request_sha256": live.execution["request"]["request_sha256"],
        "intent": {"path": _BASE._relative(intent_path),
                   "sha256": intent_digest, "bytes": intent_size,
                   "intent_sha256": intent["intent_sha256"]},
        "resource_stop": stop,
        "decision_sha256": decision["decision_sha256"],
        "swap_state_sha256": decision["state_sha256"],
        "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after,
        "checkpoint_preserved": checkpoint_after is not None,
        "completed_evidence_mutation_permitted": False,
        "source_deletion_permitted": False, "recorded_at": _BASE._now(),
    }
    shed_receipt["receipt_sha256"] = _hash_value(shed_receipt)
    shed_path = live.execution["output_dir"] / "aggressive_swap_shed_receipt.json"
    _BASE._atomic_json(shed_path, shed_receipt)
    shed_digest, shed_size = _BASE._sha_file(shed_path)
    shed_reference = {
        "path": _BASE._relative(shed_path), "sha256": shed_digest,
        "bytes": shed_size, "receipt_sha256": shed_receipt["receipt_sha256"],
    }
    state["last_resource_stop"] = stop
    state["last_aggressive_swap_shed"] = shed_reference
    _BASE._append_event("aggressive-swap-shed", cell_id=victim_id,
                        decision_sha256=decision["decision_sha256"],
                        receipt_sha256=shed_receipt["receipt_sha256"],
                        checkpoint_preserved=checkpoint_after is not None)
    row = state["cells"][victim_id]
    if checkpoint_after is None:
        blocker = "aggressive emergency shed produced no durable checkpoint"
        row.update({"status": "blocked-execution", "error": blocker,
                    "blockers": [blocker]})
    else:
        row.update({"status": "pending", "error": None, "blockers": []})
    _BASE._release_cell(state, live_cells, live)
    samples_by_cell.pop(victim_id, None)
    _BASE._save_state(plan, state, "waiting-resources")
    return victim_id


def _aggressive_enforce_pool_budget(plan: dict[str, Any], state: dict[str, Any],
                                    live_cells: dict[str, Any],
                                    samples_by_cell: dict[str, dict[str, Any]],
                                    aggregate: int) -> list[str]:
    # Preserve the base aggregate-RSS guard and sample receipts, but suppress its
    # legacy immediate pressure/swap shed. The aggressive controller below owns
    # that dimension and can shed at most one checkpointed lane per transition.
    original_snapshot = _BASE.ram_scheduler.resource_snapshot
    _BASE.ram_scheduler.resource_snapshot = lambda _root: {
        "pressure_level": 1, "swap_used_mb": 0.0
    }
    try:
        stopped = _ORIGINAL_ENFORCE_POOL_BUDGET(
            plan, state, live_cells, samples_by_cell, aggregate
        )
    finally:
        _BASE.ram_scheduler.resource_snapshot = original_snapshot
    decision = _advance_swap(_swap_snapshot(), force=True)
    if decision.get("shed_one") is True and live_cells:
        victim = _shed_one_checkpoint_lane(
            plan, state, live_cells, samples_by_cell, decision
        )
        if victim is not None:
            stopped.append(victim)
    return stopped


def configure(marker: dict[str, Any], overlay: dict[str, Any]) -> None:
    global _OVERLAY, _MARKER, _PROFILE_BY_CELL, _SWAP_STATE_PATH, _CPU_STATE_PATH
    plan = _BASE._load_plan()
    state = _BASE._load_state(plan)
    errors = _active_generation_errors(marker, overlay, plan=plan, state=state)
    if errors:
        raise AggressiveQueueError(
            "invalid aggressive active generation: " + "; ".join(errors)
        )
    profiles = _bound_pack_profiles(marker, overlay, plan)
    _BASE.SAFETY_MARGIN_BYTES = aggressive.GLOBAL_RESERVE_BYTES
    # The base resource gate's memory fields are replaced in-memory below. This
    # ceiling remains a last-ditch value if an unwrapped helper inspects it.
    _BASE.SWAP_TOLERANCE_MB = aggressive.SWAP_ABSOLUTE_EMERGENCY_MB
    _BASE.MAX_LANES = aggressive.MAX_LANES
    _BASE._cell_reservation = _aggressive_reservation
    _BASE._scan_runnable_heads = _aggressive_scan_heads
    _BASE._resource_gate = _aggressive_resource_gate
    _BASE._enforce_pool_budget = _aggressive_enforce_pool_budget
    _BASE._owner_alive = _owner_alive
    _BASE.start_queue = _start_queue
    _OVERLAY = overlay
    _MARKER = marker
    _PROFILE_BY_CELL = profiles
    _SWAP_STATE_PATH = Path(marker["swap_state"]["path"]).resolve(strict=True)
    _CPU_STATE_PATH = Path(marker["cpu_state"]["path"]).resolve(strict=True)


def _owner_alive(record: Any, plan: dict[str, Any]) -> bool:
    if not isinstance(record, dict) or record.get("schema") != _BASE.PID_SCHEMA \
            or record.get("version") != _BASE.VERSION \
            or record.get("plan_sha256") != plan.get("plan_sha256") \
            or record.get("pid_record_sha256") != _BASE._hash_value(
                _BASE._without(record, "pid_record_sha256")):
        return False
    nonce = record.get("ownership_nonce")
    identity = _BASE._process_identity(record.get("pid"))
    if identity is None or not isinstance(nonce, str) \
            or _BASE.NONCE_RE.fullmatch(nonce) is None:
        return False
    command, started = identity
    entrypoint = (
        "doctor_v5_ultra_aggressive_queue.py run" in command
        or "doctor_v5_ultra_queue.py run" in command
    )
    return (started == record.get("process_started")
            and hashlib.sha256(command.encode("utf-8")).hexdigest()
            == record.get("process_command_sha256")
            and entrypoint and f"--nonce {nonce}" in command)


def _start_queue() -> int:
    if _MARKER is None or _OVERLAY is None:
        raise AggressiveQueueError("aggressive queue is not configured")
    plan = _BASE._load_plan()
    owner = _BASE._read_json(_BASE.PID_FILE, {})
    if _owner_alive(owner, plan):
        print(f"[doctor-v5-ultra-aggressive] already active pid={owner['pid']}")
        return 0
    control = _BASE._load_control(plan)
    if control["mode"] != "run":
        _BASE.set_control("run")
    nonce = secrets.token_hex(16)
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--nonce", nonce]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    env = os.environ.copy()
    env[ENV_OVERLAY] = _MARKER["overlay"]["path"]
    env[ENV_OVERLAY_SHA256] = _OVERLAY["overlay_sha256"]
    env[ENV_MARKER_SHA256] = _MARKER["marker_sha256"]
    _BASE.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _BASE.LOG_FILE.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True, shell=False,
        )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = _BASE._read_json(_BASE.PID_FILE, {})
        if record.get("ownership_nonce") == nonce and _owner_alive(record, plan):
            print(f"[doctor-v5-ultra-aggressive] detached pid={record['pid']} "
                  f"log={_BASE.LOG_FILE}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise AggressiveQueueError("aggressive detached ownership handshake failed")


def _promotion_preflight(marker: dict[str, Any], overlay: dict[str, Any],
                         plan: dict[str, Any], state: dict[str, Any]) -> list[str]:
    try:
        rows = aggressive._load_resource_rows(aggressive.CHILD_RESOURCES)
        snapshot = _swap_snapshot()
        gate = aggressive.promotion_gate(
            overlay, plan, state, rows, snapshot=snapshot
        )
    except (OSError, aggressive.PolicyError) as exc:
        return [f"cannot run aggressive promotion gate: {exc}"]
    errors = list(gate.get("blockers", []))
    errors.extend(_active_generation_errors(marker, overlay, plan=plan, state=state))
    return errors


def _resume_preflight(marker: dict[str, Any], overlay: dict[str, Any],
                      plan: dict[str, Any], state: dict[str, Any]) -> list[str]:
    errors = _marker_errors(marker, verify_files=True)
    errors.extend(_active_generation_errors(marker, overlay, plan=plan, state=state))
    snapshot = _swap_snapshot()
    if "error" in snapshot:
        errors.append("resource snapshot is unavailable")
    power = str(snapshot.get("power_source", ""))
    if power and "AC Power" not in power:
        errors.append("AC power is not confirmed")
    return errors


def _requires_promotion_preflight(overlay: dict[str, Any],
                                  state: dict[str, Any]) -> bool:
    return (state.get("status") in {"paused", "drained"}
            and not state.get("active_children")
            and state.get("state_sha256") == overlay.get("state_sha256_at_stage"))


def main() -> int:
    _verify_static_bindings()
    marker, overlay = _load_activated_marker()
    plan = _BASE._load_plan()
    state = _BASE._load_state(plan)
    errors = (_promotion_preflight(marker, overlay, plan, state)
              if _requires_promotion_preflight(overlay, state)
              else _resume_preflight(marker, overlay, plan, state))
    if errors:
        raise AggressiveQueueError(
            "aggressive queue activation refused: " + "; ".join(errors)
        )
    configure(marker, overlay)
    return int(_BASE.main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AggressiveQueueError, aggressive.PolicyError,
            accel_loader.AccelerationBindingError) as exc:
        print(f"doctor_v5_ultra_aggressive_queue: {exc}", file=sys.stderr)
        raise SystemExit(2)
