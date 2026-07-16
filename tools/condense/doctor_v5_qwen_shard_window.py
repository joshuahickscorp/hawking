#!/usr/bin/env python3.12
"""Unbound two-shard Qwen prepare/finalize overlap coordinator.

This module is not imported by the live Doctor worker.  It defines a future
two-phase adapter in which shard N finalization may overlap shard N+1 bounded
read/RHT/block preparation.  Commit remains strictly shard-ordered and requires
independent validation of both child artifacts and receipts.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable

import doctor_v5_aggressive_admission_policy as admission_policy


REQUEST_SCHEMA = "hawking.doctor_v5_qwen_shard_window_request.v1"
CHILD_RECEIPT_SCHEMA = "hawking.doctor_v5_qwen_shard_window_child_receipt.v1"
COMMIT_RECEIPT_SCHEMA = "hawking.doctor_v5_qwen_shard_window_commit_receipt.v1"
STATE_SCHEMA = "hawking.doctor_v5_qwen_shard_window_state.v1"
CAPABILITY_SCHEMA = "hawking.doctor_v5_qwen_shard_window_capability.v1"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 64 * 1024 * 1024
TERMINAL_STAGE = {"complete", "failed", "cancelled"}
ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = Path(admission_policy.__file__).resolve(strict=True)


class WindowError(RuntimeError):
    """The shard-window execution or evidence contract failed."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise WindowError(f"cannot open regular artifact {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise WindowError(f"artifact is not regular: {path}")
        digest, total = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block); total += len(block)
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if identity(before) != identity(after) or total != after.st_size:
            raise WindowError(f"artifact changed while hashing: {path}")
        return digest.hexdigest(), total
    finally:
        os.close(fd)


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _reference_path(reference: dict[str, Any]) -> Path:
    path = Path(reference["path"])
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def _reference_artifact(reference: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"} \
            or not isinstance(reference.get("path"), str) \
            or not isinstance(reference.get("sha256"), str) \
            or SHA_RE.fullmatch(reference["sha256"]) is None \
            or isinstance(reference.get("bytes"), bool) \
            or not isinstance(reference.get("bytes"), int) or reference["bytes"] < 0:
        raise WindowError(f"{label} artifact identity is invalid")
    try:
        observed = _artifact(_reference_path(reference))
    except (KeyError, OSError, ValueError, WindowError) as exc:
        raise WindowError(f"cannot verify {label} artifact: {exc}") from exc
    if observed["sha256"] != reference["sha256"] \
            or observed["bytes"] != reference["bytes"]:
        raise WindowError(f"{label} artifact changed")
    return observed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise WindowError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WindowError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WindowError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _phase_environment() -> dict[str, str]:
    return {
        "RAYON_NUM_THREADS": "selected_phase_threads",
        "OMP_NUM_THREADS": "selected_phase_threads",
        "VECLIB_MAXIMUM_THREADS": "selected_phase_threads",
    }


def stub_thread_profile(*, maximum_in_flight_shards: int = 2,
                        prepare_threads: int = 8) -> dict[str, Any]:
    """Explicit synthetic profile for cheap orchestration tests only."""
    if maximum_in_flight_shards not in {1, 2} or prepare_threads not in \
            admission_policy.REQUIRED_THREAD_PARITY:
        raise WindowError("stub thread profile is outside the reviewed test envelope")
    profile: dict[str, Any] = {
        "scope": "stub-test-only", "name": "synthetic-two-shard-window",
        "tier": None, "rate": None,
        "required_threads": list(admission_policy.REQUIRED_THREAD_PARITY),
        "prepare_threads": prepare_threads,
        "prepare_thread_source": "explicit-stub-sample",
        "finalize_threads": 1,
        "finalize_thread_source": "explicit-canonical-serial-finalizer",
        "maximum_in_flight_shards": maximum_in_flight_shards,
        "phase_thread_environment": _phase_environment(),
        "qualified_overlay": None, "qualification_sha256": None,
        "selection_sha256": None, "profile_artifact": None,
        "runtime_binary": None, "thread_contract": None,
        "exact_parity_approved": False,
    }
    profile["profile_sha256"] = _hash_value(profile)
    return profile


def serial_thread_profile() -> dict[str, Any]:
    """Explicit one-thread, one-shard fallback; it makes no production claim."""
    profile = stub_thread_profile(maximum_in_flight_shards=1, prepare_threads=8)
    profile.update({
        "scope": "local-serial-fallback", "name": "local-serial-fallback",
        "required_threads": [], "prepare_threads": 1,
        "prepare_thread_source": "explicit-local-serial-fallback",
    })
    profile["profile_sha256"] = _hash_value(_without(profile, "profile_sha256"))
    return profile


def _verified_overlay(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay_path = Path(path).resolve(strict=True)
    overlay = _read_json(overlay_path)
    errors = admission_policy.validate_overlay(overlay)
    if errors:
        raise WindowError("aggressive admission overlay is invalid: " + "; ".join(errors))
    bindings = overlay.get("source_bindings")
    if not isinstance(bindings, dict):
        raise WindowError("aggressive admission overlay source bindings are missing")
    controller = _reference_artifact(
        bindings.get("policy_module"), label="aggressive admission controller"
    )
    if controller != _artifact(CONTROLLER_PATH):
        raise WindowError("aggressive admission overlay binds another controller")
    contract = _reference_artifact(
        bindings.get("thread_profile_contract"), label="vendor thread-profile contract"
    )
    if contract != _artifact(admission_policy.THREAD_PROFILE_CONTRACT_PATH):
        raise WindowError("aggressive admission overlay binds another thread contract")
    return overlay, _artifact(overlay_path)


def load_qualified_thread_profile(overlay_path: Path, *, tier: str,
                                  rate: str) -> dict[str, Any]:
    """Load, re-hash, and re-qualify one exact measured tier/rate winner."""
    if not isinstance(tier, str) or not tier or tier != tier.strip() \
            or not isinstance(rate, str) or not rate or rate != rate.strip():
        raise WindowError("qualified thread profile requires exact tier and rate")
    overlay, overlay_artifact = _verified_overlay(overlay_path)
    qualification = overlay.get("thread_profile_qualification")
    if not isinstance(qualification, dict) or qualification.get("status") != "qualified":
        raise WindowError("overlay has no qualified exact 8/12/16/20 profile")
    try:
        profile_artifact = _reference_artifact(
            qualification.get("profile"), label="qualified vendor thread profile"
        )
        binary_artifact = _reference_artifact(
            qualification.get("binary"), label="qualified runtime binary"
        )
        cell = {"model_label": tier, "rate_id": rate}
        current = admission_policy.qualify_thread_profile(
            [cell], profile_path=Path(profile_artifact["path"]),
            binary_path=Path(binary_artifact["path"]),
        )
        selected = admission_policy.selected_thread_profile(cell, current)
        key = json.dumps([tier, rate], separators=(",", ":"), ensure_ascii=False)
        staged_selection = qualification.get("selections", {}).get(key)
        current_selection = current.get("selections", {}).get(key)
    except (KeyError, OSError, TypeError, ValueError,
            admission_policy.PolicyError) as exc:
        raise WindowError(f"cannot requalify exact tier/rate thread profile: {exc}") from exc
    if staged_selection != current_selection:
        raise WindowError("staged and requalified tier/rate thread selections differ")
    threads = selected["threads"]
    if threads not in admission_policy.REQUIRED_THREAD_PARITY:
        raise WindowError("qualified winner is outside exact 8/12/16/20")
    profile: dict[str, Any] = {
        "scope": "production-qualified", "name": selected["profile"],
        "tier": tier, "rate": rate,
        "required_threads": list(admission_policy.REQUIRED_THREAD_PARITY),
        "prepare_threads": threads,
        "prepare_thread_source": "qualified-vendor-thread-profile-contract",
        # The canonical merge/finalizer is deliberately serial.  This is an
        # explicit phase boundary, not an invented second production profile.
        "finalize_threads": 1,
        "finalize_thread_source": "explicit-canonical-serial-finalizer",
        "maximum_in_flight_shards": 2,
        "phase_thread_environment": _phase_environment(),
        "qualified_overlay": overlay_artifact,
        "qualification_sha256": qualification["qualification_sha256"],
        "selection_sha256": current_selection["selection_sha256"],
        "profile_artifact": profile_artifact,
        "runtime_binary": binary_artifact,
        "thread_contract": _artifact(admission_policy.THREAD_PROFILE_CONTRACT_PATH),
        "exact_parity_approved": True,
    }
    profile["profile_sha256"] = _hash_value(profile)
    return profile


def _baseline_seal(baseline_swap_mb: float) -> str:
    return _hash_value({
        "sealed_baseline_swap_mb": round(float(baseline_swap_mb), 3),
        "baseline_can_ratchet": False,
        "swap_policy": admission_policy.swap_policy(),
    })


def stub_admission_binding(*, sealed_baseline_swap_mb: float = 0.0) -> dict[str, Any]:
    if isinstance(sealed_baseline_swap_mb, bool) \
            or not isinstance(sealed_baseline_swap_mb, (int, float)) \
            or not math.isfinite(float(sealed_baseline_swap_mb)) \
            or float(sealed_baseline_swap_mb) < 0:
        raise WindowError("stub swap baseline is invalid")
    baseline = round(float(sealed_baseline_swap_mb), 3)
    state = admission_policy.initial_swap_state(
        {"pressure_level": 1, "swap_used_mb": baseline}, now_epoch=0.0
    )
    binding: dict[str, Any] = {
        "scope": "stub-sample-injection", "controller": _artifact(CONTROLLER_PATH),
        "overlay": None, "overlay_sha256": None,
        "sealed_baseline_swap_mb": baseline,
        "baseline_can_ratchet": False, "baseline_seal_sha256": _baseline_seal(baseline),
        "swap_policy": admission_policy.swap_policy(),
        "initial_swap_state": state, "synthetic_samples_permitted": True,
    }
    binding["binding_sha256"] = _hash_value(binding)
    return binding


def load_production_admission_binding(overlay_path: Path) -> dict[str, Any]:
    overlay, overlay_artifact = _verified_overlay(overlay_path)
    policy = overlay.get("resource_policy")
    baseline = policy.get("sealed_swap_baseline_mb") if isinstance(policy, dict) else None
    state = overlay.get("initial_swap_state")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or float(baseline) < 0 \
            or admission_policy.validate_swap_state(
                state, sealed_baseline_swap_mb=float(baseline)
            ):
        raise WindowError("overlay sealed non-ratcheting swap baseline is invalid")
    baseline = round(float(baseline), 3)
    binding: dict[str, Any] = {
        "scope": "production-qualified-aggressive-overlay",
        "controller": _artifact(CONTROLLER_PATH), "overlay": overlay_artifact,
        "overlay_sha256": overlay["overlay_sha256"],
        "sealed_baseline_swap_mb": baseline,
        "baseline_can_ratchet": False, "baseline_seal_sha256": _baseline_seal(baseline),
        "swap_policy": admission_policy.swap_policy(),
        "initial_swap_state": state, "synthetic_samples_permitted": False,
    }
    binding["binding_sha256"] = _hash_value(binding)
    return binding


def capabilities() -> dict[str, Any]:
    return {
        "schema": CAPABILITY_SCHEMA, "status": "unbound-scaffold-only",
        "implemented": {
            "maximum_two_in_flight_shards": True,
            "qualified_exact_tier_rate_thread_profile_loader": True,
            "production_thread_profile_without_qualified_overlay": False,
            "aggressive_overlay_controller_artifact_binding": True,
            "sealed_non_ratcheting_swap_controller": True,
            "aggregate_process_tree_resource_admission": True,
            "unique_temporary_outputs": True,
            "binary_source_request_hash_binding": True,
            "strict_canonical_commit_order": True,
            "child_output_and_receipt_validation": True,
            "crash_resume_and_completed_output_adoption": True,
            "failure_cancels_later_uncommitted_work": True,
            "local_serial_fallback": True,
        },
        "reviewed_for_live_execution": False,
        "promotion_requires": [
            "owner_free_machine", "real_qwen_artifact_two_phase_binary",
            "serial_vs_window_exact_archive_sha256_parity",
            "serial_vs_window_exact_receipt_parity",
            "measured_process_tree_ram_swap_thermal_admission",
            "qualified_exact_8_12_16_20_tier_rate_profile",
            "hash_bound_aggressive_overlay_and_controller",
            "sealed_non_ratcheting_swap_baseline",
            "crash_recovery_adversarial_pass", "rollback_point",
        ],
        "source_deletion_permitted": False, "quality_claims_permitted": False,
    }


def build_request(
    *, shards: list[dict[str, Any]], output_root: Path,
    prepare_program: dict[str, Any], finalize_program: dict[str, Any],
    thread_profile: dict[str, Any], admission_binding: dict[str, Any],
    process_budget_bytes: int,
    prepare_reservation_bytes: int, finalize_reservation_bytes: int,
    execution_mode: str = "stub-test-only",
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": REQUEST_SCHEMA, "created_at": _now(),
        "execution_mode": execution_mode,
        "output_root": str(Path(output_root).resolve()), "shards": shards,
        "programs": {"prepare": prepare_program, "finalize": finalize_program},
        "thread_profile": thread_profile,
        "admission_binding": admission_binding,
        "resources": {
            "process_budget_bytes": process_budget_bytes,
            "prepare_reservation_bytes": prepare_reservation_bytes,
            "finalize_reservation_bytes": finalize_reservation_bytes,
            "memory_pressure_required": "normal",
            "allowed_thermal_states": ["nominal", "fair"],
            "aggregate_process_tree_required": True,
        },
        "lifecycle": {
            "maximum_in_flight_shards": thread_profile["maximum_in_flight_shards"],
            "strict_commit_order": True, "source_deletion_permitted": False,
            "evidence_deletion_permitted": False,
            "later_uncommitted_cancelled_on_failure": True,
            "local_serial_fallback": True,
        },
        "promotion": {
            "live_execution_permitted": False,
            "owner_free_real_artifact_parity_required": True,
            "runtime_defaults_changed": False,
        },
    }
    doc["request_sha256"] = _hash_value(doc)
    errors = validate_request(doc, verify_files=False)
    if errors:
        raise WindowError("generated window request invalid: " + "; ".join(errors))
    return doc


def _artifact_reference_errors(reference: Any, *, label: str,
                               verify_files: bool) -> list[str]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"} \
            or not isinstance(reference.get("path"), str) \
            or not isinstance(reference.get("sha256"), str) \
            or SHA_RE.fullmatch(reference["sha256"]) is None \
            or isinstance(reference.get("bytes"), bool) \
            or not isinstance(reference.get("bytes"), int) or reference["bytes"] < 0:
        return [f"{label} artifact identity is invalid"]
    if verify_files:
        try:
            observed = _artifact(_reference_path(reference))
        except (KeyError, OSError, ValueError, WindowError) as exc:
            return [f"{label} artifact verification failed: {exc}"]
        if observed["sha256"] != reference["sha256"] \
                or observed["bytes"] != reference["bytes"]:
            return [f"{label} artifact differs"]
    return []


def _thread_profile_errors(profile: Any, *, production: bool,
                           verify_files: bool) -> list[str]:
    if not isinstance(profile, dict) or profile.get("profile_sha256") != _hash_value(
            _without(profile, "profile_sha256")):
        return ["window thread profile seal is invalid"]
    errors: list[str] = []
    if profile.get("maximum_in_flight_shards") not in {1, 2} \
            or profile.get("phase_thread_environment") != _phase_environment() \
            or profile.get("finalize_threads") != 1 \
            or profile.get("finalize_thread_source") \
            != "explicit-canonical-serial-finalizer":
        errors.append("window phase thread envelope is invalid")
    if production:
        if profile.get("scope") != "production-qualified" \
                or profile.get("required_threads") \
                != list(admission_policy.REQUIRED_THREAD_PARITY) \
                or profile.get("prepare_threads") \
                not in admission_policy.REQUIRED_THREAD_PARITY \
                or profile.get("prepare_thread_source") \
                != "qualified-vendor-thread-profile-contract" \
                or profile.get("maximum_in_flight_shards") != 2 \
                or profile.get("exact_parity_approved") is not True \
                or not isinstance(profile.get("tier"), str) or not profile.get("tier") \
                or not isinstance(profile.get("rate"), str) or not profile.get("rate") \
                or not isinstance(profile.get("qualification_sha256"), str) \
                or SHA_RE.fullmatch(profile["qualification_sha256"]) is None \
                or not isinstance(profile.get("selection_sha256"), str) \
                or SHA_RE.fullmatch(profile["selection_sha256"]) is None:
            errors.append("production thread profile is not exact qualified tier/rate evidence")
        for field in ("qualified_overlay", "profile_artifact", "runtime_binary",
                      "thread_contract"):
            errors.extend(_artifact_reference_errors(
                profile.get(field), label=f"thread profile {field}",
                verify_files=verify_files,
            ))
        if verify_files and not errors:
            try:
                current = load_qualified_thread_profile(
                    Path(profile["qualified_overlay"]["path"]),
                    tier=profile["tier"], rate=profile["rate"],
                )
            except (OSError, WindowError) as exc:
                errors.append(f"production thread profile revalidation failed: {exc}")
            else:
                if current != profile:
                    errors.append("production thread profile differs from current qualification")
    elif profile.get("scope") == "stub-test-only":
        if profile.get("prepare_threads") not in admission_policy.REQUIRED_THREAD_PARITY \
                or profile.get("prepare_thread_source") != "explicit-stub-sample" \
                or profile.get("required_threads") \
                != list(admission_policy.REQUIRED_THREAD_PARITY) \
                or profile.get("exact_parity_approved") is not False:
            errors.append("stub thread profile is invalid")
    elif profile.get("scope") == "local-serial-fallback":
        if profile.get("prepare_threads") != 1 \
                or profile.get("prepare_thread_source") \
                != "explicit-local-serial-fallback" \
                or profile.get("required_threads") != [] \
                or profile.get("maximum_in_flight_shards") != 1 \
                or profile.get("exact_parity_approved") is not False:
            errors.append("serial fallback thread profile is invalid")
    else:
        errors.append("stub request carries a production or unknown thread profile")
    return errors


def _admission_binding_errors(binding: Any, *, production: bool,
                              verify_files: bool) -> list[str]:
    if not isinstance(binding, dict) or binding.get("binding_sha256") != _hash_value(
            _without(binding, "binding_sha256")):
        return ["admission binding seal is invalid"]
    errors = _artifact_reference_errors(
        binding.get("controller"), label="aggressive admission controller",
        verify_files=verify_files,
    )
    baseline = binding.get("sealed_baseline_swap_mb")
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or float(baseline) < 0:
        errors.append("sealed swap baseline is invalid")
        baseline = 0.0
    if binding.get("baseline_can_ratchet") is not False \
            or binding.get("swap_policy") != admission_policy.swap_policy() \
            or binding.get("baseline_seal_sha256") != _baseline_seal(float(baseline)):
        errors.append("swap controller is not bound to the reviewed non-ratcheting deltas")
    if admission_policy.validate_swap_state(
            binding.get("initial_swap_state"),
            sealed_baseline_swap_mb=float(baseline)):
        errors.append("initial swap controller state is invalid or re-baselined")
    if verify_files and not errors:
        try:
            if _artifact(_reference_path(binding["controller"])) != _artifact(CONTROLLER_PATH):
                errors.append("admission binding selects another controller")
        except (KeyError, OSError, ValueError, WindowError) as exc:
            errors.append(f"admission controller revalidation failed: {exc}")
    if production:
        if binding.get("scope") != "production-qualified-aggressive-overlay" \
                or binding.get("synthetic_samples_permitted") is not False \
                or not isinstance(binding.get("overlay_sha256"), str) \
                or SHA_RE.fullmatch(binding["overlay_sha256"]) is None:
            errors.append("production admission binding is not overlay/controller qualified")
        errors.extend(_artifact_reference_errors(
            binding.get("overlay"), label="aggressive admission overlay",
            verify_files=verify_files,
        ))
        if verify_files and not errors:
            try:
                current = load_production_admission_binding(Path(binding["overlay"]["path"]))
            except (OSError, WindowError) as exc:
                errors.append(f"production admission binding revalidation failed: {exc}")
            else:
                if current != binding:
                    errors.append("production admission binding differs from current overlay")
    elif binding.get("scope") != "stub-sample-injection" \
            or binding.get("synthetic_samples_permitted") is not True \
            or binding.get("overlay") is not None \
            or binding.get("overlay_sha256") is not None:
        errors.append("stub admission binding is invalid")
    return errors


def validate_request(doc: Any, *, verify_files: bool) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != REQUEST_SCHEMA:
        return ["window request schema mismatch"]
    errors: list[str] = []
    if doc.get("request_sha256") != _hash_value(_without(doc, "request_sha256")):
        errors.append("window request hash mismatch")
    mode = doc.get("execution_mode")
    if mode not in {"stub-test-only", "production-parity-gated"}:
        errors.append("window execution mode is invalid")
    production = mode == "production-parity-gated"
    shards = doc.get("shards")
    if not isinstance(shards, list) or not shards:
        errors.append("window request has no shards")
        shards = []
    for ordinal, row in enumerate(shards):
        source = row.get("source") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("ordinal") != ordinal \
                or not isinstance(source, dict) or not isinstance(source.get("path"), str) \
                or not isinstance(source.get("sha256"), str) \
                or SHA_RE.fullmatch(source["sha256"]) is None \
                or not isinstance(source.get("bytes"), int) or source["bytes"] < 0 \
                or not isinstance(row.get("shard_request_sha256"), str) \
                or SHA_RE.fullmatch(row["shard_request_sha256"]) is None:
            errors.append(f"window shard identity invalid: {ordinal}")
            continue
        if verify_files:
            try:
                if _artifact(Path(source["path"])) != source:
                    errors.append(f"window source file differs: {ordinal}")
            except (OSError, WindowError) as exc:
                errors.append(f"window source verification failed: {exc}")
    programs = doc.get("programs")
    if not isinstance(programs, dict):
        errors.append("window child programs are missing")
        programs = {}
    for phase in ("prepare", "finalize"):
        row = programs.get(phase)
        if not isinstance(row, dict) or not isinstance(row.get("launcher_argv"), list) \
                or not row["launcher_argv"] or not isinstance(row.get("program"), dict):
            errors.append(f"window {phase} program contract invalid")
            continue
        program = row["program"]
        if not isinstance(program.get("path"), str) \
                or not isinstance(program.get("sha256"), str) \
                or SHA_RE.fullmatch(program["sha256"]) is None \
                or not isinstance(program.get("bytes"), int):
            errors.append(f"window {phase} program identity invalid")
        elif verify_files:
            try:
                if _artifact(Path(program["path"])) != program:
                    errors.append(f"window {phase} program file differs")
            except (OSError, WindowError) as exc:
                errors.append(f"window program verification failed: {exc}")
    profile = doc.get("thread_profile")
    errors.extend(_thread_profile_errors(
        profile, production=production, verify_files=verify_files
    ))
    binding = doc.get("admission_binding")
    errors.extend(_admission_binding_errors(
        binding, production=production, verify_files=verify_files
    ))
    if production and isinstance(profile, dict) and isinstance(binding, dict) \
            and profile.get("qualified_overlay") != binding.get("overlay"):
        errors.append("thread profile and admission controller bind different overlays")
    resources = doc.get("resources")
    if not isinstance(resources, dict) or any(
            isinstance(resources.get(field), bool)
            or not isinstance(resources.get(field), int)
            or resources[field] < 0
            for field in ("process_budget_bytes", "prepare_reservation_bytes",
                          "finalize_reservation_bytes")):
        errors.append("window resource envelope is invalid")
    lifecycle = doc.get("lifecycle")
    if not isinstance(lifecycle, dict) \
            or lifecycle.get("source_deletion_permitted") is not False \
            or lifecycle.get("evidence_deletion_permitted") is not False \
            or lifecycle.get("strict_commit_order") is not True:
        errors.append("window lifecycle boundary is invalid")
    elif isinstance(profile, dict) and lifecycle.get("maximum_in_flight_shards") \
            != profile.get("maximum_in_flight_shards"):
        errors.append("window lifecycle and thread-profile lane counts differ")
    promotion = doc.get("promotion")
    if not isinstance(promotion, dict) \
            or promotion.get("live_execution_permitted") is not False \
            or promotion.get("owner_free_real_artifact_parity_required") is not True \
            or promotion.get("runtime_defaults_changed") is not False:
        errors.append("window promotion boundary is invalid")
    return errors


def write_request(path: Path, doc: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise WindowError(f"refusing to overwrite window request: {path}")
    _atomic_json(path, doc)


def default_resource_probe(pids: set[int]) -> dict[str, Any]:
    """Cheap macOS process-tree/system sample; production promotion must audit it."""
    rss = 0
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        rows = []
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 3:
                rows.append(tuple(map(int, fields)))
        descendants = set(pids)
        changed = True
        while changed:
            changed = False
            for pid, ppid, _rss in rows:
                if ppid in descendants and pid not in descendants:
                    descendants.add(pid); changed = True
        rss = sum(row_rss * 1024 for pid, _ppid, row_rss in rows if pid in descendants)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "unavailable", "aggregate_process_tree_rss_bytes": None,
                "swap_used_bytes": None, "memory_pressure": "unknown",
                "thermal_state": "unknown"}
    try:
        swap_text = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", swap_text)
        if match is None:
            raise ValueError("swap usage is unparseable")
        multiplier = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)]
        swap_used = round(float(match.group(1)) * multiplier)
        pressure_text = subprocess.run(
            ["memory_pressure", "-Q"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        match = re.search(r"free percentage:\s*([0-9]+)%", pressure_text)
        if match is None:
            raise ValueError("memory pressure is unparseable")
        free_pct = int(match.group(1))
        pressure = "normal" if free_pct >= 10 else "warning" if free_pct >= 5 else "critical"
        thermal_text = subprocess.run(
            ["pmset", "-g", "therm"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        thermal = "nominal" if "No thermal warning level" in thermal_text else "fair"
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "unavailable", "aggregate_process_tree_rss_bytes": rss,
                "swap_used_bytes": None, "memory_pressure": "unknown",
                "thermal_state": "unknown"}
    return {"status": "sampled", "aggregate_process_tree_rss_bytes": rss,
            "swap_used_bytes": swap_used, "memory_pressure": pressure,
            "thermal_state": thermal}


class WindowCoordinator:
    def __init__(
        self, request_path: Path, *,
        resource_probe: Callable[[set[int]], dict[str, Any]] = default_resource_probe,
        poll_seconds: float = 0.02, verify_source_files: bool = True,
    ) -> None:
        self.request_path = Path(request_path).resolve(strict=True)
        self.request = _read_json(self.request_path)
        errors = validate_request(self.request, verify_files=verify_source_files)
        if errors:
            raise WindowError("invalid window request: " + "; ".join(errors))
        self.request_file = _artifact(self.request_path)
        self.output_root = Path(self.request["output_root"])
        if self.output_root.is_symlink():
            raise WindowError("symlinked window output root is forbidden")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = self.output_root.resolve(strict=True)
        self.temp_root = self.output_root / "window-temp"
        self.commit_root = self.output_root / "committed-shards"
        self.temp_root.mkdir(exist_ok=True); self.commit_root.mkdir(exist_ok=True)
        self.state_path = self.output_root / "window-state.json"
        self.resource_probe = resource_probe
        self.poll_seconds = poll_seconds
        self.processes: dict[tuple[int, str], subprocess.Popen[bytes]] = {}
        self.state = self._load_or_initialize()

    def _initial_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "schema": STATE_SCHEMA, "created_at": _now(), "updated_at": _now(),
            "request_sha256": self.request["request_sha256"],
            "request_file_sha256": self.request_file["sha256"],
            "next_commit_ordinal": 0, "status": "running", "shards": [],
            "source_files_deleted": False, "evidence_deleted": False,
            "admission_controller_state": json.loads(json.dumps(
                self.request["admission_binding"]["initial_swap_state"]
            )),
        }
        for row in self.request["shards"]:
            state["shards"].append({
                "ordinal": row["ordinal"], "prepare": {"status": "pending", "attempt": 0},
                "finalize": {"status": "pending", "attempt": 0},
                "commit": {"status": "pending"},
            })
        state["state_sha256"] = _hash_value(state)
        return state

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._initial_state(); _atomic_json(self.state_path, state); return state
        state = _read_json(self.state_path)
        if state.get("schema") != STATE_SCHEMA \
                or state.get("state_sha256") != _hash_value(_without(state, "state_sha256")) \
                or state.get("request_sha256") != self.request["request_sha256"] \
                or state.get("request_file_sha256") != self.request_file["sha256"] \
                or admission_policy.validate_swap_state(
                    state.get("admission_controller_state"),
                    sealed_baseline_swap_mb=self.request["admission_binding"][
                        "sealed_baseline_swap_mb"
                    ]):
            raise WindowError("window resume state identity differs")
        self._adopt_completed_temps(state)
        return state

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        self.state["state_sha256"] = _hash_value(_without(self.state, "state_sha256"))
        _atomic_json(self.state_path, self.state)

    def _stage_paths(self, ordinal: int, phase: str, attempt: int,
                     token: str) -> tuple[Path, Path, Path]:
        base = f"shard-{ordinal:05d}.{phase}.attempt-{attempt}.{token}"
        return (self.temp_root / f".{base}.output.partial",
                self.temp_root / f".{base}.receipt.partial.json",
                self.temp_root / f".{base}.log")

    def _program(self, phase: str) -> dict[str, Any]:
        return self.request["programs"][phase]

    def _source(self, ordinal: int) -> dict[str, Any]:
        return self.request["shards"][ordinal]["source"]

    def _input_artifact(self, ordinal: int, phase: str) -> dict[str, Any]:
        if phase == "prepare":
            return self._source(ordinal)
        prepare = self.state["shards"][ordinal]["prepare"]
        return prepare["output"]

    def _admitted(self, phase: str) -> tuple[bool, dict[str, Any]]:
        pids = {os.getpid()} | {process.pid for process in self.processes.values()
                if process.poll() is None}
        sample = self.resource_probe(pids)
        reservation = self.request["resources"][f"{phase}_reservation_bytes"]
        rss = sample.get("aggregate_process_tree_rss_bytes")
        declared_running = sum(
            self.request["resources"][f"{running_phase}_reservation_bytes"]
            for (_ordinal, running_phase), process in self.processes.items()
            if process.poll() is None
        )
        swap = sample.get("swap_used_bytes")
        effective = max(rss, declared_running) if isinstance(rss, int) else None
        pressure_level = {"normal": 1, "warning": 2, "critical": 4}.get(
            sample.get("memory_pressure")
        )
        swap_mb = (float(swap) / (1024.0 * 1024.0)
                   if isinstance(swap, int) and not isinstance(swap, bool) else None)
        controller_state, decision = admission_policy.advance_swap_state(
            self.state["admission_controller_state"],
            {"pressure_level": pressure_level, "swap_used_mb": swap_mb},
            now_epoch=time.time(),
            sealed_baseline_swap_mb=self.request["admission_binding"][
                "sealed_baseline_swap_mb"
            ],
        )
        self.state["admission_controller_state"] = controller_state
        sample = dict(sample)
        sample["aggressive_admission_decision"] = decision
        admitted = sample.get("status") == "sampled" and isinstance(effective, int) \
            and effective + reservation <= self.request["resources"]["process_budget_bytes"] \
            and decision.get("allow_launch") is True \
            and len([process for process in self.processes.values()
                     if process.poll() is None]) < decision.get("launch_limit", 0) \
            and sample.get("memory_pressure") == "normal" \
            and sample.get("thermal_state") in self.request["resources"][
                "allowed_thermal_states"
            ]
        self.state["last_admission_decision"] = {
            "sampled_at": _now(), "phase": phase, "admitted": admitted,
            "sample": sample,
        }
        # Persist the non-ratcheting controller transition and decisive sample
        # before returning.  A fail-closed denial may have no later launch/save.
        self._save()
        return admitted, sample

    def _render_argv(self, ordinal: int, phase: str, output: Path,
                     receipt: Path) -> list[str]:
        program = self._program(phase)
        values = {
            "phase": phase, "ordinal": str(ordinal),
            "input": self._input_artifact(ordinal, phase)["path"],
            "output": str(output), "receipt": str(receipt),
        }
        return [str(value).format(**values) for value in program["launcher_argv"]]

    def _launch(self, ordinal: int, phase: str) -> None:
        stage = self.state["shards"][ordinal][phase]
        attempt = stage["attempt"] + 1
        token = secrets.token_hex(8)
        output, receipt, log = self._stage_paths(ordinal, phase, attempt, token)
        for path in (output, receipt, log):
            if path.exists() or path.is_symlink():
                raise WindowError(f"non-unique window temporary path: {path}")
        threads = self.request["thread_profile"][f"{phase}_threads"]
        env = dict(os.environ)
        env.update({
            "RAYON_NUM_THREADS": str(threads), "OMP_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
            "HAWKING_WINDOW_STAGE": phase, "HAWKING_WINDOW_ORDINAL": str(ordinal),
            "HAWKING_WINDOW_REQUEST_SHA256": self.request["request_sha256"],
            "HAWKING_WINDOW_REQUEST_FILE_SHA256": self.request_file["sha256"],
            "HAWKING_WINDOW_SOURCE_SHA256": self._source(ordinal)["sha256"],
            "HAWKING_WINDOW_SHARD_REQUEST_SHA256": self.request["shards"][ordinal][
                "shard_request_sha256"
            ],
            "HAWKING_WINDOW_PROGRAM_SHA256": self._program(phase)["program"]["sha256"],
            "HAWKING_WINDOW_INPUT_SHA256": self._input_artifact(ordinal, phase)["sha256"],
            "HAWKING_WINDOW_ATTEMPT": str(attempt), "HAWKING_WINDOW_TOKEN": token,
            "HAWKING_WINDOW_OUTPUT": str(output), "HAWKING_WINDOW_RECEIPT": str(receipt),
        })
        handle = log.open("wb")
        try:
            process = subprocess.Popen(
                self._render_argv(ordinal, phase, output, receipt),
                stdout=handle, stderr=subprocess.STDOUT, env=env,
                start_new_session=True,
            )
        finally:
            handle.close()
        stage.update({"status": "running", "attempt": attempt, "token": token,
                      "pid": process.pid, "output_path": str(output),
                      "receipt_path": str(receipt), "log_path": str(log),
                      "started_at": _now()})
        self.processes[(ordinal, phase)] = process
        self._save()

    def _validate_child(self, ordinal: int, phase: str, stage: dict[str, Any]) \
            -> tuple[dict[str, Any], dict[str, Any]]:
        output, receipt_path = Path(stage["output_path"]), Path(stage["receipt_path"])
        receipt = _read_json(receipt_path)
        if receipt.get("schema") != CHILD_RECEIPT_SCHEMA \
                or receipt.get("receipt_sha256") != _hash_value(
                    _without(receipt, "receipt_sha256")
                ) or receipt.get("status") != "complete" \
                or receipt.get("stage") != phase or receipt.get("ordinal") != ordinal \
                or receipt.get("attempt") != stage["attempt"] \
                or receipt.get("token") != stage["token"] \
                or receipt.get("request_sha256") != self.request["request_sha256"] \
                or receipt.get("request_file_sha256") != self.request_file["sha256"] \
                or receipt.get("source_sha256") != self._source(ordinal)["sha256"] \
                or receipt.get("shard_request_sha256") != self.request["shards"][ordinal][
                    "shard_request_sha256"
                ] or receipt.get("program_sha256") != self._program(phase)["program"][
                    "sha256"
                ] or receipt.get("input_sha256") != self._input_artifact(
                    ordinal, phase
                )["sha256"]:
            raise WindowError(f"{phase} child receipt authority differs for shard {ordinal}")
        output_artifact = _artifact(output)
        if receipt.get("output") != output_artifact:
            raise WindowError(f"{phase} child output identity differs for shard {ordinal}")
        return output_artifact, _artifact(receipt_path)

    def _complete_stage(self, ordinal: int, phase: str) -> None:
        stage = self.state["shards"][ordinal][phase]
        output, receipt = self._validate_child(ordinal, phase, stage)
        stage.update({"status": "complete", "output": output, "receipt": receipt,
                      "completed_at": _now()})
        stage.pop("pid", None)
        self._save()

    def _adopt_completed_temps(self, state: dict[str, Any]) -> None:
        original = self.state if hasattr(self, "state") else None
        self.state = state
        try:
            for shard in state["shards"]:
                for phase in ("prepare", "finalize"):
                    stage = shard[phase]
                    if stage.get("status") != "running":
                        continue
                    pid = stage.get("pid")
                    alive = False
                    if isinstance(pid, int):
                        try:
                            os.kill(pid, 0); alive = True
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            alive = True
                    if alive:
                        continue
                    try:
                        self._complete_stage(shard["ordinal"], phase)
                        stage["adopted_after_coordinator_restart"] = True
                    except (OSError, WindowError):
                        stage["status"] = "failed"
                        stage["error"] = "orphaned child lacked valid output/receipt"
        finally:
            if original is not None:
                self.state = original
        state["state_sha256"] = _hash_value(_without(state, "state_sha256"))
        _atomic_json(self.state_path, state)

    def _cancel_later(self, failed_ordinal: int, reason: str) -> None:
        cancelled_processes: list[subprocess.Popen[bytes]] = []
        for (ordinal, phase), process in list(self.processes.items()):
            if ordinal > failed_ordinal and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                cancelled_processes.append(process)
                self.state["shards"][ordinal][phase]["status"] = "cancelled"
                self.state["shards"][ordinal][phase]["error"] = reason
        for process in cancelled_processes:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired as exc:
                    raise WindowError(
                        f"cannot reap cancelled shard-window child {process.pid}"
                    ) from exc
        for shard in self.state["shards"]:
            if shard["ordinal"] > failed_ordinal and shard["commit"]["status"] != "complete":
                for phase in ("prepare", "finalize"):
                    if shard[phase]["status"] not in TERMINAL_STAGE:
                        shard[phase]["status"] = "cancelled"
                        shard[phase]["error"] = reason
                shard["commit"]["status"] = "cancelled"
        self.state["status"] = "failed"; self._save()

    def _commit_ready(self) -> None:
        while self.state["next_commit_ordinal"] < len(self.state["shards"]):
            ordinal = self.state["next_commit_ordinal"]
            shard = self.state["shards"][ordinal]
            if shard["prepare"]["status"] != "complete" \
                    or shard["finalize"]["status"] != "complete":
                return
            source_name = Path(self._source(ordinal)["path"]).name
            final_path = self.commit_root / f"{ordinal:05d}-{source_name}.strand"
            receipt_path = self.commit_root / f"{ordinal:05d}-{source_name}.receipt.json"
            if receipt_path.exists() and not final_path.exists():
                raise WindowError(f"canonical receipt exists without shard output: {ordinal}")
            if not final_path.exists():
                temporary = Path(shard["finalize"]["output"]["path"])
                os.replace(temporary, final_path)
                fd = os.open(final_path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            artifact = _artifact(final_path)
            if artifact["sha256"] != shard["finalize"]["output"]["sha256"]:
                raise WindowError(f"canonical commit changed shard output: {ordinal}")
            receipt: dict[str, Any] = {
                "schema": COMMIT_RECEIPT_SCHEMA, "committed_at": _now(),
                "ordinal": ordinal, "request_sha256": self.request["request_sha256"],
                "request_file_sha256": self.request_file["sha256"],
                "source": self._source(ordinal),
                "shard_request_sha256": self.request["shards"][ordinal][
                    "shard_request_sha256"
                ],
                "prepare_program": self._program("prepare")["program"],
                "finalize_program": self._program("finalize")["program"],
                "prepare_output": shard["prepare"]["output"],
                "prepare_receipt": shard["prepare"]["receipt"],
                "finalize_child_receipt": shard["finalize"]["receipt"],
                "canonical_output": artifact,
                "validators": {"prepare_child": True, "finalize_child": True,
                               "canonical_output_hash": True},
                "source_files_deleted": False, "evidence_deleted": False,
                "quality_claims_permitted": False,
            }
            receipt["receipt_sha256"] = _hash_value(receipt)
            if receipt_path.exists():
                existing = _read_json(receipt_path)
                if existing.get("schema") != COMMIT_RECEIPT_SCHEMA \
                        or existing.get("receipt_sha256") != _hash_value(
                            _without(existing, "receipt_sha256")
                        ) or existing.get("ordinal") != ordinal \
                        or existing.get("request_sha256") != self.request["request_sha256"] \
                        or existing.get("request_file_sha256") != self.request_file["sha256"] \
                        or existing.get("source") != self._source(ordinal) \
                        or existing.get("prepare_output") != shard["prepare"]["output"] \
                        or existing.get("prepare_receipt") != shard["prepare"]["receipt"] \
                        or existing.get("finalize_child_receipt") \
                        != shard["finalize"]["receipt"] \
                        or existing.get("canonical_output") != artifact \
                        or existing.get("validators") != {
                            "prepare_child": True, "finalize_child": True,
                            "canonical_output_hash": True,
                        }:
                    raise WindowError(f"existing canonical commit receipt differs: {ordinal}")
            else:
                _atomic_json(receipt_path, receipt)
            shard["commit"] = {"status": "complete", "output": artifact,
                               "receipt": _artifact(receipt_path),
                               "committed_at": _now()}
            self.state["next_commit_ordinal"] += 1
            self._save()

    def _in_flight_ordinals(self) -> set[int]:
        result = set()
        for shard in self.state["shards"]:
            if shard["commit"]["status"] == "complete":
                continue
            if shard["prepare"]["status"] in {"running", "complete"} \
                    or shard["finalize"]["status"] == "running":
                result.add(shard["ordinal"])
        return result

    def run(self, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
        if self.request["execution_mode"] == "production-parity-gated":
            raise WindowError("production shard window remains blocked pending owner-free parity")
        deadline = time.monotonic() + timeout_seconds
        while self.state["status"] == "running":
            if time.monotonic() >= deadline:
                raise WindowError("window coordinator timeout")
            progress = False
            for key, process in list(self.processes.items()):
                code = process.poll()
                if code is None:
                    continue
                ordinal, phase = key; del self.processes[key]
                if code != 0:
                    stage = self.state["shards"][ordinal][phase]
                    stage["status"] = "failed"; stage["exit_code"] = code
                    self._cancel_later(ordinal, f"earlier shard {phase} failed")
                    raise WindowError(f"shard {ordinal} {phase} child exited {code}")
                try:
                    self._complete_stage(ordinal, phase)
                except (OSError, WindowError) as exc:
                    self.state["shards"][ordinal][phase]["status"] = "failed"
                    self._cancel_later(ordinal, f"earlier shard validator failed: {exc}")
                    raise
                progress = True
            self._commit_ready()
            if self.state["next_commit_ordinal"] == len(self.state["shards"]):
                self.state["status"] = "complete"; self._save(); break

            max_in_flight = self.request["thread_profile"]["maximum_in_flight_shards"]
            # Prioritize finalization so slot N can overlap prepare N+1.
            for shard in self.state["shards"]:
                if any(running_phase == "finalize" and process.poll() is None
                       for (_running_ordinal, running_phase), process
                       in self.processes.items()):
                    break
                ordinal = shard["ordinal"]
                if shard["prepare"]["status"] == "complete" \
                        and shard["finalize"]["status"] == "pending" \
                        and len(self.processes) < max_in_flight:
                    admitted, sample = self._admitted("finalize")
                    shard["finalize"]["last_admission_sample"] = sample
                    if admitted:
                        self._launch(ordinal, "finalize"); progress = True
            for shard in self.state["shards"]:
                if any(running_phase == "prepare" and process.poll() is None
                       for (_running_ordinal, running_phase), process
                       in self.processes.items()):
                    break
                if len(self._in_flight_ordinals()) >= max_in_flight \
                        or len(self.processes) >= max_in_flight:
                    break
                ordinal = shard["ordinal"]
                if shard["prepare"]["status"] != "pending":
                    continue
                admitted, sample = self._admitted("prepare")
                shard["prepare"]["last_admission_sample"] = sample
                if admitted:
                    self._launch(ordinal, "prepare"); progress = True
                break
            if not progress:
                if not self.processes:
                    raise WindowError("resource gate prevents all pending shard progress")
                time.sleep(self.poll_seconds)
        return self.state
