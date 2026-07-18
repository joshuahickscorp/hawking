#!/usr/bin/env python3
"""Build host-native/PGO candidates into sealed non-live target directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA = "hawking.strand.native-build.v1"
MERGE_SCHEMA = "hawking.strand.native-pgo-merge.v1"
TRAINING_SCHEMA = "hawking.strand.native-pgo-training.v1"
EXECUTION_SCHEMA = "hawking.strand.native-pgo-execution.v1"
TRAINING_ADMISSION_SCHEMA = "hawking.strand.native-pgo-training-admission.v1"
TRAINING_OUTPUT_SCHEMA = "hawking.strand.native-pgo-training-output.v1"
TRAINING_PARITY_SCHEMA = "hawking.strand.native-pgo-training-parity.v1"
CRATE = Path(__file__).resolve().parents[1]
REPO = CRATE.parents[1]
ADMITTED_ROOT = (REPO / "build" / "native-execution").resolve()
OWNER_PATTERN_SOURCE = (
    REPO / "tools" / "condense" / "doctor_v5_single_device_sprint_audit.py"
).resolve()


class BuildError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def required_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label} must be a non-empty path string")
    return Path(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, text=True, capture_output=True).stdout.strip()


def admitted_path(path: Path, *, label: str, require_file: bool = False) -> Path:
    if path.is_symlink():
        raise BuildError(f"{label} must not be a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(ADMITTED_ROOT)
    except ValueError as error:
        raise BuildError(f"{label} must be below {ADMITTED_ROOT}") from error
    if require_file and not resolved.is_file():
        raise BuildError(f"{label} must be an existing regular file")
    return resolved


def read_receipt(path: Path, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = admitted_path(path, label=label, require_file=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"{label} root must be an object")
    return resolved, value, sha256_file(resolved)


def validate_sealed_document(
    value: dict[str, Any], *, schema: str, status: str, label: str
) -> None:
    sealed = value.get("document_sha256")
    body = {key: item for key, item in value.items() if key != "document_sha256"}
    if (
        value.get("schema") != schema
        or value.get("status") != status
        or not valid_sha256(sealed)
        or sealed != canonical_sha256(body)
    ):
        raise BuildError(f"{label} schema/status/self-hash is invalid")


def source_manifest() -> dict[str, Any]:
    paths = [CRATE / "Cargo.toml", Path(__file__).resolve()]
    for optional in [CRATE / "Cargo.lock", CRATE / "build.rs"]:
        if optional.is_file():
            paths.append(optional)
    paths.extend(sorted((CRATE / "src").rglob("*.rs")))
    entries = [
        {"path": str(path.relative_to(CRATE)), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(entry["path"].encode())
        aggregate.update(b"\0")
        aggregate.update(entry["sha256"].encode())
        aggregate.update(b"\0")
    return {"sha256": aggregate.hexdigest(), "files": entries}


def host_contract() -> dict[str, Any]:
    def optional(argv: list[str]) -> str | None:
        try:
            return command_output(argv)
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_brand": optional(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpu": optional(["sysctl", "-n", "hw.logicalcpu"]),
        "physical_cpu": optional(["sysctl", "-n", "hw.physicalcpu"]),
        "memory_bytes": optional(["sysctl", "-n", "hw.memsize"]),
        "rustc_vv": command_output(["rustc", "-vV"]),
        "cargo_version": command_output(["cargo", "-V"]),
    }


def sync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path = admitted_path(path, label="receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise BuildError(f"refusing to replace existing receipt: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        sync_directory(path.parent)
        os.link(temporary, path, follow_symlinks=False)
        sync_directory(path.parent)
        temporary.unlink()
        sync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def admitted_target(path: Path) -> Path:
    resolved = admitted_path(path, label="target-dir")
    if resolved in {(REPO / "target").resolve(), (CRATE / "target").resolve()}:
        raise BuildError("refusing a live or shared target directory")
    if resolved.exists() or resolved.is_symlink():
        raise BuildError("native build target directory must be fresh")
    return resolved


def validate_instrumented_build(path: Path) -> dict[str, Any]:
    resolved, receipt, receipt_sha256 = read_receipt(
        path, label="instrumented build receipt"
    )
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "instrumented"
        or receipt.get("mode") != "pgo-generate"
    ):
        raise BuildError("instrumented build receipt has the wrong schema/status/mode")
    program = admitted_path(
        required_path(receipt.get("program"), label="instrumented program"),
        label="instrumented program",
        require_file=True,
    )
    if receipt.get("program_sha256") != sha256_file(program):
        raise BuildError("instrumented program differs from its build receipt")
    current_source = source_manifest()
    if receipt.get("source_manifest") != current_source:
        raise BuildError("instrumented build source manifest is no longer current")
    current_host = host_contract()
    if receipt.get("host") != current_host:
        raise BuildError("instrumented build host differs from the current host")
    return {
        "path": str(resolved),
        "receipt_sha256": receipt_sha256,
        "program": str(program),
        "program_sha256": receipt["program_sha256"],
        "source_manifest_sha256": current_source["sha256"],
        "host_sha256": canonical_sha256(current_host),
    }


def profile_identities(paths: list[Path]) -> list[dict[str, Any]]:
    resolved = sorted(
        (
            admitted_path(path, label="raw profile", require_file=True)
            for path in paths
        ),
        key=str,
    )
    if not resolved:
        raise BuildError("at least one raw profile is required")
    if len(set(resolved)) != len(resolved):
        raise BuildError("raw profile inputs must be unique")
    return [
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in resolved
    ]


def require_exact_keys(value: dict[str, Any], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise BuildError(f"{label} fields differ (missing={missing}, extra={extra})")


def artifact_identity(
    row: Any,
    *,
    label: str,
    confined_required: bool,
    include_mtime: bool = False,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise BuildError(f"{label} identity must be an object")
    keys = {"path", "sha256", "bytes"}
    if include_mtime:
        keys.add("mtime_ns")
    require_exact_keys(row, keys, label=f"{label} identity")
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BuildError(f"{label} path is invalid")
    candidate = Path(raw_path)
    if candidate.is_symlink():
        raise BuildError(f"{label} must not be a symlink")
    path = (
        admitted_path(candidate, label=label, require_file=True)
        if confined_required
        else candidate.resolve()
    )
    if not path.is_file() or path.is_symlink():
        raise BuildError(f"{label} must be an existing regular file")
    stat_result = path.stat()
    actual: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": stat_result.st_size,
    }
    if include_mtime:
        actual["mtime_ns"] = stat_result.st_mtime_ns
    if row != actual:
        raise BuildError(f"{label} identity differs from the current artifact")
    return actual


def sealed_receipt_artifact(
    row: Any, *, label: str, schema: str, status: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = artifact_identity(
        row, label=label, confined_required=True, include_mtime=False
    )
    _, document, _ = read_receipt(Path(identity["path"]), label=label)
    validate_sealed_document(document, schema=schema, status=status, label=label)
    return identity, document


def positive_number(value: Any, *, label: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BuildError(f"{label} must be numeric")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise BuildError(f"{label} is outside its admitted envelope") from error
    if not (numeric >= 0 if allow_zero else numeric > 0) or numeric == float("inf"):
        raise BuildError(f"{label} is outside its admitted envelope")
    return numeric


def validate_pgo_execution_receipt(
    path: Path, *, build: dict[str, Any]
) -> dict[str, Any]:
    resolved, receipt, receipt_sha256 = read_receipt(path, label="PGO execution receipt")
    validate_sealed_document(
        receipt, schema=EXECUTION_SCHEMA, status="pass", label="PGO execution receipt"
    )
    require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "document_sha256",
            "instrumented_build_receipt",
            "program",
            "invocation",
            "input_bundle",
            "admission_receipt",
            "output_receipt",
            "parity_receipt",
            "run",
            "profile_generation",
            "resources",
            "source_files_deleted",
            "runtime_defaults_changed",
        },
        label="PGO execution receipt",
    )

    expected_build_row = {"path": build["path"], "sha256": build["receipt_sha256"]}
    if receipt.get("instrumented_build_receipt") != expected_build_row:
        raise BuildError("PGO execution receipt does not bind the instrumented build")
    program = artifact_identity(
        receipt.get("program"),
        label="instrumented training program",
        confined_required=True,
    )
    if program["path"] != build["program"] or program["sha256"] != build["program_sha256"]:
        raise BuildError("PGO execution program differs from the instrumented build")

    invocation = receipt.get("invocation")
    if not isinstance(invocation, dict):
        raise BuildError("PGO execution invocation must be an object")
    require_exact_keys(
        invocation, {"argv", "cwd", "environment", "sha256"}, label="PGO invocation"
    )
    argv = invocation.get("argv")
    environment = invocation.get("environment")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
        or not isinstance(environment, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        )
    ):
        raise BuildError("PGO invocation argv/environment is invalid")
    executable = Path(argv[0])
    if not executable.is_absolute() or executable.resolve() != Path(program["path"]):
        raise BuildError("PGO invocation does not execute the bound instrumented program")
    cwd_text = invocation.get("cwd")
    if not isinstance(cwd_text, str):
        raise BuildError("PGO invocation cwd is invalid")
    cwd = admitted_path(Path(cwd_text), label="PGO invocation cwd")
    if not cwd.is_dir() or cwd.is_symlink():
        raise BuildError("PGO invocation cwd must be a staged directory")
    invocation_body = {"argv": argv, "cwd": str(cwd), "environment": environment}
    if invocation.get("sha256") != canonical_sha256(invocation_body):
        raise BuildError("PGO invocation hash differs from its exact argv/cwd/environment")

    input_bundle = artifact_identity(
        receipt.get("input_bundle"),
        label="training input bundle",
        confined_required=False,
    )
    run = receipt.get("run")
    if not isinstance(run, dict):
        raise BuildError("PGO run evidence must be an object")
    require_exact_keys(
        run,
        {"started_unix_ns", "finished_unix_ns", "exit_code", "skipped"},
        label="PGO run evidence",
    )
    started = run.get("started_unix_ns")
    finished = run.get("finished_unix_ns")
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or started <= 0
        or finished < started
        or finished > time.time_ns() + 5_000_000_000
        or run.get("exit_code") != 0
        or run.get("skipped") is not False
    ):
        raise BuildError("PGO run was skipped, failed, or has an invalid time interval")

    admission_identity, admission = sealed_receipt_artifact(
        receipt.get("admission_receipt"),
        label="PGO training admission receipt",
        schema=TRAINING_ADMISSION_SCHEMA,
        status="admitted",
    )
    require_exact_keys(
        admission,
        {
            "schema",
            "status",
            "document_sha256",
            "generated_unix_ns",
            "instrumented_build_receipt_sha256",
            "instrumented_program_sha256",
            "invocation_sha256",
            "active_heavy_owner_count",
            "owners_rechecked_under_lease",
            "exclusive_heavy_lease_held",
            "resource_health_ok",
            "owner_pattern_source",
            "owner_snapshot_sha256",
            "resource_snapshot_sha256",
        },
        label="PGO training admission receipt",
    )
    admission_time = admission.get("generated_unix_ns")
    if (
        isinstance(admission_time, bool)
        or not isinstance(admission_time, int)
        or admission_time > started
        or started - admission_time > 60_000_000_000
        or admission.get("instrumented_build_receipt_sha256")
        != build["receipt_sha256"]
        or admission.get("instrumented_program_sha256") != program["sha256"]
        or admission.get("invocation_sha256") != invocation["sha256"]
        or admission.get("active_heavy_owner_count") != 0
        or admission.get("owners_rechecked_under_lease") is not True
        or admission.get("exclusive_heavy_lease_held") is not True
        or admission.get("resource_health_ok") is not True
    ):
        raise BuildError("PGO training admission is stale or not owner-free/lease-held")
    expected_owner_source = {
        "path": str(OWNER_PATTERN_SOURCE),
        "sha256": sha256_file(OWNER_PATTERN_SOURCE),
        "bytes": OWNER_PATTERN_SOURCE.stat().st_size,
    }
    if admission.get("owner_pattern_source") != expected_owner_source or not valid_sha256(
        admission.get("owner_snapshot_sha256")
    ) or not valid_sha256(admission.get("resource_snapshot_sha256")):
        raise BuildError("PGO admission does not bind authoritative owner/resource snapshots")

    generation = receipt.get("profile_generation")
    if not isinstance(generation, dict):
        raise BuildError("PGO profile-generation evidence must be an object")
    require_exact_keys(
        generation,
        {"directory", "before_entries", "after_entries"},
        label="PGO profile-generation evidence",
    )
    directory_text = generation.get("directory")
    if not isinstance(directory_text, str):
        raise BuildError("PGO raw-profile directory is invalid")
    profile_directory = admitted_path(Path(directory_text), label="raw-profile directory")
    if not profile_directory.is_dir() or profile_directory.is_symlink():
        raise BuildError("PGO raw-profile directory must be a staged directory")
    if generation.get("before_entries") != []:
        raise BuildError("PGO raw-profile directory was not empty before training")
    template = environment.get("LLVM_PROFILE_FILE")
    if not isinstance(template, str) or "%p" not in template or ".profraw" not in template:
        raise BuildError("PGO invocation must use a collision-safe LLVM_PROFILE_FILE template")
    template_parent = admitted_path(Path(template).parent, label="profile template parent")
    if template_parent != profile_directory:
        raise BuildError("LLVM_PROFILE_FILE does not target the recorded profile directory")
    after_rows = generation.get("after_entries")
    if not isinstance(after_rows, list) or not after_rows:
        raise BuildError("PGO training produced no raw profiles")
    profiles = [
        artifact_identity(
            row,
            label="generated raw profile",
            confined_required=True,
            include_mtime=True,
        )
        for row in after_rows
    ]
    profile_paths = [row["path"] for row in profiles]
    if len(set(profile_paths)) != len(profile_paths):
        raise BuildError("PGO execution receipt repeats a raw profile")
    directory_entries = list(profile_directory.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".profraw"
        for path in directory_entries
    ):
        raise BuildError("PGO raw-profile directory contains a non-profraw entry")
    actual_paths = sorted(
        str(path.resolve())
        for path in directory_entries
    )
    if sorted(profile_paths) != actual_paths:
        raise BuildError("PGO raw-profile directory contents differ from execution evidence")
    if any(
        row["bytes"] <= 0
        or row["mtime_ns"] < started
        or row["mtime_ns"] > finished
        for row in profiles
    ):
        raise BuildError("PGO raw profile was not physically generated during the run")

    resources = receipt.get("resources")
    if not isinstance(resources, dict):
        raise BuildError("PGO resource envelope must be an object")
    require_exact_keys(
        resources,
        {
            "owner_free_before",
            "owner_free_after",
            "active_heavy_owner_count_before",
            "active_heavy_owner_count_after",
            "exclusive_heavy_lease_held_throughout",
            "memory_pressure_start",
            "memory_pressure_end",
            "thermal_start",
            "thermal_end",
            "swap_start_bytes",
            "swap_end_bytes",
            "peak_rss_bytes",
            "cpu_seconds",
            "wall_seconds",
            "disk_free_start_bytes",
            "disk_free_end_bytes",
            "scratch_peak_bytes",
        },
        label="PGO resource envelope",
    )
    if (
        resources.get("owner_free_before") is not True
        or resources.get("owner_free_after") is not True
        or resources.get("active_heavy_owner_count_before") != 0
        or resources.get("active_heavy_owner_count_after") != 0
        or resources.get("exclusive_heavy_lease_held_throughout") is not True
        or resources.get("memory_pressure_start") != "normal"
        or resources.get("memory_pressure_end") != "normal"
        or not isinstance(resources.get("thermal_start"), str)
        or resources.get("thermal_start") not in {"nominal", "fair"}
        or not isinstance(resources.get("thermal_end"), str)
        or resources.get("thermal_end") not in {"nominal", "fair"}
    ):
        raise BuildError("PGO resource envelope is not owner-free and healthy")
    for field in (
        "swap_start_bytes",
        "swap_end_bytes",
        "scratch_peak_bytes",
    ):
        positive_number(resources.get(field), label=field, allow_zero=True)
    for field in (
        "peak_rss_bytes",
        "cpu_seconds",
        "wall_seconds",
        "disk_free_start_bytes",
        "disk_free_end_bytes",
    ):
        positive_number(resources.get(field), label=field)

    output_identity, output = sealed_receipt_artifact(
        receipt.get("output_receipt"),
        label="PGO training output receipt",
        schema=TRAINING_OUTPUT_SCHEMA,
        status="pass",
    )
    require_exact_keys(
        output,
        {
            "schema",
            "status",
            "document_sha256",
            "program_sha256",
            "invocation_sha256",
            "input_bundle_sha256",
            "output_bundle",
        },
        label="PGO training output receipt",
    )
    output_bundle = artifact_identity(
        output.get("output_bundle"),
        label="PGO training output bundle",
        confined_required=True,
    )
    if (
        output.get("program_sha256") != program["sha256"]
        or output.get("invocation_sha256") != invocation["sha256"]
        or output.get("input_bundle_sha256") != input_bundle["sha256"]
    ):
        raise BuildError("PGO training output receipt authority differs")

    parity_identity, parity = sealed_receipt_artifact(
        receipt.get("parity_receipt"),
        label="PGO training parity receipt",
        schema=TRAINING_PARITY_SCHEMA,
        status="pass",
    )
    require_exact_keys(
        parity,
        {
            "schema",
            "status",
            "document_sha256",
            "program_sha256",
            "invocation_sha256",
            "input_bundle_sha256",
            "output_bundle_sha256",
            "exact_output",
            "skipped_cases",
        },
        label="PGO training parity receipt",
    )
    if (
        parity.get("program_sha256") != program["sha256"]
        or parity.get("invocation_sha256") != invocation["sha256"]
        or parity.get("input_bundle_sha256") != input_bundle["sha256"]
        or parity.get("output_bundle_sha256") != output_bundle["sha256"]
        or parity.get("exact_output") is not True
        or parity.get("skipped_cases") != 0
    ):
        raise BuildError("PGO training parity is not exact and non-skipping")
    if receipt.get("source_files_deleted") is not False:
        raise BuildError("PGO execution receipt permits source deletion")
    if receipt.get("runtime_defaults_changed") is not False:
        raise BuildError("PGO execution receipt permits runtime-default mutation")

    return {
        "path": str(resolved),
        "receipt_sha256": receipt_sha256,
        "program": program,
        "invocation": invocation,
        "input_bundle": input_bundle,
        "profiles": [
            {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in profiles
        ],
        "admission_receipt": admission_identity,
        "output_receipt": output_identity,
        "output_bundle": output_bundle,
        "parity_receipt": parity_identity,
        "resources": resources,
        "run": run,
    }


def validate_training_receipt(
    path: Path, *, expected_build: dict[str, Any] | None = None
) -> dict[str, Any]:
    resolved, receipt, receipt_sha256 = read_receipt(path, label="training receipt")
    validate_sealed_document(
        receipt, schema=TRAINING_SCHEMA, status="pass", label="training receipt"
    )
    require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "document_sha256",
            "scope",
            "generated_unix_ns",
            "instrumented_build_receipt",
            "execution_receipt",
            "instrumented_program",
            "program_sha256",
            "source_manifest_sha256",
            "host_sha256",
            "input_bundle",
            "invocation_sha256",
            "profiles",
            "admission_receipt",
            "output_receipt",
            "output_bundle",
            "parity_receipt",
            "resources",
            "run",
            "runtime_activation",
        },
        label="training receipt",
    )
    build_receipt_row = receipt.get("instrumented_build_receipt")
    if not isinstance(build_receipt_row, dict):
        raise BuildError("training receipt instrumented build authority is invalid")
    build_path = required_path(
        build_receipt_row.get("path"), label="instrumented build receipt"
    )
    build = validate_instrumented_build(build_path)
    expected_build_row = {
        "path": build["path"],
        "sha256": build["receipt_sha256"],
    }
    if receipt.get("instrumented_build_receipt") != expected_build_row:
        raise BuildError("training receipt does not bind its instrumented build")
    execution_row = receipt.get("execution_receipt")
    if not isinstance(execution_row, dict):
        raise BuildError("training receipt execution authority is invalid")
    execution = validate_pgo_execution_receipt(
        required_path(execution_row.get("path"), label="PGO execution receipt"),
        build=build,
    )
    if receipt.get("execution_receipt") != {
        "path": execution["path"],
        "sha256": execution["receipt_sha256"],
    }:
        raise BuildError("training receipt does not bind its execution receipt")
    if receipt.get("instrumented_program") != build["program"]:
        raise BuildError("training receipt names a different instrumented program")
    for field in (
        "program_sha256",
        "source_manifest_sha256",
        "host_sha256",
    ):
        if receipt.get(field) != build[field]:
            raise BuildError(f"training receipt {field} differs from its build")
    for field, expected in (
        ("input_bundle", execution["input_bundle"]),
        ("invocation_sha256", execution["invocation"]["sha256"]),
        ("profiles", execution["profiles"]),
        ("admission_receipt", execution["admission_receipt"]),
        ("output_receipt", execution["output_receipt"]),
        ("output_bundle", execution["output_bundle"]),
        ("parity_receipt", execution["parity_receipt"]),
        ("resources", execution["resources"]),
        ("run", execution["run"]),
    ):
        if receipt.get(field) != expected:
            raise BuildError(f"training receipt {field} differs from execution evidence")
    if receipt.get("runtime_activation") is not False:
        raise BuildError("training receipt must remain runtime-inactive")
    if expected_build is not None and build != expected_build:
        raise BuildError("training receipt is for a different instrumented build")
    return {
        "path": str(resolved),
        "receipt_sha256": receipt_sha256,
        "build": build,
        "profiles": execution["profiles"],
        "input_bundle_sha256": execution["input_bundle"]["sha256"],
        "invocation_sha256": receipt["invocation_sha256"],
        "execution": execution,
    }


def validate_profile_authority(profile: Path, merge_receipt_path: Path) -> dict[str, Any]:
    profile = admitted_path(profile, label="profile data", require_file=True)
    merge_path, receipt, receipt_sha256 = read_receipt(
        merge_receipt_path, label="profile merge receipt"
    )
    merge_body = {
        key: value for key, value in receipt.items() if key != "document_sha256"
    }
    if (
        receipt.get("schema") != MERGE_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("document_sha256") != canonical_sha256(merge_body)
    ):
        raise BuildError("profile merge receipt schema/status is invalid")
    recorded_output = required_path(receipt.get("output"), label="merged profile output")
    if recorded_output.resolve() != profile or receipt.get(
        "output_sha256"
    ) != sha256_file(profile) or receipt.get("requested_output") != str(profile):
        raise BuildError("profile data differs from the merge receipt")
    tool_row = receipt.get("llvm_profdata")
    if not isinstance(tool_row, dict):
        raise BuildError("llvm-profdata identity differs from the merge receipt")
    tool = required_path(tool_row.get("path"), label="llvm-profdata").resolve()
    if (
        not tool.is_file()
        or tool_row.get("sha256") != sha256_file(tool)
        or tool_row.get("version") != command_output([str(tool), "--version"])
    ):
        raise BuildError("llvm-profdata identity differs from the merge receipt")
    build_row = receipt.get("instrumented_build_receipt")
    training_row = receipt.get("training_receipt")
    if not isinstance(build_row, dict) or not isinstance(training_row, dict):
        raise BuildError("profile merge receipt build/training authority is invalid")
    build = validate_instrumented_build(
        required_path(build_row.get("path"), label="instrumented build receipt")
    )
    training = validate_training_receipt(
        required_path(training_row.get("path"), label="training receipt"),
        expected_build=build,
    )
    if receipt.get("instrumented_build_receipt") != {
        "path": build["path"],
        "sha256": build["receipt_sha256"],
    } or receipt.get("training_receipt") != {
        "path": training["path"],
        "sha256": training["receipt_sha256"],
    }:
        raise BuildError("profile merge receipt build/training authority differs")
    for field in ("program_sha256", "source_manifest_sha256", "host_sha256"):
        if receipt.get(field) != build[field]:
            raise BuildError(f"profile merge receipt {field} differs")
    if receipt.get("inputs") != training["profiles"]:
        raise BuildError("profile merge inputs differ from the sealed training profiles")
    return {
        "profile": str(profile),
        "profile_sha256": receipt["output_sha256"],
        "merge_receipt": str(merge_path),
        "merge_receipt_sha256": receipt_sha256,
        "build": build,
        "training": training,
    }


def build(args: argparse.Namespace) -> int:
    target_dir = admitted_target(args.target_dir)
    receipt_path = admitted_path(args.receipt, label="build receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise BuildError("build receipt path must be fresh")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_dir.mkdir()
    except FileExistsError as error:
        raise BuildError("native build target directory lost its freshness race") from error
    before = source_manifest()
    rustflags = ["-C", "target-cpu=native"]
    profile_sha256 = None
    profile_authority = None
    if args.mode == "pgo-generate":
        raw_dir = admitted_path(args.raw_profile_dir, label="raw-profile-dir")
        try:
            raw_dir.relative_to(target_dir)
        except ValueError as error:
            raise BuildError("raw-profile-dir must be inside the fresh target-dir") from error
        if raw_dir.exists() or raw_dir.is_symlink():
            raise BuildError("raw-profile-dir must be fresh")
        raw_dir.mkdir(parents=True, exist_ok=True)
        rustflags.extend(["-C", f"profile-generate={raw_dir}"])
    elif args.mode == "pgo-use":
        if args.profile_data is None:
            raise BuildError("pgo-use requires an existing --profile-data file")
        if args.profile_merge_receipt is None:
            raise BuildError("pgo-use requires --profile-merge-receipt authority")
        profile_authority = validate_profile_authority(
            args.profile_data, args.profile_merge_receipt
        )
        profile = Path(profile_authority["profile"])
        profile_sha256 = profile_authority["profile_sha256"]
        rustflags.extend(
            ["-C", f"profile-use={profile}", "-C", "llvm-args=-pgo-warn-missing-function"]
        )
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["RUSTFLAGS"] = " ".join(rustflags)
    argv = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--features",
        "native-execution",
        "--bin",
        "quantize-model-native",
    ]
    started = time.monotonic_ns()
    completed = subprocess.run(argv, cwd=CRATE, env=env, text=True, capture_output=True)
    wall_ns = time.monotonic_ns() - started
    if completed.returncode != 0:
        raise BuildError(f"native build failed:\n{completed.stderr[-8000:]}")
    after = source_manifest()
    if before["sha256"] != after["sha256"]:
        raise BuildError("source tree changed during native build")
    binary = target_dir / "release" / "quantize-model-native"
    if not binary.is_file():
        raise BuildError(f"native binary missing: {binary}")
    receipt = {
        "schema": SCHEMA,
        "status": "instrumented" if args.mode == "pgo-generate" else "built_candidate",
        "scope": "build_only",
        "generated_unix_ns": time.time_ns(),
        "mode": args.mode,
        "host": host_contract(),
        "source_manifest": before,
        "program": str(binary),
        "program_sha256": sha256_file(binary),
        "invocation_identity": {
            "argv": argv,
            "cwd": str(CRATE),
            "CARGO_TARGET_DIR": str(target_dir),
            "RUSTFLAGS": env["RUSTFLAGS"],
            "allocator": "system",
            "features": ["native-execution"],
        },
        "profile_data_sha256": profile_sha256,
        "profile_merge_receipt_sha256": (
            profile_authority["merge_receipt_sha256"]
            if profile_authority is not None
            else None
        ),
        "pgo_authority": (
            {
                "instrumented_program_sha256": profile_authority["build"][
                    "program_sha256"
                ],
                "instrumented_build_receipt_sha256": profile_authority["build"][
                    "receipt_sha256"
                ],
                "training_receipt_sha256": profile_authority["training"][
                    "receipt_sha256"
                ],
                "host_sha256": profile_authority["build"]["host_sha256"],
                "source_manifest_sha256": profile_authority["build"][
                    "source_manifest_sha256"
                ],
            }
            if profile_authority is not None
            else None
        ),
        "measurements": {
            "build_wall_ns": wall_ns,
            "read_decode_ns": None,
            "rht_preprocess_ns": None,
            "encode_ns": None,
            "finalize_write_ns": None,
            "end_to_end_wall_ns": None,
            "cpu_time_ns": None,
            "gpu_time_ns": None,
            "peak_rss_bytes": None,
            "swap_delta_bytes": None,
            "scratch_peak_bytes": None,
            "disk_read_bytes": None,
            "disk_write_bytes": None,
            "thermal_start": None,
            "thermal_end": None,
        },
        "input_bundle_sha256": None,
        "output_bundle_sha256": None,
        "scientific_receipt_bundle_sha256": None,
        "component_speedup_is_eta_evidence": False,
        "exact_output_receipt": None,
        "runtime_activation": False,
        "production_promotion_allowed": False,
    }
    atomic_write(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def seal_training(args: argparse.Namespace) -> int:
    receipt_path = admitted_path(args.receipt, label="training receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise BuildError("training receipt path must be fresh")
    build = validate_instrumented_build(args.instrumented_build_receipt)
    execution = validate_pgo_execution_receipt(args.execution_receipt, build=build)
    document = {
        "schema": TRAINING_SCHEMA,
        "status": "pass",
        "scope": "profile-training-only",
        "generated_unix_ns": time.time_ns(),
        "instrumented_build_receipt": {
            "path": build["path"],
            "sha256": build["receipt_sha256"],
        },
        "instrumented_program": build["program"],
        "program_sha256": build["program_sha256"],
        "source_manifest_sha256": build["source_manifest_sha256"],
        "host_sha256": build["host_sha256"],
        "execution_receipt": {
            "path": execution["path"],
            "sha256": execution["receipt_sha256"],
        },
        "input_bundle": execution["input_bundle"],
        "invocation_sha256": execution["invocation"]["sha256"],
        "profiles": execution["profiles"],
        "admission_receipt": execution["admission_receipt"],
        "output_receipt": execution["output_receipt"],
        "output_bundle": execution["output_bundle"],
        "parity_receipt": execution["parity_receipt"],
        "resources": execution["resources"],
        "run": execution["run"],
        "runtime_activation": False,
    }
    document["document_sha256"] = canonical_sha256(document)
    atomic_write(receipt_path, document)
    print(json.dumps(document, sort_keys=True))
    return 0


def publish_profile(temporary: Path, output: Path) -> None:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, output)
    directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
        temporary.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)


def merge(args: argparse.Namespace) -> int:
    output = admitted_path(args.output, label="merged profile")
    receipt_path = admitted_path(args.receipt, label="profile merge receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise BuildError("profile merge receipt path must be fresh")
    build = validate_instrumented_build(args.instrumented_build_receipt)
    training = validate_training_receipt(args.training_receipt, expected_build=build)
    inputs = profile_identities(args.profraw)
    if inputs != training["profiles"]:
        raise BuildError("merge inputs differ from the sealed training receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_tool = shutil.which(args.llvm_profdata)
    if resolved_tool is None:
        raise BuildError(f"llvm-profdata not found: {args.llvm_profdata}")
    tool = Path(resolved_tool).resolve()
    if output.exists() or output.is_symlink():
        raise BuildError("refusing to replace existing merged profile data")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".pgo-merge-") as directory:
        temporary = Path(directory) / output.name
        argv = [
            str(tool),
            "merge",
            "-o",
            str(temporary),
            *(row["path"] for row in inputs),
        ]
        subprocess.run(argv, check=True)
        publish_profile(temporary, output)
    receipt = {
        "schema": MERGE_SCHEMA,
        "status": "pass",
        "scope": "profile_data_only",
        "generated_unix_ns": time.time_ns(),
        "argv": argv,
        "requested_output": str(output),
        "llvm_profdata": {
            "path": str(tool),
            "sha256": sha256_file(tool),
            "version": command_output([str(tool), "--version"]),
        },
        "inputs": inputs,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "instrumented_build_receipt": {
            "path": build["path"],
            "sha256": build["receipt_sha256"],
        },
        "training_receipt": {
            "path": training["path"],
            "sha256": training["receipt_sha256"],
        },
        "program_sha256": build["program_sha256"],
        "source_manifest_sha256": build["source_manifest_sha256"],
        "host_sha256": build["host_sha256"],
        "runtime_activation": False,
    }
    receipt["document_sha256"] = canonical_sha256(receipt)
    atomic_write(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--target-dir", type=Path, required=True)
    build_parser.add_argument("--receipt", type=Path, required=True)
    build_parser.add_argument(
        "--mode", choices=("native", "pgo-generate", "pgo-use"), default="native"
    )
    build_parser.add_argument("--raw-profile-dir", type=Path)
    build_parser.add_argument("--profile-data", type=Path)
    build_parser.add_argument("--profile-merge-receipt", type=Path)
    training_parser = commands.add_parser("seal-training")
    training_parser.add_argument("--instrumented-build-receipt", type=Path, required=True)
    training_parser.add_argument("--execution-receipt", type=Path, required=True)
    training_parser.add_argument("--receipt", type=Path, required=True)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--llvm-profdata", default="llvm-profdata")
    merge_parser.add_argument("--profraw", type=Path, action="append", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--receipt", type=Path, required=True)
    merge_parser.add_argument("--instrumented-build-receipt", type=Path, required=True)
    merge_parser.add_argument("--training-receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            if args.mode == "pgo-generate" and args.raw_profile_dir is None:
                raise BuildError("pgo-generate requires --raw-profile-dir")
            return build(args)
        if args.command == "seal-training":
            return seal_training(args)
        return merge(args)
    except (BuildError, OSError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
