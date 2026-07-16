#!/usr/bin/env python3.12
"""Default-off direct-counter barrier for an external Doctor A/B arm.

The Doctor physical A/B executor launches the benchmark itself.  This process
is the separate collector-side barrier: it must be ready before the arm is
released, observe an exact PID/nonce/program interval, and remain live until
the executor writes the stop receipt.  It is intentionally distinct from the
Appendix device/spec counter normalizer and executor.

No trustworthy process-attributed joule backend is registered today.
``powermetrics --show-process-energy`` reports an energy-impact score and the
tool itself describes subsystem power as estimated; neither can satisfy the
Doctor ``energy_j``/non-estimated contract.  Consequently ``collect`` exits 75
before reading a request or creating a barrier file.  A future backend must be
added to the source-bound registry below and pass the exact raw-capture and
attestation validators; no manifest can opt around that source review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import stat
import sys
from typing import Any, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"
SCHEMA = "hawking.doctor_v5_physical_counter_barrier.v1"
STATUS_SCHEMA = "hawking.doctor_v5_physical_counter_barrier_status.v1"
DRY_RUN_SCHEMA = "hawking.doctor_v5_physical_counter_barrier_dry_run.v1"
BACKEND_SCHEMA = "hawking.doctor_v5_direct_counter_backend.v1"
COUNTER_SCHEMA = "hawking.doctor_v5_physical_ab_counters.v1"
ATTESTATION_SCHEMA = "hawking.doctor_v5_physical_ab_counter_attestation.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EX_TEMPFAIL = 75
COUNTER_FIELDS = (
    "energy_j", "cpu_time_ns", "read_bytes", "write_bytes", "peak_rss_bytes",
)

# Deliberately empty.  A backend entry is reviewed source, not caller data.  It
# must name an exact executable/source hash and implement concurrent, direct,
# process-attributed joule + proc-rusage/I/O/RSS capture.  The empty registry is
# the present honest answer: no such backend has been physically qualified.
DIRECT_BACKEND_REGISTRY: dict[str, dict[str, Any]] = {}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def build_config() -> dict[str, Any]:
    return _stamp({
        "schema": SCHEMA,
        "default_off": True,
        "external_arm_barrier_abi": "hawking.doctor_v5_physical_counter_barrier.v1",
        "counter_payload_schema": COUNTER_SCHEMA,
        "counter_attestation_schema": ATTESTATION_SCHEMA,
        "required_direct_fields": list(COUNTER_FIELDS),
        "estimated_values_permitted": False,
        "process_attribution_required": True,
        "wall_and_continuous_interval_coverage_required": True,
        "raw_captures_immutable": True,
        "inherited_shared_heavy_lease_required": True,
        "opens_shared_heavy_lease": False,
        "shell_permitted": False,
        "powermetrics_energy_impact_accepted_as_joules": False,
        "powermetrics_estimated_subsystem_power_accepted": False,
        "backend_registry_ids": sorted(DIRECT_BACKEND_REGISTRY),
    }, "config_sha256")


def _fd_present(env: Mapping[str, str]) -> bool:
    try:
        descriptor = int(env.get(HEAVY_LEASE_FD_ENV, ""))
        info = os.fstat(descriptor)
    except (OSError, TypeError, ValueError):
        return False
    return stat.S_ISREG(info.st_mode)


def _program_identity(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink():
        raise OSError("backend program is a symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 \
            or not os.access(resolved, os.X_OK):
        raise OSError("backend program is absent, empty, non-regular, or non-executable")
    raw = resolved.read_bytes()
    after = resolved.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise OSError("backend program changed while hashing")
    return {"path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _backend_errors() -> list[str]:
    if len(DIRECT_BACKEND_REGISTRY) != 1:
        return ["exactly one source-reviewed direct counter backend must be registered"]
    backend_id, entry = next(iter(DIRECT_BACKEND_REGISTRY.items()))
    expected = {
        "schema", "backend_id", "program", "barrier_abi",
        "direct_fields", "estimated_values_permitted", "process_attributed",
        "wall_and_continuous_coverage", "immutable_raw_captures", "uses_shell",
        "opens_shared_heavy_lease", "entry_sha256",
    }
    errors: list[str] = []
    if not isinstance(entry, dict) or set(entry) != expected \
            or entry.get("schema") != BACKEND_SCHEMA \
            or entry.get("backend_id") != backend_id:
        return ["direct counter backend registry entry schema is invalid"]
    try:
        observed = _program_identity(pathlib.Path(entry.get("program", {}).get("path", "")))
    except (OSError, TypeError) as exc:
        errors.append(f"direct counter backend program cannot be verified: {exc}")
    else:
        if observed != entry.get("program"):
            errors.append("direct counter backend program identity differs")
    if entry.get("barrier_abi") != SCHEMA \
            or entry.get("direct_fields") != list(COUNTER_FIELDS) \
            or entry.get("estimated_values_permitted") is not False \
            or entry.get("process_attributed") is not True \
            or entry.get("wall_and_continuous_coverage") is not True \
            or entry.get("immutable_raw_captures") is not True \
            or entry.get("uses_shell") is not False \
            or entry.get("opens_shared_heavy_lease") is not False:
        errors.append("direct counter backend weakens the exact physical ABI")
    unstamped = copy.deepcopy(entry)
    claimed = unstamped.pop("entry_sha256", None)
    if not isinstance(claimed, str) or HEX64.fullmatch(claimed) is None \
            or claimed != canonical_sha256(unstamped):
        errors.append("direct counter backend registry hash mismatch")
    return errors


def build_status(
    *, env: Mapping[str, str] | None = None, system: str | None = None,
    euid: int | None = None, powermetrics_path: str | None = None,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    operating_system = platform.system() if system is None else system
    effective_uid = os.geteuid() if euid is None else euid
    power_tool = shutil.which("powermetrics") if powermetrics_path is None else powermetrics_path
    blockers: list[str] = []
    if operating_system != "Darwin":
        blockers.append("direct Doctor counters require the reviewed Darwin backend")
    if not _fd_present(environment):
        blockers.append("inherited shared-heavy-lease descriptor is absent")
    if not power_tool:
        blockers.append("powermetrics is unavailable")
    if effective_uid != 0:
        blockers.append("unattended powermetrics privilege is unavailable")
    backend_errors = _backend_errors()
    if backend_errors:
        blockers.append(
            "no reviewed direct process-joule backend is registered; "
            "powermetrics energy-impact/estimated power is not admissible"
        )
        blockers.extend(backend_errors)
    return _stamp({
        "schema": STATUS_SCHEMA,
        "config_sha256": build_config()["config_sha256"],
        "platform": operating_system,
        "powermetrics_path": power_tool,
        "powermetrics_privilege_available": effective_uid == 0,
        "inherited_shared_heavy_lease_present": _fd_present(environment),
        "registered_backend_ids": sorted(DIRECT_BACKEND_REGISTRY),
        "direct_process_energy_joules_available": not backend_errors,
        "direct_cpu_io_rss_required": True,
        "estimated_values_permitted": False,
        "shared_heavy_lease_opened": False,
        "execution_ready": not blockers,
        "blockers": blockers,
    }, "status_sha256")


def build_dry_run() -> dict[str, Any]:
    status = build_status()
    return _stamp({
        "schema": DRY_RUN_SCHEMA,
        "config_sha256": status["config_sha256"],
        "status_sha256": status["status_sha256"],
        "would_collect": False,
        "would_create_ready_barrier": False,
        "would_open_shared_heavy_lease": False,
        "would_spawn_backend": False,
        "would_use_shell": False,
        "blockers": status["blockers"],
    }, "dry_run_sha256")


def validate_backend_result(
    counter: Any, attestation: Any, *, facet: str, role: str, repeat: int,
    run_nonce: str, program_sha256: str, started_at_unix_ns: int,
    ended_at_unix_ns: int, started_at_continuous_ns: int,
    ended_at_continuous_ns: int,
) -> list[str]:
    """Validate a future backend result; never derive or estimate counters."""
    errors: list[str] = []
    if not isinstance(counter, dict):
        errors.append("counter payload is absent")
    else:
        expected_counter = {
            "schema", "facet", "run_nonce", "energy_j", "cpu_time_ns",
            "read_bytes", "write_bytes", "peak_rss_bytes", "sample_count",
            "directly_measured", "estimated", "counter_payload_sha256",
        }
        if set(counter) != expected_counter or counter.get("schema") != COUNTER_SCHEMA:
            errors.append("counter payload fields/schema are incomplete or unexpected")
        if counter.get("facet") != facet or counter.get("run_nonce") != run_nonce:
            errors.append("counter payload is not bound to the exact arm")
        for field in COUNTER_FIELDS:
            value = counter.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not __import__("math").isfinite(float(value)) or value < 0 \
                    or field in {"energy_j", "cpu_time_ns", "peak_rss_bytes"} and value <= 0:
                errors.append(f"counter payload {field} is invalid")
        if not isinstance(counter.get("sample_count"), int) \
                or isinstance(counter.get("sample_count"), bool) \
                or counter.get("sample_count", 0) <= 0:
            errors.append("counter sample count is invalid")
        if counter.get("directly_measured") is not True or counter.get("estimated") is not False:
            errors.append("counter payload is estimated")
        unstamped = copy.deepcopy(counter)
        claimed = unstamped.pop("counter_payload_sha256", None)
        if not isinstance(claimed, str) or HEX64.fullmatch(claimed) is None \
                or claimed != canonical_sha256(unstamped):
            errors.append("counter payload self-hash mismatch")
    if not isinstance(attestation, dict):
        errors.append("counter attestation is absent")
        return errors
    expected_attestation = {
        "schema", "plan_sha256", "contract_sha256", "facet", "phase", "role",
        "repeat", "run_nonce", "collector_authority_sha256",
        "collector_program_sha256", "benchmark_program_sha256", "invocation_sha256",
        "execution_interval", "capture_interval", "directly_measured", "estimated",
        "domains", "counter_payload_sha256", "output_sha256", "scientific_sha256",
        "stdout_sha256", "stderr_sha256", "raw_captures", "attestation_sha256",
    }
    if set(attestation) != expected_attestation or attestation.get("schema") != ATTESTATION_SCHEMA:
        errors.append("counter attestation schema is invalid")
    for field, expected in (
        ("facet", facet), ("role", role), ("repeat", repeat),
        ("run_nonce", run_nonce), ("benchmark_program_sha256", program_sha256),
    ):
        if attestation.get(field) != expected:
            errors.append(f"counter attestation {field} binding differs")
    if attestation.get("directly_measured") is not True \
            or attestation.get("estimated") is not False \
            or attestation.get("domains") != list(COUNTER_FIELDS):
        errors.append("counter attestation does not prove exact direct domains")
    for field in (
        "plan_sha256", "contract_sha256", "collector_authority_sha256",
        "collector_program_sha256", "invocation_sha256", "counter_payload_sha256",
        "output_sha256", "scientific_sha256", "stdout_sha256", "stderr_sha256",
    ):
        if not isinstance(attestation.get(field), str) \
                or HEX64.fullmatch(attestation[field]) is None:
            errors.append(f"counter attestation {field} is invalid")
    if isinstance(counter, dict) and attestation.get("counter_payload_sha256") \
            != counter.get("counter_payload_sha256"):
        errors.append("counter attestation is not bound to its counter payload")
    execution = attestation.get("execution_interval")
    if execution != {
        "started_at_unix_ns": started_at_unix_ns,
        "ended_at_unix_ns": ended_at_unix_ns,
        "started_at_continuous_ns": started_at_continuous_ns,
        "ended_at_continuous_ns": ended_at_continuous_ns,
    }:
        errors.append("counter attestation execution interval differs")
    capture = attestation.get("capture_interval")
    if not isinstance(capture, dict) or not (
        capture.get("started_at_unix_ns", 1 << 80) <= started_at_unix_ns
        < ended_at_unix_ns <= capture.get("ended_at_unix_ns", -1)
        and capture.get("started_at_continuous_ns", 1 << 80) <= started_at_continuous_ns
        < ended_at_continuous_ns <= capture.get("ended_at_continuous_ns", -1)
    ):
        errors.append("counter capture does not cover the exact arm interval")
    raw_captures = attestation.get("raw_captures")
    if not isinstance(raw_captures, list) or not raw_captures:
        errors.append("counter attestation lacks immutable raw captures")
        raw_captures = []
    hashes: list[Any] = []
    for index, artifact in enumerate(raw_captures):
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "bytes"}:
            errors.append(f"counter raw capture {index} identity is invalid")
            continue
        path = artifact.get("path")
        if not isinstance(path, str) or not pathlib.Path(path).is_absolute() \
                or not isinstance(artifact.get("sha256"), str) \
                or HEX64.fullmatch(artifact["sha256"]) is None \
                or not isinstance(artifact.get("bytes"), int) \
                or isinstance(artifact.get("bytes"), bool) or artifact["bytes"] <= 0:
            errors.append(f"counter raw capture {index} identity is invalid")
        else:
            try:
                target = pathlib.Path(path)
                if target.is_symlink():
                    raise OSError("symlink")
                raw = target.read_bytes()
                if len(raw) != artifact["bytes"] \
                        or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                    raise OSError("byte identity mismatch")
            except OSError as exc:
                errors.append(f"counter raw capture {index} cannot be verified: {exc}")
        hashes.append(artifact.get("sha256"))
    if len(set(hashes)) != len(hashes):
        errors.append("counter raw capture hash is reused")
    unstamped_attestation = copy.deepcopy(attestation)
    claimed_attestation = unstamped_attestation.pop("attestation_sha256", None)
    if not isinstance(claimed_attestation, str) or HEX64.fullmatch(claimed_attestation) is None \
            or claimed_attestation != canonical_sha256(unstamped_attestation):
        errors.append("counter attestation self-hash mismatch")
    return errors


def _selftest() -> int:
    status = build_status(env={}, system="Darwin", euid=0, powermetrics_path="/usr/bin/powermetrics")
    assert status["execution_ready"] is False
    assert status["direct_process_energy_joules_available"] is False
    assert build_dry_run()["would_collect"] is False
    print("doctor_v5_physical_counter_barrier.py selftest OK")
    return 0


def _dispatch_backend(args: argparse.Namespace) -> int:
    """Replace this barrier with the sole exact backend; never invoke a shell."""
    errors = _backend_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    _backend_id, entry = next(iter(DIRECT_BACKEND_REGISTRY.items()))
    program = entry["program"]["path"]
    argv = [
        program, "collect",
        "--request", str(args.request),
        "--ready", str(args.ready),
        "--arm-started", str(args.arm_started),
        "--stop", str(args.stop),
        "--counter-output", str(args.counter_output),
        "--counter-attestation", str(args.counter_attestation),
    ]
    lease_fd = int(os.environ[HEAVY_LEASE_FD_ENV])
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
        HEAVY_LEASE_FD_ENV: str(lease_fd),
        "HAWKING_DOCTOR_COUNTER_BARRIER_ADMITTED": "1",
    }
    # The backend inherits only stdio and the already-held heavy lease.  The
    # anchored cwd survives descriptor closure and supplies all relative output
    # authority; arbitrary caller FDs and ambient environment never cross exec.
    try:
        open_descriptors = [int(name) for name in os.listdir("/dev/fd") if name.isdigit()]
    except OSError:
        open_descriptors = []
    for descriptor in open_descriptors:
        if descriptor not in {0, 1, 2, lease_fd}:
            try:
                os.set_inheritable(descriptor, False)
            except OSError:
                pass
    os.execve(program, argv, environment)
    return EX_TEMPFAIL


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("dry-run")
    collect = subparsers.add_parser("collect")
    collect.add_argument("--request", required=True, type=pathlib.Path)
    collect.add_argument("--ready", required=True, type=pathlib.Path)
    collect.add_argument("--arm-started", required=True, type=pathlib.Path)
    collect.add_argument("--stop", required=True, type=pathlib.Path)
    collect.add_argument("--counter-output", required=True, type=pathlib.Path)
    collect.add_argument("--counter-attestation", required=True, type=pathlib.Path)
    subparsers.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(build_status(), indent=2, sort_keys=True))
        return 0
    if args.command == "dry-run":
        print(json.dumps(build_dry_run(), indent=2, sort_keys=True))
        return 0
    if args.command == "selftest":
        return _selftest()
    # Fail before reading the request or creating any path.  This branch is
    # replaced only when a source-reviewed backend is registered and tested.
    status = build_status()
    if not status["execution_ready"]:
        print(json.dumps({
            "ok": False, "exit_code": EX_TEMPFAIL,
            "reason": "direct-counter barrier admission is closed",
            "blockers": status["blockers"],
        }, indent=2, sort_keys=True), file=sys.stderr)
        return EX_TEMPFAIL
    try:
        return _dispatch_backend(args)
    except (OSError, RuntimeError) as exc:
        print(f"direct counter backend dispatch failed: {exc}", file=sys.stderr)
        return EX_TEMPFAIL


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
