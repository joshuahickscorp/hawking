"""Outer-reap one CPU-only source-token L0 router-plan regeneration.

The child is the direct-packed Rust discriminator.  It makes exactly one
strict artifact admission scan, validates the sealed token-1/zero-state
prefix baseline, and produces raw material.  This outer records child
stdout/stderr/exit, seals a distinct source-token route authority only after
all thirty exact payload descriptors validate, and writes its terminal record
last.  There is intentionally no Metal, lease, watcher, server, or retry
path here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_l0_route_plan as authority
from lab.receipts import seal


SCHEMA = "hawking.ascension.qwen80_source_token_l0_route_plan_outer.v1"
EXPECTED_PROBE = "ascension_qwen80_source_token_l0_router_discriminator"
MATERIAL_FILENAME = "source-token-route-material.json"
AUTHORITY_FILENAME = "source-token-route-plan-authority.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"


class SourceTokenRoutePlanOuterError(RuntimeError):
    """The source-token route-plan outer cannot safely start or seal."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise SourceTokenRoutePlanOuterError(f"refusing to overwrite {path}")
    raw = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_evidence_if_present(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    try:
        return authority._file_evidence(path, label)
    except authority.SourceTokenRoutePlanError as exc:
        return {"path": str(path), "present": False, "error": str(exc)}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if not args.capture_dir.is_absolute() or args.capture_dir.exists() or not args.capture_dir.parent.is_dir():
        raise SourceTokenRoutePlanOuterError("--capture-dir must be new, absolute, and have an existing parent")
    try:
        probe = authority._file_evidence(args.probe_bin, "--probe-bin", executable=True)
        if Path(str(probe["path"])).name != EXPECTED_PROBE:
            raise SourceTokenRoutePlanOuterError(f"--probe-bin must be {EXPECTED_PROBE}")
        manifest, manifest_seal = authority._bind_manifest(args.manifest)
        admission, pointer_seal, immutable_admission, immutable_admission_seal = authority._bind_admission(
            args.admission_current, manifest, manifest_seal
        )
        prefix, prefix_seal, _ = authority._bind_prefix(
            args.first_residual_receipt,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission_receipt_seal=immutable_admission_seal,
        )
        old_plan = authority._bind_old_fixture_plan(args.old_route_plan)
    except authority.SourceTokenRoutePlanError as exc:
        raise SourceTokenRoutePlanOuterError(str(exc)) from exc
    return {
        "probe_binary": probe,
        "manifest": manifest,
        "manifest_seal_sha256": manifest_seal,
        "admission_current": admission,
        "admission_pointer_seal_sha256": pointer_seal,
        "admission_receipt": immutable_admission,
        "admission_receipt_seal_sha256": immutable_admission_seal,
        "first_residual_outer_receipt": prefix,
        "first_residual_outer_seal_sha256": prefix_seal,
        "historical_fixture_route_plan": old_plan,
    }


def _command(args: argparse.Namespace, material: Path) -> list[str]:
    return [
        str(args.probe_bin.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--admission-current",
        str(args.admission_current.resolve()),
        "--first-residual-receipt",
        str(args.first_residual_receipt.resolve()),
        "--old-route-plan",
        str(args.old_route_plan.resolve()),
        "--out",
        str(material),
    ]


def _kill_group(child: subprocess.Popen[bytes]) -> int | None:
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait(timeout=10)


def run_attempt(args: argparse.Namespace) -> dict[str, Any]:
    context = _preflight(args)
    identity = _sha256(context)
    args.capture_dir.mkdir(mode=0o750)
    inner = args.capture_dir / "inner"
    inner.mkdir(mode=0o750)
    material = inner / MATERIAL_FILENAME
    sealed_authority = inner / AUTHORITY_FILENAME
    command = _command(args, material)
    _atomic_new(
        args.capture_dir / "active.json",
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_ONE_SHOT",
                "recorded_at": _utc_now(),
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {
                    "cpu_artifact_scan_only": True,
                    "metal_or_gpu_allowed": False,
                    "automatic_retry_disabled": True,
                },
            }
        ),
    )
    started = _utc_now()
    child_pid: int | None = None
    returncode: int | None = None
    timed_out = False
    spawn_error: str | None = None
    with (args.capture_dir / "outer.stdout.log").open("xb") as stdout, (
        args.capture_dir / "outer.stderr.log"
    ).open("xb") as stderr:
        try:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            spawn_error = f"{type(exc).__name__}: {exc}"
        else:
            child_pid = child.pid
            _atomic_new(
                args.capture_dir / "child.json",
                seal(
                    {
                        "schema": SCHEMA,
                        "status": "RUNNING_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_ONE_SHOT",
                        "recorded_at": _utc_now(),
                        "launch_identity_sha256": identity,
                        "pid": child_pid,
                        "parent_pid": os.getpid(),
                        "command": command,
                        "metal_or_gpu_allowed": False,
                    }
                ),
            )
            try:
                returncode = child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = _kill_group(child)

    seal_error: str | None = None
    authority_evidence: dict[str, Any] = {"path": str(sealed_authority), "present": False}
    if returncode == 0 and not timed_out and spawn_error is None:
        try:
            document = authority.build_authority(
                manifest_path=args.manifest,
                admission_path=args.admission_current,
                first_residual_path=args.first_residual_receipt,
                old_plan_path=args.old_route_plan,
                material_path=material,
            )
            authority.write_new(sealed_authority, document)
            authority_evidence = authority._file_evidence(sealed_authority, "sealed source-token route authority")
            authority_evidence.update(
                {"schema": document["schema"], "status": document["status"], "seal_sha256": document["seal_sha256"]}
            )
        except authority.SourceTokenRoutePlanError as exc:
            seal_error = str(exc)

    if spawn_error is not None:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_CHILD_SPAWN_ERROR"
    elif timed_out:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_CHILD_TIMEOUT"
    elif returncode != 0:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_CHILD_NONZERO"
    elif seal_error is not None or authority_evidence.get("present") is not True:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_WITHOUT_SEALED_AUTHORITY"
    else:
        status = "CAPTURED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_SEALED_FOR_NEW_TYPED_BRIDGE"
    terminal = seal(
        {
            "schema": SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "launch_identity_sha256": identity,
            "one_shot": {
                "automatic_retry_disabled": True,
                "terminal_receipt_written_last": True,
                "same_capture_dir_never_starts_a_second_child": True,
            },
            "source_binding": context,
            "child": {
                "pid": child_pid,
                "started_at": started,
                "finished_at": _utc_now(),
                "command": command,
                "terminal": {
                    "reaped": returncode is not None,
                    "returncode": returncode,
                    "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
                    "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
                    "timed_out": timed_out,
                    "spawn_error": spawn_error,
                },
            },
            "outer_capture": {
                "directory": str(args.capture_dir),
                "stdout": _file_evidence_if_present(args.capture_dir / "outer.stdout.log", "outer stdout"),
                "stderr": _file_evidence_if_present(args.capture_dir / "outer.stderr.log", "outer stderr"),
                "raw_material": _file_evidence_if_present(material, "raw source-token material"),
                "sealed_source_token_route_authority": authority_evidence,
            },
            "claim_boundary": {
                "cpu_artifact_scan_and_source_token_route_authority_only": True,
                "metal_device_or_dispatch_performed": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
                "historical_fixture_plan_preserved_as_negative_science": True,
            },
            **({"seal_error": seal_error} if seal_error is not None else {}),
        }
    )
    _atomic_new(args.capture_dir / TERMINAL_FILENAME, terminal)
    return terminal


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--old-route-plan", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        terminal = run_attempt(args)
    except SourceTokenRoutePlanOuterError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_SOURCE_TOKEN_L0_ROUTE_PLAN_CPU_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(terminal, sort_keys=True))
    return 0 if str(terminal["status"]).startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
