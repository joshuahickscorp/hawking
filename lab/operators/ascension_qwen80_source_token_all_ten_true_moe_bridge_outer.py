"""Outer-reap one CPU-only Qwen80 source-token all-ten bridge build.

The child performs one strict admitted compact-artifact scan.  The outer
records its terminal evidence, seals a typed bridge only after exact current
source-token authority validation, and writes its own terminal receipt last.
It has no device, lease, watcher, or retry path.
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

from lab.operators import ascension_qwen80_source_token_all_ten_true_moe_bridge as bridge
from lab.operators import ascension_qwen80_source_token_l0_route_plan as route_plan
from lab.receipts import seal


SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_bridge_outer.v1"
EXPECTED_PROBE = "ascension_qwen80_source_token_all_ten_true_moe_bridge"
MATERIAL_FILENAME = "source-token-bridge-material.json"
BRIDGE_FILENAME = "typed-source-token-bridge.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"


class SourceTokenBridgeOuterError(RuntimeError):
    """The source-token bridge outer cannot safely launch or seal."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise SourceTokenBridgeOuterError(f"refusing to overwrite {path}")
    raw = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_evidence_if_present(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    try:
        return route_plan._file_evidence(path, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        return {"path": str(path), "present": False, "error": str(exc)}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if not args.capture_dir.is_absolute() or args.capture_dir.exists() or not args.capture_dir.parent.is_dir():
        raise SourceTokenBridgeOuterError("--capture-dir must be new, absolute, and have an existing parent")
    try:
        probe = route_plan._file_evidence(args.probe_bin, "--probe-bin", executable=True)
        if Path(str(probe["path"])).name != EXPECTED_PROBE:
            raise SourceTokenBridgeOuterError(f"--probe-bin must be {EXPECTED_PROBE}")
        manifest, manifest_seal = route_plan._bind_manifest(args.manifest)
        admission, pointer_seal, immutable_admission, immutable_admission_seal = route_plan._bind_admission(
            args.admission_current, manifest, manifest_seal
        )
        prefix, prefix_seal, _ = route_plan._bind_prefix(
            args.first_residual_receipt,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission_receipt_seal=immutable_admission_seal,
        )
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenBridgeOuterError(str(exc)) from exc
    try:
        source_authority, source_authority_seal, _ = bridge._bind_source_authority(
            args.source_token_route_authority,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission=admission,
            admission_receipt=immutable_admission,
            admission_receipt_seal=immutable_admission_seal,
            prefix=prefix,
            prefix_seal=prefix_seal,
        )
    except bridge.SourceTokenBridgeSealError as exc:
        raise SourceTokenBridgeOuterError(str(exc)) from exc
    return {
        "probe_binary": probe,
        "manifest": manifest,
        "manifest_seal_sha256": manifest_seal,
        "admission_current": admission,
        "admission_pointer_seal_sha256": pointer_seal,
        "admission_receipt": immutable_admission,
        "admission_receipt_seal_sha256": immutable_admission_seal,
        "source_token_route_authority": source_authority,
        "source_token_route_authority_seal_sha256": source_authority_seal,
        "first_residual_receipt": prefix,
        "first_residual_receipt_seal_sha256": prefix_seal,
    }


def _command(args: argparse.Namespace, material: Path) -> list[str]:
    return [
        str(args.probe_bin.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--admission-current",
        str(args.admission_current.resolve()),
        "--source-token-route-authority",
        str(args.source_token_route_authority.resolve()),
        "--first-residual-receipt",
        str(args.first_residual_receipt.resolve()),
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
    identity = _identity(context)
    args.capture_dir.mkdir(mode=0o750)
    inner = args.capture_dir / "inner"
    inner.mkdir(mode=0o750)
    material = inner / MATERIAL_FILENAME
    typed_bridge = inner / BRIDGE_FILENAME
    command = _command(args, material)
    _atomic_new(
        args.capture_dir / "active.json",
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_ONE_SHOT",
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
                        "status": "RUNNING_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_ONE_SHOT",
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
    bridge_evidence: dict[str, Any] = {"path": str(typed_bridge), "present": False}
    if returncode == 0 and not timed_out and spawn_error is None:
        try:
            document = bridge.build_receipt(
                manifest_path=args.manifest,
                admission_path=args.admission_current,
                source_authority_path=args.source_token_route_authority,
                first_residual_path=args.first_residual_receipt,
                material_path=material,
            )
            bridge.write_new(typed_bridge, document)
            bridge_evidence = route_plan._file_evidence(typed_bridge, "typed source-token bridge")
            bridge_evidence.update(
                {"schema": document["schema"], "status": document["status"], "seal_sha256": document["seal_sha256"]}
            )
        except bridge.SourceTokenBridgeSealError as exc:
            seal_error = str(exc)

    if spawn_error is not None:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_CHILD_SPAWN_ERROR"
    elif timed_out:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_CHILD_TIMEOUT"
    elif returncode != 0:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_CHILD_NONZERO"
    elif seal_error is not None or bridge_evidence.get("present") is not True:
        status = "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_WITHOUT_SEALED_TYPED_BRIDGE"
    else:
        status = "CAPTURED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_SEALED_FOR_OUTER_PREFLIGHT"
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
                "raw_material": _file_evidence_if_present(material, "raw source-token bridge material"),
                "typed_source_token_bridge": bridge_evidence,
            },
            "claim_boundary": {
                "cpu_artifact_scan_and_source_token_bridge_only": True,
                "metal_device_or_dispatch_performed": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
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
    parser.add_argument("--source-token-route-authority", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        terminal = run_attempt(args)
    except SourceTokenBridgeOuterError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_BRIDGE_CPU_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(terminal, sort_keys=True))
    return 0 if str(terminal["status"]).startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
