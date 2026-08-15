"""Outer-reaped CPU-only Qwen80 all-ten source-bridge material scan.

The child performs exactly one strict compact-artifact admission scan and
writes raw bridge material.  This outer process captures stdout/stderr/exit,
seals the typed bridge only after the child succeeds, and writes its terminal
receipt last.  It intentionally has no Metal, watcher, server, or benchmark
path.
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

from lab.operators import ascension_qwen80_all_ten_true_moe_source_bridge as bridge
from lab.operators import ascension_qwen80_true_input_all_ten_moe_graph_launcher as authority
from lab.receipts import seal


SCHEMA = "hawking.ascension.qwen80_all_ten_true_moe_source_bridge_outer.v1"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
MATERIAL_FILENAME = "bridge-material.json"
BRIDGE_FILENAME = "typed-bridge.json"
EXPECTED_BINARY = "ascension_qwen80_all_ten_true_moe_source_bridge"


class SourceBridgeOuterError(RuntimeError):
    """The CPU-only source-bridge outer cannot safely start."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_new(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise SourceBridgeOuterError(f"refusing to overwrite {path}")
    raw = json.dumps(dict(document), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.capture_dir.exists() or not args.capture_dir.is_absolute() or not args.capture_dir.parent.is_dir():
        raise SourceBridgeOuterError("--capture-dir must be a new absolute directory with an existing parent")
    try:
        probe = authority._file_evidence(args.probe_bin, "--probe-bin", executable=True)
        if Path(str(probe["path"])).name != EXPECTED_BINARY:
            raise SourceBridgeOuterError(f"--probe-bin must be {EXPECTED_BINARY}")
        manifest, manifest_seal = authority._bind_manifest(args.manifest)
        admission, pointer_seal, admission_receipt_seal = authority._bind_admission(
            args.admission_current, manifest, manifest_seal
        )
        router, router_outer, router_outer_seal = authority._bind_router(
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission=admission,
            admission_receipt_seal=admission_receipt_seal,
            router_path=args.router_receipt,
            router_outer_path=args.router_outer_receipt,
        )
        route_plan = authority._file_evidence(args.route_plan, "--route-plan")
        route_ids, route_weights = authority._route_ids_and_weights(
            authority._read_json(args.route_plan, "--route-plan")
        )
        first_residual, first_residual_seal, output_sha = authority._bind_first_residual(
            args.first_residual_receipt,
            manifest=manifest,
            manifest_seal=manifest_seal,
            admission=admission,
            admission_receipt_seal=admission_receipt_seal,
        )
    except authority.TrueInputAllTenMoeGraphLauncherError as exc:
        raise SourceBridgeOuterError(str(exc)) from exc
    return {
        "probe_binary": probe,
        "manifest": manifest,
        "manifest_seal_sha256": manifest_seal,
        "admission_current": admission,
        "admission_pointer_seal_sha256": pointer_seal,
        "admission_receipt_seal_sha256": admission_receipt_seal,
        "router_receipt": router,
        "router_outer_receipt": router_outer,
        "router_outer_receipt_seal_sha256": router_outer_seal,
        "route_plan": route_plan,
        "route_ids": list(route_ids),
        "route_weights": list(route_weights),
        "first_residual_receipt": first_residual,
        "first_residual_receipt_seal_sha256": first_residual_seal,
        "first_residual_output_sha256": output_sha,
    }


def _command(args: argparse.Namespace, material: Path) -> list[str]:
    return [
        str(args.probe_bin.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--admission-current",
        str(args.admission_current.resolve()),
        "--router-receipt",
        str(args.router_receipt.resolve()),
        "--router-outer-receipt",
        str(args.router_outer_receipt.resolve()),
        "--route-plan",
        str(args.route_plan.resolve()),
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


def _file_evidence_if_present(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    try:
        return authority._file_evidence(path, label)
    except authority.TrueInputAllTenMoeGraphLauncherError as exc:
        return {"path": str(path), "present": False, "error": str(exc)}


def run_attempt(args: argparse.Namespace) -> dict[str, Any]:
    context = _preflight(args)
    identity = _sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    args.capture_dir.mkdir(mode=0o750)
    inner = args.capture_dir / "inner"
    inner.mkdir(mode=0o750)
    material = inner / MATERIAL_FILENAME
    bridge_path = inner / BRIDGE_FILENAME
    command = _command(args, material)
    _atomic_new(
        args.capture_dir / "active.json",
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_ONE_SHOT",
                "recorded_at": _utc_now(),
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {
                    "cpu_artifact_scan_only": True,
                    "automatic_retry_disabled": True,
                    "metal_or_gpu_allowed": False,
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
                        "status": "RUNNING_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_ONE_SHOT",
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

    bridge_error: str | None = None
    typed_bridge: dict[str, Any] = {"path": str(bridge_path), "present": False}
    if returncode == 0 and not timed_out and spawn_error is None:
        try:
            typed = bridge.build_receipt(
                manifest_path=args.manifest,
                admission_path=args.admission_current,
                router_path=args.router_receipt,
                router_outer_path=args.router_outer_receipt,
                route_plan_path=args.route_plan,
                first_residual_path=args.first_residual_receipt,
                material_path=material,
            )
            bridge._write_new(bridge_path, typed)
            typed_bridge = authority._file_evidence(bridge_path, "typed bridge")
            typed_bridge.update({"schema": typed["schema"], "status": typed["status"], "seal_sha256": typed["seal_sha256"]})
        except (bridge.SourceBridgeSealError, authority.TrueInputAllTenMoeGraphLauncherError) as exc:
            bridge_error = str(exc)

    if spawn_error is not None:
        status = "REFUSED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_CHILD_SPAWN_ERROR"
    elif timed_out:
        status = "REFUSED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_CHILD_TIMEOUT"
    elif returncode != 0:
        status = "REFUSED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_CHILD_NONZERO"
    elif bridge_error is not None or typed_bridge.get("present") is not True:
        status = "REFUSED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_WITHOUT_SEALED_TYPED_BRIDGE"
    else:
        status = "CAPTURED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_SEALED_FOR_FUTURE_DEVICE_LEASE"
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
                "raw_material": _file_evidence_if_present(material, "raw material"),
                "typed_bridge": typed_bridge,
            },
            "claim_boundary": {
                "cpu_artifact_scan_and_source_bridge_authority_only": True,
                "metal_device_or_dispatch_performed": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            },
            **({"bridge_error": bridge_error} if bridge_error is not None else {}),
        }
    )
    _atomic_new(args.capture_dir / TERMINAL_FILENAME, terminal)
    return terminal


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--router-receipt", type=Path, required=True)
    parser.add_argument("--router-outer-receipt", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        receipt = run_attempt(args)
    except SourceBridgeOuterError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_ALL_TEN_SOURCE_BRIDGE_CPU_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
