"""One-shot, outer-reaped Qwen80 all-ten routed-expert CPU oracle capture.

This launcher owns only immutable input evidence, process lifetime, and a
receipt-last terminal envelope.  The Rust child is the sole authority for the
strict complete-artifact admission scan and direct-packed route bodies.  The
capture deliberately stops before shared-expert, routed/shared combine,
residual, a complete layer, a token, or any device work.

It is intentionally a one-shot launcher: a directory with a terminal receipt
is replayed, while a directory without one is refused rather than reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.ascension.qwen80_all_ten_routed_expert_cpu_outer_launcher.v1"
INNER_SCHEMA = "hawking.ascension.qwen80_all_ten_routed_expert_cpu_oracle.v1"
INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_ORACLE_"
    "READY_FOR_SEPARATE_DEVICE_LEASE"
)
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"


class AllTenCpuLauncherError(RuntimeError):
    """The one-shot CPU-only capture cannot safely proceed."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    expected_manifest_seal_sha256: str
    expected_source_audit_seal_sha256: str
    expected_source_revision: str
    route_plan: Path
    router_inner: Path
    router_outer: Path
    capture_dir: Path
    timeout_seconds: float


@dataclass(frozen=True)
class LaunchContext:
    probe_binary: dict[str, Any]
    manifest: dict[str, Any]
    route_plan: dict[str, Any]
    router_inner: dict[str, Any]
    router_outer: dict[str, Any]
    router_outer_seal_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64:
        raise AllTenCpuLauncherError(f"{label} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AllTenCpuLauncherError(f"{label} must be hexadecimal") from exc
    if value != value.lower():
        raise AllTenCpuLauncherError(f"{label} must be lowercase")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise AllTenCpuLauncherError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise AllTenCpuLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AllTenCpuLauncherError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise AllTenCpuLauncherError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise AllTenCpuLauncherError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    canonical = _canonical_regular(path, label, executable=executable)
    return {
        "path": str(canonical),
        "present": True,
        "bytes": canonical.stat().st_size,
        "sha256": _file_sha256(canonical),
    }


def _read_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    clean = _canonical_regular(path, label)
    try:
        raw = clean.read_bytes()
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AllTenCpuLauncherError(f"cannot read JSON {label} at {clean}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise AllTenCpuLauncherError(f"{label} is not a JSON object")
    return raw, dict(parsed)


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise AllTenCpuLauncherError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise AllTenCpuLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_config(config: LaunchConfig) -> LaunchContext:
    if config.timeout_seconds <= 0:
        raise AllTenCpuLauncherError("timeout must be positive")
    if config.capture_dir.exists():
        raise AllTenCpuLauncherError("capture directory must not exist before a new attempt")
    if not config.capture_dir.is_absolute() or not config.capture_dir.parent.is_dir():
        raise AllTenCpuLauncherError("capture directory must be absolute with an existing parent")
    for label, value in (
        ("expected manifest seal", config.expected_manifest_seal_sha256),
        ("expected source audit seal", config.expected_source_audit_seal_sha256),
    ):
        _require_sha256(value, label)
    if not config.expected_source_revision:
        raise AllTenCpuLauncherError("expected source revision must not be empty")
    probe_binary = _file_evidence(config.probe_bin, "probe binary", executable=True)
    manifest = _file_evidence(config.manifest, "manifest")
    route_plan = _file_evidence(config.route_plan, "all-ten route plan")
    router_inner = _file_evidence(config.router_inner, "router inner receipt")
    router_outer = _file_evidence(config.router_outer, "router outer receipt")
    _, outer = _read_json(config.router_outer, "router outer receipt")
    try:
        verify(outer, label="router outer receipt")
    except ValueError as exc:
        raise AllTenCpuLauncherError(f"router outer receipt is not sealed: {exc}") from exc
    if outer.get("schema") != (
        "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1"
    ) or outer.get("status") != (
        "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"
    ):
        raise AllTenCpuLauncherError("router outer receipt schema/status is not the strict component")
    outer_inner = outer.get("inner_probe_capture")
    if not isinstance(outer_inner, Mapping):
        raise AllTenCpuLauncherError("router outer receipt lacks inner component evidence")
    if outer_inner.get("path") != router_inner["path"] or outer_inner.get("sha256") != router_inner["sha256"]:
        raise AllTenCpuLauncherError("router outer receipt does not bind supplied router inner receipt")
    outer_seal = outer.get("seal_sha256")
    if not isinstance(outer_seal, str):
        raise AllTenCpuLauncherError("router outer receipt has no seal")
    _require_sha256(outer_seal, "router outer seal")
    return LaunchContext(
        probe_binary=probe_binary,
        manifest=manifest,
        route_plan=route_plan,
        router_inner=router_inner,
        router_outer=router_outer,
        router_outer_seal_sha256=outer_seal,
    )


def _identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "schema": SCHEMA,
        "mode": "cpu-oracle",
        "probe_binary": context.probe_binary,
        "manifest": context.manifest,
        "expected_manifest_seal_sha256": config.expected_manifest_seal_sha256,
        "expected_source_audit_seal_sha256": config.expected_source_audit_seal_sha256,
        "expected_source_revision": config.expected_source_revision,
        "route_plan": context.route_plan,
        "router_inner": context.router_inner,
        "router_outer": context.router_outer,
        "router_outer_seal_sha256": context.router_outer_seal_sha256,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _child_command(config: LaunchConfig, inner_capture: Path) -> list[str]:
    return [
        str(_canonical_regular(config.probe_bin, "probe binary", executable=True)),
        "--manifest",
        str(_canonical_regular(config.manifest, "manifest")),
        "--expected-manifest-seal-sha256",
        config.expected_manifest_seal_sha256,
        "--expected-source-audit-seal-sha256",
        config.expected_source_audit_seal_sha256,
        "--expected-source-revision",
        config.expected_source_revision,
        "--route-plan",
        str(_canonical_regular(config.route_plan, "all-ten route plan")),
        "--router-inner",
        str(_canonical_regular(config.router_inner, "router inner receipt")),
        "--router-outer",
        str(_canonical_regular(config.router_outer, "router outer receipt")),
        "--capture-dir",
        str(inner_capture),
    ]


def _terminate_group(child: subprocess.Popen[bytes]) -> int | None:
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


def _inner_evidence(capture_dir: Path) -> dict[str, Any]:
    receipt = capture_dir / INNER_CAPTURE / "receipt.json"
    evidence: dict[str, Any] = {"path": str(receipt), "present": receipt.is_file()}
    if not receipt.is_file():
        return evidence
    try:
        raw, parsed = _read_json(receipt, "inner receipt")
    except AllTenCpuLauncherError as exc:
        evidence["binding_error"] = str(exc)
        return evidence
    evidence.update({"sha256": _sha256_bytes(raw), "schema": parsed.get("schema"), "status": parsed.get("status"), "mode": parsed.get("mode")})
    boundary = parsed.get("claim_boundary")
    if (
        parsed.get("schema") == INNER_SCHEMA
        and parsed.get("status") == INNER_STATUS
        and parsed.get("mode") == "cpu-oracle"
        and isinstance(boundary, Mapping)
        and boundary.get("no_metal_context_or_gpu_dispatch_performed") is True
        and boundary.get("no_shared_expert_route_combine_or_residual_performed") is True
        and boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim") is True
    ):
        evidence["binding_valid"] = True
    else:
        evidence["binding_valid"] = False
    return evidence


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
    return "CAPTURED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_TERMINAL_PRE_SHARED_PRE_COMBINE_PRE_RESIDUAL"


def _terminal_record(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    finished_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None,
) -> dict[str, Any]:
    inner = _inner_evidence(config.capture_dir)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": finished_at,
        "mode": "cpu-oracle",
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": context.probe_binary,
            "manifest": context.manifest,
            "expected_manifest_seal_sha256": config.expected_manifest_seal_sha256,
            "expected_source_audit_seal_sha256": config.expected_source_audit_seal_sha256,
            "expected_source_revision": config.expected_source_revision,
            "route_plan": context.route_plan,
            "router_inner": context.router_inner,
            "router_outer": context.router_outer,
            "router_outer_seal_sha256": context.router_outer_seal_sha256,
        },
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": list(command),
            "terminal": dict(terminal),
        },
        "outer_capture": {
            "directory": str(config.capture_dir),
            "stdout": _file_evidence(config.capture_dir / OUTER_STDOUT, "outer stdout"),
            "stderr": _file_evidence(config.capture_dir / OUTER_STDERR, "outer stderr"),
        },
        "inner_probe_capture": inner,
        "claim_boundary": {
            "cpu_oracle_only": True,
            "no_metal_gpu_watcher_or_server_action": True,
            "pre_shared_expert_pre_combine_pre_residual": True,
            "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_result": True,
        },
    }
    if capture_error is not None:
        payload["capture_error"] = capture_error
    return seal(payload)


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    context = _validate_config(config)
    identity = _identity(config, context)
    config.capture_dir.mkdir(mode=0o750)
    command = _child_command(config, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {"cpu_oracle_only": True, "automatic_retry_disabled": True},
            }
        ),
    )
    child_pid: int | None = None
    capture_error: str | None = None
    terminal: dict[str, Any]
    with (config.capture_dir / OUTER_STDOUT).open("xb") as stdout, (
        config.capture_dir / OUTER_STDERR
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
            terminal = {
                "reaped": False,
                "timed_out": False,
                "returncode": None,
                "exit_code": None,
                "signal": None,
                "spawn_error": f"{type(exc).__name__}: {exc}",
            }
        else:
            child_pid = child.pid
            try:
                _atomic_json_new(
                    config.capture_dir / CHILD_FILENAME,
                    seal(
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                            "mode": "cpu-oracle",
                            "metal_or_gpu_allowed": False,
                        }
                    ),
                )
            except AllTenCpuLauncherError as exc:
                capture_error = str(exc)
                code = _terminate_group(child)
                terminal = {
                    "reaped": code is not None,
                    "timed_out": False,
                    "returncode": code,
                    "exit_code": code if isinstance(code, int) and code >= 0 else None,
                    "signal": -code if isinstance(code, int) and code < 0 else None,
                }
            else:
                try:
                    code = child.wait(timeout=config.timeout_seconds)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    code = _terminate_group(child)
                    timed_out = True
                terminal = {
                    "reaped": code is not None,
                    "timed_out": timed_out,
                    "returncode": code,
                    "exit_code": code if isinstance(code, int) and code >= 0 else None,
                    "signal": -code if isinstance(code, int) and code < 0 else None,
                }
    receipt = _terminal_record(
        config,
        context,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        finished_at=_utc_now(),
        terminal=terminal,
        capture_error=capture_error,
    )
    _atomic_json_new(config.capture_dir / TERMINAL_FILENAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-seal-sha256", required=True)
    parser.add_argument("--expected-source-audit-seal-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--router-inner", type=Path, required=True)
    parser.add_argument("--router-outer", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        expected_manifest_seal_sha256=parsed.expected_manifest_seal_sha256,
        expected_source_audit_seal_sha256=parsed.expected_source_audit_seal_sha256,
        expected_source_revision=parsed.expected_source_revision,
        route_plan=parsed.route_plan,
        router_inner=parsed.router_inner,
        router_outer=parsed.router_outer,
        capture_dir=parsed.capture_dir,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except AllTenCpuLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
