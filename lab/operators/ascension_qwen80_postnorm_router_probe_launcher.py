"""One-shot outer capture for Qwen80 postnorm→router→top-10 device parity.

This is intentionally a small launcher rather than a watcher or retry loop.
It starts one explicitly built component probe, durably captures its terminal
streams and exit status, reaps the child, and seals an outer receipt whether
the inner probe earns component parity or refuses.  It cannot promote a
component result to a Qwen80 layer, token, decoder, HCLI, TPS, TG, or
tournament result.
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"
EXPECTED_PROBE_BASENAME = "ascension_qwen80_direct_packed_postnorm_router_top10"
EXPECTED_METAL_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)
EXPECTED_CPU_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_"
    "CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
)


class PostnormRouterLauncherError(RuntimeError):
    """The terminal capture cannot safely continue."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    admission_current: Path
    capture_dir: Path
    mode: str
    workers: int
    timeout_seconds: float
    lease_receipt: Path | None = None


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


def _file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable new receipt without replacing old terminal evidence."""

    if path.exists():
        raise PostnormRouterLauncherError(f"refusing to overwrite {path}")
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
        raise PostnormRouterLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostnormRouterLauncherError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PostnormRouterLauncherError(f"JSON document {path} is not an object")
    return dict(payload)


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise PostnormRouterLauncherError(f"{label} must be absolute: {path}")


def _require_regular(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise PostnormRouterLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or (executable and not os.access(path, os.X_OK)):
        raise PostnormRouterLauncherError(
            f"{label} must be a {'executable ' if executable else ''}regular file: {path}"
        )


def _validate_config(config: LaunchConfig) -> None:
    for path, label, executable in (
        (config.probe_bin, "--probe-bin", True),
        (config.manifest, "--manifest", False),
        (config.admission_current, "--admission-current", False),
    ):
        _require_absolute(path, label)
        _require_regular(path, label, executable=executable)
    _require_absolute(config.capture_dir, "--capture-dir")
    if config.probe_bin.name != EXPECTED_PROBE_BASENAME:
        raise PostnormRouterLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {config.probe_bin.name!r}"
        )
    if config.mode not in {"cpu-oracle", "metal"}:
        raise PostnormRouterLauncherError(f"unsupported --mode {config.mode!r}")
    if config.workers < 1:
        raise PostnormRouterLauncherError("--workers must be positive")
    if not config.timeout_seconds > 0:
        raise PostnormRouterLauncherError("--timeout-seconds must be positive")
    if config.mode == "metal":
        if config.lease_receipt is None:
            raise PostnormRouterLauncherError("--mode metal requires --lease-receipt")
        _require_absolute(config.lease_receipt, "--lease-receipt")
        _require_regular(config.lease_receipt, "--lease-receipt")
    elif config.lease_receipt is not None:
        raise PostnormRouterLauncherError("--lease-receipt is valid only with --mode metal")


def _launch_identity(config: LaunchConfig) -> str:
    evidence: dict[str, Any] = {
        "probe_bin": str(config.probe_bin),
        "probe_binary_sha256": _file_sha256(config.probe_bin),
        "manifest": _file_evidence(config.manifest),
        "admission_current": _file_evidence(config.admission_current),
        "mode": config.mode,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
    }
    if config.lease_receipt is not None:
        evidence["lease_receipt"] = _file_evidence(config.lease_receipt)
    return _sha256_bytes(
        json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _child_command(config: LaunchConfig, inner_capture: Path) -> list[str]:
    command = [
        str(config.probe_bin),
        "--manifest",
        str(config.manifest),
        "--admission-current",
        str(config.admission_current),
        "--capture-dir",
        str(inner_capture),
        "--mode",
        config.mode,
        "--workers",
        str(config.workers),
    ]
    if config.lease_receipt is not None:
        command.extend(("--lease-receipt", str(config.lease_receipt)))
    return command


def _inner_evidence(inner_capture: Path) -> dict[str, Any]:
    receipt = inner_capture / "receipt.json"
    evidence = _file_evidence(receipt)
    evidence["capture_dir"] = str(inner_capture)
    if not evidence["present"]:
        evidence["invocation"] = _file_evidence(inner_capture / "invocation.json")
        evidence["stdout"] = _file_evidence(inner_capture / "stdout.jsonl")
        evidence["stderr"] = _file_evidence(inner_capture / "stderr.log")
        return evidence
    try:
        document = _read_json(receipt)
    except PostnormRouterLauncherError as exc:
        evidence["parse_error"] = str(exc)
        return evidence
    evidence["schema"] = document.get("schema")
    evidence["status"] = document.get("status")
    evidence["mode"] = document.get("mode")
    evidence["metal_performed"] = document.get("metal_device_or_dispatch_performed")
    return evidence


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        result["spawn_error"] = spawn_error
        result["reaped"] = False
    return result


def _terminal_status(config: LaunchConfig, terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_CHILD_NONZERO"
    expected = EXPECTED_METAL_STATUS if config.mode == "metal" else EXPECTED_CPU_STATUS
    if inner.get("present") is not True or inner.get("status") != expected:
        return "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_ZERO_EXIT_WITHOUT_EXPECTED_INNER_RECEIPT"
    return "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") == "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_receipt(
    config: LaunchConfig,
    *,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    finished_at: str,
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    inner = _inner_evidence(config.capture_dir / INNER_CAPTURE)
    receipt = {
        "schema": SCHEMA,
        "status": _terminal_status(config, terminal, inner),
        "recorded_at": finished_at,
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": _file_evidence(config.probe_bin),
            "manifest": _file_evidence(config.manifest),
            "admission_current": _file_evidence(config.admission_current),
            "lease_receipt": _file_evidence(config.lease_receipt)
            if config.lease_receipt is not None
            else None,
            "mode": config.mode,
            "workers": config.workers,
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
            "stdout": _sync_evidence(config.capture_dir / OUTER_STDOUT),
            "stderr": _sync_evidence(config.capture_dir / OUTER_STDERR),
        },
        "inner_probe_capture": inner,
        "claim_boundary": {
            "outer_terminal_capture_only": True,
            "does_not_validate_or_promote_inner_component_parity": True,
            "does_not_execute_a_complete_layer_or_decoder": True,
            "does_not_generate_tokens_expose_hcli_or_measure_tps": True,
            "does_not_claim_tg10_tg3_or_tournament_qualification": True,
        },
    }
    return seal(receipt)


def _replay_existing(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise PostnormRouterLauncherError(
            f"capture directory exists without a terminal receipt: {config.capture_dir}"
        )
    receipt = _read_json(terminal_path)
    try:
        verify(receipt, label=str(terminal_path))
    except ValueError as exc:
        raise PostnormRouterLauncherError(f"outer terminal receipt is not sealed: {exc}") from exc
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise PostnormRouterLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run exactly one child, or replay its sealed terminal record."""

    _validate_config(config)
    identity = _launch_identity(config)
    if config.capture_dir.exists():
        return _replay_existing(config, identity)
    if not config.capture_dir.parent.is_dir():
        raise PostnormRouterLauncherError(
            f"capture parent does not exist: {config.capture_dir.parent}"
        )
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay_existing(config, identity)
    command = _child_command(config, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {"automatic_retry_disabled": True, "component_only": True},
            }
        ),
    )
    child_pid: int | None = None
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
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            _atomic_json_new(
                config.capture_dir / "child.json",
                seal(
                    {
                        "schema": SCHEMA,
                        "status": "RUNNING_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_ONE_SHOT",
                        "recorded_at": _utc_now(),
                        "launch_identity_sha256": identity,
                        "pid": child_pid,
                        "command": command,
                        "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                    }
                ),
            )
            try:
                returncode = child.wait(timeout=config.timeout_seconds)
                terminal = _terminal(returncode, timed_out=False)
            except subprocess.TimeoutExpired:
                terminal = _terminal(_terminate_process_group(child), timed_out=True)
    receipt = _terminal_receipt(
        config,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        finished_at=_utc_now(),
        terminal=terminal,
    )
    _atomic_json_new(config.capture_dir / TERMINAL_FILENAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-oracle", "metal"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lease-receipt", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        admission_current=parsed.admission_current,
        capture_dir=parsed.capture_dir,
        mode=parsed.mode,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
        lease_receipt=parsed.lease_receipt,
    )
    try:
        receipt = run_attempt(config)
    except PostnormRouterLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
