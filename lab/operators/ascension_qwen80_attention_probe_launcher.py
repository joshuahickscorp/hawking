"""One-shot outer terminal capture for the Qwen80 layer-3 GQA component probe.

This is deliberately a bounded launcher, not a watcher, optimizer, admission
authority, or retry controller.  It launches exactly one already-built probe
under a fresh capture directory, preserves child PID/command/terminal streams,
reaps the child, and seals an outer terminal receipt even if the child exits
before its own inner receipt can be written.

The receipt records terminal evidence only.  It never upgrades a component
probe into a layer, decoder, generation, HCLI, TPS, TG, or tournament claim.
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
SCHEMA = "hawking.ascension.qwen80_direct_packed_layer3_gqa_outer_launcher.v1"
TERMINAL_RECEIPT = "outer-terminal-receipt.json"
ACTIVE_RECEIPT = "active.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"
EXPECTED_PROBE_BASENAME = "ascension_qwen80_direct_packed_attention_stage"


class AttentionProbeLauncherError(RuntimeError):
    """The one-shot component-probe launcher cannot safely continue."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    expected_manifest_seal_sha256: str
    expected_source_audit_seal_sha256: str
    expected_source_revision: str
    capture_dir: Path
    mode: str
    timeout_seconds: float


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
    """Durably create a receipt without replacing prior attempt evidence."""

    if path.exists():
        raise AttentionProbeLauncherError(f"refusing to overwrite receipt {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Link publication is create-new: a concurrent or stale target is a
        # refusal rather than an accidental replacement of terminal evidence.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise AttentionProbeLauncherError(f"refusing to overwrite receipt {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttentionProbeLauncherError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AttentionProbeLauncherError(f"JSON document {path} is not an object")
    return dict(value)


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise AttentionProbeLauncherError(f"{label} must be an absolute path: {path}")


def _validate_config(config: LaunchConfig) -> None:
    for path, label in (
        (config.probe_bin, "--probe-bin"),
        (config.manifest, "--manifest"),
        (config.capture_dir, "--capture-dir"),
    ):
        _require_absolute(path, label)
    if config.probe_bin.name != EXPECTED_PROBE_BASENAME:
        raise AttentionProbeLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {config.probe_bin.name!r}"
        )
    try:
        probe_stat = config.probe_bin.stat()
    except OSError as exc:
        raise AttentionProbeLauncherError(f"cannot stat --probe-bin {config.probe_bin}: {exc}") from exc
    if not stat.S_ISREG(probe_stat.st_mode) or not os.access(config.probe_bin, os.X_OK):
        raise AttentionProbeLauncherError(
            f"--probe-bin must be an executable regular file: {config.probe_bin}"
        )
    if config.mode not in {"cpu-oracle", "metal"}:
        raise AttentionProbeLauncherError(f"unsupported mode {config.mode!r}")
    if not config.timeout_seconds > 0:
        raise AttentionProbeLauncherError("--timeout-seconds must be positive")
    for value, label in (
        (config.expected_manifest_seal_sha256, "manifest seal"),
        (config.expected_source_audit_seal_sha256, "source-audit seal"),
    ):
        if len(value) != 64:
            raise AttentionProbeLauncherError(f"{label} must be a 64-character SHA-256")
        try:
            int(value, 16)
        except ValueError as exc:
            raise AttentionProbeLauncherError(f"{label} must be hexadecimal") from exc


def _launch_identity(config: LaunchConfig) -> str:
    payload = {
        "probe_bin": str(config.probe_bin),
        "probe_binary_sha256": _file_sha256(config.probe_bin),
        "manifest": str(config.manifest),
        "expected_manifest_seal_sha256": config.expected_manifest_seal_sha256,
        "expected_source_audit_seal_sha256": config.expected_source_audit_seal_sha256,
        "expected_source_revision": config.expected_source_revision,
        "mode": config.mode,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _child_command(config: LaunchConfig, inner_capture_dir: Path) -> list[str]:
    return [
        str(config.probe_bin),
        "--manifest",
        str(config.manifest),
        "--expected-manifest-seal-sha256",
        config.expected_manifest_seal_sha256,
        "--expected-source-audit-seal-sha256",
        config.expected_source_audit_seal_sha256,
        "--expected-source-revision",
        config.expected_source_revision,
        "--capture-dir",
        str(inner_capture_dir),
        "--mode",
        config.mode,
    ]


def _inner_receipt_evidence(inner_capture_dir: Path) -> dict[str, Any]:
    receipt_path = inner_capture_dir / "receipt.json"
    evidence = _file_evidence(receipt_path)
    evidence["capture_dir"] = str(inner_capture_dir)
    if not evidence["present"]:
        evidence["invocation"] = _file_evidence(inner_capture_dir / "invocation.json")
        return evidence
    try:
        document = _read_json(receipt_path)
    except AttentionProbeLauncherError as exc:
        evidence["parse_error"] = str(exc)
        return evidence
    evidence["schema"] = document.get("schema")
    evidence["status"] = document.get("status")
    evidence["receipt_unsigned_json_sha256"] = document.get("receipt_unsigned_json_sha256")
    return evidence


def _sync_and_evidence(path: Path) -> dict[str, Any]:
    # The child has been reaped before this call. Opening write-only and fsyncing
    # makes the outer terminal receipt bind the exact terminal bytes it reads.
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path)


def _terminal_from_returncode(
    returncode: int | None, *, timed_out: bool, spawn_error: str | None = None
) -> dict[str, Any]:
    terminal: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        terminal["spawn_error"] = spawn_error
        terminal["reaped"] = False
    return terminal


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_LAYER3_GQA_OUTER_CAPTURE_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_LAYER3_GQA_OUTER_CAPTURE_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_LAYER3_GQA_OUTER_CAPTURE_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_LAYER3_GQA_OUTER_CAPTURE_CHILD_NONZERO"
    if inner.get("present") is not True:
        return "REFUSED_QWEN80_LAYER3_GQA_OUTER_CAPTURE_ZERO_EXIT_WITHOUT_INNER_RECEIPT"
    return (
        "CAPTURED_QWEN80_LAYER3_GQA_OUTER_TERMINAL_CHILD_ZERO_INNER_RECEIPT_PRESENT_"
        "NOT_A_PARITY_OR_RUNTIME_CLAIM"
    )


def _seal_terminal_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return seal(dict(receipt))


def _terminal_receipt(
    config: LaunchConfig,
    *,
    launch_identity_sha256: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    finished_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None = None,
) -> dict[str, Any]:
    stdout_path = config.capture_dir / OUTER_STDOUT
    stderr_path = config.capture_dir / OUTER_STDERR
    inner_capture_dir = config.capture_dir / INNER_CAPTURE
    streams = {
        "stdout": _sync_and_evidence(stdout_path),
        "stderr": _sync_and_evidence(stderr_path),
    }
    inner = _inner_receipt_evidence(inner_capture_dir)
    status = _terminal_status(terminal, inner)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "recorded_at": finished_at,
        "one_shot": {
            "automatic_retry_disabled": True,
            "replay_of_same_capture_dir_never_launches_a_second_child": True,
            "terminal_receipt_is_written_last": True,
        },
        "launch_identity_sha256": launch_identity_sha256,
        "source_binding": {
            "manifest": str(config.manifest),
            "expected_manifest_seal_sha256": config.expected_manifest_seal_sha256,
            "expected_source_audit_seal_sha256": config.expected_source_audit_seal_sha256,
            "expected_source_revision": config.expected_source_revision,
            "mode": config.mode,
        },
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": list(command),
            "probe_binary_sha256": _file_sha256(config.probe_bin),
            "terminal": dict(terminal),
        },
        "outer_capture": {
            "directory": str(config.capture_dir),
            "active_receipt": str(config.capture_dir / ACTIVE_RECEIPT),
            "stdout": streams["stdout"],
            "stderr": streams["stderr"],
        },
        "inner_probe_capture": inner,
        "claim_boundary": {
            "outer_terminal_capture_only": True,
            "does_not_validate_or_promote_inner_parity": True,
            "does_not_execute_a_complete_layer_or_decoder": True,
            "does_not_generate_tokens_expose_hcli_or_measure_tps": True,
            "does_not_claim_tg10_tg3_or_tournament_qualification": True,
        },
    }
    if capture_error is not None:
        receipt["capture_error"] = capture_error
    return _seal_terminal_receipt(receipt)


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") == (
        "CAPTURED_QWEN80_LAYER3_GQA_OUTER_TERMINAL_CHILD_ZERO_INNER_RECEIPT_PRESENT_"
        "NOT_A_PARITY_OR_RUNTIME_CLAIM"
    )


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


def _replay_existing(config: LaunchConfig, launch_identity_sha256: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_RECEIPT
    if not terminal_path.is_file():
        raise AttentionProbeLauncherError(
            f"capture directory already exists without a terminal receipt: {config.capture_dir}"
        )
    receipt = _read_json(terminal_path)
    try:
        verify(receipt, label=str(terminal_path))
    except ValueError as exc:
        raise AttentionProbeLauncherError(f"terminal receipt seal is invalid: {exc}") from exc
    if receipt.get("schema") != SCHEMA:
        raise AttentionProbeLauncherError(f"terminal receipt schema drift in {terminal_path}")
    if receipt.get("launch_identity_sha256") != launch_identity_sha256:
        raise AttentionProbeLauncherError(
            "capture directory belongs to a different command/binding; refusing a second child"
        )
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run or sealed-replay exactly one bounded child process."""

    _validate_config(config)
    launch_identity_sha256 = _launch_identity(config)
    if config.capture_dir.exists():
        return _replay_existing(config, launch_identity_sha256)
    parent = config.capture_dir.parent
    if not parent.is_dir():
        raise AttentionProbeLauncherError(
            f"--capture-dir parent is not an existing directory: {parent}"
        )
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay_existing(config, launch_identity_sha256)
    command = _child_command(config, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_RECEIPT,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_LAYER3_GQA_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": launch_identity_sha256,
                "source_binding": {
                    "manifest": str(config.manifest),
                    "expected_manifest_seal_sha256": config.expected_manifest_seal_sha256,
                    "expected_source_audit_seal_sha256": config.expected_source_audit_seal_sha256,
                    "expected_source_revision": config.expected_source_revision,
                    "mode": config.mode,
                },
                "command": command,
                "claim_boundary": {
                    "launch_is_not_a_parity_or_runtime_success_receipt": True,
                    "automatic_retry_disabled": True,
                },
            }
        ),
    )
    stdout_path = config.capture_dir / OUTER_STDOUT
    stderr_path = config.capture_dir / OUTER_STDERR
    child_pid: int | None = None
    terminal: dict[str, Any]
    capture_error: str | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            terminal = _terminal_from_returncode(
                None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}"
            )
        else:
            child_pid = process.pid
            try:
                _atomic_json_new(
                    config.capture_dir / "child.json",
                    seal(
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING_QWEN80_LAYER3_GQA_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": launch_identity_sha256,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "stdout_path": str(stdout_path),
                            "stderr_path": str(stderr_path),
                            "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                        }
                    ),
                )
            except AttentionProbeLauncherError as exc:
                capture_error = str(exc)
                returncode = _terminate_process_group(process)
                terminal = _terminal_from_returncode(returncode, timed_out=False)
            else:
                try:
                    returncode = process.wait(timeout=config.timeout_seconds)
                    terminal = _terminal_from_returncode(returncode, timed_out=False)
                except subprocess.TimeoutExpired:
                    returncode = _terminate_process_group(process)
                    terminal = _terminal_from_returncode(returncode, timed_out=True)
    finished_at = _utc_now()
    receipt = _terminal_receipt(
        config,
        launch_identity_sha256=launch_identity_sha256,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        finished_at=finished_at,
        terminal=terminal,
        capture_error=capture_error,
    )
    _atomic_json_new(config.capture_dir / TERMINAL_RECEIPT, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-seal-sha256", required=True)
    parser.add_argument("--expected-source-audit-seal-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-oracle", "metal"), required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        expected_manifest_seal_sha256=parsed.expected_manifest_seal_sha256,
        expected_source_audit_seal_sha256=parsed.expected_source_audit_seal_sha256,
        expected_source_revision=parsed.expected_source_revision,
        capture_dir=parsed.capture_dir,
        mode=parsed.mode,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except AttentionProbeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_LAYER3_GQA_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
