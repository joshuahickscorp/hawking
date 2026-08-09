#!/usr/bin/env python3
"""One-shot outer runner for the strict Qwen80 L0→L1 prefix component.

The only live mode is deliberately explicit: ``--execute-one-shot``.  It
requires a freshly sealed resource admission, the sealed joint outer
preflight/execution binding, and the held Q80 watcher.  It then creates one
new lease, lets the lifecycle module create the replay guard and outer launch
authority, reaps exactly one strict host child, and writes a separate release
record.  Tests inject a disposable child; this module never defaults to a
device action.

This remains a component capture runner.  It cannot start a Q80 server, run a
decoder token loop, execute an L1 suffix/MoE block, or make TG/TPS/tournament
claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_l0_l1_same_runtime_prefix_lifecycle as lifecycle
from lab.receipts import SealIntegrityError, seal, verify


RESOURCE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_resource_admission.v1"
RESOURCE_STATUS = "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_RESOURCE_ADMISSION_NOT_LEASED_OR_EXECUTED"
RESOURCE_REFUSED_STATUS = "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_RESOURCE_ADMISSION"
RUNNER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_outer_runner.v1"
# This runner records the lifecycle terminal, not a capture success.  A child
# refusal must remain a durable, released terminal without being mislabeled as
# an executed component result.
RUNNER_STATUS = "TERMINAL_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_ONE_SHOT_REAPED"

MIN_FREE_PERCENT = 80
MAX_RESOURCE_AGE_SECONDS = 300
MAX_TIMEOUT_SECONDS = 7200.0
# Only a direct process invocation of this executable is a competing strict
# joint child.  A substring match would flag the caller's shell merely because
# it passed ``--host-binary`` (or a checksum command named the binary).
Q80_HOST_EXECUTABLE = str(
    REPO_ROOT
    / "workspace/ops/build/rust/debug/examples/ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device"
)
Q80_WATCHER_FRAGMENT = "ascension_qwen80_bootstrap_lanes runtime --watch"
Q30_CAPTURE_FRAGMENTS = (
    "ascension_qwen30_quality_repack_all_layer_current_trace_diagnostic",
    "ascension_qwen30_streamed_source_teacher_child",
    "ascension_qwen30_guarded_streamed_source_oracle",
)


class JointOuterRunnerError(RuntimeError):
    """A resource or lifecycle condition prevents a one-shot capture."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JointOuterRunnerError(f"{label} must be an object")
    return dict(value)


def _regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise JointOuterRunnerError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise JointOuterRunnerError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise JointOuterRunnerError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise JointOuterRunnerError(f"{label} must be executable")
    return path.resolve(strict=True)


def _new_output(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.exists():
        raise JointOuterRunnerError(f"{label} must be a new absolute path")
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError as exc:
        raise JointOuterRunnerError(f"cannot stat {label} parent: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise JointOuterRunnerError(f"{label} parent must be a real directory")
    return path


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    path = _new_output(path, "output")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(dict(document), handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _sealed(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = seal(dict(document))
    try:
        checked = verify(payload, label="joint outer runner document")
    except SealIntegrityError as exc:  # pragma: no cover - defensive
        raise JointOuterRunnerError(f"cannot self-verify sealed output: {exc}") from exc
    return dict(checked)


def _run_text(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise JointOuterRunnerError(f"resource observation command failed: {exc}") from exc
    return completed.stdout


def _parse_memory_free_percent(output: str) -> int:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if not match:
        raise JointOuterRunnerError("memory_pressure output lacks free percentage")
    return int(match.group(1))


def _parse_swap_used_bytes(output: str) -> int:
    match = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", output, re.IGNORECASE)
    if not match:
        raise JointOuterRunnerError("vm.swapusage output lacks used value")
    magnitude = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[unit]
    return int(magnitude * multiplier)


def _process_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in output.splitlines():
        fields = raw.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid, parent_pid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        rows.append({"pid": pid, "ppid": parent_pid, "command": fields[2]})
    return rows


def _is_strict_joint_host_process(row: Mapping[str, Any]) -> bool:
    command = row.get("command")
    if not isinstance(command, str):
        return False
    normalized = command.lstrip()
    return normalized == Q80_HOST_EXECUTABLE or normalized.startswith(Q80_HOST_EXECUTABLE + " ")


def collect_live_snapshot() -> dict[str, Any]:
    """Collect a small read-only resource/process observation on macOS."""
    pressure = _parse_memory_free_percent(_run_text(["/usr/bin/memory_pressure", "-Q"]))
    swap = _parse_swap_used_bytes(_run_text(["/usr/sbin/sysctl", "vm.swapusage"]))
    rows = _process_rows(_run_text(["/bin/ps", "-axo", "pid=,ppid=,command="]))
    watcher = [row["pid"] for row in rows if Q80_WATCHER_FRAGMENT in row["command"]]
    q80_host = [row for row in rows if _is_strict_joint_host_process(row)]
    q30_capture = [
        row
        for row in rows
        if any(fragment in row["command"] for fragment in Q30_CAPTURE_FRAGMENTS)
    ]
    return {
        "observed_at": _utc_now(),
        "memory_free_percent": pressure,
        "swap_used_bytes": swap,
        "q80_watcher_parent_pids": watcher,
        "q80_strict_joint_host_children": q80_host,
        "q30_metal_or_capture_children": q30_capture,
    }


def evaluate_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    """Return every admission blocker without changing process state."""
    record = _mapping(snapshot, "resource snapshot")
    blockers: list[str] = []
    free = record.get("memory_free_percent")
    if isinstance(free, bool) or not isinstance(free, int) or free < MIN_FREE_PERCENT:
        blockers.append(f"memory free percentage is below {MIN_FREE_PERCENT}")
    swap = record.get("swap_used_bytes")
    if isinstance(swap, bool) or not isinstance(swap, int) or swap != 0:
        blockers.append("swap must be exactly zero")
    watchers = record.get("q80_watcher_parent_pids")
    if not isinstance(watchers, list) or len(watchers) != 1 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in watchers
    ):
        blockers.append("exactly one held Q80 watcher parent is required")
    for key, label in (
        ("q80_strict_joint_host_children", "Q80 strict joint host child"),
        ("q30_metal_or_capture_children", "Q30 Metal/capture child"),
    ):
        rows = record.get(key)
        if not isinstance(rows, list):
            blockers.append(f"{label} observation is invalid")
        elif rows:
            blockers.append(f"{label} is already active")
    return blockers


def _identity(bound: lifecycle.BoundDocument) -> dict[str, Any]:
    return {
        "path": str(bound.path),
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def build_resource_admission(
    *,
    outer_preflight: Path,
    execution_binding: Path,
    watcher_hold: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a sealed resource decision; this never issues a lease."""
    outer = lifecycle._read_outer_preflight(outer_preflight)
    execution = lifecycle._read_execution_binding(execution_binding, outer)
    hold_path, _, hold_evidence = lifecycle._read_watcher_hold(watcher_hold, outer)
    blockers = evaluate_snapshot(snapshot)
    status = RESOURCE_STATUS if not blockers else RESOURCE_REFUSED_STATUS
    scope = _mapping(outer.document.get("exact_joint_scope"), "joint outer scope")
    host_sha = scope.get("host_binary_sha256")
    if not _is_sha256(host_sha):
        raise JointOuterRunnerError("joint outer preflight host SHA is invalid")
    return _sealed(
        {
            "schema": RESOURCE_SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "prepared": not blockers,
            "blockers": blockers,
            "minimum_memory_free_percent": MIN_FREE_PERCENT,
            "maximum_resource_age_seconds": MAX_RESOURCE_AGE_SECONDS,
            "outer_preflight": _identity(outer),
            "execution_binding": _identity(execution),
            "watcher_hold": {"path": str(hold_path), **hold_evidence},
            "host_binary_sha256": host_sha,
            "resource_snapshot": dict(snapshot),
            "claim_boundary": {
                "cpu_file_only_resource_admission": True,
                "lease_issued_or_consumed": False,
                "metal_or_gpu_activity_performed": False,
                "server_watcher_hcli_tps_tg_or_tournament_action": False,
                "cross_process_pinned_buffer_transfer_authorized": False,
            },
        }
    )


def write_resource_admission(
    *,
    outer_preflight: Path,
    execution_binding: Path,
    watcher_hold: Path,
    out: Path,
    snapshot_provider: Callable[[], Mapping[str, Any]] = collect_live_snapshot,
) -> dict[str, Any]:
    document = build_resource_admission(
        outer_preflight=outer_preflight,
        execution_binding=execution_binding,
        watcher_hold=watcher_hold,
        snapshot=snapshot_provider(),
    )
    _write_new(out, document)
    return document


def _parse_recorded_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise JointOuterRunnerError("resource admission recorded_at is invalid")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JointOuterRunnerError("resource admission recorded_at is invalid") from exc


def _read_resource_admission(
    *,
    path: Path,
    outer_preflight: Path,
    execution_binding: Path,
    watcher_hold: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    clean = _regular(path, "resource admission")
    try:
        document = verify(json.loads(clean.read_text(encoding="utf-8")), label="resource admission")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise JointOuterRunnerError(f"resource admission is absent or invalid: {exc}") from exc
    root = _mapping(document, "resource admission")
    if root.get("schema") != RESOURCE_SCHEMA or root.get("status") != RESOURCE_STATUS:
        raise JointOuterRunnerError("resource admission schema/status is not a PASS")
    if root.get("prepared") is not True or root.get("blockers") != []:
        raise JointOuterRunnerError("resource admission is not green")
    observed = _parse_recorded_at(root.get("recorded_at"))
    reference = now or datetime.now(timezone.utc)
    if observed.tzinfo is None or (reference - observed).total_seconds() < 0 or (
        reference - observed
    ).total_seconds() > MAX_RESOURCE_AGE_SECONDS:
        raise JointOuterRunnerError("resource admission is stale")
    outer = lifecycle._read_outer_preflight(outer_preflight)
    execution = lifecycle._read_execution_binding(execution_binding, outer)
    hold_path, _, hold_evidence = lifecycle._read_watcher_hold(watcher_hold, outer)
    for name, expected in (("outer_preflight", _identity(outer)), ("execution_binding", _identity(execution))):
        if _mapping(root.get(name), f"resource admission.{name}") != expected:
            raise JointOuterRunnerError(f"resource admission {name} drifted")
    expected_hold = {"path": str(hold_path), **hold_evidence}
    if _mapping(root.get("watcher_hold"), "resource admission.watcher_hold") != expected_hold:
        raise JointOuterRunnerError("resource admission watcher hold drifted")
    scope = _mapping(outer.document.get("exact_joint_scope"), "joint outer scope")
    if root.get("host_binary_sha256") != scope.get("host_binary_sha256"):
        raise JointOuterRunnerError("resource admission host SHA drifted")
    return root


def _run_host_source_admission_preflight(*, host: Path, outer_preflight: Path, workers: int) -> None:
    """Validate the host's immutable source chain before any joint lease exists."""
    command = (
        str(host),
        "--mode",
        "source-admission-preflight",
        "--joint-outer-preflight",
        str(outer_preflight),
        "--workers",
        str(workers),
    )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise JointOuterRunnerError(f"read-only host source-admission preflight failed: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr[-4_000:]
        raise JointOuterRunnerError(
            "read-only host source-admission preflight refused before lease issuance: " + stderr
        )


def execute_one_shot(
    *,
    outer_preflight: Path,
    execution_binding: Path,
    watcher_hold: Path,
    resource_admission: Path,
    lease_out: Path,
    capture_dir: Path,
    release_out: Path,
    host_binary: Path,
    workers: int,
    timeout_seconds: float,
    snapshot_provider: Callable[[], Mapping[str, Any]] = collect_live_snapshot,
    child_command_for_test: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Issue and consume one new joint lease after a fresh resource recheck.

    ``child_command_for_test`` is intentionally not exposed through the CLI;
    it lets the focused suite test the production terminal/release path without
    a Metal child.
    """
    if not 1 <= workers <= 4 or not 1.0 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise JointOuterRunnerError("workers/timeout are outside the bounded one-shot policy")
    host = _regular(host_binary, "strict joint host", executable=True)
    if capture_dir.exists():
        raise JointOuterRunnerError("capture directory must be new before lease issuance")
    _new_output(lease_out, "lease output")
    _new_output(release_out, "release output")
    resource = _read_resource_admission(
        path=resource_admission,
        outer_preflight=outer_preflight,
        execution_binding=execution_binding,
        watcher_hold=watcher_hold,
    )
    recheck_blockers = evaluate_snapshot(snapshot_provider())
    if recheck_blockers:
        raise JointOuterRunnerError("fresh resource recheck refused: " + "; ".join(recheck_blockers))
    if resource.get("host_binary_sha256") != _sha256_bytes(host.read_bytes()):
        raise JointOuterRunnerError("strict joint host bytes drifted from resource admission")
    _run_host_source_admission_preflight(
        host=host,
        outer_preflight=outer_preflight,
        workers=workers,
    )
    lease = lifecycle.issue_lease(
        outer_preflight=outer_preflight,
        execution_binding=execution_binding,
        watcher_hold=watcher_hold,
        out=lease_out,
    )
    if child_command_for_test is None:
        command: tuple[str, ...] = (
            str(host),
            "--mode",
            "metal",
            "--joint-outer-preflight",
            str(outer_preflight),
            "--lease-receipt",
            str(lease_out),
            "--outer-launch-authority",
            str(capture_dir / lifecycle.OUTER_LAUNCH_FILENAME),
            "--outer-capture-dir",
            str(capture_dir),
            "--capture-dir",
            str(capture_dir / lifecycle.INNER_DIRNAME),
            "--workers",
            str(workers),
        )
    else:
        command = tuple(child_command_for_test)
    # Focused tests may inject a disposable command, but that hook is not
    # reachable from the CLI.  Preserve the lifecycle's explicit test-only
    # boundary so temporary refusal coverage never resembles a Metal capture.
    terminal = lifecycle.run_one_shot(
        lifecycle.CaptureConfig(
            lease_receipt=lease_out,
            outer_preflight=outer_preflight,
            execution_binding=execution_binding,
            watcher_hold=watcher_hold,
            capture_dir=capture_dir,
            timeout_seconds=timeout_seconds,
            workers=workers,
            child_command=command,
        ),
        test_only=child_command_for_test is not None,
    )
    release = lifecycle.release_after_terminal(
        outer_terminal=capture_dir / lifecycle.OUTER_TERMINAL_FILENAME,
        lease_receipt=lease_out,
        out=release_out,
        release_issuer_identity_sha256=_sha256_bytes(
            f"joint-release:{lease.lease_id}:{release_out}".encode("utf-8")
        ),
    )
    return _sealed(
        {
            "schema": RUNNER_SCHEMA,
            "status": RUNNER_STATUS,
            "resource_admission": {
                "path": str(resource_admission),
                "seal_sha256": resource["seal_sha256"],
            },
            "lease": {"path": str(lease_out), "seal_sha256": lease.lease.document_seal_sha256},
            "outer_terminal": {
                "path": str(capture_dir / lifecycle.OUTER_TERMINAL_FILENAME),
                "status": terminal["status"],
                "seal_sha256": terminal["seal_sha256"],
            },
            "release": {"path": str(release_out), "seal_sha256": release["seal_sha256"]},
            "claim_boundary": {
                "component_only": True,
                "l1_suffix_or_moe_executed": False,
                "complete_layer_executed": False,
                "token_generated": False,
                "decoder_started": False,
                "server_or_watcher_started": False,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute-one-shot"), default="preflight")
    parser.add_argument("--outer-preflight", required=True, type=Path)
    parser.add_argument("--execution-binding", required=True, type=Path)
    parser.add_argument("--watcher-hold", required=True, type=Path)
    parser.add_argument("--resource-admission", type=Path)
    parser.add_argument("--out", type=Path, help="new resource-admission output for --mode preflight")
    parser.add_argument("--lease-out", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--release-out", type=Path)
    parser.add_argument("--host-binary", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=MAX_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "preflight":
            if args.out is None:
                raise JointOuterRunnerError("--out is required for --mode preflight")
            result = write_resource_admission(
                outer_preflight=args.outer_preflight,
                execution_binding=args.execution_binding,
                watcher_hold=args.watcher_hold,
                out=args.out,
            )
            print(json.dumps({"out": str(args.out), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
            return 0 if result["status"] == RESOURCE_STATUS else 2
        required = {
            "--resource-admission": args.resource_admission,
            "--lease-out": args.lease_out,
            "--capture-dir": args.capture_dir,
            "--release-out": args.release_out,
            "--host-binary": args.host_binary,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise JointOuterRunnerError("execute-one-shot requires " + ", ".join(missing))
        result = execute_one_shot(
            outer_preflight=args.outer_preflight,
            execution_binding=args.execution_binding,
            watcher_hold=args.watcher_hold,
            resource_admission=args.resource_admission,
            lease_out=args.lease_out,
            capture_dir=args.capture_dir,
            release_out=args.release_out,
            host_binary=args.host_binary,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
        )
    except JointOuterRunnerError as exc:
        print(f"Q80 strict joint outer runner refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
