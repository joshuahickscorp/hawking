#!/usr/bin/env python3
"""Run one long-lived command under fail-closed local resource limits.

The runner is intentionally generic: it owns a process group, samples the
whole descendant tree, publishes an atomic heartbeat, and kills the group if
RSS, disk, or swap policy is violated.  It is suitable for detached model
capture jobs where a shell ``ulimit`` is neither portable nor auditable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "hawking.ascension.bounded_process_runner.v1"
DEFAULT_RSS_CAP = 5 * 1024**3
DEFAULT_DISK_FLOOR = 25 * 1024**3
_SWAP_RE = re.compile(r"used\s*=\s*([0-9.]+)([KMGTP])", re.IGNORECASE)
_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_ps_rows(text: str) -> list[tuple[int, int, int, float]]:
    rows: list[tuple[int, int, int, float]] = []
    for raw in text.splitlines():
        fields = raw.split()
        if len(fields) != 4:
            continue
        try:
            pid, ppid, rss_kib = map(int, fields[:3])
            cpu = float(fields[3])
        except ValueError:
            continue
        rows.append((pid, ppid, rss_kib * 1024, cpu))
    return rows


def descendant_pids(root_pid: int, rows: Iterable[tuple[int, int, int, float]]) -> set[int]:
    materialized = list(rows)
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _rss, _cpu in materialized:
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def process_tree_sample(root_pid: int) -> dict[str, Any]:
    proc = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,%cpu="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = parse_ps_rows(proc.stdout)
    pids = descendant_pids(root_pid, rows)
    selected = [row for row in rows if row[0] in pids]
    return {
        "root_pid": root_pid,
        "pids": sorted(pids),
        "process_count": len(selected),
        "rss_bytes": sum(row[2] for row in selected),
        "cpu_percent": sum(row[3] for row in selected),
    }


def swap_used_bytes() -> int:
    proc = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = _SWAP_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"cannot parse vm.swapusage: {proc.stdout!r}")
    return int(float(match.group(1)) * _UNITS[match.group(2).upper()])


def terminate_group(proc: subprocess.Popen[Any], grace_seconds: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("command is required after --")
    disk_path = args.disk_path.resolve()
    status_path = args.status.resolve()
    receipt_path = args.receipt.resolve()
    baseline_swap = swap_used_bytes()
    started = time.monotonic()
    child = subprocess.Popen(args.command, start_new_session=True)
    stop_reason: str | None = None
    samples = 0
    peak_rss = 0
    min_free_disk = shutil.disk_usage(disk_path).free

    def publish(state: str, sample: dict[str, Any] | None = None) -> None:
        payload = {
            "schema": SCHEMA,
            "state": state,
            "updated_at": utc_now(),
            "runner_pid": os.getpid(),
            "child_pid": child.pid,
            "command": args.command,
            "policy": {
                "rss_cap_bytes": args.rss_cap_bytes,
                "disk_floor_bytes": args.disk_floor_bytes,
                "disk_path": str(disk_path),
                "swap_growth_allowed": False,
                "baseline_swap_bytes": baseline_swap,
                "poll_seconds": args.poll_seconds,
            },
            "sample": sample,
            "samples": samples,
            "peak_rss_bytes": peak_rss,
            "minimum_free_disk_bytes": min_free_disk,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stop_reason": stop_reason,
        }
        atomic_json(status_path, payload)

    publish("STARTED")
    try:
        while child.poll() is None:
            tree = process_tree_sample(child.pid)
            free_disk = shutil.disk_usage(disk_path).free
            swap_now = swap_used_bytes()
            samples += 1
            peak_rss = max(peak_rss, int(tree["rss_bytes"]))
            min_free_disk = min(min_free_disk, free_disk)
            sample = {
                **tree,
                "free_disk_bytes": free_disk,
                "swap_used_bytes": swap_now,
                "sampled_at": utc_now(),
            }
            if tree["rss_bytes"] > args.rss_cap_bytes:
                stop_reason = "RSS_CAP_EXCEEDED"
            elif free_disk < args.disk_floor_bytes:
                stop_reason = "DISK_FLOOR_BREACHED"
            elif swap_now > baseline_swap:
                stop_reason = "SWAP_GROWTH_DETECTED"
            if stop_reason:
                publish("TERMINATING_POLICY_BREACH", sample)
                terminate_group(child, args.terminate_grace_seconds)
                break
            publish("RUNNING", sample)
            time.sleep(args.poll_seconds)
    except BaseException:
        stop_reason = stop_reason or "RUNNER_INTERRUPTED"
        terminate_group(child, args.terminate_grace_seconds)
        raise

    exit_code = child.wait()
    final_state = "COMPLETE" if exit_code == 0 and stop_reason is None else "FAILED"
    receipt = {
        "schema": SCHEMA,
        "state": final_state,
        "started_at_monotonic": started,
        "completed_at": utc_now(),
        "runner_pid": os.getpid(),
        "child_pid": child.pid,
        "command": args.command,
        "exit_code": exit_code,
        "stop_reason": stop_reason,
        "samples": samples,
        "peak_rss_bytes": peak_rss,
        "minimum_free_disk_bytes": min_free_disk,
        "baseline_swap_bytes": baseline_swap,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "policy": {
            "rss_cap_bytes": args.rss_cap_bytes,
            "disk_floor_bytes": args.disk_floor_bytes,
            "disk_path": str(disk_path),
            "swap_growth_allowed": False,
        },
    }
    atomic_json(receipt_path, receipt)
    publish(final_state)
    return exit_code if exit_code != 0 else (78 if stop_reason else 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--rss-cap-bytes", type=int, default=DEFAULT_RSS_CAP)
    parser.add_argument("--disk-floor-bytes", type=int, default=DEFAULT_DISK_FLOOR)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.rss_cap_bytes <= 0 or args.disk_floor_bytes <= 0 or args.poll_seconds <= 0:
        raise SystemExit("resource limits and poll interval must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
