#!/usr/bin/env python3
"""Launch a Flash FastPath benchmark with durable detached state."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    if ns.monitor:
        state_path = ns.out_dir / "PROCESS_STATE.json"
        state = json.loads(state_path.read_text())
        pid = int(state["pid"])
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                state.update({"status": "COMPLETE", "completed_unix_ns": time.time_ns()})
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                summary = ns.out_dir / "FAST_CHAIN_SUMMARY.json"
                receipts = sorted(
                    str(path.relative_to(ns.out_dir))
                    for path in ns.out_dir.rglob("*.json")
                    if path.name in {"receipt.json", "terminal.json"}
                )
                result = {
                    "schema": "hawking.flash.detached_result.v1",
                    "status": "COMPLETE" if summary.is_file() else "FAILED_OR_INCOMPLETE",
                    "summary": str(summary) if summary.is_file() else None,
                    "receipts": receipts,
                    "completed_unix_ns": state["completed_unix_ns"],
                }
                (ns.out_dir / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
                timing = {}
                if summary.is_file():
                    try:
                        summary_doc = json.loads(summary.read_text())
                        timing = {
                            "elapsed_wall_ns": summary_doc.get("elapsed_wall_ns"),
                            "start_layer": summary_doc.get("start_layer"),
                            "end_layer": summary_doc.get("end_layer"),
                        }
                    except json.JSONDecodeError:
                        timing = {"parse_error": True}
                (ns.out_dir / "TIMING.json").write_text(json.dumps(timing, indent=2) + "\n")
                return 0
            time.sleep(1)
    command = ns.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("a command is required after --")
    out = ns.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "EXPERIMENT_SPEC.json").write_text(json.dumps({
        "schema": "hawking.flash.detached_experiment.v1",
        "command": command,
        "cwd": os.getcwd(),
        "mode": "PROTECTED_FAST",
        "created_unix_ns": time.time_ns(),
    }, indent=2) + "\n")
    stdout = (out / "stdout.log").open("w")
    stderr = (out / "stderr.log").open("w")
    started = time.time_ns()
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    state = {
        "schema": "hawking.flash.detached_process.v1",
        "pid": proc.pid,
        "started_unix_ns": started,
        "status": "RUNNING",
    }
    (out / "PROCESS_STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    monitor = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(out), "--monitor"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    state["monitor_pid"] = monitor.pid
    (out / "PROCESS_STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"pid": proc.pid, "out_dir": str(out), "status": "RUNNING"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
