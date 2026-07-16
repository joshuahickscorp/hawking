#!/usr/bin/env python3.12
"""Canonical frontier engine, queue observer, evidence profile, and conductor."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Any

import condense_common as common
import condense_profiles
import doctor_v5_local_observer


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.frontier_condensed_runtime.v1"
PS_PATH = doctor_v5_local_observer.AUTHORITY_TOOL_PATHS["ps"]
HEAVY_PATTERNS = doctor_v5_local_observer.HEAVY_COMMAND_PATTERNS
BASE_FOR = {"mp-4a3f": "mp-4a3f", "3-AWQ": "3-AWQ", "4-AWQ": "4-AWQ"}
PROFILE = {
    "minimum_gain_pct": 0.25,
    "verifier_windows": 5,
    "verifier_tolerance_pct_points": 5.0,
    "verifier_poll_seconds": 300,
    "disk_reserve_bytes": 150_000_000_000,
    "required_pressure_level": 1,
    "required_swap_bytes": 0,
    "canonical_commands": [
        "frontier.runtime", "frontier.conductor", "frontier.verifier",
        "frontier.autopilot", "frontier.launcher", "frontier.ops",
    ],
}


def _sysctl_raw(name: str) -> bytes:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.sysctlbyname
    function.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    size = ctypes.c_size_t()
    encoded = name.encode("ascii")
    if function(encoded, None, ctypes.byref(size), None, 0) != 0 or size.value <= 0:
        raise OSError(ctypes.get_errno(), f"sysctl size probe failed for {name}")
    buffer = ctypes.create_string_buffer(size.value)
    if function(encoded, buffer, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), f"sysctl value probe failed for {name}")
    return bytes(buffer.raw[:size.value])


def _sysctl_uint(name: str) -> int:
    raw = _sysctl_raw(name)
    if len(raw) not in {1, 2, 4, 8}:
        raise ValueError(f"unexpected integer sysctl width for {name}: {len(raw)}")
    return int.from_bytes(raw, byteorder=sys.byteorder, signed=False)


def _power_source() -> str:
    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    iokit.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
    iokit.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]
    iokit.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
    cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32,
    ]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    blob = iokit.IOPSCopyPowerSourcesInfo()
    if not blob:
        raise RuntimeError("IOKit power-source snapshot unavailable")
    try:
        value = iokit.IOPSGetProvidingPowerSourceType(blob)
        buffer = ctypes.create_string_buffer(128)
        if not value or not cf.CFStringGetCString(value, buffer, len(buffer), 0x08000100):
            raise RuntimeError("cannot decode IOKit power source")
        return buffer.value.decode("utf-8")
    finally:
        cf.CFRelease(blob)


def _thermal_state() -> int:
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    if not address:
        raise RuntimeError("objc_msgSend unavailable")
    send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
    send_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(address)
    cls = objc.objc_getClass(b"NSProcessInfo")
    process_sel = objc.sel_registerName(b"processInfo")
    thermal_sel = objc.sel_registerName(b"thermalState")
    if not cls or not process_sel or not thermal_sel:
        raise RuntimeError("NSProcessInfo thermal selectors unavailable")
    process = send_object(cls, process_sel)
    state = int(send_integer(process, thermal_sel)) if process else -1
    if state not in {0, 1, 2, 3}:
        raise RuntimeError(f"unknown NSProcessInfo thermal state {state}")
    return state


def resource_snapshot(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    pressure = swap = physical = power = thermal = None
    if sys.platform != "darwin":
        errors.append("resource probes support Darwin only")
    else:
        for label, function in (
            ("memory pressure", lambda: _sysctl_uint("kern.memorystatus_vm_pressure_level")),
            ("physical-memory", lambda: _sysctl_uint("hw.memsize")),
            ("power-source", _power_source),
            ("thermal-state", _thermal_state),
        ):
            try:
                value = function()
                if label == "memory pressure":
                    pressure = value
                elif label == "physical-memory":
                    physical = value
                elif label == "power-source":
                    power = value
                else:
                    thermal = value
            except Exception as exc:
                errors.append(f"{label} probe failed: {exc}")
        try:
            raw = _sysctl_raw("vm.swapusage")
            if len(raw) < struct.calcsize("@QQQII"):
                raise ValueError(f"unexpected vm.swapusage width {len(raw)}")
            _total, _available, swap, _page, _encrypted = struct.unpack_from("@QQQII", raw)
        except Exception as exc:
            errors.append(f"swap probe failed: {exc}")
    try:
        usage = shutil.disk_usage(root)
        disk_free, disk_total = int(usage.free), int(usage.total)
    except OSError as exc:
        errors.append(f"disk probe failed: {exc}")
        disk_free = disk_total = None
    return {
        "probe_ok": not errors,
        "errors": errors,
        "pressure_level": pressure,
        "swap_used_bytes": int(swap) if isinstance(swap, int) else None,
        "physical_memory_bytes": int(physical) if isinstance(physical, int) else None,
        "power_source": power,
        "thermal_state": thermal,
        "disk_free_bytes": disk_free,
        "disk_total_bytes": disk_total,
    }


_resource_snapshot = resource_snapshot


def owners_from_ps(output: str, *, own_pid: int) -> list[dict[str, Any]]:
    owners = []
    for line in output.splitlines():
        pid_text, separator, command = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        command = command.strip() if separator else ""
        matches = sorted({
            pattern.pattern for pattern in HEAVY_PATTERNS
            if pattern.search(command.lower())
        })
        if pid != own_pid and command and matches:
            owners.append({"pid": pid, "command": command, "matched_patterns": matches})
    return sorted(owners, key=lambda row: row["pid"])


def active_heavy_owners() -> list[dict[str, Any]]:
    try:
        process = subprocess.run(
            [str(PS_PATH), "-axo", "pid=,command="],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if process.returncode:
            raise subprocess.SubprocessError(f"ps exited with status {process.returncode}")
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        return [{
            "pid": 0, "command": "owner-probe-unavailable",
            "matched_patterns": [], "probe_error": f"{type(exc).__name__}: {exc}",
        }]
    return owners_from_ps(process.stdout, own_pid=os.getpid())


def read_records(outbase: str) -> dict[str, dict[str, Any]]:
    path = ROOT / f"{outbase}.jsonl"
    records: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("config"), str):
                records[row["config"]] = row
    return records


def _number(row: dict[str, Any], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def pareto(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row for row in records.values()
        if all(_number(row, key) is not None for key in ("eff_bpw", "degr_pct"))
    ]
    return sorted([
        row for row in rows
        if not any(
            other is not row
            and _number(other, "eff_bpw") <= _number(row, "eff_bpw")
            and _number(other, "degr_pct") <= _number(row, "degr_pct")
            and (
                _number(other, "eff_bpw") < _number(row, "eff_bpw")
                or _number(other, "degr_pct") < _number(row, "degr_pct")
            )
            for other in rows
        )
    ], key=lambda row: (_number(row, "eff_bpw"), _number(row, "degr_pct")))


def parent_for(config: str) -> str | None:
    if config in BASE_FOR.values():
        return None
    return next((
        base for prefix, base in BASE_FOR.items()
        if config.startswith(prefix + "+") or config.startswith(prefix + "-")
    ), "mp-4a3f" if config.startswith("mp-4a3f") else None)


def classify(
    records: dict[str, dict[str, Any]], row: dict[str, Any], minimum_gain: float,
) -> dict[str, Any]:
    config = str(row.get("config", ""))
    parent = parent_for(config)
    degradation = _number(row, "degr_pct")
    base = _number(records.get(parent, {}), "degr_pct") if parent else None
    gain = base - degradation if base is not None and degradation is not None else None
    verdict = (
        "error" if "error" in row else
        "baseline" if gain is None else
        "excellent" if gain >= max(1.0, minimum_gain * 4) else
        "good" if gain >= minimum_gain else
        "small" if gain > 0 else "bad"
    )
    return {
        "config": config, "parent": parent, "eff_bpw": _number(row, "eff_bpw"),
        "degr_pct": degradation, "parent_degr_pct": base,
        "gain_pct_points": gain, "verdict": verdict,
    }


def conductor_step(outbase: str) -> dict[str, Any]:
    records = read_records(outbase)
    minimum = float(os.environ.get("AUTOPILOT_MIN_GAIN_PCT", "0.25"))
    frontier = pareto(records)
    promotions = [
        classify(records, row, minimum)
        for row in records.values()
        if "+dr" in str(row.get("config", ""))
    ]
    state = {
        "schema": SCHEMA,
        "observed_at": time.time(),
        "records": len(records),
        "pareto": [row.get("config") for row in frontier],
        "good": [row["config"] for row in promotions if row["verdict"] in {"good", "excellent"}],
        "branch": (
            "good-results-expand"
            if any(row["verdict"] in {"good", "excellent"} for row in promotions[-4:])
            else "mid-results-probe"
            if any(row["verdict"] == "small" for row in promotions[-4:])
            else "bad-results-prune"
            if any(row["verdict"] == "bad" for row in promotions[-4:])
            else "waiting-for-results"
        ),
    }
    common.atomic_write_json(ROOT / f"{outbase}_conductor_state.json", state)
    return state


def _conductor(args: argparse.Namespace) -> int:
    if args.once:
        print(json.dumps(conductor_step(args.outbase), indent=2, sort_keys=True))
        return 0
    pid_path = ROOT / f"{args.outbase}_conductor.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        while True:
            conductor_step(args.outbase)
            time.sleep(args.interval)
    finally:
        if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()


def _autopilot_module() -> Any:
    module = condense_profiles.archived_module("frontier_autopilot")

    def write_rearm(outbase: str) -> None:
        path = ROOT / f"{outbase}_inject.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "from tools.condense import frontier_runtime as _frontier_runtime\n"
            "_frontier_runtime.activate_autopilot(globals())\n",
            encoding="utf-8",
        )

    module._write_rearm = write_rearm
    return module


def activate_autopilot(namespace: dict[str, Any], rearm: bool = True) -> list[dict]:
    """Run the archived frontier policy while rearming through this runtime."""
    module = _autopilot_module()
    prior = Path.cwd()
    os.chdir(ROOT)
    try:
        return module.activate(namespace, rearm=rearm)
    finally:
        os.chdir(prior)


def _autopilot(args: argparse.Namespace) -> int:
    module = _autopilot_module()
    prior = Path.cwd()
    os.chdir(ROOT)
    try:
        records = module._read_records(args.outbase)
        candidates, skipped = module.plan(
            records, set(records), max_new=args.max_new,
        )
        state = {
            "records": len(records),
            "candidates": candidates,
            "skipped": skipped[:40],
        }
        state_path = ROOT / f"{args.outbase}_autopilot_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        if args.emit_inject and candidates:
            module._write_rearm(args.outbase)
            print(f"emitted {args.outbase}_inject.py with {len(candidates)} candidate(s)")
            return 10
        print(json.dumps(state, indent=2))
        return 0
    finally:
        os.chdir(prior)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profile")
    status = sub.add_parser("status")
    status.add_argument("--outbase", default="reports/cron/7b_frontier")
    conductor = sub.add_parser("conductor")
    conductor.add_argument("--outbase", default="reports/cron/7b_frontier")
    conductor.add_argument("--interval", type=int, default=180)
    conductor.add_argument("--once", action="store_true")
    autopilot = sub.add_parser("autopilot")
    autopilot.add_argument("--outbase", default="reports/cron/7b_frontier")
    autopilot.add_argument("--emit-inject", action="store_true")
    autopilot.add_argument(
        "--max-new", type=int,
        default=int(os.environ.get("AUTOPILOT_MAX_NEW", "8")),
    )
    verifier = sub.add_parser("verifier")
    verifier.add_argument("arguments", nargs=argparse.REMAINDER)
    launcher = sub.add_parser("launcher")
    launcher.add_argument("arguments", nargs=argparse.REMAINDER)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "profile":
        print(json.dumps({"schema": SCHEMA, **PROFILE}, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        print(json.dumps(conductor_step(args.outbase), indent=2, sort_keys=True))
        return 0
    if args.command == "conductor":
        return _conductor(args)
    if args.command == "autopilot":
        return _autopilot(args)
    if args.command == "verifier":
        return condense_profiles.invoke("frontier_verifier", args.arguments)
    if args.command == "launcher":
        return condense_profiles.invoke("ladder_launch", args.arguments)
    sample = {
        "mp-4a3f": {"config": "mp-4a3f", "eff_bpw": 4.0, "degr_pct": 10.0},
        "mp-4a3f+dr-r64": {
            "config": "mp-4a3f+dr-r64", "eff_bpw": 4.0, "degr_pct": 8.0,
        },
    }
    assert classify(sample, sample["mp-4a3f+dr-r64"], 0.25)["verdict"] == "excellent"
    assert pareto(sample)[0]["config"] == "mp-4a3f+dr-r64"
    assert owners_from_ps("1 python quantize-model\n", own_pid=2)
    print("frontier_runtime.py selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
