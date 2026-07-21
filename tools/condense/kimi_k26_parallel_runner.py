#!/usr/bin/env python3.12
"""Run one Kimi heavy lane with bounded light/CPU lanes and seal utilization evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_final_chapter_manager as chapter  # noqa: E402


MIN_AVAILABLE_MEMORY = 24 * 1024**3
MAX_SWAP_GROWTH = 512 * 1024**2


def memory_available() -> int:
    output = subprocess.run(["/usr/bin/memory_pressure", "-Q"], text=True,
                            capture_output=True, check=False).stdout
    match = re.search(r"free percentage:\s*(\d+)%", output)
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return int(total * int(match.group(1)) / 100) if match else 0


def swap_used() -> int:
    output = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True,
                            capture_output=True, check=False).stdout
    match = re.search(r"used = ([0-9.]+)([MG])", output)
    if not match:
        return 0
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * scale)


def thermal_green() -> bool:
    output = subprocess.run(["/usr/bin/pmset", "-g", "therm"], text=True,
                            capture_output=True, check=False).stdout.lower()
    return "warning level" not in output or "no thermal warning" in output


def gpu_utilization() -> float | None:
    output = subprocess.run(["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                            text=True, capture_output=True, check=False).stdout
    values = [float(value) for value in re.findall(r'"Device Utilization %"=(\d+)', output)]
    return max(values) if values else None


def process_sample(pid: int) -> tuple[float, int] | None:
    result = subprocess.run(["/bin/ps", "-o", "%cpu=,rss=", "-p", str(pid)], text=True,
                            capture_output=True, check=False)
    fields = result.stdout.split()
    if len(fields) < 2:
        return None
    return float(fields[0]), int(fields[1]) * 1024


def validate_plan(plan: dict[str, Any]) -> None:
    rows = plan.get("lanes", [])
    heavy_count = sum(row.get("lane") == "HEAVY_LANE" for row in rows)
    if heavy_count != 1:
        raise RuntimeError(f"parallel plan requires exactly one heavy lane; found {heavy_count}")
    for row in rows:
        if not row.get("command") or not isinstance(row["command"], list):
            raise RuntimeError(f"invalid lane command: {row}")


def run(repo: Path, plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    guard = chapter.audit(repo)
    if guard["status"] != "PASS":
        raise RuntimeError(f"parallel preflight failed: {guard['failures']}")
    max_write = max((int(row.get("maximum_atomic_write_bytes", 0))
                     for row in plan["lanes"]), default=0)
    if guard["resources"]["free_disk_bytes"] - max_write <= chapter.FLOOR_BYTES:
        raise RuntimeError("parallel plan would cross disk floor plus atomic-write requirement")
    if memory_available() < MIN_AVAILABLE_MEMORY or not thermal_green():
        raise RuntimeError("parallel preflight resource floor is not green")
    start_swap = swap_used()
    group_started = chapter.f1.now()
    output_root = chapter.RUNTIME / "final_parallel"
    output_root.mkdir(parents=True, exist_ok=True)
    processes: list[dict[str, Any]] = []
    for row in plan["lanes"]:
        log_path = output_root / f"{plan['execution_id']}_{row['name']}.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            [str(item) for item in row["command"]], cwd=repo,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        processes.append({**row, "process": process, "log_handle": handle,
                          "log_path": log_path, "pid": process.pid,
                          "started_at": chapter.f1.now(),
                          "started_wall": time.time(), "cpu_samples": [],
                          "rss_samples": [], "gpu_samples": [],
                          "ended_wall": None, "ended_at": None,
                          "return_code": None})
    backed_off = []
    while any(item["process"].poll() is None for item in processes):
        current_swap = swap_used()
        available = memory_available()
        gpu = gpu_utilization()
        disk_green = shutil.disk_usage(Path.home()).free - max_write > chapter.FLOOR_BYTES
        contention = (available < MIN_AVAILABLE_MEMORY or
                      current_swap - start_swap > MAX_SWAP_GROWTH or
                      not thermal_green() or not disk_green)
        for item in processes:
            return_code = item["process"].poll()
            if return_code is not None:
                if item["ended_wall"] is None:
                    item["ended_wall"] = time.time()
                    item["ended_at"] = chapter.f1.now()
                    item["return_code"] = return_code
                continue
            sample = process_sample(item["pid"])
            if sample:
                item["cpu_samples"].append(sample[0])
                item["rss_samples"].append(sample[1])
            if gpu is not None:
                item["gpu_samples"].append(gpu)
            if (contention and item["lane"] != "HEAVY_LANE" and
                    item["name"] not in backed_off):
                item["process"].terminate()
                backed_off.append(item["name"])
        time.sleep(0.5)
    ended = chapter.f1.now()
    lane_records = []
    for item in processes:
        if item["ended_wall"] is None:
            item["ended_wall"] = time.time()
            item["ended_at"] = chapter.f1.now()
            item["return_code"] = item["process"].returncode
        item["log_handle"].close()
        duration = item["ended_wall"] - item["started_wall"]
        log_bytes = item["log_path"].stat().st_size
        record = chapter.append_parallel({
            "event": "LANE_COMPLETE",
            "execution_id": plan["execution_id"], "lane_name": item["name"],
            "lane": item["lane"], "task": item["task"], "pid": item["pid"],
            "command": item["command"], "started_at": item["started_at"],
            "ended_at": item["ended_at"], "duration_seconds": duration,
            "exit_code": item["return_code"],
            "cpu_percent_peak": max(item["cpu_samples"], default=0.0),
            "cpu_percent_mean": (sum(item["cpu_samples"]) / len(item["cpu_samples"])
                                 if item["cpu_samples"] else 0.0),
            "resident_memory_peak_bytes": max(item["rss_samples"], default=0),
            "gpu_device_utilization_peak_percent": max(item["gpu_samples"], default=None),
            "gpu_measurement_scope": "system IOAccelerator counter; heavy lane owns Metal",
            "logical_disk_read_bytes": int(item.get("logical_disk_read_bytes", 0)),
            "logical_disk_write_bytes": int(item.get("logical_disk_write_bytes", 0)) + log_bytes,
            "swap_delta_group_bytes": swap_used() - start_swap,
            "backed_off_for_contention": item["name"] in backed_off,
            "contention_effect": item.get("contention_effect", "MEASURE_FROM_TASK_ARTIFACT"),
            "log_path": str(item["log_path"]), "log_bytes": log_bytes,
        })
        lane_records.append(record)
    after = chapter.audit(repo)
    group = chapter.append_parallel({
        "event": "PARALLEL_EXECUTION_COMPLETE", "execution_id": plan["execution_id"],
        "started_at": group_started, "ended_at": ended,
        "lane_count": len(lane_records),
        "heavy_lane_count": sum(item["lane"] == "HEAVY_LANE" for item in processes),
        "backed_off_lanes": backed_off,
        "swap_delta_bytes": swap_used() - start_swap,
        "memory_available_minimum_policy_bytes": MIN_AVAILABLE_MEMORY,
        "guard_after_status": after["status"], "guard_after_failures": after["failures"],
        "lane_record_seals": [record["seal_sha256"] for record in lane_records],
        "decision": "PARALLELISM_DEMONSTRATED" if (
            not backed_off and all(item["return_code"] == 0 for item in processes)
        ) else "BACKOFF_OR_FAILURE_REVIEW_REQUIRED",
    })
    return {"group": group, "lanes": lane_records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo.resolve(strict=True), args.plan.resolve(strict=True))
    print(json.dumps({
        "status": "PASS" if result["group"]["decision"] == "PARALLELISM_DEMONSTRATED" else "FAIL",
        "decision": result["group"]["decision"],
        "execution_id": result["group"]["execution_id"],
        "seal_sha256": result["group"]["seal_sha256"],
    }, sort_keys=True))
    return 0 if result["group"]["decision"] == "PARALLELISM_DEMONSTRATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
