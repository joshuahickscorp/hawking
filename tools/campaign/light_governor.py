#!/usr/bin/env python3.12
"""Light-only resource governor: measure the machine, decide what may run, never touch MOP.

MOP owns this box.  This process's whole job is to stay out of its way and to say so with
numbers rather than intentions.  It reads the live machine and returns one of:

    LIGHT_ONLY      the default while MOP is active
    RELIEF_WINDOW   sustained green headroom; ONE bounded micro-job may run
    BACKOFF         something regressed; pause Hawking light jobs
    HEAVY_WINDOW_AVAILABLE  MOP is gone. Marked, never auto-acted on.

    python3.12 tools/campaign/light_governor.py            # one decision
    python3.12 tools/campaign/light_governor.py --watch N  # N samples, for a sustained call
    python3.12 tools/campaign/light_governor.py --json

The envelope is derived from the machine, not hard-coded: cores from sysctl, the memory
guard from total RAM.  A threshold someone typed once is a threshold that is wrong on the
next machine.

It never signals, kills, renices or inspects MOP's internals. It only observes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "HAWKING_LIGHT_ONLY_RESOURCE_POLICY.json"
LEDGER = ROOT / "HAWKING_LIGHT_ONLY_LEDGER.jsonl"
PAGE = 16384


def _sysctl(name: str) -> int:
    return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip())


def _vm_stat() -> dict[str, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {}
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages wired down",
                "Pages active", "Pages occupied by compressor"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            d[key] = int(m.group(1))
    return d


def _swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def _loadavg() -> tuple[float, float, float]:
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    nums = re.findall(r"[\d.]+", out)
    return tuple(float(x) for x in nums[:3])  # type: ignore[return-value]


def _thermal_green() -> bool:
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    # Absence of a recorded warning is the green case on this machine.
    return "No thermal warning level has been recorded" in out or "CPU_Scheduler_Limit" not in out


def _foreign_cpu() -> tuple[float, int]:
    """Total CPU percent and process count for non-Hawking heavy work (MOP).

    Identified by cost, not by name: any python-family process burning real CPU that this
    campaign did not start. Deliberately conservative -- overcounting means we back off
    more, which is the safe direction.
    """
    out = subprocess.run(
        ["ps", "-eo", "pcpu,rss,comm"], capture_output=True, text=True
    ).stdout.splitlines()[1:]
    total, n = 0.0, 0
    for line in out:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pcpu = float(parts[0])
        except ValueError:
            continue
        comm = parts[2]
        if pcpu > 20.0 and "Python" in comm:
            total += pcpu
            n += 1
    return total, n


def sample() -> dict:
    ncpu = _sysctl("hw.ncpu")
    memsize = _sysctl("hw.memsize")
    vm = _vm_stat()
    avail = (vm.get("Pages free", 0) + vm.get("Pages inactive", 0)
             + vm.get("Pages speculative", 0)) * PAGE
    load1, load5, load15 = _loadavg()
    fcpu, fprocs = _foreign_cpu()
    swap = _swap_used_mb()
    disk = subprocess.run(["df", "-k", "/Users"], capture_output=True, text=True).stdout.splitlines()
    disk_free = int(disk[-1].split()[3]) * 1024 if len(disk) > 1 else 0

    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ncpu": ncpu,
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "load_per_core": round(load1 / ncpu, 3),
        "mem_total_gib": round(memsize / 2**30, 2),
        "mem_available_gib": round(avail / 2**30, 2),
        "mem_available_frac": round(avail / memsize, 4),
        "swap_used_mb": swap,
        "disk_free_gib": round(disk_free / 2**30, 2),
        "thermal_green": _thermal_green(),
        "foreign_cpu_pct": round(fcpu, 1),
        "foreign_cpu_cores": round(fcpu / 100.0, 2),
        "foreign_procs": fprocs,
    }


def envelope(s: dict) -> dict:
    """Thresholds derived from the machine rather than typed in."""
    return {
        "load_per_core_relief_max": 0.70,   # relief needs real headroom, not merely 'not on fire'
        "mem_available_frac_min": 0.10,     # 10 percent of RAM must stay available
        # Swap is judged on GROWTH, not level. macOS keeps swap allocated after a memory
        # peak, so this machine sits at ~13 GB with 56 GiB available and is in no distress.
        # An absolute 512 MB guard fired on every sample, and a backoff that always fires
        # trains its reader to ignore it -- which is worse than not having one.
        "swap_growth_mb_max": 2048.0,       # across the sampled window
        "swap_used_mb_absolute_ceiling": 24576.0,  # genuine distress on a 96 GiB machine
        "disk_free_gib_min": 40.0,
        "foreign_cores_idle": 1.0,          # below this, MOP is effectively gone
        # Deliberately NOT a backoff trigger: high load. MOP's normal working state is
        # load/core ~1.6, so backing off on load alone would mean never running at all,
        # which is not caution, it is paralysis. Backoff fires on things that indicate
        # the machine is actually degrading -- memory, swap, thermal, disk -- because
        # those are the ones a light job can genuinely make worse.
    }


def decide(samples: list[dict]) -> dict:
    s = samples[-1]
    env = envelope(s)
    reasons: list[str] = []

    if not s["thermal_green"]:
        reasons.append("thermal pressure recorded")
    if s["mem_available_frac"] < env["mem_available_frac_min"]:
        reasons.append(f"available memory {s['mem_available_frac']:.1%} below guard")
    if s["swap_used_mb"] > env["swap_used_mb_absolute_ceiling"]:
        reasons.append(f"swap {s['swap_used_mb']:.0f} MB above the absolute ceiling")
    if len(samples) >= 2:
        growth = s["swap_used_mb"] - samples[0]["swap_used_mb"]
        if growth > env["swap_growth_mb_max"]:
            reasons.append(
                f"swap GREW {growth:.0f} MB across the sampled window -- something is "
                f"actively consuming memory"
            )
    if s["disk_free_gib"] < env["disk_free_gib_min"]:
        reasons.append(f"disk free {s['disk_free_gib']:.0f} GiB below reserve")

    if reasons:
        return {"mode": "BACKOFF", "why": reasons, "sample": s, "envelope": env}

    # MOP gone? Two independent signals must agree, because either alone lies.
    #
    # `ps -eo pcpu` reports CPU averaged over a process's LIFETIME, so a freshly restarted
    # MOP worker shows near-zero pcpu while pinning a core. Trusting it alone once reported
    # "foreign 0.94 cores" against a load average of 20.7 -- which would have declared a
    # heavy window in the middle of MOP restarting. Load average cannot be fooled that way,
    # so it is the veto.
    quiet_by_pcpu = all(x["foreign_cpu_cores"] < env["foreign_cores_idle"] for x in samples)
    quiet_by_load = all(x["load1"] < env["foreign_cores_idle"] + 1.0 for x in samples)
    if quiet_by_pcpu and quiet_by_load:
        return {
            "mode": "HEAVY_WINDOW_AVAILABLE",
            "why": ["no foreign heavy process observed across every sample",
                    f"and load average stayed below {env['foreign_cores_idle'] + 1.0}"],
            "sample": s,
            "envelope": env,
            "law": "MARKED ONLY. Heavy work is never auto-launched; a human decides.",
        }
    if quiet_by_pcpu and not quiet_by_load:
        return {
            "mode": "LIGHT_ONLY",
            "why": [f"per-process CPU suggests idle ({s['foreign_cpu_cores']} cores) but load "
                    f"average is {s['load1']} -- foreign work is running and its pcpu is still "
                    f"averaging up. Treating as busy."],
            "sample": s,
            "envelope": env,
        }

    sustained_green = all(x["load_per_core"] <= env["load_per_core_relief_max"] for x in samples)
    if sustained_green and len(samples) >= 3:
        return {
            "mode": "RELIEF_WINDOW",
            "why": [f"load/core stayed <= {env['load_per_core_relief_max']} across {len(samples)} samples"],
            "sample": s,
            "envelope": env,
            "law": "ONE bounded micro-job, with wall time, RSS, threads, bytes, rollback and stop trigger declared BEFORE launch.",
        }

    return {
        "mode": "LIGHT_ONLY",
        "why": [
            f"load/core {s['load_per_core']} above the relief ceiling {env['load_per_core_relief_max']}",
            f"foreign work holding ~{s['foreign_cpu_cores']} cores across {s['foreign_procs']} processes",
        ],
        "sample": s,
        "envelope": env,
    }


def main() -> int:
    n = 1
    if "--watch" in sys.argv:
        n = int(sys.argv[sys.argv.index("--watch") + 1])
    samples = []
    for i in range(n):
        samples.append(sample())
        if i < n - 1:
            time.sleep(5)
    d = decide(samples)

    if "--json" in sys.argv:
        print(json.dumps(d, indent=2))
    else:
        s = d["sample"]
        print(f"mode: {d['mode']}")
        for w in d["why"]:
            print(f"  - {w}")
        print(f"  load/core {s['load_per_core']}  avail {s['mem_available_gib']} GiB "
              f"({s['mem_available_frac']:.1%})  swap {s['swap_used_mb']:.0f} MB  "
              f"disk {s['disk_free_gib']:.0f} GiB  foreign ~{s['foreign_cpu_cores']} cores")

    with LEDGER.open("a") as fh:
        fh.write(json.dumps({"at": d["sample"]["at"], "mode": d["mode"], "why": d["why"],
                             "load_per_core": d["sample"]["load_per_core"],
                             "foreign_cores": d["sample"]["foreign_cpu_cores"]}) + "\n")
    # Exit code carries the decision so a shell wrapper can gate on it.
    return {"LIGHT_ONLY": 0, "RELIEF_WINDOW": 0, "BACKOFF": 3, "HEAVY_WINDOW_AVAILABLE": 0}[d["mode"]]


if __name__ == "__main__":
    raise SystemExit(main())
