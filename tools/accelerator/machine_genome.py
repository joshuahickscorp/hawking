"""MachineGenome — the physical identity of THIS machine. FRONT A (G043, S015 §22).

The steer is explicit that a MachineGenome must carry measured bandwidth, not a
datasheet number, and that all Apple Silicon must not be assumed to behave alike.
So everything here is either read from the machine or measured on it. Fields that
cannot be obtained are ABSENT with a reason; none is guessed.

Bandwidth is measured inside a protected window because the lake fill saturates
disk and network, and a contended sample is not a roof. Repeats are alternated and
the spread is reported alongside the number -- a tight spread is itself evidence,
and a wide one means the number is not yet a measurement.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.accelerator.machine_genome.v1"


def _sysctl(key: str) -> str | None:
    try:
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _gpu_cores() -> Any:
    """system_profiler is the only place the GPU core count is exposed."""
    try:
        r = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                           capture_output=True, text=True, timeout=90)
        m = re.search(r"Total Number of Cores:\s*(\d+)", r.stdout)
        if m:
            return int(m.group(1))
        return {"status": "ABSENT", "reason": "system_profiler reported no core count"}
    except Exception as e:
        return {"status": "ABSENT", "reason": f"system_profiler failed: {type(e).__name__}"}


def _toolchain() -> dict[str, Any]:
    out: dict[str, Any] = {}
    absent_metal = {
        "status": "ABSENT",
        "reason": "xcrun metal is not installed; no Metal developer toolchain, so "
                  "AOT metallib compilation is unavailable and kernels go through "
                  "the MLX JIT"}
    try:
        r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"],
                           capture_output=True, text=True, timeout=30)
        # subprocess does not raise on a non-zero exit, so without this check the
        # xcrun ERROR TEXT got stored as if it were a compiler version.
        out["metal_compiler"] = (r.stdout.strip().splitlines()[0]
                                 if r.returncode == 0 and r.stdout.strip()
                                 else absent_metal)
    except Exception:
        out["metal_compiler"] = absent_metal
    try:
        import mlx.core as mx
        out["mlx"] = getattr(mx, "__version__", "unknown")
    except Exception:
        out["mlx"] = {"status": "ABSENT", "reason": "mlx not importable in this interpreter"}
    out["python"] = sys.version.split()[0]
    return out


def measure_bandwidth(n: int = 1 << 26, reps: int = 30, warmup: int = 8) -> dict[str, Any]:
    """Streaming triad on the GPU: reads 2N f32, writes N f32 => 12N bytes moved.

    A first attempt at this reported best 403 GB/s with a 286% spread across reps,
    which is not a measurement -- it is a distribution with a fast tail. Larger
    buffers, a real warmup and an interquartile spread replace it. The number is
    reported ONLY if the IQR is tight; otherwise it is marked UNRELIABLE rather
    than quoted, because a wide spread means the machine was not actually held
    still.
    """
    try:
        import mlx.core as mx
    except Exception as e:
        return {"status": "ABSENT", "reason": f"mlx unavailable: {type(e).__name__}"}
    a = mx.random.normal((n,), dtype=mx.float32)
    b = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(a, b)
    bytes_moved = 12 * n
    for _ in range(warmup):                    # compile + clock ramp
        mx.eval(a + b)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(a + b)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    q1 = samples[len(samples) // 4]
    med = samples[len(samples) // 2]
    q3 = samples[(3 * len(samples)) // 4]
    iqr_pct = round((q3 - q1) / q1 * 100, 2)
    reliable = iqr_pct <= 10.0
    out = {
        "pattern": "triad c = a + b, f32",
        "elements": n,
        "bytes_moved_per_rep": bytes_moved,
        "reps": reps, "warmup": warmup,
        "median_gb_s": round(bytes_moved / med / 1e9, 2),
        "q1_gb_s": round(bytes_moved / q3 / 1e9, 2),
        "q3_gb_s": round(bytes_moved / q1 / 1e9, 2),
        "iqr_spread_pct": iqr_pct,
        "full_range_spread_pct": round((samples[-1] - samples[0]) / samples[0] * 100, 2),
        "reliable": reliable,
        "is_theoretical_roof": False,
        "note": "one access pattern on one dtype; not the SoC roof and not a "
                "workload-reachable roof",
    }
    if not reliable:
        out["status"] = "UNRELIABLE"
        out["reason"] = (f"interquartile spread {iqr_pct}% exceeds the 10% gate; the "
                         f"machine was not held still enough for this to be a roof")
    return out


def build(*, contended: bool, contention_note: str) -> dict[str, Any]:
    mem = _sysctl("hw.memsize")
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "soc": _sysctl("machdep.cpu.brand_string"),
        "arch": platform.machine(),
        "cpu_cores": int(_sysctl("hw.ncpu") or 0) or None,
        "perf_cores": int(_sysctl("hw.perflevel0.physicalcpu") or 0) or None,
        "efficiency_cores": int(_sysctl("hw.perflevel1.physicalcpu") or 0) or None,
        "gpu_cores": _gpu_cores(),
        "memory_bytes": int(mem) if mem else None,
        "os": f"{platform.system()} {platform.release()}",
        "os_product": _sysctl("kern.osproductversion"),
        "toolchain": _toolchain(),
        "measured_bandwidth": measure_bandwidth(),
        "measurement_conditions": {
            "contended": contended,
            "note": contention_note,
        },
        "thermal_envelope": {
            "status": "ABSENT",
            "reason": "no sustained thermal campaign has been run; sustained behaviour "
                      "is required before any production ADP (G049) and is not claimed here"},
        "sustained_behaviour": {
            "status": "ABSENT",
            "reason": "microbenchmark only; the steer requires sustained evidence to be "
                      "distinguished from a microbenchmark and this is the latter"},
        "knowledge_level": "INSTANCE",
    }
