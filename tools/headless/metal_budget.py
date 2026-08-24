#!/usr/bin/env python3
"""Metal working-set budget — the real admission constraint on Apple Silicon.

Free system RAM is the WRONG gate for admitting model runtimes on this machine,
and the mistake is easy to make because the page accounting looks encouraging.

Measured on this M3 Ultra with 96 GiB, admitting llama-server runtimes one at a
time at ctx 8192:

    runtime 0:  rss 19.1 GiB    marginal free-RAM cost  9.89 GiB
    runtime 1:  rss 19.1 GiB    marginal free-RAM cost  1.93 GiB
    runtime 2:  rss 19.1 GiB    marginal free-RAM cost  1.22 GiB
    ...
    runtime 5:  rss 19.2 GiB    marginal free-RAM cost  1.56 GiB
                                free RAM still 39.9 GiB, swap 0

By that measure a seventh runtime looks obviously safe. It is not. With six
resident, decode started failing:

    ggml_metal_synchronize: error: command buffer failed with status 5
    error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)

Because llama.cpp mmaps the weights, the *pages* really are shared between
processes — which is what the free-RAM delta is seeing. But each process wraps
those pages in its own MTLBuffers, and Metal charges every process separately
against the device's working-set budget. GPU accounting does not share.

    recommendedMaxWorkingSetSize = 77.76 GiB   (83494174720 bytes)
    maxBufferLength              = 58.32 GiB
    77.76 / 19.535 GiB per runtime = 3.98

Which is exactly the "memory-safe resident runtime capacity: 4" the operator
already had. That figure was right; free-RAM reasoning is what was wrong.

So: gate admission on the Metal working set, not on free RAM. RSS is a closer
proxy for GPU cost than the free-RAM delta is, which is the opposite of the usual
advice.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile

SWIFT = r'''
import Metal
import Foundation
guard let d = MTLCreateSystemDefaultDevice() else {
  print("{\"error\":\"no metal device\"}"); exit(1)
}
let out: [String: Any] = [
  "name": d.name,
  "hasUnifiedMemory": d.hasUnifiedMemory,
  "recommendedMaxWorkingSetSize": d.recommendedMaxWorkingSetSize,
  "maxBufferLength": d.maxBufferLength,
  "currentAllocatedSize": d.currentAllocatedSize,
]
let j = try! JSONSerialization.data(withJSONObject: out)
print(String(data: j, encoding: .utf8)!)
'''

GIB = 1024 ** 3


_DEVICE_CACHE: dict | None = None


def metal_device(force: bool = False) -> dict:
    """Ask Metal directly. Falls back to the documented ~75% heuristic if the
    Swift toolchain is absent, and SAYS which one it used — an estimate silently
    presented as a measurement is how a gate ends up meaning nothing.

    Cached, because every uncached call COMPILES AND RUNS a Swift source file:
    measured at 229 ms median, and MemGate.consider(refresh_metal=True) drove it
    on a path the latency ledger ranked third at 277.7 ms. What that spawn
    re-reads -- name, hasUnifiedMemory, recommendedMaxWorkingSetSize,
    maxBufferLength -- are constants of the machine and cannot change while this
    process lives. The one genuinely dynamic field, currentAllocatedSize, is read
    inside a throwaway subprocess and therefore describes THAT helper's device
    allocation, never this process's, so refreshing it was never buying the
    freshness it appeared to buy. Pass force=True to re-probe anyway.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None and not force:
        return dict(_DEVICE_CACHE)
    if shutil.which("swift"):
        with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
            f.write(SWIFT)
            path = f.name
        try:
            p = subprocess.run(["swift", path], capture_output=True, text=True, timeout=180)
            if p.returncode == 0 and p.stdout.strip():
                d = json.loads(p.stdout.strip().splitlines()[-1])
                d["source"] = "MTLDevice.recommendedMaxWorkingSetSize (measured)"
                d["currentAllocatedSize_scope"] = (
                    "helper subprocess, not this process — do not read as our allocation"
                )
                _DEVICE_CACHE = dict(d)
                return d
        except Exception:
            pass
        finally:
            os.unlink(path)
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True).stdout.strip())
    info = {
        "name": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
        "hasUnifiedMemory": True,
        "recommendedMaxWorkingSetSize": int(total * 0.75),
        "maxBufferLength": None,
        "currentAllocatedSize": None,
        "source": "ESTIMATE: 75% of hw.memsize — swift unavailable, so this is NOT measured",
    }
    _DEVICE_CACHE = dict(info)
    return info


def wired_limit_override() -> dict:
    """iogpu.wired_limit_mb raises the budget above the default when nonzero."""
    out = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                         capture_output=True, text=True).stdout.strip()
    try:
        mb = int(out)
    except ValueError:
        mb = 0
    return {"iogpu_wired_limit_mb": mb,
            "in_effect": mb > 0,
            "note": ("0 means the OS default applies. A nonzero value raises the working-set "
                     "budget and would change every admission number below.")}


def resident_capacity(model_bytes: int, budget_bytes: int, per_runtime_overhead_bytes: int,
                      headroom_frac: float) -> dict:
    per = model_bytes + per_runtime_overhead_bytes
    usable = int(budget_bytes * (1.0 - headroom_frac))
    n = max(0, usable // per)
    return {
        "per_runtime_gpu_bytes": per,
        "per_runtime_gpu_gib": round(per / GIB, 2),
        "budget_gib": round(budget_bytes / GIB, 2),
        "headroom_frac": headroom_frac,
        "usable_gib": round(usable / GIB, 2),
        "resident_capacity": int(n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"))
    ap.add_argument("--per-runtime-overhead-gib", type=float, default=1.6,
                    help="KV cache plus compute buffers per runtime, measured by machine_probe")
    ap.add_argument("--headroom-frac", type=float, default=0.10,
                    help="fraction of the budget left unallocated; Metal needs room for "
                         "transient command-buffer allocations or it OOMs mid-decode")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    dev = metal_device()
    override = wired_limit_override()
    model_bytes = os.path.getsize(args.model) if os.path.isfile(args.model) else None

    doc = {
        "schema": "hawking.headless.metal_budget.v1",
        "device": dev,
        "recommendedMaxWorkingSetSize_gib": round(dev["recommendedMaxWorkingSetSize"] / GIB, 2),
        "maxBufferLength_gib": (round(dev["maxBufferLength"] / GIB, 2)
                                if dev.get("maxBufferLength") else None),
        "wired_limit_override": override,
        "model": args.model,
        "model_bytes": model_bytes,
        "model_gib": round(model_bytes / GIB, 2) if model_bytes else None,
        "why_this_and_not_free_ram": (
            "llama.cpp mmaps the weights so the PAGES are shared between processes, which makes "
            "the marginal free-RAM cost of runtime N+1 look like ~1.5 GiB. But each process wraps "
            "those pages in its own MTLBuffers and Metal charges every process separately against "
            "the device working set. GPU accounting does not share. Admitting on free RAM "
            "over-admits and the failure surfaces as "
            "kIOGPUCommandBufferCallbackErrorOutOfMemory during decode, not at spawn."),
    }
    if model_bytes:
        doc["capacity"] = resident_capacity(
            model_bytes, dev["recommendedMaxWorkingSetSize"],
            int(args.per_runtime_overhead_gib * GIB), args.headroom_frac)

    if args.json:
        print(json.dumps(doc, indent=1))
        return 0

    print(f"device                {dev['name']}")
    print(f"source                {dev['source']}")
    print(f"working set budget    {doc['recommendedMaxWorkingSetSize_gib']} GiB")
    print(f"max single buffer     {doc['maxBufferLength_gib']} GiB")
    print(f"iogpu.wired_limit_mb  {override['iogpu_wired_limit_mb']} "
          f"({'OVERRIDE ACTIVE' if override['in_effect'] else 'OS default'})")
    if model_bytes:
        c = doc["capacity"]
        print(f"model                 {doc['model_gib']} GiB")
        print(f"per runtime on GPU    {c['per_runtime_gpu_gib']} GiB "
              f"(model + {args.per_runtime_overhead_gib} GiB KV/compute)")
        print(f"usable after {int(args.headroom_frac*100)}% headroom  {c['usable_gib']} GiB")
        print(f"RESIDENT CAPACITY     {c['resident_capacity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
