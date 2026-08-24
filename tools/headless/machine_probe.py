#!/usr/bin/env python3
"""MachineProbe / MemGate calibration (directive §3, §4, §5, §11, §25).

Measures the two quantities the directive insists are different, and never
conflates them:

  RESIDENT_RUNTIME_LIMIT  how many llama-server runtimes can stay resident
                          without pushing the box into swap
  ACTIVE_DECODE_LIMIT     how many of them may decode AT ONCE before aggregate
                          throughput stops improving

"4 runtimes fit in RAM" is not "4 runtimes should decode at once".  These are
measured by two different experiments here.

Residency is measured against *real* resident cost, not a paper estimate.
llama.cpp mmaps the weight file, so N runtimes of the SAME model share one set
of file-backed pages: the marginal cost of runtime N+1 is its KV cache and
compute buffers, not another copy of the weights.  The probe measures the
marginal footprint empirically by admitting one runtime at a time and watching
free RAM, compressed memory, and swap.

Concurrency is measured as an aggregate: k runtimes each decode the same
bounded workload simultaneously, and we report aggregate tok/s.  If aggregate
at k+1 is not meaningfully above aggregate at k, the GPU/unified-memory path is
saturated and the extra decoder buys nothing.

Everything is written to receipts/headless/MACHINE_GENOME.json with the exact
reprofile command, so a later reader can rerun it rather than trust it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_DEFAULT = os.path.expanduser(
    "~/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf")
MIB = 1024 * 1024
GIB = 1024 ** 3

# The GPU budget is the FIRST admission gate on Apple Silicon, not free RAM.
# See receipts/headless/GPU_MEMORY_GATE.json: mmap makes the marginal free-RAM
# cost of runtime N+1 look like ~1.5 GiB, but each process wraps the shared pages
# in its own MTLBuffers and Metal charges every process separately. Admitting on
# free RAM over-admits, and the failure surfaces mid-decode as
# kIOGPUCommandBufferCallbackErrorOutOfMemory rather than at spawn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from metal_budget import metal_device, wired_limit_override
except Exception:  # pragma: no cover - metal_budget is beside this file
    metal_device = None
    wired_limit_override = None


# ---------------------------------------------------------------- machine state

def vm_stat() -> dict:
    """Page-level memory truth.  macOS 'used' counts purgeable and compressed
    pages, so read the page counters directly instead of trusting Activity
    Monitor style numbers."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    d = {}
    for line in out.splitlines():
        mm = re.match(r'"?([A-Za-z][^:"]*)"?:\s+(\d+)', line.strip())
        if mm:
            d[mm.group(1).strip()] = int(mm.group(2)) * page
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True).stdout.strip())
    free = d.get("Pages free", 0) + d.get("Pages inactive", 0) + d.get("Pages speculative", 0)
    return {
        "total_bytes": total,
        "free_bytes": free,
        "wired_bytes": d.get("Pages wired down", 0),
        "compressed_bytes": d.get("Pages occupied by compressor", 0),
        "file_backed_bytes": d.get("File-backed pages", 0),
        "anonymous_bytes": d.get("Anonymous pages", 0),
        "swapins": d.get("Swapins", 0),
        "swapouts": d.get("Swapouts", 0),
    }


def swap_used_bytes() -> int:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
    if not m:
        return 0
    v = float(m.group(1))
    return int(v * (GIB if m.group(2) == "G" else MIB))


def memory_free_pct() -> float:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)", out)
    return float(m.group(1)) if m else -1.0


def snapshot() -> dict:
    s = vm_stat()
    s["swap_used_bytes"] = swap_used_bytes()
    s["free_pct"] = memory_free_pct()
    s["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return s


# ---------------------------------------------------------------- runtime

class Runtime:
    def __init__(self, idx: int, model: str, port: int, ctx: int, log_dir: Path):
        self.idx, self.model, self.port, self.ctx = idx, model, port, ctx
        self.log = open(log_dir / f"runtime-{idx}-{port}.log", "wb")
        self.proc = subprocess.Popen(
            ["llama-server", "-m", model, "--port", str(port), "-c", str(ctx),
             "-ngl", "999", "--host", "127.0.0.1", "-np", "1", "--jinja"],
            stdout=self.log, stderr=subprocess.STDOUT)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def ready(self, timeout: float = 600.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                time.sleep(1.5)
        return False

    def rss(self) -> int:
        out = subprocess.run(["ps", "-p", str(self.pid), "-o", "rss="],
                             capture_output=True, text=True).stdout.strip()
        return int(out) * 1024 if out else 0

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        try:
            self.log.close()
        except Exception:
            pass


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------- decode work

def decode_once(port: int, prompt: str, n: int, timeout: float) -> dict:
    """One bounded decode.  ignore_eos + fixed n so every arm does the SAME
    amount of GPU work — otherwise a chattier arm looks slower for free."""
    payload = {"prompt": prompt, "n_predict": n, "temperature": 0.0,
               "ignore_eos": True, "cache_prompt": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.time() - t0, 3)}
    wall = time.time() - t0
    tim = body.get("timings", {}) or {}
    return {
        "ok": True,
        "wall_s": round(wall, 3),
        "predicted_n": tim.get("predicted_n"),
        "predicted_ms": tim.get("predicted_ms"),
        "predicted_per_second": tim.get("predicted_per_second"),
        "prompt_n": tim.get("prompt_n"),
        "prompt_ms": tim.get("prompt_ms"),
        "prompt_per_second": tim.get("prompt_per_second"),
    }


def concurrent_decode(ports, prompt: str, n: int, timeout: float) -> dict:
    """Fire one decode at each port simultaneously; aggregate over the window in
    which they overlap.  Aggregate tok/s is computed against the WALL of the
    whole batch, not the sum of per-call rates, so stragglers are charged."""
    results = [None] * len(ports)

    def run(i, p):
        results[i] = decode_once(p, prompt, n, timeout)

    threads = [threading.Thread(target=run, args=(i, p)) for i, p in enumerate(ports)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results if r and r.get("ok")]
    tokens = sum(r.get("predicted_n") or 0 for r in ok)
    return {
        "k": len(ports),
        "batch_wall_s": round(wall, 3),
        "ok_count": len(ok),
        "total_tokens": tokens,
        "aggregate_tps": round(tokens / wall, 3) if wall > 0 else None,
        "per_runtime_tps": [r.get("predicted_per_second") for r in results],
        "per_runtime_wall_s": [r.get("wall_s") if r else None for r in results],
        "failures": [r for r in results if not (r and r.get("ok"))],
    }


# ---------------------------------------------------------------- main


def foreign_model_load() -> dict:
    """Model processes on this box that this probe did not start.

    A contended GPU makes every bandwidth number below a lie, so the probe
    refuses by default. `--allow-foreign-load` permits measuring anyway, but the
    permission is not silent: the foreign processes, and how busy their slots
    were, are recorded into the receipt so a later reader can see the numbers
    were taken under load rather than discovering it themselves."""
    import json as _json, re as _re, urllib.request as _u
    out = subprocess.run(
        ["bash", "-lc",
         "ps -eo pid,etime,pcpu,rss,command | grep -E 'llama-server|mlx_lm' | grep -v grep"],
        capture_output=True, text=True).stdout.strip()
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, etime, pcpu, rss, cmd = parts
        port = None
        m = _re.search(r"--port\s+(\d+)", cmd)
        if m:
            port = int(m.group(1))
        busy = None
        total = None
        if port:
            try:
                with _u.urlopen(f"http://127.0.0.1:{port}/slots", timeout=3) as r:
                    slots = _json.loads(r.read().decode("utf-8", "replace"))
                total = len(slots)
                busy = sum(1 for s in slots if s.get("is_processing"))
            except Exception:
                pass
        procs.append({"pid": int(pid), "etime": etime, "pcpu": float(pcpu),
                      "rss_bytes": int(rss) * 1024, "port": port,
                      "slots_total": total, "slots_processing": busy,
                      "command": cmd[:220]})
    return {"count": len(procs), "processes": procs,
            "any_actively_decoding": any((p.get("slots_processing") or 0) > 0 for p in procs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-runtimes", type=int, default=6)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu-headroom-frac", type=float, default=0.10,
                    help="fraction of the Metal working set left unallocated; without it Metal "
                         "OOMs mid-decode on transient command-buffer allocations")
    ap.add_argument("--per-runtime-overhead-gib", type=float, default=1.6,
                    help="KV cache plus compute buffers per runtime, above the model bytes")
    ap.add_argument("--reserve-gib", type=float, default=8.0,
                    help="stop admitting runtimes when free RAM would drop below this")
    ap.add_argument("--swap-ceiling-gib", type=float, default=2.0,
                    help="stop admitting if swap grows past this above the start value")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--allow-foreign-load", action="store_true",
                    help="measure even though model processes this probe did not "
                         "start are resident; their identity and slot activity are "
                         "recorded into the receipt")
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/Downloads/hawking-copy/receipts/headless"))
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"FAIL: model not found: {args.model}", file=sys.stderr)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "machine_probe_logs"
    log_dir.mkdir(exist_ok=True)

    # A stale model process makes every number below a lie.  Refuse rather than
    # silently measure a contended machine.
    foreign = foreign_model_load()
    if foreign["count"] and not args.allow_foreign_load:
        print("FAIL: model processes this probe did not start are resident. Every number below "
              "would be taken under an unknown load.\n"
              "Stop them, or re-run with --allow-foreign-load to measure anyway (the load is then "
              "recorded into the receipt):", file=sys.stderr)
        for pr in foreign["processes"]:
            print(f"  pid={pr['pid']} port={pr['port']} cpu={pr['pcpu']}% "
                  f"slots_busy={pr['slots_processing']}/{pr['slots_total']} {pr['command'][:100]}",
                  file=sys.stderr)
        return 3
    if foreign["count"]:
        print(f"WARNING: measuring with {foreign['count']} foreign model process(es) resident; "
              f"actively_decoding={foreign['any_actively_decoding']}", flush=True)

    # ---- GPU admission ceiling, computed before anything is spawned ----
    gpu = {}
    gpu_max = args.max_runtimes
    if metal_device is not None:
        dev = metal_device()
        model_bytes = os.path.getsize(args.model)
        per_runtime = model_bytes + int(args.per_runtime_overhead_gib * GIB)
        usable = int(dev["recommendedMaxWorkingSetSize"] * (1.0 - args.gpu_headroom_frac))
        gpu_max = max(1, usable // per_runtime)
        gpu = {
            "device": dev.get("name"),
            "source": dev.get("source"),
            "recommendedMaxWorkingSetSize_bytes": dev["recommendedMaxWorkingSetSize"],
            "recommendedMaxWorkingSetSize_gib": round(dev["recommendedMaxWorkingSetSize"] / GIB, 2),
            "maxBufferLength_gib": (round(dev["maxBufferLength"] / GIB, 2)
                                    if dev.get("maxBufferLength") else None),
            "wired_limit": wired_limit_override() if wired_limit_override else None,
            "per_runtime_gpu_bytes": per_runtime,
            "per_runtime_gpu_gib": round(per_runtime / GIB, 2),
            "headroom_frac": args.gpu_headroom_frac,
            "gpu_admission_ceiling": int(gpu_max),
        }
        print(f"GPU gate: working set {gpu['recommendedMaxWorkingSetSize_gib']} GiB, "
              f"{gpu['per_runtime_gpu_gib']} GiB per runtime, "
              f"{int(args.gpu_headroom_frac*100)}% headroom -> ceiling {gpu_max}", flush=True)
        if gpu_max < args.max_runtimes:
            print(f"  capping --max-runtimes {args.max_runtimes} -> {gpu_max} "
                  f"(exceeding it OOMs the GPU mid-decode, not at spawn)", flush=True)
            args.max_runtimes = int(gpu_max)
    else:
        print("WARNING: metal_budget unavailable; admitting on host memory alone, which "
              "over-admits on Apple Silicon", flush=True)

    prompt = ("You are a benchmark harness. Write a long, plain description of "
              "how a compiler lowers a for-loop into basic blocks.\n" * 4)

    start = snapshot()
    print(f"start: free={start['free_bytes']/GIB:.1f} GiB  swap={start['swap_used_bytes']/GIB:.2f} GiB "
          f"free_pct={start['free_pct']}", flush=True)

    runtimes: list[Runtime] = []
    admission = []
    residency_stop_reason = "reached --max-runtimes"

    try:
        # ---------- experiment 1: progressive residency admission ----------
        for i in range(args.max_runtimes):
            before = snapshot()
            port = free_port()
            rt = Runtime(i, args.model, port, args.ctx, log_dir)
            t0 = time.time()
            if not rt.ready():
                rt.stop()
                residency_stop_reason = f"runtime {i} failed to become ready"
                print(f"  runtime {i}: FAILED to start", flush=True)
                break
            load_s = time.time() - t0
            # one decode so KV and compute buffers are actually allocated;
            # an idle server under-reports its true resident cost
            warm = decode_once(port, prompt, 16, args.timeout)
            after = snapshot()

            marginal = before["free_bytes"] - after["free_bytes"]
            swap_growth = after["swap_used_bytes"] - start["swap_used_bytes"]
            row = {
                "index": i, "pid": rt.pid, "port": port,
                "load_s": round(load_s, 2),
                "rss_bytes": rt.rss(),
                "marginal_free_ram_cost_bytes": marginal,
                "marginal_free_ram_cost_gib": round(marginal / GIB, 3),
                "free_after_gib": round(after["free_bytes"] / GIB, 3),
                "swap_growth_gib": round(swap_growth / GIB, 3),
                "compressed_gib": round(after["compressed_bytes"] / GIB, 3),
                "free_pct_after": after["free_pct"],
                "warm_decode_tps": warm.get("predicted_per_second"),
            }
            admission.append(row)
            runtimes.append(rt)
            print(f"  runtime {i}: pid={rt.pid} port={port} load={load_s:.1f}s "
                  f"rss={row['rss_bytes']/GIB:.1f}GiB marginal_free={row['marginal_free_ram_cost_gib']:.2f}GiB "
                  f"free_after={row['free_after_gib']:.1f}GiB swap+={row['swap_growth_gib']:.2f}GiB "
                  f"tps={row['warm_decode_tps']}", flush=True)

            if after["free_bytes"] < args.reserve_gib * GIB:
                residency_stop_reason = (
                    f"free RAM {after['free_bytes']/GIB:.1f} GiB fell below "
                    f"--reserve-gib {args.reserve_gib}")
                break
            if swap_growth > args.swap_ceiling_gib * GIB:
                residency_stop_reason = (
                    f"swap grew {swap_growth/GIB:.2f} GiB past "
                    f"--swap-ceiling-gib {args.swap_ceiling_gib}")
                break

        resident_limit = len(runtimes)
        print(f"\nRESIDENT_RUNTIME_LIMIT = {resident_limit}  ({residency_stop_reason})\n", flush=True)

        # ---------- experiment 2: active-decode concurrency curve ----------
        # Runtimes stay resident throughout; only how many DECODE at once varies.
        curve = []
        for k in range(1, resident_limit + 1):
            ports = [rt.port for rt in runtimes[:k]]
            reps = []
            for rep in range(args.reps):
                r = concurrent_decode(ports, prompt, args.n_predict, args.timeout)
                r["rep"] = rep
                r["mem"] = {"free_gib": round(snapshot()["free_bytes"] / GIB, 2),
                            "swap_gib": round(swap_used_bytes() / GIB, 2)}
                # A rep where not every stream finished is not a slower
                # measurement of the same thing — it is a different, broken
                # thing. Including it silently is how an OOM gets reported as
                # "saturation".
                r["valid"] = (r["ok_count"] == k)
                reps.append(r)
                print(f"  k={k} rep{rep}: agg={r['aggregate_tps']} tok/s "
                      f"wall={r['batch_wall_s']}s ok={r['ok_count']}/{k}"
                      + ("" if r["valid"] else "   INVALID (stream failed)"), flush=True)
            valid = [r for r in reps if r["valid"]]
            if not valid:
                print(f"  k={k}: every rep had a failed stream — stopping the curve here "
                      f"rather than reporting a degraded number as a limit", flush=True)
                curve.append({"k": k, "aggregate_tps_min": None, "aggregate_tps_median": None,
                              "aggregate_tps_max": None, "spread_pct": None,
                              "all_reps_invalid": True, "reps": reps})
                break
            aggs = sorted(r["aggregate_tps"] for r in valid if r["aggregate_tps"])
            curve.append({
                "k": k,
                "aggregate_tps_min": aggs[0] if aggs else None,
                "aggregate_tps_median": aggs[len(aggs) // 2] if aggs else None,
                "aggregate_tps_max": aggs[-1] if aggs else None,
                "spread_pct": (round(100 * (aggs[-1] - aggs[0]) / aggs[0], 1)
                               if aggs and aggs[0] else None),
                "valid_reps": len(valid),
                "invalid_reps": len(reps) - len(valid),
                "reps": reps,
            })

        # knee: the largest k whose median aggregate beats k-1 by > noise.
        # noise is taken as the worst observed within-k spread, so we never call
        # a difference real that is smaller than the measurement's own jitter.
        curve = [c for c in curve if not c.get("all_reps_invalid")]
        spreads = [c["spread_pct"] for c in curve if c["spread_pct"] is not None]
        noise_pct = max(spreads) if spreads else 5.0
        active_limit, why = 1, "only one runtime measured"
        for i in range(1, len(curve)):
            prev = curve[i - 1]["aggregate_tps_median"]
            cur = curve[i]["aggregate_tps_median"]
            if not prev or not cur:
                continue
            gain = 100 * (cur - prev) / prev
            if gain > noise_pct:
                active_limit = curve[i]["k"]
                why = (f"k={curve[i]['k']} improved aggregate by {gain:.1f}% over "
                       f"k={curve[i-1]['k']}, above the {noise_pct:.1f}% measurement noise floor")
            else:
                why = (f"k={curve[i]['k']} gained only {gain:.1f}% over k={curve[i-1]['k']}, "
                       f"within the {noise_pct:.1f}% noise floor -> saturated")
                break

        single = curve[0]["aggregate_tps_median"] if curve else None
        best = max((c["aggregate_tps_median"] or 0) for c in curve) if curve else 0

        genome = {
            "schema": "hawking.headless.machine_genome.v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reprofile_command":
                f"python3 tools/headless/machine_probe.py --ctx {args.ctx} "
                f"--max-runtimes {args.max_runtimes} --n-predict {args.n_predict} "
                f"--reps {args.reps}",
            "machine": {
                "hw_model": subprocess.run(["sysctl", "-n", "hw.model"],
                                           capture_output=True, text=True).stdout.strip(),
                "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      capture_output=True, text=True).stdout.strip(),
                "ncpu": int(subprocess.run(["sysctl", "-n", "hw.ncpu"],
                                           capture_output=True, text=True).stdout.strip()),
                "mem_bytes": start["total_bytes"],
            },
            "runtime_identity": {
                "llama_server": subprocess.run(["which", "llama-server"],
                                               capture_output=True, text=True).stdout.strip(),
                "llama_version": subprocess.run(["llama-server", "--version"],
                                                capture_output=True, text=True).stderr.strip()[:200],
                "model_path": args.model,
                "model_size_bytes": os.path.getsize(args.model),
                "ctx": args.ctx,
                "n_predict_per_decode": args.n_predict,
                "decode_flags": "temperature=0 ignore_eos=true cache_prompt=false -np 1",
            },
            "gpu_gate": gpu,
            "RESIDENT_RUNTIME_LIMIT": resident_limit,
            "resident_limit_reason": residency_stop_reason,
            "resident_gate_params": {"reserve_gib": args.reserve_gib,
                                     "swap_ceiling_gib": args.swap_ceiling_gib},
            "admission": admission,
            "ACTIVE_DECODE_LIMIT": active_limit,
            "active_decode_reason": why,
            "measurement_noise_floor_pct": noise_pct,
            "single_decoder_tps": single,
            "best_aggregate_tps": best,
            "aggregate_scaling_vs_1": round(best / single, 4) if single else None,
            "concurrency_curve": curve,
            "foreign_load_at_start": foreign,
            "foreign_load_at_end": foreign_model_load(),
            "memory_start": start,
            "memory_end": snapshot(),
            "caveats": [
                "llama.cpp mmaps weights, so runtimes of the same model share file-backed "
                "pages; marginal_free_ram_cost is the true admission cost, RSS is not.",
                "aggregate_tps is tokens over the whole batch wall, so a straggler is "
                "charged against the batch rather than hidden by per-call averaging.",
                "ACTIVE_DECODE_LIMIT is judged against the worst observed within-k spread, "
                "so a gain smaller than the harness's own jitter is not counted as real.",
            ] + ([f"MEASURED UNDER FOREIGN LOAD: {foreign['count']} model process(es) this probe "
                  f"did not start were resident (actively decoding: "
                  f"{foreign['any_actively_decoding']}). Treat these figures as a floor, not a "
                  f"clean-room result, and re-run on an idle machine before citing them as a roof."]
                 if foreign["count"] else []),
        }
        (out / "MACHINE_GENOME.json").write_text(json.dumps(genome, indent=1))

        print("\n=== MACHINE GENOME ===")
        print(f"  RESIDENT_RUNTIME_LIMIT = {resident_limit}   ({residency_stop_reason})")
        print(f"  ACTIVE_DECODE_LIMIT    = {active_limit}   ({why})")
        print(f"  single decoder         = {single} tok/s")
        print(f"  best aggregate         = {best} tok/s  "
              f"({genome['aggregate_scaling_vs_1']}x vs 1)")
        for c in curve:
            print(f"    k={c['k']}  agg_median={c['aggregate_tps_median']}  "
                  f"spread={c['spread_pct']}%")
        print(f"\n-> {out/'MACHINE_GENOME.json'}")
        return 0
    finally:
        for rt in runtimes:
            rt.stop()
        # prove we left nothing behind — an orphan model child is the exact
        # failure this whole campaign is trying to eliminate
        leftover = subprocess.run(
            ["bash", "-lc", "ps -eo pid,command | grep llama-server | grep -v grep"],
            capture_output=True, text=True).stdout.strip()
        print(f"\ncleanup: leftover llama-server processes: {leftover or '(none)'}")


if __name__ == "__main__":
    sys.exit(main())
