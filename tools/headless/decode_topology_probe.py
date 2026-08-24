#!/usr/bin/env python3
"""Decode topology probe — the operator's stated secondary target:

  "even be able to run parellel models like ports without aggregate being
   similar if possibel"

Today HCLI spawns N *separate* llama-server processes on N ports and decodes in
all of them at once.  The prior calibration measured 2 concurrent decoders at
only 1.208x the aggregate of 1.  The question this probe answers is whether that
ceiling is physics or topology.

The hypothesis is that it is topology.  Decode on this box is dominated by
streaming the weights once per token:

    19.535 GB of Q5_K weights x ~21 tok/s = ~410 GB/s for ONE decoder,
    against an M3 Ultra roof of ~819 GB/s.

N independent processes each stream the full weight set per token, so they
contend for the same finite bandwidth and aggregate saturates almost
immediately.  One process with N continuous-batching slots streams the weights
ONCE per step and applies them to N sequences, so the dominant cost is
amortised and aggregate should scale far closer to N.

    PROCESS topology:  N servers x 1 slot   -> weight traffic scales with N
    SLOT topology:     1 server x N slots   -> weight traffic is shared

Both arms do identical total GPU work: N sequences x --n-predict tokens, same
prompt, temperature 0, ignore_eos so every sequence emits exactly n tokens.
Arms alternate rep by rep so page-cache and thermal drift cancel.

Writes receipts/headless/DECODE_TOPOLOGY.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

GIB = 1024 ** 3

# The PROCESS arm is bounded by the Metal working set, because each process
# charges the full model against the device budget independently (see
# receipts/headless/GPU_MEMORY_GATE.json). The SLOT arm is not: one process
# charges the weights once and adds only KV per slot. Running the process arm
# past its ceiling does not measure a slower topology, it measures an OOM.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from metal_budget import metal_device
except Exception:  # pragma: no cover
    metal_device = None

MODEL_DEFAULT = os.path.expanduser(
    "~/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def swap_gib() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v if m.group(2) == "G" else v / 1024.0


def free_gib() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    tot = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        mm = re.search(rf'{key}:\s+(\d+)', out)
        if mm:
            tot += int(mm.group(1)) * page
    return tot / GIB


class Server:
    def __init__(self, model: str, port: int, ctx_total: int, slots: int, log: Path):
        self.port, self.slots = port, slots
        self.fh = open(log, "wb")
        # -c is the TOTAL context split across slots, so each slot gets
        # ctx_total/slots.  Sizing it as slots*per_slot keeps per-slot context
        # identical between the two topologies — otherwise the slot arm would be
        # quietly running with a smaller window and the comparison would be rigged.
        self.proc = subprocess.Popen(
            ["llama-server", "-m", model, "--port", str(port),
             "-c", str(ctx_total), "-np", str(slots),
             "-ngl", "999", "--host", "127.0.0.1", "--cont-batching"],
            stdout=self.fh, stderr=subprocess.STDOUT)

    def ready(self, timeout: float = 900.0) -> bool:
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

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        try:
            self.fh.close()
        except Exception:
            pass


def decode(port: int, prompt: str, n: int, timeout: float) -> dict:
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
    t = body.get("timings", {}) or {}
    return {"ok": True, "wall_s": round(time.time() - t0, 3),
            "predicted_n": t.get("predicted_n"),
            "predicted_per_second": t.get("predicted_per_second"),
            "prompt_n": t.get("prompt_n"),
            "prompt_per_second": t.get("prompt_per_second")}


def fan(ports_repeated, prompt, n, timeout) -> dict:
    """One in-flight decode per entry in ports_repeated (the same port may appear
    more than once — that is exactly how the slot topology is exercised)."""
    res = [None] * len(ports_repeated)

    def run(i, p):
        res[i] = decode(p, prompt, n, timeout)

    ths = [threading.Thread(target=run, args=(i, p)) for i, p in enumerate(ports_repeated)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ok = [r for r in res if r and r.get("ok")]
    tok = sum(r.get("predicted_n") or 0 for r in ok)
    return {"n_streams": len(ports_repeated), "batch_wall_s": round(wall, 3),
            "ok": len(ok), "tokens": tok,
            "aggregate_tps": round(tok / wall, 3) if wall > 0 else None,
            "per_stream_tps": [r.get("predicted_per_second") if r else None for r in res],
            "failures": [r for r in res if not (r and r.get("ok"))]}


def kill_strays() -> str:
    return subprocess.run(
        ["bash", "-lc", "ps -eo pid,command | grep -E 'llama-server' | grep -v grep"],
        capture_output=True, text=True).stdout.strip()



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
    ap.add_argument("--per-slot-ctx", type=int, default=4096)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--streams", default="1,2,3,4",
                    help="comma list of concurrent stream counts to test")
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument("--gpu-headroom-frac", type=float, default=0.10,
                    help="fraction of the Metal working set left unallocated")
    ap.add_argument("--allow-foreign-load", action="store_true",
                    help="measure even though model processes this probe did not "
                         "start are resident; their identity and slot activity are "
                         "recorded into the receipt")
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/Downloads/hawking-copy/receipts/headless"))
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"FAIL: model missing: {args.model}", file=sys.stderr)
        return 2
    foreign = foreign_model_load()
    if foreign["count"] and not args.allow_foreign_load:
        print("FAIL: model processes this probe did not start are resident. The whole point of "
              "this probe is a bandwidth comparison, and an unknown co-tenant invalidates it.\n"
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "topology_logs"
    logs.mkdir(exist_ok=True)

    stream_counts = [int(x) for x in args.streams.split(",")]
    kmax = max(stream_counts)

    gpu = {}
    process_counts = list(stream_counts)
    if metal_device is not None:
        dev = metal_device()
        per_proc = os.path.getsize(args.model) + int(1.6 * GIB)
        usable = int(dev["recommendedMaxWorkingSetSize"] * (1.0 - args.gpu_headroom_frac))
        proc_ceiling = max(1, usable // per_proc)
        gpu = {
            "device": dev.get("name"), "source": dev.get("source"),
            "recommendedMaxWorkingSetSize_gib": round(dev["recommendedMaxWorkingSetSize"] / GIB, 2),
            "per_process_gpu_gib": round(per_proc / GIB, 2),
            "headroom_frac": args.gpu_headroom_frac,
            "process_arm_ceiling": int(proc_ceiling),
            "slot_arm_not_capped_because": (
                "one process charges the weights against the device budget once; extra slots "
                "add only KV, so the slot arm is not bounded by the same ceiling"),
        }
        process_counts = [k for k in stream_counts if k <= proc_ceiling]
        dropped = [k for k in stream_counts if k > proc_ceiling]
        print(f"GPU gate: working set {gpu['recommendedMaxWorkingSetSize_gib']} GiB, "
              f"{gpu['per_process_gpu_gib']} GiB per process -> process arm ceiling "
              f"{proc_ceiling}", flush=True)
        if dropped:
            print(f"  process arm will SKIP k={dropped} (would exceed the Metal working set "
                  f"and measure an OOM, not a topology)", flush=True)
        gpu["process_counts_measured"] = process_counts
        gpu["process_counts_skipped"] = dropped
    kproc = max(process_counts) if process_counts else 1
    prompt = ("Explain, in ordinary prose and at length, how a compiler turns a "
              "for-loop into basic blocks and then into machine code.\n")

    model_bytes = os.path.getsize(args.model)
    results = {"process": {}, "slot": {}}
    servers: list[Server] = []

    try:
        # ---------------- arm A: N processes x 1 slot ----------------
        print(f"arm PROCESS: starting {kproc} servers, 1 slot each "
              f"(ctx {args.per_slot_ctx} per slot)", flush=True)
        proc_ports = []
        for i in range(kproc):
            p = free_port()
            s = Server(args.model, p, args.per_slot_ctx, 1, logs / f"proc-{i}-{p}.log")
            if not s.ready():
                print(f"FAIL: process-arm server {i} never became ready", file=sys.stderr)
                s.stop()
                return 4
            servers.append(s)
            proc_ports.append(p)
            print(f"  server {i} pid={s.proc.pid} port={p} free={free_gib():.1f}GiB "
                  f"swap={swap_gib():.2f}GiB", flush=True)
        # warm every server once so no arm pays a first-touch penalty
        for p in proc_ports:
            decode(p, prompt, 8, args.timeout)

        for k in process_counts:
            reps = []
            for rep in range(args.reps):
                r = fan(proc_ports[:k], prompt, args.n_predict, args.timeout)
                r["rep"] = rep
                r["free_gib"] = round(free_gib(), 2)
                r["swap_gib"] = round(swap_gib(), 2)
                reps.append(r)
                print(f"  PROCESS k={k} rep{rep}: agg={r['aggregate_tps']} tok/s "
                      f"wall={r['batch_wall_s']}s ok={r['ok']}/{k}", flush=True)
            results["process"][k] = reps

        for s in servers:
            s.stop()
        servers = []
        left = kill_strays()
        print(f"  process arm torn down; leftover: {left or '(none)'}", flush=True)

        # ---------------- arm B: 1 process x N slots ----------------
        print(f"\narm SLOT: starting 1 server with {kmax} slots "
              f"(-c {args.per_slot_ctx * kmax}, so {args.per_slot_ctx} per slot)", flush=True)
        p = free_port()
        s = Server(args.model, p, args.per_slot_ctx * kmax, kmax, logs / f"slot-{p}.log")
        if not s.ready():
            print("FAIL: slot-arm server never became ready", file=sys.stderr)
            s.stop()
            return 5
        servers.append(s)
        print(f"  server pid={s.proc.pid} port={p} slots={kmax} "
              f"free={free_gib():.1f}GiB swap={swap_gib():.2f}GiB", flush=True)
        for _ in range(kmax):
            decode(p, prompt, 8, args.timeout)

        for k in stream_counts:
            reps = []
            for rep in range(args.reps):
                r = fan([p] * k, prompt, args.n_predict, args.timeout)
                r["rep"] = rep
                r["free_gib"] = round(free_gib(), 2)
                r["swap_gib"] = round(swap_gib(), 2)
                reps.append(r)
                print(f"  SLOT k={k} rep{rep}: agg={r['aggregate_tps']} tok/s "
                      f"wall={r['batch_wall_s']}s ok={r['ok']}/{k}", flush=True)
            results["slot"][k] = reps

    finally:
        for s in servers:
            s.stop()
        left = kill_strays()
        print(f"\ncleanup: leftover llama-server: {left or '(none)'}")

    def med(reps, key="aggregate_tps"):
        v = sorted(r[key] for r in reps if r.get(key))
        return v[len(v) // 2] if v else None

    summary = {}
    for arm in ("process", "slot"):
        base = med(results[arm].get(1, []))
        summary[arm] = {
            str(k): {
                "aggregate_tps_median": med(results[arm][k]),
                "scaling_vs_1": (round(med(results[arm][k]) / base, 4)
                                 if base and med(results[arm][k]) else None),
                "spread_pct": (lambda v: round(100 * (v[-1] - v[0]) / v[0], 1) if v and v[0] else None)(
                    sorted(r["aggregate_tps"] for r in results[arm][k] if r.get("aggregate_tps"))),
            } for k in results[arm]
        }

    # bandwidth accounting: how much weight traffic each topology implies
    single = summary["process"].get("1", {}).get("aggregate_tps_median")
    bw = None
    if single:
        bw = {
            "model_bytes": model_bytes,
            "single_decoder_tps": single,
            "implied_weight_stream_GB_per_s": round(model_bytes * single / 1e9, 1),
            "m3_ultra_peak_GB_per_s": 819,
            "single_decoder_pct_of_peak": round(100 * model_bytes * single / 1e9 / 819, 1),
            "note": ("PROCESS topology multiplies this traffic by k because each process "
                     "streams the full weight set per token. SLOT topology does not: one "
                     "pass over the weights serves every slot in the step."),
        }

    doc = {
        "schema": "hawking.headless.decode_topology.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": ("Is the ~1.21x aggregate ceiling at 2 concurrent decoders a hardware "
                     "bandwidth limit, or an artefact of running N separate processes?"),
        "reprofile_command":
            f"python3 tools/headless/decode_topology_probe.py --per-slot-ctx {args.per_slot_ctx} "
            f"--n-predict {args.n_predict} --reps {args.reps} --streams {args.streams}",
        "model": args.model,
        "model_bytes": model_bytes,
        "llama_version": subprocess.run(["llama-server", "--version"],
                                        capture_output=True, text=True).stderr.strip()[:200],
        "params": {"per_slot_ctx": args.per_slot_ctx, "n_predict": args.n_predict,
                   "reps": args.reps, "streams": stream_counts,
                   "decode_flags": "temperature=0 ignore_eos=true cache_prompt=false",
                   "design": "identical total work per arm; per-slot context held equal"},
        "gpu_gate": gpu,
        "foreign_load_at_start": foreign,
        "foreign_load_at_end": foreign_model_load(),
        "measurement_is_clean_room": not foreign["count"],
        "clean_room_caveat": (None if not foreign["count"] else
            f"{foreign['count']} model process(es) this probe did not start were resident "
            f"(actively decoding: {foreign['any_actively_decoding']}). The PROCESS-vs-SLOT "
            "comparison is still meaningful because both arms saw the same co-tenant, but the "
            "absolute tok/s and the percent-of-peak figure are a floor, not a roof."),
        "summary": summary,
        "bandwidth_accounting": bw,
        "raw": results,
    }
    (out / "DECODE_TOPOLOGY.json").write_text(json.dumps(doc, indent=1))

    print("\n=== DECODE TOPOLOGY ===")
    print(f"{'k':>3}  {'PROCESS agg':>12} {'x1':>6}   {'SLOT agg':>12} {'x1':>6}   {'slot/process':>12}")
    for k in stream_counts:
        pr = summary["process"].get(str(k), {})
        sl = summary["slot"].get(str(k), {})
        ratio = (round(sl["aggregate_tps_median"] / pr["aggregate_tps_median"], 3)
                 if pr.get("aggregate_tps_median") and sl.get("aggregate_tps_median") else None)
        print(f"{k:>3}  {str(pr.get('aggregate_tps_median')):>12} {str(pr.get('scaling_vs_1')):>6}   "
              f"{str(sl.get('aggregate_tps_median')):>12} {str(sl.get('scaling_vs_1')):>6}   {str(ratio):>12}")
    if bw:
        print(f"\n  single decoder streams ~{bw['implied_weight_stream_GB_per_s']} GB/s "
              f"= {bw['single_decoder_pct_of_peak']}% of the ~819 GB/s roof")
    print(f"\n-> {out/'DECODE_TOPOLOGY.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
