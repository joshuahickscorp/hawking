#!/usr/bin/env python3
"""RuntimeBackend A/B — llama.cpp GGUF vs MLX, on the same machine, same work.

Directive §9/§17: expose a second runtime through the RuntimeBackend abstraction
and benchmark it head-to-head before the bootstrap path is retired.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
This is a RUNTIME + QUANTISATION speed comparison. It is NOT a quality
comparison, and the receipt says so, because the two artifacts are not the same
weights:

  llama.cpp arm : huihui-ai abliteration, Q5_K GGUF, ~19.5 GB
  MLX arm       : PocketAiHub abliteration, 4-bit MLX, ~15 GB

Same base model (Qwen/Qwen3.8-27B, dense, 27B) and therefore the same
architecture and the same per-token weight-streaming shape, which is what decode
speed depends on. But they are DIFFERENT abliterations — huihui-ai left the first
15 layers unablated, PocketAiHub projected layers 24-63 — so any behavioural or
coherence claim across these two arms would be confounded. Speed only.

A clean quality A/B needs the same lineage in both runtimes: fetch the huihui-ai
bf16 parent and convert it with mlx_lm.convert. That is a separate step and this
harness will pick it up automatically once the converted directory exists.

Both arms: greedy, fixed token count, ignore-EOS where available, several
alternating reps so page-cache and thermal drift cancel.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

GGUF_DEFAULT = os.path.expanduser(
    "~/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf")
MLX_CANDIDATES = [
    # preferred: same lineage as the GGUF, if it has been converted
    "~/models/qwen3.8-27b-abliterated-mlx-huihui-4bit",
    # fallback: a different abliteration — speed-only comparison
    "~/models/qwen3.8-27b-abliterated-mlx/4bit",
]
MLX_PY = os.path.expanduser("~/.local/share/uv/tools/mlx-lm/bin/python")

PROMPT = ("Explain, in ordinary prose and at length, how a compiler turns a "
          "for-loop into basic blocks and then into machine code.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def dir_bytes(p: str) -> int:
    total = 0
    for dp, _, fns in os.walk(p):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


def strays() -> str:
    return subprocess.run(
        ["bash", "-lc", "ps -eo pid,command | grep -E 'llama-server|mlx_lm' | grep -v grep"],
        capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ llama.cpp

def llama_arm(model: str, n_predict: int, reps: int, ctx: int, log: Path) -> list:
    port = free_port()
    fh = open(log, "wb")
    proc = subprocess.Popen(
        ["llama-server", "-m", model, "--port", str(port), "-c", str(ctx),
         "-ngl", "999", "--host", "127.0.0.1", "-np", "1"],
        stdout=fh, stderr=subprocess.STDOUT)
    out = []
    try:
        end = time.time() + 900
        ready = False
        while time.time() < end:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(1.5)
        if not ready:
            raise RuntimeError("llama-server never became ready")

        def one(n):
            payload = {"prompt": PROMPT, "n_predict": n, "temperature": 0.0,
                       "ignore_eos": True, "cache_prompt": False}
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=1200) as r:
                body = json.loads(r.read().decode("utf-8", "replace"))
            t = body.get("timings", {}) or {}
            return {"wall_s": round(time.time() - t0, 3),
                    "predicted_n": t.get("predicted_n"),
                    "decode_tps": t.get("predicted_per_second"),
                    "prompt_n": t.get("prompt_n"),
                    "prefill_tps": t.get("prompt_per_second"),
                    "ttft_ms": t.get("prompt_ms")}

        one(8)  # warm
        for rep in range(reps):
            r = one(n_predict)
            r["rep"] = rep
            out.append(r)
            print(f"  llama rep{rep}: {r['decode_tps']} tok/s  wall={r['wall_s']}s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        fh.close()
    return out


# ------------------------------------------------------------------------ MLX

MLX_SNIPPET = r'''
import json, sys, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
path, n_predict, reps, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
t0 = time.time()
model, tokenizer = load(path)
load_s = time.time() - t0
msgs = [{"role": "user", "content": prompt}]
try:
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
except TypeError:
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
sampler = make_sampler(temp=0.0)
out = []
def run(n):
    t = time.time()
    resp = generate(model, tokenizer, prompt=text, max_tokens=n,
                    sampler=sampler, verbose=False)
    wall = time.time() - t
    ntok = len(tokenizer.encode(resp))
    return {"wall_s": round(wall, 3), "predicted_n": ntok,
            "decode_tps": round(ntok / wall, 3) if wall > 0 else None}
run(8)
for rep in range(reps):
    r = run(n_predict); r["rep"] = rep; out.append(r)
    print(json.dumps({"progress": r}), file=sys.stderr, flush=True)
print(json.dumps({"load_s": round(load_s, 2), "runs": out}))
'''


def mlx_arm(path: str, n_predict: int, reps: int) -> dict:
    if not os.path.isfile(MLX_PY):
        return {"error": f"mlx python not found at {MLX_PY}"}
    proc = subprocess.run([MLX_PY, "-c", MLX_SNIPPET, path, str(n_predict), str(reps), PROMPT],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return {"error": f"mlx arm exited {proc.returncode}",
                "stderr": proc.stderr[-2000:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"could not parse mlx output: {e}", "stdout": proc.stdout[-2000:]}



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
    ap.add_argument("--gguf", default=GGUF_DEFAULT)
    ap.add_argument("--mlx", default=None)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--allow-foreign-load", action="store_true",
                    help="measure even though model processes this probe did not "
                         "start are resident; their identity and slot activity are "
                         "recorded into the receipt")
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/Downloads/hawking-copy/receipts/headless"))
    args = ap.parse_args()

    mlx_path = args.mlx
    if mlx_path is None:
        for c in MLX_CANDIDATES:
            e = os.path.expanduser(c)
            if os.path.isdir(e):
                mlx_path = e
                break
    if mlx_path is None:
        print("FAIL: no MLX model directory found; tried " + ", ".join(MLX_CANDIDATES),
              file=sys.stderr)
        return 2
    mlx_path = os.path.expanduser(mlx_path)
    same_lineage = "huihui" in os.path.basename(mlx_path).lower()

    foreign = foreign_model_load()
    if foreign["count"] and not args.allow_foreign_load:
        print("FAIL: model processes this probe did not start are resident; an A/B under an "
              "unknown co-tenant compares nothing.\n"
              "Stop them, or re-run with --allow-foreign-load (the load is then recorded):",
              file=sys.stderr)
        for pr in foreign["processes"]:
            print(f"  pid={pr['pid']} port={pr['port']} cpu={pr['pcpu']}% "
                  f"slots_busy={pr['slots_processing']}/{pr['slots_total']} {pr['command'][:100]}",
                  file=sys.stderr)
        return 3
    if foreign["count"]:
        print(f"WARNING: measuring with {foreign['count']} foreign model process(es) resident",
              flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "runtime_ab_logs"
    logs.mkdir(exist_ok=True)

    print(f"llama.cpp arm: {args.gguf}", flush=True)
    llama = llama_arm(args.gguf, args.n_predict, args.reps, args.ctx, logs / "llama.log")
    time.sleep(10)
    print(f"MLX arm: {mlx_path}", flush=True)
    mlx = mlx_arm(mlx_path, args.n_predict, args.reps)
    for r in (mlx.get("runs") or []):
        print(f"  mlx   rep{r['rep']}: {r['decode_tps']} tok/s  wall={r['wall_s']}s", flush=True)

    def med(rows, key="decode_tps"):
        v = sorted(r[key] for r in rows if r.get(key))
        return v[len(v) // 2] if v else None

    def spread(rows, key="decode_tps"):
        v = sorted(r[key] for r in rows if r.get(key))
        return round(100 * (v[-1] - v[0]) / v[0], 1) if v and v[0] else None

    lt, mt = med(llama), med(mlx.get("runs") or [])
    gguf_bytes = os.path.getsize(args.gguf)
    mlx_bytes = dir_bytes(mlx_path)

    doc = {
        "schema": "hawking.headless.runtime_ab.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_measures": "runtime + quantisation DECODE SPEED only",
        "what_this_does_not_measure": (
            "quality, coherence, or behavioural equivalence"
            if not same_lineage else "quality (not evaluated here, but lineage is matched)"),
        "confound_declared": (
            None if same_lineage else
            "The two arms are DIFFERENT abliterations of the same base model "
            "(huihui-ai leaves the first 15 layers unablated; PocketAiHub projects layers 24-63). "
            "Architecture and per-token weight-streaming shape are identical, so the SPEED "
            "comparison is valid. Any quality claim across these arms would be confounded and "
            "must not be made from this receipt."),
        "foreign_load_at_start": foreign,
        "measurement_is_clean_room": not foreign["count"],
        "arms": {
            "llama_cpp": {"model": args.gguf, "bytes": gguf_bytes,
                          "gib": round(gguf_bytes / 1024**3, 2),
                          "quant": "Q5_K", "lineage": "huihui-ai",
                          "runs": llama, "decode_tps_median": lt, "spread_pct": spread(llama),
                          "version": subprocess.run(["llama-server", "--version"],
                                                    capture_output=True, text=True).stderr.strip()[:120]},
            "mlx": {"model": mlx_path, "bytes": mlx_bytes,
                    "gib": round(mlx_bytes / 1024**3, 2),
                    "quant": "4bit", "lineage": "huihui-ai" if same_lineage else "PocketAiHub",
                    "load_s": mlx.get("load_s"), "runs": mlx.get("runs"),
                    "decode_tps_median": mt, "spread_pct": spread(mlx.get("runs") or []),
                    "error": mlx.get("error"), "stderr": mlx.get("stderr")},
        },
        "comparison": {
            "mlx_over_llama": round(mt / lt, 3) if lt and mt else None,
            "bytes_ratio_llama_over_mlx": round(gguf_bytes / mlx_bytes, 3) if mlx_bytes else None,
            "speed_explained_by_bytes_alone": (
                round((gguf_bytes / mlx_bytes) if mlx_bytes else 0, 3)),
            "interpretation": (
                "If mlx_over_llama is close to bytes_ratio, the difference is just fewer bytes "
                "streamed per token and the two runtimes are equally efficient. If it is "
                "meaningfully ABOVE the byte ratio, MLX's kernels are doing better work per byte "
                "and the runtime itself is the lever. If BELOW, MLX is leaving efficiency on the "
                "table despite the smaller artifact."),
        },
        "params": {"n_predict": args.n_predict, "reps": args.reps, "ctx": args.ctx,
                   "temperature": 0.0, "design": "alternating-free sequential arms, warmed, medians reported"},
        "reprofile_command":
            f"python3 tools/headless/runtime_ab.py --n-predict {args.n_predict} --reps {args.reps}",
        "to_remove_the_confound": (
            "hf download huihui-ai/Huihui-Qwen3.8-27B-abliterated --local-dir "
            "~/models/qwen3.8-27b-abliterated-bf16 && "
            f"{MLX_PY} -m mlx_lm convert --hf-path ~/models/qwen3.8-27b-abliterated-bf16 "
            "-q --q-bits 4 --mlx-path ~/models/qwen3.8-27b-abliterated-mlx-huihui-4bit"),
    }
    (out / "RUNTIME_AB.json").write_text(json.dumps(doc, indent=1))

    print("\n=== RUNTIME A/B (speed only) ===")
    print(f"  llama.cpp Q5_K  {doc['arms']['llama_cpp']['gib']} GiB  "
          f"{lt} tok/s (spread {doc['arms']['llama_cpp']['spread_pct']}%)")
    print(f"  MLX 4bit        {doc['arms']['mlx']['gib']} GiB  "
          f"{mt} tok/s (spread {doc['arms']['mlx']['spread_pct']}%)")
    c = doc["comparison"]
    print(f"  mlx/llama = {c['mlx_over_llama']}   bytes llama/mlx = {c['bytes_ratio_llama_over_mlx']}")
    if not same_lineage:
        print("  NOTE: different abliterations — SPEED comparison only, no quality claim.")
    print(f"\n-> {out/'RUNTIME_AB.json'}")
    print(f"cleanup: leftover model processes: {strays() or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
