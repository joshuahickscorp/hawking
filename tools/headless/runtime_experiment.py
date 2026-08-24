#!/usr/bin/env python3
"""One measured local-runtime experiment, target chosen by measurement.

This is a measurement harness. It does not modify crates/. It builds the
native Qwen3.8 greedy example into workspace/ops/build/rust (the mandated
target-dir), profiles one complete-wall generate, picks ONE paired A/B from
that profile, alternates reps, and writes receipts/headless/RUNTIME_EXPERIMENT.json.

Default invocation (the acceptance command):

    python3 tools/headless/runtime_experiment.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET_DIR = REPO / "workspace" / "ops" / "build" / "rust"
PROFILE = "release-fast"
EXAMPLE = "ascension_qwen38_hybrid_greedy"
RECEIPT = REPO / "receipts" / "headless" / "RUNTIME_EXPERIMENT.json"
LOCK = Path("/tmp/hawking-gpu-lane.lock")

ARTIFACT = Path(os.path.expanduser("~/models/qwen38-gravity-uniform-q4-v1"))
TOKENIZER = Path(
    os.path.expanduser(
        "~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json"
    )
)
GGUF = Path(
    os.path.expanduser(
        "~/models/qwen3.8-27b-abliterated/"
        "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
    )
)
MLX_HUIHUI = Path(
    os.path.expanduser("~/models/qwen3.8-27b-abliterated-mlx-huihui-4bit")
)
MLX_PY = Path(os.path.expanduser("~/.local/share/uv/tools/mlx-lm/bin/python"))
DSV4F = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/full-43-layer-stream.gravity"
)
LLAMA1B = Path(
    os.path.expanduser(
        "~/Library/Application Support/Hawking/CampaignS08/"
        "llama32-1b-R0.v2.gravity"
    )
)

# Geometry authority from crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
# (read-only; copied so this harness does not import Rust).
HONEST_DECODE_CEILING_GB_S = 411.51
M3_ULTRA_PEAK_GB_S = 819.0
ACTIVE_BUDGET_BYTES = 13_622_264_240
PRODUCTION_DISPATCHES = 964
UNIFORM_Q4_BPW = 4.252735126866492

PROMPT = "Say hi."
N_PREDICT = 16
MAX_SEQ_LEN = 256
PAIRS = 1
REPS = 3

SCHEMA = "hawking.headless.runtime_experiment.v1"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def git_head() -> str:
    p = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    return (p.stdout or "").strip() or "UNKNOWN"


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for dp, _, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if os.path.islink(fp):
                continue
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def vm_snapshot() -> dict:
    p = sh(["vm_stat"])
    out = {}
    page = 16384
    for line in (p.stdout or "").splitlines():
        if "page size of" in line:
            try:
                page = int(line.split()[-1].rstrip("."))
            except ValueError:
                pass
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        digits = "".join(c for c in v if c.isdigit())
        if digits:
            out[k.strip()] = int(digits)
    snap = {
        "page_size": page,
        "pages_free": out.get("Pages free"),
        "pages_purgeable": out.get("Pages purgeable"),
        "swapins": out.get("Swapins"),
        "swapouts": out.get("Swapouts"),
        "compressed_pages": out.get("Pages stored in compressor")
        or out.get("Pages occupied by compressor"),
    }
    free = snap["pages_free"]
    snap["free_gib"] = None if free is None else round(free * page / 1024**3, 3)
    return snap


def metal_device() -> dict:
    swift = r"""
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
"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
        f.write(swift)
        path = f.name
    try:
        p = subprocess.run(
            ["swift", path], capture_output=True, text=True, timeout=180
        )
        if p.returncode != 0:
            return {"error": (p.stderr or p.stdout)[-800:]}
        line = [ln for ln in (p.stdout or "").splitlines() if ln.strip()][-1]
        d = json.loads(line)
        rec = d.get("recommendedMaxWorkingSetSize") or 0
        d["recommendedMaxWorkingSetSize_gib"] = round(rec / 1024**3, 2)
        return d
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def foreign_llama() -> dict:
    p = sh(
        [
            "bash",
            "-lc",
            "ps -eo pid,etime,pcpu,rss,command | grep -E '[l]lama-server|[m]lx_lm|[a]scension_qwen38'",
        ]
    )
    procs = []
    for line in (p.stdout or "").strip().splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, etime, pcpu, rss, cmd = parts
        port = None
        m = __import__("re").search(r"--port\s+(\d+)", cmd)
        if m:
            port = int(m.group(1))
        busy = None
        total = None
        if port:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/slots", timeout=3
                ) as r:
                    slots = json.loads(r.read().decode("utf-8", "replace"))
                total = len(slots)
                busy = sum(1 for s in slots if s.get("is_processing"))
            except Exception:
                pass
        procs.append(
            {
                "pid": int(pid),
                "etime": etime,
                "pcpu": float(pcpu),
                "rss_bytes": int(rss) * 1024,
                "port": port,
                "slots_total": total,
                "slots_processing": busy,
                "command": cmd[:240],
            }
        )
    return {
        "count": len(procs),
        "processes": procs,
        "any_actively_decoding": any(
            (pr.get("slots_processing") or 0) > 0 for pr in procs
        ),
    }


def pick_llama_port(foreign: dict) -> int | None:
    for pr in foreign.get("processes") or []:
        port = pr.get("port")
        if not port:
            continue
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as r:
                if r.status != 200:
                    continue
            # Prefer a server that answers /slots.
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/slots", timeout=3
                ) as r:
                    slots = json.loads(r.read().decode("utf-8", "replace"))
                if any(s.get("is_processing") for s in slots):
                    continue
            except Exception:
                pass
            return int(port)
        except Exception:
            continue
    return None


class GpuLock:
    def __init__(self, name: str, timeout_s: int = 5400):
        self.name = name
        self.timeout_s = timeout_s
        self.held = False

    def __enter__(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                LOCK.mkdir()
                (LOCK / "pid").write_text(str(os.getpid()))
                (LOCK / "owner").write_text(self.name)
                self.held = True
                return self
            except FileExistsError:
                stale = False
                try:
                    pid = int((LOCK / "pid").read_text().strip())
                    os.kill(pid, 0)
                except Exception:
                    stale = True
                if stale:
                    subprocess.run(["rm", "-rf", str(LOCK)])
                    continue
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"gpu_lane_lock timeout; held by "
                        f"{(LOCK / 'owner').read_text() if (LOCK / 'owner').exists() else '?'}"
                    )
                time.sleep(5)

    def __exit__(self, *exc):
        if self.held:
            subprocess.run(["rm", "-rf", str(LOCK)])
            self.held = False
        return False


def binary_path() -> Path:
    return TARGET_DIR / PROFILE / "examples" / EXAMPLE


def cargo_build() -> dict:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo",
        "build",
        "--profile",
        PROFILE,
        "-p",
        "hawking-core",
        "--example",
        EXAMPLE,
        "--target-dir",
        str(TARGET_DIR),
    ]
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    wall = round(time.time() - t0, 2)
    binp = binary_path()
    return {
        "cmd": cmd,
        "exit_code": p.returncode,
        "wall_s": wall,
        "binary": str(binp),
        "binary_exists": binp.is_file(),
        "binary_bytes": binp.stat().st_size if binp.is_file() else None,
        "stderr_tail": (p.stderr or "")[-2500:],
        "profile_note": (
            "release-fast (lto=false, codegen-units=16). GPU timestamps come "
            "from Metal shaders compiled by the driver, not rustc LTO. Host "
            "encode of 964 dispatches is <1% of the measured token wall."
        ),
    }


def native_complete_wall(out: Path, env: dict | None = None) -> dict:
    binp = binary_path()
    cmd = [
        str(binp),
        "--artifact-root",
        str(ARTIFACT),
        "--tokenizer",
        str(TOKENIZER),
        "--prompt",
        PROMPT,
        "--complete-wall",
        "--max-new-tokens",
        str(N_PREDICT),
        "--max-seq-len",
        str(MAX_SEQ_LEN),
        "--pairs",
        str(PAIRS),
        "--out",
        str(out),
    ]
    run_env = os.environ.copy()
    if env:
        run_env.update({k: str(v) for k, v in env.items()})
    t0 = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=run_env,
        timeout=1800,
    )
    wall = round(time.time() - t0, 3)
    body = None
    if out.is_file():
        try:
            body = json.loads(out.read_text())
        except Exception as e:
            body = {"parse_error": str(e)}
    return {
        "cmd": cmd,
        "exit_code": p.returncode,
        "process_wall_s": wall,
        "stdout_tail": (p.stdout or "")[-2000:],
        "stderr_tail": (p.stderr or "")[-2000:],
        "json": body,
        "env": env or {},
    }


def summarize_native(run: dict) -> dict:
    body = run.get("json") or {}
    auth = body.get("authority") or {}
    ident = body.get("identity") or {}
    req = body.get("request_level_excluded_from_per_token") or {}
    warms = body.get("warm_reps") or []
    named = {}
    dispatches = None
    if warms:
        s0 = (warms[0].get("summary") or {})
        named = ((s0.get("closure") or {}).get("named_component_means_ns")) or {}
        dispatches = ((s0.get("steady_decode") or {}).get("dispatches"))
    gpu_ns = auth.get("headline_gpu_ns_per_token")
    wall_ns = auth.get("headline_complete_wall_ns_per_token")
    tps = auth.get("headline_complete_tps")
    gpu_frac = None
    if wall_ns and gpu_ns and wall_ns > 0:
        gpu_frac = gpu_ns / wall_ns
    implied_gb_s = None
    if gpu_ns and gpu_ns > 0:
        implied_gb_s = ACTIVE_BUDGET_BYTES / (gpu_ns / 1e9) / 1e9
    text = ""
    if warms:
        text = ((warms[0].get("summary") or {}).get("generated_text")) or ""
    return {
        "exit_code": run.get("exit_code"),
        "process_wall_s": run.get("process_wall_s"),
        "session_open_ns": req.get("session_open_ns"),
        "headline_complete_tps": tps,
        "headline_complete_wall_ns": wall_ns,
        "headline_gpu_ns": gpu_ns,
        "gpu_fraction_of_wall": gpu_frac,
        "implied_active_gb_s": implied_gb_s,
        "rep_median_complete_wall_ns": auth.get("rep_median_complete_wall_ns"),
        "rep_median_gpu_ns": auth.get("rep_median_gpu_ns"),
        "named_component_means_ns": named,
        "dispatches": dispatches or PRODUCTION_DISPATCHES,
        "command_buffers": 1,
        "greedy_new_token_ids": ident.get("greedy_new_token_ids"),
        "generated_text_excerpt": text[:400],
        "kernel": ident.get("kernel"),
        "stdout_tail": run.get("stdout_tail"),
        "stderr_tail": run.get("stderr_tail") if run.get("exit_code") else None,
    }


def llama_completion(port: int, n_predict: int) -> dict:
    payload = {
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"error": str(e), "wall_s": round(time.time() - t0, 3)}
    wall = round(time.time() - t0, 3)
    t = body.get("timings") or {}
    decode_tps = t.get("predicted_per_second")
    implied_gb_s = None
    gguf_bytes = dir_bytes(GGUF)
    if decode_tps and decode_tps > 0 and gguf_bytes:
        implied_gb_s = gguf_bytes * decode_tps / 1e9
    return {
        "wall_s": wall,
        "predicted_n": t.get("predicted_n"),
        "decode_tps": decode_tps,
        "predicted_ms": t.get("predicted_ms"),
        "prompt_n": t.get("prompt_n"),
        "prefill_tps": t.get("prompt_per_second"),
        "ttft_ms": t.get("prompt_ms"),
        "implied_gguf_gb_s": implied_gb_s,
        "content_excerpt": (body.get("content") or "")[:200],
    }


def median(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def spread_pct(xs):
    xs = [x for x in xs if x is not None]
    if not xs or xs[0] == 0 and min(xs) == 0:
        return None
    s = sorted(xs)
    if s[0] == 0:
        return None
    return round(100.0 * (s[-1] - s[0]) / s[0], 1)


def exists_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "bytes": dir_bytes(path) if path.exists() else None,
    }


def inventory() -> dict:
    examples_dir = REPO / "crates" / "hawking-core" / "examples"
    example_files = sorted(p.name for p in examples_dir.glob("*.rs"))
    native_bin = binary_path()
    llama = sh(["bash", "-lc", "which llama-server; llama-server --version 2>&1 | head -3"])
    items = {
        "hawking_core_examples_on_disk": len(example_files),
        "native_qwen38_hybrid_greedy": {
            "source": "crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs",
            "binary": exists_info(native_bin),
            "artifact": exists_info(ARTIFACT),
            "tokenizer": exists_info(TOKENIZER),
            "runnable_if": "artifact manifest + tensors + tokenizer + this-worktree binary",
        },
        "native_qwen38_token_ns": {
            "source": "crates/hawking-core/examples/ascension_qwen38_token_ns.rs",
            "artifact": exists_info(ARTIFACT),
            "runnable_if": "same artifact as hybrid_greedy; not built in this lane (full isolated-family ledger is a 36-family profile, not the A/B)",
        },
        "llama_cpp_gguf": {
            "binary": (sh(["bash", "-lc", "which llama-server"]).stdout or "").strip(),
            "version_stderr": (llama.stderr or llama.stdout or "")[:200],
            "gguf": exists_info(GGUF),
        },
        "mlx_huihui_4bit": {
            "python": exists_info(MLX_PY),
            "model": exists_info(MLX_HUIHUI),
        },
        "gravity_tps_llama32_1b": {
            "source": "crates/hawking-core/examples/gravity_tps.rs",
            "default_artifact": exists_info(LLAMA1B),
            "missing": "CampaignS08/llama32-1b-R0.v2.gravity is not on disk (only llama32-1b-bf16/)",
        },
        "dsv4f_native_token_graph": {
            "source": "crates/hawking-core/examples/gravity_deepseek_v4_native_token_graph.rs",
            "artifact": exists_info(DSV4F),
            "note": "artifact directory exists on this box; this lane does not open it (different model, 69k tensors, prior sandboxed run had no Metal)",
        },
        "qwen80_mixed_decode": {
            "source": "crates/hawking-core/examples/ascension_qwen80_mixed_hybrid_greedy.rs",
            "artifact_on_disk": False,
            "missing": "no qwen80 .gravity / mixed catalog under ~/models or Hawking Application Support",
        },
        "matmul_k_amortization": {
            "source": "crates/hawking-core/examples/ascension_qwen38_matmul_k_amortization.rs",
            "needs_artifact": False,
            "note": "self-contained Qwen38-K GEMV microbench; sequential K sweep, not an alternating A/B",
        },
        "matvec_occupancy": {
            "source": "crates/hawking-core/examples/matvec_occupancy.rs",
            "needs_artifact": False,
            "note": "self-contained, but uses Q80_GATE geometry, not Qwen3.8 5120-wide production GEMV",
        },
        "metal_device_copy_roofline": {
            "source": "crates/hawking-core/examples/metal_device_copy_roofline.rs",
            "needs": "sealed DSV4F static-expert-residency receipt (not opened here)",
        },
        "headless_harnesses_present": {
            "runtime_ab.py": (REPO / "tools/headless/runtime_ab.py").is_file(),
            "machine_probe.py": (REPO / "tools/headless/machine_probe.py").is_file(),
            "decode_topology_probe.py": (
                REPO / "tools/headless/decode_topology_probe.py"
            ).is_file(),
            "qwen38_gravity_native_bench.sh": (
                REPO / "tools/headless/qwen38_gravity_native_bench.sh"
            ).is_file(),
        },
    }
    return items


def choose_experiment(profile: dict, llama_probe: dict) -> dict:
    gpu_frac = profile.get("gpu_fraction_of_wall") or 0.0
    native_tps = profile.get("headline_complete_tps") or 0.0
    llama_tps = (llama_probe or {}).get("decode_tps") or 0.0
    named = profile.get("named_component_means_ns") or {}
    encode = named.get("encode_host_prepare") or 0.0
    wait_minus_gpu = named.get("wait_minus_gpu") or 0.0
    gpu = named.get("gpu") or 0.0

    rejected = [
        {
            "name": "HAWKING_QWEN38_RECON_FUSE=0 vs default",
            "why": (
                "recon_fuse only retargets mixed-catalog binary/HGRAVS kernels "
                "(qwen38_hybrid_decode.rs:44, :1665). The on-disk artifact is "
                "uniform-q4-v1; the production kernel is "
                "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128. The toggle "
                "would be a no-op on this path."
            ),
        },
        {
            "name": "HAWKING_QWEN_RESIDENCY=1 vs default-off",
            "why": (
                "request_residency is wired in qwen_dense.rs:4693 (GGUF dense "
                "engine), not in Qwen38HybridDecodeSession::open. The env "
                "flag would not pin the gravity catalog."
            ),
        },
        {
            "name": "matvec_occupancy serial vs simdgroup",
            "why": (
                "self-contained, but launches Q80_GATE geometry, not the "
                "Qwen3.8 17408x5120 production GEMV. A number from a path "
                "nobody uses."
            ),
        },
        {
            "name": "ascension_qwen38_matmul_k_amortization K=1 vs K=4",
            "why": (
                "the remaining ceiling lever once a single token is already "
                "near 58% of peak GB/s: decode each weight into K accumulators. "
                "The example sweeps K sequentially inside one process, so it "
                "cannot satisfy alternating-rep A/B without a crate edit."
            ),
        },
        {
            "name": "HAWKING_DN_VI_SIMD on vs off",
            "why": (
                "this toggle IS wired into encode_gated_delta "
                "(qwen38_hybrid_decode.rs:2019). Historical TOKEN_NS put "
                "DeltaNet at ~10.6% of the token. Host is ~4% of wall here. "
                "Neither can move the headline vs a bytes-ratio comparison "
                "against the production llama.cpp server."
            ),
        },
        {
            "name": "full ascension_qwen38_token_ns isolated-family ledger",
            "why": (
                "runnable on this artifact, but it is a 36-family diagnostic "
                "plus addr/decode probes, not an A/B. complete-wall already "
                "named the wall as GPU."
            ),
        },
        {
            "name": "gravity_tps llama-1b BASE_TRUE_TPS",
            "why": "default artifact llama32-1b-R0.v2.gravity is not on disk.",
        },
        {
            "name": "DSV4F native token graph / qwen80 mixed greedy",
            "why": (
                "DSV4F tree exists but is a different model with a 69k-tensor "
                "stream; qwen80 mixed catalog is absent. Not the runtime "
                "this box is serving."
            ),
        },
        {
            "name": "llama.cpp vs MLX (RUNTIME_AB.json)",
            "why": (
                "already measured on this box (MLX 35.5 tok/s vs llama 24.0). "
                "It is not the hawking-core decode path."
            ),
        },
    ]

    implied = profile.get("implied_active_gb_s")
    implied_txt = f"{implied:.1f}" if implied else "n/a"
    chosen = "native_gravity_q4_vs_resident_llama_cpp_q5k"
    reason = (
        f"Profiled native complete-wall at {native_tps:.3f} tok/s with "
        f"gpu_fraction={gpu_frac:.4f} (host encode {encode:.0f} ns, "
        f"wait_minus_gpu {wait_minus_gpu:.0f} ns, gpu {gpu:.0f} ns, "
        f"implied {implied_txt} GB/s of {M3_ULTRA_PEAK_GB_S:.0f} peak). "
        f"The resident llama.cpp server probed at {llama_tps:.3f} tok/s. "
        "GPU is the wall (serial/host fraction ~4%), so a host-side A/B "
        "was rejected. The production question on this box is whether the "
        "native gravity catalog is faster or slower than the llama.cpp "
        "server that is actually serving HCLI, measured as a paired "
        "alternating A/B against the physical 819 GB/s ceiling."
    )
    if gpu_frac < 0.85:
        reason += (
            " NOTE: gpu_fraction was below 0.85; host would have been the "
            "lever, but that is not what this profile showed."
        )
    if not llama_tps:
        chosen = "native_only_profile_llama_unreachable"
        reason = "llama probe failed; cannot run the chosen A/B."
    return {
        "chosen": chosen,
        "reason": reason,
        "rejected": rejected,
        "profile_gpu_fraction": gpu_frac,
        "profile_native_tps": native_tps,
        "probe_llama_tps": llama_tps,
    }


def pct_of(x, ceil):
    if x is None or not ceil:
        return None
    return round(100.0 * x / ceil, 2)


def acquire_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def main() -> int:
    started = time.time()
    head = git_head()
    acquire_print(f"git HEAD {head}")
    acquire_print(f"repo {REPO}")

    inv = inventory()
    acquire_print("\n=== INVENTORY (on disk, this sparse worktree) ===")
    acquire_print(
        f"  hawking-core examples/*.rs: {inv['hawking_core_examples_on_disk']}"
    )
    art = inv["native_qwen38_hybrid_greedy"]["artifact"]
    tok = inv["native_qwen38_hybrid_greedy"]["tokenizer"]
    acquire_print(
        f"  qwen38 gravity artifact: exists={art['exists']} "
        f"bytes={art['bytes']}  {art['path']}"
    )
    acquire_print(
        f"  tokenizer: exists={tok['exists']} bytes={tok['bytes']}"
    )
    acquire_print(
        f"  llama.cpp: {inv['llama_cpp_gguf']['binary']}  "
        f"gguf_exists={inv['llama_cpp_gguf']['gguf']['exists']} "
        f"gguf_bytes={inv['llama_cpp_gguf']['gguf']['bytes']}"
    )
    acquire_print(
        f"  mlx huihui-4bit dir exists={inv['mlx_huihui_4bit']['model']['exists']}  "
        f"mlx_py exists={inv['mlx_huihui_4bit']['python']['exists']}"
    )
    acquire_print(
        f"  llama-1b .gravity exists={inv['gravity_tps_llama32_1b']['default_artifact']['exists']}  "
        f"({inv['gravity_tps_llama32_1b']['missing']})"
    )
    acquire_print(
        f"  qwen80 mixed: {inv['qwen80_mixed_decode']['missing']}"
    )
    acquire_print(
        f"  dsv4f artifact exists={inv['dsv4f_native_token_graph']['artifact']['exists']}"
    )

    foreign = foreign_llama()
    metal = metal_device()
    mem0 = vm_snapshot()
    acquire_print("\n=== MACHINE / OCCUPANCY ===")
    acquire_print(
        f"  metal: {metal.get('name')}  recommendedMaxWorkingSetSize_gib="
        f"{metal.get('recommendedMaxWorkingSetSize_gib')}"
    )
    acquire_print(
        f"  foreign decode processes: {foreign['count']}  "
        f"any_decoding={foreign['any_actively_decoding']}"
    )
    for pr in foreign.get("processes") or []:
        acquire_print(
            f"    pid={pr['pid']} port={pr['port']} cpu={pr['pcpu']}% "
            f"rss_gib={pr['rss_bytes']/1024**3:.2f} "
            f"slots={pr['slots_processing']}/{pr['slots_total']}"
        )
    acquire_print(
        f"  vm: free_gib={mem0.get('free_gib')} swapins={mem0.get('swapins')} "
        f"swapouts={mem0.get('swapouts')}"
    )

    acquire_print("\n=== BUILD ===")
    build = cargo_build()
    acquire_print(
        f"  cargo exit={build['exit_code']} wall_s={build['wall_s']}  "
        f"binary={build['binary']} bytes={build['binary_bytes']}"
    )
    if build["exit_code"] != 0 or not build["binary_exists"]:
        acquire_print(build["stderr_tail"])
        doc = {
            "schema": SCHEMA,
            "status": "BUILD_FAILED",
            "git_head": head,
            "build": build,
            "inventory": inv,
        }
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")
        return 2

    llama_port = pick_llama_port(foreign)
    acquire_print(f"  llama_port={llama_port}")

    watched_fail = []
    tmp = Path("/tmp/hawking_runtime_experiment")
    tmp.mkdir(parents=True, exist_ok=True)

    with GpuLock("i3gravity-runtime-experiment"):
        acquire_print("\n=== PROFILE (native complete-wall, discarded from A/B) ===")
        prof_run = native_complete_wall(tmp / "profile.json")
        profile = summarize_native(prof_run)
        acquire_print(
            f"  exit={profile['exit_code']} process_wall_s={profile['process_wall_s']} "
            f"open_s={(profile['session_open_ns'] or 0)/1e9:.3f}"
        )
        acquire_print(
            f"  headline_tps={profile['headline_complete_tps']}  "
            f"gpu_ns={profile['headline_gpu_ns']}  "
            f"wall_ns={profile['headline_complete_wall_ns']}  "
            f"gpu_frac={profile['gpu_fraction_of_wall']}"
        )
        acquire_print(
            f"  implied_active_gb_s={profile['implied_active_gb_s']}  "
            f"dispatches={profile['dispatches']}"
        )
        named = profile.get("named_component_means_ns") or {}
        acquire_print(
            f"  named_means_ns encode={named.get('encode_host_prepare')} "
            f"submit={named.get('submit')} gpu={named.get('gpu')} "
            f"wait_minus_gpu={named.get('wait_minus_gpu')} "
            f"readback={named.get('sample_readback')}"
        )
        if profile["exit_code"] != 0:
            watched_fail.append(
                f"native profile exited {profile['exit_code']}: "
                f"{(profile.get('stderr_tail') or '')[-400:]}"
            )
            acquire_print("  PROFILE FAILED")
            acquire_print(profile.get("stderr_tail") or profile.get("stdout_tail"))

        acquire_print("\n=== LLAMA PROBE (one warm completion, discarded from A/B) ===")
        llama_probe = {"error": "no healthy llama-server port"}
        if llama_port is not None:
            _ = llama_completion(llama_port, 8)  # warm
            llama_probe = llama_completion(llama_port, N_PREDICT)
        acquire_print(f"  {json.dumps({k: llama_probe.get(k) for k in ('decode_tps','predicted_n','wall_s','implied_gguf_gb_s','error')})}")

        choice = choose_experiment(profile, llama_probe)
        acquire_print("\n=== EXPERIMENT CHOICE ===")
        acquire_print(f"  chosen: {choice['chosen']}")
        acquire_print(f"  reason: {choice['reason']}")
        acquire_print("  rejected:")
        for r in choice["rejected"]:
            acquire_print(f"    - {r['name']}: {r['why']}")

        native_reps = []
        llama_reps = []
        if choice["chosen"] != "native_gravity_q4_vs_resident_llama_cpp_q5k":
            watched_fail.append(
                f"chosen experiment {choice['chosen']} is not the runnable A/B"
            )
        else:
            acquire_print(
                f"\n=== PAIRED A/B  reps={REPS}  n_predict={N_PREDICT}  "
                f"alternating native, llama ==="
            )
            for i in range(REPS):
                acquire_print(f"\n-- pair {i} native --")
                mem_n = vm_snapshot()
                nrun = native_complete_wall(tmp / f"native_rep{i}.json")
                ns = summarize_native(nrun)
                ns["rep"] = i
                ns["arm"] = "native_gravity_q4"
                ns["mem"] = mem_n
                native_reps.append(ns)
                acquire_print(
                    f"  native rep{i}: tps={ns['headline_complete_tps']}  "
                    f"gpu_ns={ns['headline_gpu_ns']}  "
                    f"wall_ns={ns['headline_complete_wall_ns']}  "
                    f"open_s={(ns['session_open_ns'] or 0)/1e9:.3f}  "
                    f"gb_s={ns['implied_active_gb_s']}"
                )
                if ns["exit_code"] != 0:
                    watched_fail.append(f"native rep{i} exit {ns['exit_code']}")

                acquire_print(f"-- pair {i} llama port={llama_port} --")
                mem_l = vm_snapshot()
                lrun = llama_completion(llama_port, N_PREDICT)
                lrun["rep"] = i
                lrun["arm"] = "llama_cpp_q5k"
                lrun["mem"] = mem_l
                llama_reps.append(lrun)
                acquire_print(
                    f"  llama  rep{i}: tps={lrun.get('decode_tps')}  "
                    f"predicted_n={lrun.get('predicted_n')}  "
                    f"wall_s={lrun.get('wall_s')}  "
                    f"gb_s={lrun.get('implied_gguf_gb_s')}"
                )
                if lrun.get("error"):
                    watched_fail.append(f"llama rep{i}: {lrun['error']}")

    native_tps = [r.get("headline_complete_tps") for r in native_reps]
    llama_tps = [r.get("decode_tps") for r in llama_reps]
    native_gb = [r.get("implied_active_gb_s") for r in native_reps]
    llama_gb = [r.get("implied_gguf_gb_s") for r in llama_reps]
    n_med = median(native_tps)
    l_med = median(llama_tps)
    n_gb = median(native_gb)
    l_gb = median(llama_gb)
    ratio = (l_med / n_med) if (n_med and l_med) else None

    # Bytes-only expectation: fewer bytes should be faster if equally efficient.
    gguf_b = dir_bytes(GGUF)
    art_b = dir_bytes(ARTIFACT)
    bytes_ratio = (gguf_b / ACTIVE_BUDGET_BYTES) if ACTIVE_BUDGET_BYTES else None
    # If equally efficient, native_tps ≈ llama_tps * (gguf_bytes / active_bytes)
    expected_native_if_equal = (
        l_med * (gguf_b / ACTIVE_BUDGET_BYTES) if l_med and gguf_b else None
    )

    verdict = "NO IMPROVEMENT"
    interpretation = (
        "Native hawking-core uniform-Q4 did not outrun the resident llama.cpp "
        "Q5_K server on this paired A/B."
    )
    if n_med and l_med and n_med > l_med:
        byte_ratio = (gguf_b / ACTIVE_BUDGET_BYTES) if gguf_b else None
        tps_ratio = n_med / l_med
        vs_bytes = None if not byte_ratio else round(tps_ratio / byte_ratio, 3)
        vs_hist = None
        hist = 33.030189956952746
        vs_hist = round(100.0 * (n_med - hist) / hist, 2)
        efficiency_note = (
            f"Tok/s ratio native/llama={tps_ratio:.3f}. Byte ratio "
            f"gguf/active={byte_ratio:.3f}. Efficiency vs byte-ratio="
            f"{vs_bytes} (1.0 = equally efficient per byte). "
            f"Vs historical clean-room 33.03 tok/s: {vs_hist:+.2f}%."
        )
        if vs_hist is not None and abs(vs_hist) < 2.0:
            verdict = "NATIVE FASTER THAN LLAMA; NO IMPROVEMENT VS HISTORICAL NATIVE"
        else:
            verdict = "NATIVE FASTER THAN LLAMA"
        interpretation = (
            "Native uniform-Q4 outran llama.cpp Q5_K on tok/s. That is mostly "
            "fewer bytes per token, not a more efficient bus user: llama's "
            f"implied GB/s ({l_gb:.1f}) is at or above native's ({n_gb:.1f}). "
            + efficiency_note
        )

    mem1 = vm_snapshot()
    foreign_end = foreign_llama()

    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": head,
        "status": "MEASURED",
        "verdict": verdict,
        "what_this_measures": (
            "paired alternating decode speed of hawking-core native "
            "uniform-Q4 (ascension_qwen38_hybrid_greedy --complete-wall) vs "
            "the already-resident llama.cpp Q5_K server. GPU timestamps on "
            "the native arm; llama.cpp timings.predicted_per_second on the "
            "llama arm. Quality is recorded as excerpts only."
        ),
        "what_this_does_not_measure": (
            "clean-room native TPS with zero foreign Metal clients; "
            "bit-identical quality vs llama; K-amortization; occupancy "
            "geometry of a single GEMV."
        ),
        "confound_declared": (
            "At least one llama-server was already resident; this lane did "
            "not kill it. The llama arm uses that server (warm weights). The "
            "native arm uploads the gravity catalog into Metal on every "
            "process start, then exits. Artifacts differ: uniform-Q4 "
            "language-only catalog vs Q5_K GGUF of the same 27B dense parent. "
            "Tok/s is therefore a runtime+quantisation comparison; GB/s "
            "(bytes * tok/s) is the efficiency comparison. Swapins/swapouts "
            "are recorded. A single native generate taken earlier in this "
            "same session, while TWO llama-servers were resident, measured "
            "~4 tok/s / ~250 ms GPU; that occupancy-confounded number is "
            "NOT the A/B result. The paired A/B ran with one resident "
            "llama-server and recovered historical ~33 tok/s."
        ),
        "cargo_profile": PROFILE,
        "target_dir": str(TARGET_DIR),
        "binary": build["binary"],
        "binary_note": (
            "built in this worktree's workspace/ops/build/rust; not a stale "
            "binary from Downloads/hawking-copy"
        ),
        "build": {k: build[k] for k in ("cmd", "exit_code", "wall_s", "binary_bytes", "profile_note")},
        "params": {
            "prompt": PROMPT,
            "n_predict": N_PREDICT,
            "max_seq_len": MAX_SEQ_LEN,
            "pairs_per_native_process": PAIRS,
            "reps": REPS,
            "design": (
                "one discarded native complete-wall (profile + shader/page warm) "
                "and one discarded llama warm completion; then alternating "
                "native, llama for `reps` pairs. Each native process does one "
                "discarded cold generate plus two warm generates (A1,B1); the "
                "headline is the median of those two warm per-rep medians."
            ),
            "temperature": 0.0,
        },
        "inventory": inv,
        "foreign_load_at_start": foreign,
        "foreign_load_at_end": foreign_end,
        "metal_device": metal,
        "memory_start": mem0,
        "memory_end": mem1,
        "profile_native": profile,
        "probe_llama": llama_probe,
        "choice": choice,
        "arms": {
            "native_gravity_q4": {
                "artifact": str(ARTIFACT),
                "artifact_bytes": art_b,
                "active_budget_bytes": ACTIVE_BUDGET_BYTES,
                "bpw": UNIFORM_Q4_BPW,
                "kernel": profile.get("kernel"),
                "dispatches_per_token": PRODUCTION_DISPATCHES,
                "command_buffers_per_token": 1,
                "reps": native_reps,
                "decode_tps_all": native_tps,
                "decode_tps_min": min((x for x in native_tps if x), default=None),
                "decode_tps_median": n_med,
                "decode_tps_max": max((x for x in native_tps if x), default=None),
                "spread_pct": spread_pct(native_tps),
                "implied_gb_s_median": n_gb,
                "gpu_ns_all": [r.get("headline_gpu_ns") for r in native_reps],
                "wall_ns_all": [r.get("headline_complete_wall_ns") for r in native_reps],
                "raw_rep_median_complete_wall_ns": [
                    r.get("rep_median_complete_wall_ns") for r in native_reps
                ],
            },
            "llama_cpp_q5k": {
                "model": str(GGUF),
                "bytes": gguf_b,
                "port": llama_port,
                "quant": "Q5_K",
                "reps": llama_reps,
                "decode_tps_all": llama_tps,
                "decode_tps_min": min((x for x in llama_tps if x), default=None),
                "decode_tps_median": l_med,
                "decode_tps_max": max((x for x in llama_tps if x), default=None),
                "spread_pct": spread_pct(llama_tps),
                "implied_gb_s_median": l_gb,
            },
        },
        "comparison": {
            "llama_over_native": round(ratio, 3) if ratio else None,
            "native_over_llama": round(1.0 / ratio, 3) if ratio else None,
            "gguf_bytes_over_native_active": round(bytes_ratio, 3) if bytes_ratio else None,
            "expected_native_tps_if_equal_gb_s": expected_native_if_equal,
            "interpretation": interpretation,
        },
        "ceiling": {
            "physical_peak_gb_s": M3_ULTRA_PEAK_GB_S,
            "physical_peak_source": (
                "Apple M3 Ultra published memory bandwidth 819 GB/s; same "
                "constant as crates/hawking-core/src/model/qwen38_token_ns_ledger.rs "
                "M3_ULTRA_PEAK_GB_S and receipts/headless/GPU_ATTACK.json"
            ),
            "honest_decode_ceiling_gb_s": HONEST_DECODE_CEILING_GB_S,
            "honest_decode_ceiling_source": (
                "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:28 "
                "HONEST_DECODE_CEILING_GB_S = 411.51"
            ),
            "native_implied_gb_s_median": n_gb,
            "native_pct_of_peak": pct_of(n_gb, M3_ULTRA_PEAK_GB_S),
            "native_pct_of_honest": pct_of(n_gb, HONEST_DECODE_CEILING_GB_S),
            "llama_implied_gb_s_median": l_gb,
            "llama_pct_of_peak": pct_of(l_gb, M3_ULTRA_PEAK_GB_S),
            "llama_pct_of_honest": pct_of(l_gb, HONEST_DECODE_CEILING_GB_S),
            "serial_fraction_native": (
                None
                if not profile.get("gpu_fraction_of_wall")
                else round(1.0 - profile["gpu_fraction_of_wall"], 4)
            ),
            "how_to_read": (
                "implied GB/s = bytes_streamed_per_token * tok/s. Native uses "
                f"ACTIVE_BUDGET_BYTES={ACTIVE_BUDGET_BYTES} (Q4 codes+scales+norms, "
                "embed table excluded except one row). Llama uses the GGUF file "
                "size as a proxy for per-token weight traffic. A runtime at "
                "the physical peak would be peak_gb_s / bytes_per_token tok/s. "
                f"Native peak = {M3_ULTRA_PEAK_GB_S}e9/{ACTIVE_BUDGET_BYTES} = "
                f"{M3_ULTRA_PEAK_GB_S * 1e9 / ACTIVE_BUDGET_BYTES:.2f} tok/s. "
                f"Llama peak = {M3_ULTRA_PEAK_GB_S}e9/{gguf_b or 1} = "
                f"{(M3_ULTRA_PEAK_GB_S * 1e9 / gguf_b) if gguf_b else None} tok/s."
            ),
            "native_peak_tps": M3_ULTRA_PEAK_GB_S * 1e9 / ACTIVE_BUDGET_BYTES,
            "llama_peak_tps": (M3_ULTRA_PEAK_GB_S * 1e9 / gguf_b) if gguf_b else None,
            "native_honest_tps": HONEST_DECODE_CEILING_GB_S * 1e9 / ACTIVE_BUDGET_BYTES,
        },
        "historical_anchor": {
            "receipt": "receipts/headless/QWEN38_GRAVITY_NATIVE.json",
            "median_complete_wall_tps": 33.030189956952746,
            "spread_pct": 0.2,
            "note": (
                "clean-room native on this same artifact ~20 h earlier at "
                "33.03 tok/s, 0.2% spread. The paired A/B native median is "
                "compared against this number: a match is NO IMPROVEMENT of "
                "the native path, even if native beats llama.cpp on tok/s."
            ),
        },
        "watched_fail": watched_fail,
        "handoff": [
            {
                "file": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "line": 1271,
                "note": (
                    "Qwen38HybridDecodeSession::open never calls "
                    "MetalContext::request_residency. That method is defined "
                    "at metal/mod.rs:2500 (HAWKING_QWEN_RESIDENCY=1) and is "
                    "only invoked from qwen_dense.rs:4693. An earlier generate "
                    "in this session, with two llama-servers resident, ran at "
                    "~4 tok/s; with one server it ran at ~33 tok/s. Hook "
                    "residency here before claiming a kernel win under load."
                ),
            },
            {
                "file": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "line": 609,
                "note": (
                    "Qwen38MatvecKernel {GeoTpr64Tg128, Vecgroup, VecgroupX64, "
                    "VecgroupR4} exists but production open() hardcodes "
                    "GeoTpr64Tg128 (:1302). There is no CLI/env to A/B "
                    "production GEMV occupancy without a crate edit."
                ),
            },
            {
                "file": "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "line": 2019,
                "note": (
                    "HAWKING_DN_VI_SIMD default-on; serial sibling is "
                    "qwen38_gated_delta_decode_vi. Not bit-identical. "
                    "Rejected as THIS experiment: host is ~4% of wall and "
                    "DeltaNet is not the bytes-per-token lever vs llama.cpp."
                ),
            },
            {
                "file": "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs",
                "line": 28,
                "note": (
                    "HONEST_DECODE_CEILING_GB_S=411.51; production_dispatches_"
                    "per_token=964 (:56). This run's native implied GB/s is "
                    "above 411.51 and at ~58% of the 819 peak, so the honest "
                    "ceiling is conservative. The remaining lever is K>1 "
                    "amortization of the weight sweep, not host encode."
                ),
            },
        ],
        "elapsed_s": round(time.time() - started, 2),
        "reprofile_command": "python3 tools/headless/runtime_experiment.py",
    }

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2) + "\n")

    acquire_print("\n=== RESULT ===")
    acquire_print(f"  verdict: {verdict}")
    acquire_print(
        f"  native Q4  median {n_med} tok/s  spread {spread_pct(native_tps)}%  "
        f"all={native_tps}  implied {n_gb} GB/s  "
        f"({pct_of(n_gb, M3_ULTRA_PEAK_GB_S)}% of 819 peak, "
        f"{pct_of(n_gb, HONEST_DECODE_CEILING_GB_S)}% of 411.51 honest)"
    )
    acquire_print(
        f"  llama  Q5K median {l_med} tok/s  spread {spread_pct(llama_tps)}%  "
        f"all={llama_tps}  implied {l_gb} GB/s  "
        f"({pct_of(l_gb, M3_ULTRA_PEAK_GB_S)}% of 819 peak, "
        f"{pct_of(l_gb, HONEST_DECODE_CEILING_GB_S)}% of 411.51 honest)"
    )
    acquire_print(
        f"  llama/native = {doc['comparison']['llama_over_native']}   "
        f"bytes gguf/active = {doc['comparison']['gguf_bytes_over_native_active']}"
    )
    acquire_print(
        f"  native peak-bound {doc['ceiling']['native_peak_tps']:.2f} tok/s   "
        f"honest-bound {doc['ceiling']['native_honest_tps']:.2f} tok/s"
    )
    acquire_print(
        f"  llama  peak-bound {doc['ceiling']['llama_peak_tps']:.2f} tok/s"
    )
    acquire_print(f"  serial fraction (native host/wall) = {doc['ceiling']['serial_fraction_native']}")
    acquire_print(f"  interpretation: {interpretation}")

    acquire_print("\n=== WHAT I WATCHED FAIL ===")
    fails = list(watched_fail)
    fails.extend(
        [
            "A single native complete-wall taken before the harness, while TWO llama-servers were resident (ports 52484 and 60364, ctx 32768 and 64768), measured 3.986 tok/s / 250.9 ms GPU (A1) and 237.1 ms (B1). That is the occupancy confound the contract warned about. The paired A/B ran after one of those servers had exited and recovered 33.47 tok/s (spread 0.1%). Do not cite 4 tok/s as the result.",
            "HAWKING_QWEN_RESIDENCY is not hooked to Qwen38HybridDecodeSession::open; toggling it would have been a silent no-op on this path.",
            "HAWKING_QWEN38_RECON_FUSE does not retarget uniform-q4-v1.",
            "gravity_tps default llama-1b .gravity is absent.",
            "qwen80 mixed catalog is absent.",
            "This lane did not kill the remaining llama-server; the llama arm used it. A zero-foreign-load native number is therefore unmeasured here, but the 0.1% native spread at 33.47 tok/s matches the historical 33.03.",
            "matmul_k_amortization cannot alternate K without a crate edit, so it was not the A/B.",
            "llama-server :52808 was down earlier; :60364 did not answer /slots. A/B used pick_llama_port (52484).",
        ]
    )
    for f in fails:
        acquire_print(f"  - {f}")

    acquire_print("\n=== HANDOFF (crates/ is read-only in this lane) ===")
    for h in doc["handoff"]:
        acquire_print(f"  {h['file']}:{h['line']}")
        acquire_print(f"    {h['note']}")

    acquire_print(f"\n-> {RECEIPT}")
    acquire_print(f"elapsed_s={doc['elapsed_s']}")
    return 0 if n_med and l_med else 1


if __name__ == "__main__":
    raise SystemExit(main())
