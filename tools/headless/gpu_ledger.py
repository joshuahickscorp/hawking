#!/usr/bin/env python3
"""Per-token GPU ledger for the Qwen3.8 uniform-q4 incumbent.

Rank ACTIVE_BYTES_PER_TOKEN and DRAM_BYTES_PER_TOKEN above stored size.
Doctor's objective: useful function / (active bytes × token_ns).

A representation storing 1 EBPW but reading 5 EBPW/token is worse than a
2 EBPW one reading 1.2. Parallel sessions compete for the same unified-
memory subsystem; bytes actually moved per token governs concurrency.

Anything this box cannot measure is ABSENT with a physical reason — never
0, never an estimate labelled MEASURED. Cold and warm are reported
separately with a spread.

Does not load a second 27B. Writes receipts/headless/GPU_LEDGER.json.

    python3 tools/headless/gpu_ledger.py
    python3 tools/headless/gpu_ledger.py --measure   # live Metal complete-wall
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from noetic_operation_census import (  # noqa: E402
    ANCHOR_DISPATCHES,
    ANCHOR_GPU_CORES,
    ANCHOR_PARAMS,
    ANCHOR_ROOF_GB_S,
    dram_and_temp,
    gemv_organs,
    load_geometry,
    q4_matrix_bytes,
)

SCHEMA = "hawking.headless.gpu_ledger.v1"
RECEIPT = REPO / "receipts/headless/GPU_LEDGER.json"
PROBE_RECEIPT = REPO / "receipts/headless/GPU_LEDGER_METAL_PROBE.json"
RAW_GLOB = "GPU_LEDGER_RAW.run*.json"

ARTIFACT = Path.home() / "models/qwen38-gravity-uniform-q4-v1"
TOKENIZER = Path.home() / "models/qwen3.8-27b-abliterated-bf16/tokenizer.json"
DECODE_BIN = Path(
    "/Users/scammermike/Downloads/hawking-copy/workspace/ops/build/rust/"
    "release-fast/examples/ascension_qwen38_hybrid_greedy"
)
PROMPT = (
    "Explain, in ordinary prose and at length, how a compiler turns a "
    "for-loop into basic blocks and then into machine code."
)

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"

Q80_PCT_OF_700 = 0.79
Q80_IDLE_PCT = 51.0
Q80_CEILING_GB_S = 700.0
PEAK_GB_S = 819.0
HONEST_ROOF_GB_S = ANCHOR_ROOF_GB_S  # 778.8, N017 measured sequential DRAM roof
PRIOR_TOKEN_NS_GIT = "HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"
PRIOR_TOKEN_NS_GPU_NS = 33_912_333  # median_gpu_ns in that ledger

REQUIRED_STAGES = (
    "representation_decode",
    "routing",
    "operator",
    "activation",
    "kv_state",
    "sampling",
)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def qty(
    value,
    *,
    kind: str,
    unit: str,
    command: str,
    note: str | None = None,
    absent_reason: str | None = None,
    spread=None,
):
    if kind == ABSENT:
        if value is not None:
            raise ValueError(f"ABSENT quantity must not carry a value ({command})")
        if not absent_reason:
            raise ValueError(f"ABSENT quantity needs a physical reason ({command})")
        out = {
            "value": None,
            "kind": ABSENT,
            "unit": unit,
            "command": command,
            "absent_reason": absent_reason,
        }
        if note:
            out["note"] = note
        return out
    if value is None:
        raise ValueError(f"{kind} quantity has no value ({command})")
    out = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "absent_reason": None,
    }
    if note:
        out["note"] = note
    if spread is not None:
        out["spread"] = spread
    return out


def spread_of(values: list[int | float]) -> dict:
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0, "min": None, "median": None, "max": None, "spread_pct": None, "all": []}
    s = sorted(xs)
    med = s[len(s) // 2]
    spread_pct = None if med == 0 else 100.0 * (s[-1] - s[0]) / med
    return {
        "n": len(s),
        "min": s[0],
        "median": med,
        "max": s[-1],
        "spread_pct": spread_pct,
        "all": xs,
    }


def geo_tpr64_tg(rows: int) -> dict:
    threadgroups = (rows + 1) // 2
    threads = threadgroups * 128
    return {
        "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "rows": rows,
        "threadgroups": threadgroups,
        "threads": threads,
        "threads_per_threadgroup": 128,
        "simdgroups_per_threadgroup": 4,
        "gpu_cores": ANCHOR_GPU_CORES,
        "threadgroups_per_core_if_spread": threadgroups / ANCHOR_GPU_CORES,
        "note": (
            "Not a hardware occupancy counter. Launch geometry vs 60 M3 Ultra "
            "cores. A 17408-row gate launches 8704 TGs; the kernel is "
            "bandwidth-saturated, not occupancy-starved."
        ),
    }


def load_token_ns_prior() -> dict:
    p = subprocess.run(
        ["git", "show", PRIOR_TOKEN_NS_GIT],
        capture_output=True, text=True, cwd=REPO, timeout=30,
    )
    if p.returncode != 0:
        raise SystemExit(
            f"FAIL: git show {PRIOR_TOKEN_NS_GIT} failed: {p.stderr[-400:]}"
        )
    return json.loads(p.stdout)


def measure_artifact() -> dict:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    by = {}
    for t in manifest["tensors"]:
        name = t["name"]
        b = int(t["bytes"])
        if "embed_tokens" in name:
            cls = "embed_table"
        elif "lm_head" in name or name.endswith("lm_head.weight"):
            cls = "lm_head"
        elif ".mlp." in name:
            cls = "mlp"
        elif "linear_attn" in name:
            cls = "linear_attn"
        elif "self_attn" in name:
            cls = "full_attn"
        else:
            cls = "norms_and_mixer_f32"
        slot = by.setdefault(cls, {"bytes": 0, "n": 0, "elements": 0})
        slot["bytes"] += b
        slot["n"] += 1
        slot["elements"] += int(t["elements"])
    hidden = 5120
    embed_row = q4_matrix_bytes(1, hidden)
    stored = int(manifest["tensor_payload_bytes"])
    embed = by["embed_table"]["bytes"]
    active = stored - embed + embed_row
    n_files = 0
    disk = 0
    for p in ARTIFACT.rglob("*"):
        if p.is_file():
            n_files += 1
            disk += p.stat().st_size
    return {
        "path": str(ARTIFACT),
        "manifest_schema": manifest["schema"],
        "tensor_count": manifest["tensor_count"],
        "q4_tensors": manifest["q4_tensors"],
        "f32_tensors": manifest["f32_tensors"],
        "source_weight_elements": manifest["source_weight_elements"],
        "complete_physical_bpw": manifest["complete_physical_bpw"],
        "stored_payload_bytes": stored,
        "disk_bytes": disk,
        "n_files": n_files,
        "by_class_bytes": {k: v["bytes"] for k, v in by.items()},
        "by_class_n": {k: v["n"] for k, v in by.items()},
        "embed_table_bytes": embed,
        "embed_row_bytes": embed_row,
        "ACTIVE_BYTES_PER_TOKEN": active,
        "command": (
            "python3 -c 'import json; from pathlib import Path; "
            "m=json.loads(Path.home().joinpath("
            "\"models/qwen38-gravity-uniform-q4-v1/manifest.json\").read_text()); "
            "embed=next(t[\"bytes\"] for t in m[\"tensors\"] if \"embed_tokens\" in t[\"name\"]); "
            "row=34*(5120//64); print(m[\"tensor_payload_bytes\"]-embed+row)'"
        ),
    }


def metal_probe() -> dict:
    if PROBE_RECEIPT.is_file():
        saved = json.loads(PROBE_RECEIPT.read_text())
    else:
        saved = None
    swift = """
import Metal
import Foundation
guard let d = MTLCreateSystemDefaultDevice() else {
  print("{\\"error\\":\\"no metal device\\"}"); exit(1)
}
var counterSets: [[String: Any]] = []
if let sets = d.counterSets {
  for cs in sets {
    counterSets.append(["name": cs.name, "counters": cs.counters.map { $0.name }])
  }
}
let src = \"\"\"
#include <metal_stdlib>
using namespace metal;
kernel void occ_probe(device float* out [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[4];
  float acc = float(gid) + float(simd_lane);
  if (simd_lane == 0u) { red[simd_id % 4] = acc; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (gid == 0) { out[0] = red[0]; }
}
\"\"\"
var pipe: [String: Any] = ["compiled": false]
do {
  let lib = try d.makeLibrary(source: src, options: nil)
  let fn = lib.makeFunction(name: "occ_probe")!
  let p = try d.makeComputePipelineState(function: fn)
  pipe = [
    "compiled": true,
    "maxTotalThreadsPerThreadgroup": p.maxTotalThreadsPerThreadgroup,
    "threadExecutionWidth": p.threadExecutionWidth,
    "staticThreadgroupMemoryLength": p.staticThreadgroupMemoryLength,
  ]
} catch { pipe["error"] = String(describing: error) }
let sampling: [String: Bool] = [
  "atStageBoundary": d.supportsCounterSampling(.atStageBoundary),
  "atDrawBoundary": d.supportsCounterSampling(.atDrawBoundary),
  "atDispatchBoundary": d.supportsCounterSampling(.atDispatchBoundary),
  "atTileDispatchBoundary": d.supportsCounterSampling(.atTileDispatchBoundary),
  "atBlitBoundary": d.supportsCounterSampling(.atBlitBoundary),
]
let out: [String: Any] = [
  "name": d.name,
  "hasUnifiedMemory": d.hasUnifiedMemory,
  "recommendedMaxWorkingSetSize": d.recommendedMaxWorkingSetSize,
  "maxBufferLength": d.maxBufferLength,
  "counterSets": counterSets,
  "pipeline_reflection_toy_kernel": pipe,
  "supportsCounterSampling": sampling,
]
let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
"""
    try:
        with open("/tmp/n004_gpu_ledger_probe.swift", "w") as f:
            f.write(swift)
        p = subprocess.run(
            ["swift", "/tmp/n004_gpu_ledger_probe.swift"],
            capture_output=True, text=True, timeout=120,
        )
        if p.returncode == 0 and p.stdout.strip():
            line = p.stdout.strip().splitlines()[-1]
            probed = json.loads(line)
            probed["command"] = "swift /tmp/n004_gpu_ledger_probe.swift"
            PROBE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
            PROBE_RECEIPT.write_text(json.dumps(probed, indent=1))
            return probed
    except Exception:
        pass
    if saved:
        saved = dict(saved)
        saved["command"] = f"cat {PROBE_RECEIPT}  # live swift probe unavailable; using this-session file"
        saved["reprobe_failed"] = True
        return saved
    raise SystemExit("FAIL: Metal probe produced nothing and no saved probe exists")


def raw_paths() -> list[Path]:
    return sorted((REPO / "receipts/headless").glob(RAW_GLOB))


def load_raw_runs() -> list[dict]:
    paths = raw_paths()
    if len(paths) < 3:
        raise SystemExit(
            f"FAIL: need 3 process complete-wall RAWs under receipts/headless/{RAW_GLOB}; "
            f"found {len(paths)}. Run python3 tools/headless/gpu_ledger.py --measure"
        )
    return [json.loads(p.read_text()) for p in paths]


def occupancy_snapshot() -> dict:
    p = subprocess.run(
        ["ps", "-eo", "pid,rss,command"],
        capture_output=True, text=True, timeout=10,
    )
    lines = []
    second_27b = False
    for line in p.stdout.splitlines():
        low = line.lower()
        if any(s in low for s in ("llama-server", "ascension_qwen", "mlx_lm.server")):
            if "rg " in low or "gpu_ledger" in low:
                continue
            lines.append(line.strip())
            parts = line.split()
            try:
                rss_kb = int(parts[1])
            except (IndexError, ValueError):
                rss_kb = 0
            # A 27B resident runtime is tens of GiB, not tens of MiB.
            if rss_kb > 4_000_000:
                second_27b = True
    return {
        "ps_matches": lines,
        "loaded_a_second_27b": second_27b,
        "note": (
            "mlx_lm.server on a tiny tmp 4bit (~tens of MiB RSS) is not a 27B. "
            "A second Qwen3.8-27B would show RSS in the 10+ GiB class and is refused."
        ),
    }


def stage_split(token_ns: dict, live_gpu_ns: float) -> dict:
    """Exclusive GPU partition mapped onto the requested stage names.

    Isolated-family GPU timestamps live in the TOKEN_NS ledger (diagnostic
    CBs, not production shape). Production is one mixed CB; Metal on this
    box exposes only GPUTimestamp and does not support atDispatchBoundary
    sampling, so a production-shape per-kernel split is ABSENT. Isolated
    fractions are scaled onto this session's production GPU_NS and labelled
    DERIVED.
    """
    isolated = {x["name"]: x["median_gpu_ns"] for x in token_ns["isolated"]}
    probes = {p["class"]: p for p in token_ns["probes"]}
    prior_gpu = float(token_ns["median_gpu_ns"])
    scale = live_gpu_ns / prior_gpu if prior_gpu else 0.0

    def probe_decode(cls: str) -> float:
        p = probes[cls]
        return float(p["decode_median_gpu_ns"] - p["addr_median_gpu_ns"])

    representation_decode = (
        probe_decode("mlp") + probe_decode("dn") + probe_decode("gqa") + probe_decode("lm_head")
    )
    activation = (
        isolated["silu_64"]
        + isolated["sigmoid_16"]
        + isolated["input_norms"]
        + isolated["post_norms"]
        + isolated["final_norm"]
        + isolated["gated_rmsnorm_48"]
    )
    kv_state = (
        isolated["stream_rec_state"]
        + isolated["stream_conv_state"]
        + isolated["stream_gqa_key"]
        + isolated["stream_gqa_value"]
    )
    sampling = isolated["argmax"]
    embed = isolated["embed"]
    named = representation_decode + activation + kv_state + sampling + embed
    operator = prior_gpu - named
    if operator < 0:
        operator = 0.0

    def row(name, prior_ns, kind_if_present=DERIVED):
        live = prior_ns * scale
        return {
            "stage": name,
            "prior_isolated_gpu_ns": prior_ns,
            "live_ns": live,
            "pct_of_live_gpu": (100.0 * live / live_gpu_ns) if live_gpu_ns else None,
            "kind": kind_if_present,
            "command": (
                f"git show {PRIOR_TOKEN_NS_GIT} | python3 -c "
                "'isolated family GPU medians, scaled onto this session GPU_NS'"
            ),
        }

    stages = {
        "representation_decode": {
            **row("representation_decode", representation_decode),
            "what": (
                "Nibble-unpack ALU: (decode_probe − addr_probe) on MLP/DN/GQA/lm_head "
                "isolated GEMV CBs. Packed decode stays in registers; bytes_written=0."
            ),
        },
        "routing": {
            "stage": "routing",
            "prior_isolated_gpu_ns": None,
            "live_ns": None,
            "pct_of_live_gpu": None,
            "kind": ABSENT,
            "value": None,
            "absent_reason": (
                "This parent is dense Qwen3.8 (zero expert routes). The production "
                "graph has no router dispatch. A 0 would imply a measured empty "
                "router; the physical reason is the architecture has no routing stage."
            ),
            "command": (
                "python3 -c 'from pathlib import Path; import json; "
                "d=json.loads(Path(\"receipts/headless/DOCTOR_V2_PRESCRIPTION.json\").read_text()); "
                "print(d[\"prior_science\"][\"q80\"][\"note\"])'"
            ),
        },
        "operator": {
            **row("operator", operator),
            "what": (
                "Remainder of production GPU after representation-decode, activation, "
                "KV-state, sampling, and embed are attributed out. Dominated by "
                "geo_tpr64 weight addressing (TOKEN_NS weight_addressing ~60% of wall)."
            ),
        },
        "activation": {
            **row("activation", activation),
            "what": "isolated silu + sigmoid + input/post/final RMSNorm + gated RMSNorm",
        },
        "kv_state": {
            **row("kv_state", kv_state),
            "what": "isolated sequential f32 stream of rec_state + conv_state + GQA K/V",
        },
        "sampling": {
            **row("sampling", sampling),
            "what": "isolated sample_argmax_f32; host sample_readback is a separate host field",
        },
    }
    # Closure over GPU: requested stages that are present + embed (named residual).
    present_sum = representation_decode + operator + activation + kv_state + sampling + embed
    stages["_closure"] = {
        "prior_gpu_ns": prior_gpu,
        "live_gpu_ns": live_gpu_ns,
        "scale": scale,
        "prior_named_sum_ns": present_sum,
        "prior_residual_ns": prior_gpu - present_sum,
        "embed_ns_prior": embed,
        "routing_in_sum": False,
        "identity": "representation_decode + operator + activation + kv_state + sampling + embed == prior GPU_NS; routing is ABSENT and not in the sum",
        "kind": DERIVED,
        "command": f"git show {PRIOR_TOKEN_NS_GIT}",
        "note": (
            "Isolated families are diagnostic CBs. Production is 1 mixed CB / 964 "
            "dispatches; per-dispatch GPU timestamps are ABSENT on this box "
            "(counterSets={timestamp}; atDispatchBoundary=false). Scale is "
            "live_gpu / prior_token_ns_gpu so the exclusive set partitions THIS "
            "session's GPU_NS."
        ),
    }
    return stages


def exclusive_token_ns_components(token_ns: dict, live_gpu_ns: float, live_wall_ns: float) -> list:
    prior_gpu = float(token_ns["median_gpu_ns"])
    scale = live_gpu_ns / prior_gpu if prior_gpu else 0.0
    rows = []
    for c in token_ns["components"]:
        gpu_like = c["resource_class"] == "gpu"
        live = c["ns_per_token"] * scale if gpu_like else c["ns_per_token"]
        # Host rows stay in ns as measured then; they are re-measured live
        # in the complete-wall fields, so keep prior as provenance only.
        rows.append({
            "component": c["component"],
            "prior_ns_per_token": c["ns_per_token"],
            "prior_pct_of_token_wall": c["pct_of_token_wall"],
            "resource_class": c["resource_class"],
            "bytes_read": c["bytes_read"],
            "bytes_written": c["bytes_written"],
            "dispatches": c["dispatches"],
            "scaled_live_ns_if_gpu": live if gpu_like else None,
            "kind": DERIVED if gpu_like else MEASURED,
            "method": c.get("method"),
            "command": f"git show {PRIOR_TOKEN_NS_GIT}",
        })
    return rows


def summarize_runs(runs: list[dict]) -> dict:
    process_gpu = []
    process_wall = []
    process_tps = []
    cold_gpu = []
    cold_wall = []
    cold_encode = []
    cold_wait = []
    warm_rep_gpu = []
    warm_rep_wall = []
    wait_minus = []
    encode_med = []
    submit_med = []
    readback_med = []
    dispatches = set()
    cbs = set()
    fallbacks = []
    token_ids = []
    texts = []
    concurrent = []
    for d in runs:
        a = d["authority"]
        process_gpu.append(int(a["headline_gpu_ns_per_token"]))
        process_wall.append(int(a["headline_complete_wall_ns_per_token"]))
        process_tps.append(float(a["headline_complete_tps"]))
        warm_rep_gpu.extend(int(x) for x in a["rep_median_gpu_ns"])
        warm_rep_wall.extend(int(x) for x in a["rep_median_complete_wall_ns"])
        first = d["cold_generate"]["cold_or_first_step"]
        cold_gpu.append(int(first["gpu_ns"]))
        cold_wall.append(int(first["complete_wall_ns"]))
        cold_encode.append(int(first["encode_ns"]))
        cold_wait.append(int(first["wait_ns"]))
        fallbacks.append(int(d["cold_generate"]["fallbacks"]))
        token_ids.append(d["identity"]["greedy_new_token_ids"])
        texts.append(d["cold_generate"].get("generated_text") or "")
        concurrent.append(bool(d["identity"]["concurrent_independent"]))
        for arm in [d["cold_generate"], *[w["summary"] for w in d["warm_reps"]]]:
            sd = arm["steady_decode"]
            dispatches.add(int(sd["dispatches"]))
            cbs.add(int(sd["command_buffers"]))
        # first warm arm medians for host fields
        w0 = d["warm_reps"][0]["summary"]["steady_decode"]
        wait_minus.append(int(w0["wait_minus_gpu_ns"]["median"]))
        encode_med.append(int(w0["encode_host_prepare_ns"]["median"]))
        submit_med.append(int(w0["submit_ns"]["median"]))
        readback_med.append(int(w0["sample_readback_ns"]["median"]))
    assert dispatches == {ANCHOR_DISPATCHES}, dispatches
    assert cbs == {1}, cbs
    assert all(f == 0 for f in fallbacks)
    assert len({tuple(t) for t in token_ids}) == 1
    assert all(c is False for c in concurrent)
    gpu_spread = spread_of(process_gpu)
    wall_spread = spread_of(process_wall)
    return {
        "n_process_runs": len(runs),
        "warm_process_gpu_ns": gpu_spread,
        "warm_process_wall_ns": wall_spread,
        "warm_process_tps": spread_of(process_tps),
        "warm_rep_gpu_ns": spread_of(warm_rep_gpu),
        "warm_rep_wall_ns": spread_of(warm_rep_wall),
        "cold_first_step_gpu_ns": spread_of(cold_gpu),
        "cold_first_step_wall_ns": spread_of(cold_wall),
        "cold_first_step_encode_ns": spread_of(cold_encode),
        "cold_first_step_wait_ns": spread_of(cold_wait),
        "warm_wait_minus_gpu_ns": spread_of(wait_minus),
        "warm_encode_ns": spread_of(encode_med),
        "warm_submit_ns": spread_of(submit_med),
        "warm_sample_readback_ns": spread_of(readback_med),
        "dispatches_per_token": ANCHOR_DISPATCHES,
        "command_buffers_per_token": 1,
        "fallbacks": fallbacks,
        "greedy_new_token_ids": token_ids[0],
        "generated_text_excerpt": texts[0][:240],
        "concurrent_independent": False,
        "raw_paths": [str(p.relative_to(REPO)) for p in raw_paths()],
        "gpu_timestamp_authority": (
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; "
            "never a CPU-wait proxy"
        ),
        "command": (
            "./tools/gpu_lane_lock.sh n004-gpu-ledger "
            f"{DECODE_BIN} --artifact-root {ARTIFACT} --tokenizer {TOKENIZER} "
            "--complete-wall --pairs 2 --max-new-tokens 16 --max-seq-len 256 "
            "--out receipts/headless/GPU_LEDGER_RAW.runN.json"
        ),
    }


def build() -> dict:
    art = measure_artifact()
    probe = metal_probe()
    runs = load_raw_runs()
    summ = summarize_runs(runs)
    token_ns = load_token_ns_prior()
    g = load_geometry()
    organs = gemv_organs(g)
    # Steady decode in these RAWs is 15 steps after a 34-token prompt: ~pos 42.
    seq_len = 42
    dram = dram_and_temp(g, organs, seq_len)
    occ = occupancy_snapshot()
    live_gpu = summ["warm_process_gpu_ns"]["median"]
    live_wall = summ["warm_process_wall_ns"]["median"]
    stages = stage_split(token_ns, live_gpu)
    components = exclusive_token_ns_components(token_ns, live_gpu, live_wall)

    active = art["ACTIVE_BYTES_PER_TOKEN"]
    dram_bytes = dram["executable_dram_bytes_per_token"]
    gpu_s = live_gpu / 1e9
    achieved_gb_s = (active / gpu_s) / 1e9 if gpu_s else None
    stored_ebpw = art["complete_physical_bpw"]
    active_ebpw = active * 8.0 / ANCHOR_PARAMS
    token_ns_s = live_wall
    ranking = 1.0 / (active * token_ns_s)

    counters = probe.get("counterSets") or []
    counter_names = sorted({
        c for cs in counters for c in (cs.get("counters") or [])
    })
    set_names = [cs.get("name") for cs in counters]
    only_timestamp = set_names == ["timestamp"] and counter_names == ["GPUTimestamp"]
    sampling = probe.get("supportsCounterSampling") or {}

    dram_counter_reason = (
        "MTLDevice.counterSets on this Apple M3 Ultra contains only the "
        f"'timestamp' set with counter {counter_names!r}. There is no DRAM "
        "bytes-moved, cache, or occupancy counter. Apple GPU does not expose "
        "NVIDIA-style Nsight bandwidth counters via the public Metal API."
    )
    idle_gap_reason = (
        "Production is one mixed command buffer. Intra-CB idle between the 964 "
        "dispatches sits inside GPUEndTime−GPUStartTime and cannot be split: "
        f"supportsCounterSampling.atDispatchBoundary={sampling.get('atDispatchBoundary')} "
        "(false). TCB production identity refuses ComputePassDescriptor boundary "
        "samples because they changed greedy token ids. wait_minus_gpu is host "
        "queue wait, not intra-CB GPU idle — see GPU_QUEUE_WAIT_NS."
    )
    simd_reason = (
        "No SIMD-utilization or occupancy counter is in MTLDevice.counterSets "
        f"(sets={set_names}). Instruments Metal System Trace can show occupancy "
        "but is not a CLI measurement this box can seal. Launch-geometry "
        "occupancy is labelled DERIVED, not MEASURED."
    )
    reg_reason = (
        "Apple Metal does not report register file pressure via a public GPU "
        "counter. MTLComputePipelineState.maxTotalThreadsPerThreadgroup is "
        "queryable after compiling a pipeline; the production decode binary does "
        "not emit it, and a toy kernel's 1024 is not geo_tpr64's occupancy. "
        "Xcode GPU debugger is out of band."
    )

    fields = {
        "GPU_NS": qty(
            int(live_gpu), kind=MEASURED, unit="ns/token",
            command=summ["command"],
            note="warm process-headline median of 3 fresh processes; GPUEnd−GPUStart of the production CB",
            spread=summ["warm_process_gpu_ns"],
        ),
        "GPU_BUSY_NS": qty(
            int(live_gpu), kind=MEASURED, unit="ns/token",
            command=summ["command"],
            note=(
                "Equal to GPU_NS on this 1-CB production path: the CB GPU interval "
                "includes any intra-CB bubbles, which cannot be subtracted (see "
                "GPU_IDLE_GAPS_NS ABSENT)."
            ),
            spread=summ["warm_process_gpu_ns"],
        ),
        "GPU_IDLE_GAPS_NS": qty(
            None, kind=ABSENT, unit="ns/token",
            command=(
                "python3 -c 'import json; print(json.load(open("
                "\"receipts/headless/GPU_LEDGER_METAL_PROBE.json\"))"
                "[\"supportsCounterSampling\"][\"atDispatchBoundary\"])'"
            ),
            absent_reason=idle_gap_reason,
        ),
        "GPU_QUEUE_WAIT_NS": qty(
            int(summ["warm_wait_minus_gpu_ns"]["median"]),
            kind=MEASURED, unit="ns/token",
            command=summ["command"] + "  # steady_decode.wait_minus_gpu_ns.median",
            note="host wait_until_completed minus GPUEnd−GPUStart; NOT intra-CB GPU idle",
            spread=summ["warm_wait_minus_gpu_ns"],
        ),
        "DRAM_READ_BYTES": qty(
            None, kind=ABSENT, unit="bytes/token",
            command="swift receipts/headless probe: MTLDevice.counterSets",
            absent_reason=dram_counter_reason,
        ),
        "DRAM_WRITE_BYTES": qty(
            None, kind=ABSENT, unit="bytes/token",
            command="swift receipts/headless probe: MTLDevice.counterSets",
            absent_reason=dram_counter_reason,
        ),
        "DRAM_READ_BYTES_DERIVED": qty(
            dram["executable_weight_bytes_per_token"]
            + dram["executable_rec_state_read_bytes"]
            + dram["executable_conv_state_read_bytes"]
            + dram["executable_gqa_kv_read_bytes_at_seq"]
            + dram["executable_gemv_input_read_bytes"],
            kind=DERIVED, unit="bytes/token",
            command=(
                "python3 -c 'from noetic_operation_census import load_geometry, gemv_organs, dram_and_temp; "
                "g=load_geometry(); print(dram_and_temp(g, gemv_organs(g), 42))'"
            ),
            note=f"encode-path traffic at seq_len={seq_len}; not a GPU counter",
        ),
        "DRAM_WRITE_BYTES_DERIVED": qty(
            dram["executable_activation_write_bytes"],
            kind=DERIVED, unit="bytes/token",
            command="tools/headless/noetic_operation_census.py::dram_and_temp executable_activation_write_bytes",
            note="activation + state writes the fused kernels emit; dense W write is 0",
        ),
        "dispatch_count": qty(
            ANCHOR_DISPATCHES, kind=MEASURED, unit="dispatches/token",
            command=summ["command"] + "  # steady_decode.dispatches",
            note="TokenCommandBuffer.dispatch_count: one kernel launch = one dispatch. Reproduced 964 on every cold and warm step of 3 processes.",
        ),
        "command_buffer_count": qty(
            1, kind=MEASURED, unit="command_buffers/token",
            command=summ["command"] + "  # steady_decode.command_buffers",
            note="Production shape is one CB. Reproduced 1 on every step.",
        ),
        "encoder_count": qty(
            ANCHOR_DISPATCHES, kind=DERIVED, unit="compute_encoders/token",
            command=(
                "rg -n 'new_compute_command_encoder' "
                "crates/hawking-core/src/metal/mod.rs; python3 -c "
                "'import json; print(json.load(open("
                "\"receipts/headless/GPU_LEDGER_RAW.run1.json\"))"
                "[\"identity\"][\"concurrent_independent\"])'"
            ),
            note=(
                "Off-mode TokenCommandBuffer::dispatch_threads_inner creates one "
                "compute encoder per dispatch (new_compute_command_encoder + "
                "end_encoding) unless a concurrent/serial group is open. This "
                "run has concurrent_independent=false and does not call "
                "begin_serial_group on the production token, so encoder_count = "
                "dispatch_count = 964. Not a PhysicalTraceGuard measurement "
                "(the complete-wall JSON does not emit encoder_count)."
            ),
        ),
        "synchronization_count": qty(
            1, kind=MEASURED, unit="waits/token",
            command=summ["command"] + "  # one wait_until_completed per production CB",
            note="One host wait per token, matching one command buffer.",
        ),
        "host_wait_ns": qty(
            int(summ["warm_wait_minus_gpu_ns"]["median"])
            + int(live_gpu),  # wait_ns = gpu + wait_minus_gpu
            kind=MEASURED, unit="ns/token",
            command=summ["command"] + "  # steady_decode.wait_ns.median",
            note="host Instant around wait_until_completed (includes GPU work from the host's view)",
            spread=None,
        ),
        "active_threadgroups": qty(
            geo_tpr64_tg(17408)["threadgroups"],
            kind=DERIVED, unit="threadgroups/launch",
            command="python3 tools/headless/gpu_ledger.py  # geo_tpr64_tg(17408)",
            note="Workhorse gate_proj launch: ceil(17408/2)=8704 TGs of 128 threads. Other organs differ by rows.",
        ),
        "occupancy_estimate": qty(
            geo_tpr64_tg(17408)["threadgroups_per_core_if_spread"],
            kind=DERIVED, unit="threadgroups/core",
            command="python3 tools/headless/gpu_ledger.py  # 8704/60",
            note=geo_tpr64_tg(17408)["note"],
        ),
        "SIMD_utilization": qty(
            None, kind=ABSENT, unit="fraction",
            command="swift Metal probe: MTLDevice.counterSets",
            absent_reason=simd_reason,
        ),
        "register_pressure": qty(
            None, kind=ABSENT, unit="registers/thread",
            command="MTLComputePipelineState is not emitted by the production decode JSON",
            absent_reason=reg_reason,
        ),
        "threadgroup_memory_bytes": qty(
            16, kind=DERIVED, unit="bytes/threadgroup",
            command=(
                "rg -n 'threadgroup float red\\[4\\]' "
                "crates/hawking-core/shaders/qwen_uniform_q4.metal"
            ),
            note=(
                "geo_tpr64_tg128: threadgroup float red[4] = 16 bytes. gated_delta "
                "binds set_threadgroup_memory_length(0, 128*4)=512 bytes on that "
                "organ only. Not a hardware occupancy counter."
            ),
        ),
        "ACTIVE_BYTES_PER_TOKEN": qty(
            active, kind=MEASURED, unit="bytes/token",
            command=art["command"],
            note=(
                "Sum of 755 tensor payloads minus the embedding table plus one "
                "Q4 gathered row. Dense decode streams every weight except the "
                "embed table. Rank this above stored size."
            ),
        ),
        "DRAM_BYTES_PER_TOKEN": qty(
            dram_bytes, kind=DERIVED, unit="bytes/token",
            command=(
                "python3 -c 'from noetic_operation_census import load_geometry, gemv_organs, dram_and_temp; "
                "print(dram_and_temp(load_geometry(), gemv_organs(load_geometry()), 42)"
                "[\"executable_dram_bytes_per_token\"])'"
            ),
            note=(
                "Weight stream + activation writes + rec/conv/KV reads + GEMV X "
                "reads at seq_len=42. Not a GPU DRAM counter (that is ABSENT). "
                "Rank this with ACTIVE_BYTES above stored size."
            ),
        ),
        "STORED_BYTES": qty(
            art["stored_payload_bytes"], kind=MEASURED, unit="bytes",
            command="python3 -c 'import json; from pathlib import Path; print(json.loads(Path.home().joinpath(\"models/qwen38-gravity-uniform-q4-v1/manifest.json\").read_text())[\"tensor_payload_bytes\"])'",
            note="On-disk tensor payloads including the embed table you do not stream per token. Ranked BELOW active/DRAM bytes.",
        ),
        "host_encode_ns": qty(
            int(summ["warm_encode_ns"]["median"]), kind=MEASURED, unit="ns/token",
            command=summ["command"] + "  # encode_host_prepare_ns",
            spread=summ["warm_encode_ns"],
        ),
        "host_submit_ns": qty(
            int(summ["warm_submit_ns"]["median"]), kind=MEASURED, unit="ns/token",
            command=summ["command"] + "  # submit_ns",
            spread=summ["warm_submit_ns"],
        ),
        "sample_readback_ns": qty(
            int(summ["warm_sample_readback_ns"]["median"]), kind=MEASURED, unit="ns/token",
            command=summ["command"] + "  # sample_readback_ns",
            note="Host read of the sampled u32 after the CB wait. Sampling compute is the argmax dispatch inside the CB.",
            spread=summ["warm_sample_readback_ns"],
        ),
        "COMPLETE_TOKEN_WALL_NS": qty(
            int(live_wall), kind=MEASURED, unit="ns/token",
            command=summ["command"],
            note="warm process-headline median; complete-wall = encode+submit+wait+epilogue+readback+state+tokenizer+bookkeeping+residual",
            spread=summ["warm_process_wall_ns"],
        ),
        "OS_PAGE_CACHE_COLD_GPU_NS": qty(
            None, kind=ABSENT, unit="ns/token",
            command="sudo purge && <decode>  # not run; purge requires root",
            absent_reason=(
                "A true disk-cold run requires dropping the kernel page cache "
                "(purge / posix_fadvise). This session cannot sudo purge. Three "
                "sequential processes share the ~14 GiB artifact page cache. "
                "Graph-cold first-step GPU (pipeline first-touch in a fresh "
                "process) IS measured separately as COLD_FIRST_STEP_GPU_NS."
            ),
        ),
        "COLD_FIRST_STEP_GPU_NS": qty(
            int(summ["cold_first_step_gpu_ns"]["median"]),
            kind=MEASURED, unit="ns",
            command=summ["command"] + "  # cold_generate.cold_or_first_step.gpu_ns",
            note="First step of the first generate in a fresh process. role=prefill. Graph-cold. Never averaged into the warm median.",
            spread=summ["cold_first_step_gpu_ns"],
        ),
        "hardware_occupancy_counter": qty(
            None, kind=ABSENT, unit="fraction",
            command="MTLDevice.counterSets",
            absent_reason=simd_reason,
        ),
        "per_dispatch_gpu_ns": qty(
            None, kind=ABSENT, unit="ns/dispatch",
            command="TokenCommandBuffer ProdCbGpu / atDispatchBoundary",
            absent_reason=(
                "atDispatchBoundary sampling is unsupported. ProdCbGpu boundary "
                "samples on ComputePassDescriptor changed Qwen3.8 greedy token "
                "ids vs plain new_compute_command_encoder (documented in "
                "crates/hawking-core/src/metal/mod.rs dispatch_threads_prod_cb). "
                "Correctness outranks attribution; production stays Off mode."
            ),
        ),
    }

    # host_wait_ns I set as wait_ns = gpu + wait_minus. Recompute cleanly.
    wait_ns = int(live_gpu + summ["warm_wait_minus_gpu_ns"]["median"])
    fields["host_wait_ns"] = qty(
        wait_ns, kind=MEASURED, unit="ns/token",
        command=summ["command"] + "  # wait_ns ≈ gpu_ns + wait_minus_gpu_ns",
        note="host Instant around wait_until_completed",
    )

    idle_pct_if_queue_wait = 100.0 * summ["warm_wait_minus_gpu_ns"]["median"] / live_wall
    gpu_frac = live_gpu / live_wall
    pct_of_700 = 100.0 * achieved_gb_s / Q80_CEILING_GB_S
    pct_of_819 = 100.0 * achieved_gb_s / PEAK_GB_S
    pct_of_roof = 100.0 * achieved_gb_s / HONEST_ROOF_GB_S

    q80 = {
        "anchor": {
            "pct_of_700_GBs": Q80_PCT_OF_700,
            "gpu_idle_pct": Q80_IDLE_PCT,
            "ceiling_gb_s": Q80_CEILING_GB_S,
            "kind": "PRIOR_MEASURED",
            "note": "Qwen3-80B MoE mixed decode, many CBs, host router readback. Sealed prior, not re-derived.",
        },
        "q4_incumbent": {
            "achieved_gb_s": achieved_gb_s,
            "pct_of_700_GBs": pct_of_700,
            "pct_of_819_peak": pct_of_819,
            "pct_of_595p9_roof": pct_of_roof,
            "gpu_as_fraction_of_wall": gpu_frac,
            "queue_wait_as_pct_of_wall": idle_pct_if_queue_wait,
            "kind": DERIVED,
            "command": "ACTIVE_BYTES_PER_TOKEN / (GPU_NS * 1e-9) / 1e9",
        },
        "verdict": "CONTRADICTED_FOR_THIS_INCUMBENT",
        "reading": (
            f"Q80 sat at {Q80_PCT_OF_700}% of a {Q80_CEILING_GB_S} GB/s ceiling with "
            f"{Q80_IDLE_PCT}% GPU idle because it was dispatch-bound across many CBs. "
            f"The q4 incumbent is 1 CB / 964 dispatches, streams {achieved_gb_s:.1f} GB/s "
            f"({pct_of_700:.1f}% of 700, {pct_of_roof:.1f}% of the {HONEST_ROOF_GB_S} GB/s sequential roof) "
            f"with GPU occupying {100*gpu_frac:.1f}% of complete-wall. Host queue wait is "
            f"{idle_pct_if_queue_wait:.2f}% of wall, not {Q80_IDLE_PCT}%. Applying the Q80 "
            "idle diagnosis to this vehicle is false. The remaining lever is ACTIVE BYTES "
            "per token (and encoder/dispatch ceremony), not 'the GPU is idle'."
        ),
    }

    doctor = {
        "formula": "useful_function / (ACTIVE_BYTES_PER_TOKEN × TOKEN_NS)",
        "useful_function": (
            "one coherent decode token of the qualified parent (fallbacks=0, "
            "greedy ids stable across warm reps, prose-on-compiler prompt)"
        ),
        "useful_function_scalar": {
            "value": 1.0,
            "kind": DERIVED,
            "unit": "coherent_token",
            "note": "Unit function. A capability-suite score would replace this; it is not invented here.",
        },
        "ACTIVE_BYTES_PER_TOKEN": active,
        "TOKEN_NS": int(live_wall),
        "denominator_byte_ns": active * token_ns_s,
        "ranking_quantity": ranking,
        "ranking_unit": "coherent_token / (byte · ns)",
        "rank_active_and_dram_above_stored_size": True,
        "why": (
            "A representation storing 1 EBPW but reading 5 EBPW/token is worse "
            "than a 2 EBPW one reading 1.2. Parallel sessions compete for the "
            "same unified-memory subsystem, so bytes actually moved per token "
            "is the quantity that governs concurrency."
        ),
        "stored_ebpw": stored_ebpw,
        "active_ebpw": active_ebpw,
        "stored_vs_active": (
            f"Stored complete_physical_bpw={stored_ebpw:.4f} includes the embed "
            f"table. Active EBPW={active_ebpw:.4f} is what decode actually moves. "
            "Optimizing stored size while increasing active bytes is a regression "
            "on this objective."
        ),
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "obligation": "N004 — GPU ledger per token: stop optimizing blind",
        "incumbent": {
            "artifact": str(ARTIFACT),
            "name": "qwen38-gravity-uniform-q4-v1",
            "complete_physical_bpw": stored_ebpw,
            "parameter_count": ANCHOR_PARAMS,
        },
        "did_not_load_second_27b": not occ["loaded_a_second_27b"],
        "occupancy": occ,
        "gpu_timestamp_authority": summ["gpu_timestamp_authority"],
        "measurement_label": "DIRTY_ENGINEERING",
        "measurement_label_reason": (
            "GPU lock held for the three process runs; other CPU/memory lanes may "
            "still be live. A tiny mlx_lm.server (~tens of MiB) was resident and "
            "is not a 27B. Not offered as CLEAN_CANDIDATE or BASE_TRUE_TPS."
        ),
        "ACTIVE_BYTES_PER_TOKEN": fields["ACTIVE_BYTES_PER_TOKEN"],
        "DRAM_BYTES_PER_TOKEN": fields["DRAM_BYTES_PER_TOKEN"],
        "rank_above_stored_size": [
            "ACTIVE_BYTES_PER_TOKEN",
            "DRAM_BYTES_PER_TOKEN",
        ],
        "doctor_objective": doctor,
        "fields": fields,
        "cold": {
            "what": "first step of the first generate in a fresh process (graph-cold prefill)",
            "gpu_ns": fields["COLD_FIRST_STEP_GPU_NS"],
            "wall_ns": qty(
                int(summ["cold_first_step_wall_ns"]["median"]),
                kind=MEASURED, unit="ns",
                command=summ["command"],
                spread=summ["cold_first_step_wall_ns"],
            ),
            "page_cache_cold": fields["OS_PAGE_CACHE_COLD_GPU_NS"],
            "dispatches": ANCHOR_DISPATCHES,
            "command_buffers": 1,
            "role": "prefill",
        },
        "warm": {
            "what": "in-process warm complete-wall after one discarded cold generate; 3 fresh processes × 4 warm reps",
            "gpu_ns": fields["GPU_NS"],
            "complete_token_wall_ns": fields["COMPLETE_TOKEN_WALL_NS"],
            "tps": qty(
                1e9 / live_wall, kind=DERIVED, unit="tok/s",
                command="1e9 / COMPLETE_TOKEN_WALL_NS",
                spread={
                    **summ["warm_process_tps"],
                    "note": "derived from per-process wall medians",
                },
            ),
            "wait_minus_gpu_ns": fields["GPU_QUEUE_WAIT_NS"],
            "dispatches": ANCHOR_DISPATCHES,
            "command_buffers": 1,
            "gpu_as_fraction_of_wall": gpu_frac,
            "n_process_runs": 3,
            "n_warm_reps_per_process": 4,
        },
        "stages": {k: v for k, v in stages.items() if not k.startswith("_")},
        "stages_closure": stages["_closure"],
        "exclusive_token_ns_components": components,
        "q80_anchor": q80,
        "metal_probe": probe,
        "artifact": art,
        "dram_geometry": {
            "seq_len": seq_len,
            "executable_weight_bytes_per_token": dram["executable_weight_bytes_per_token"],
            "executable_dram_bytes_per_token": dram["executable_dram_bytes_per_token"],
            "executable_activation_write_bytes": dram["executable_activation_write_bytes"],
            "dense_w_materialized_bytes_per_token": dram["dense_w_materialized_bytes_per_token"],
            "kind": DERIVED,
            "command": "tools/headless/noetic_operation_census.py::dram_and_temp",
        },
        "occupancy_launch_geometry": {
            "gate_proj": geo_tpr64_tg(17408),
            "lm_head": geo_tpr64_tg(248320),
            "kind": DERIVED,
        },
        "production_shape": {
            "dispatches_per_token": ANCHOR_DISPATCHES,
            "command_buffers_per_token": 1,
            "encoder_count_derived": ANCHOR_DISPATCHES,
            "source_unit_test": "crates/hawking-core/src/model/qwen38_token_ns_ledger.rs::production_dispatch_count_is_964",
        },
        "binary": {
            "path": str(DECODE_BIN),
            "bytes": DECODE_BIN.stat().st_size if DECODE_BIN.is_file() else None,
            "note": "Existing release-fast hybrid greedy; production graph default 964/1 CB (fusion is opt-in).",
        },
        "runs": summ,
        "anchors_reproduced": {
            "dispatches_per_token": 964,
            "command_buffers_per_token": 1,
            "reproduced": True,
        },
        "never_zero_unmeasurable": True,
        "sparse_note": (
            "This worktree is sparse. TOKEN_NS prior was read via git show "
            f"{PRIOR_TOKEN_NS_GIT}, not by writing receipts/ascent-2026-08-16."
        ),
    }
    return doc


def write_receipt(doc: dict) -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=1) + "\n")
    return RECEIPT


def measure_live(n_runs: int = 3, pairs: int = 2, max_new: int = 16) -> None:
    if not DECODE_BIN.is_file():
        raise SystemExit(f"FAIL: missing decode binary {DECODE_BIN}")
    if occupancy_snapshot()["loaded_a_second_27b"]:
        raise SystemExit("FAIL: a second 27B is resident; refuse to load q4")
    outdir = REPO / "receipts/headless"
    outdir.mkdir(parents=True, exist_ok=True)
    lock = REPO / "tools/gpu_lane_lock.sh"
    for i in range(1, n_runs + 1):
        out = outdir / f"GPU_LEDGER_RAW.run{i}.json"
        cmd = [
            str(lock), f"n004-gpu-ledger-r{i}",
            str(DECODE_BIN),
            "--artifact-root", str(ARTIFACT),
            "--tokenizer", str(TOKENIZER),
            "--prompt", PROMPT,
            "--complete-wall",
            "--pairs", str(pairs),
            "--max-new-tokens", str(max_new),
            "--max-seq-len", "256",
            "--out", str(out),
        ]
        print("RUN", " ".join(cmd[:6]), "...", flush=True)
        p = subprocess.run(cmd, cwd=str(REPO))
        if p.returncode != 0:
            raise SystemExit(f"FAIL: complete-wall run {i} exit {p.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="run 3 process complete-wall generates (loads the q4 incumbent, not a second 27B)")
    args = ap.parse_args()
    if args.measure:
        measure_live()
    doc = build()
    path = write_receipt(doc)
    a = doc["ACTIVE_BYTES_PER_TOKEN"]["value"]
    d = doc["DRAM_BYTES_PER_TOKEN"]["value"]
    g = doc["fields"]["GPU_NS"]["value"]
    w = doc["fields"]["COMPLETE_TOKEN_WALL_NS"]["value"]
    print("=== GPU LEDGER ===")
    print(f"  ACTIVE_BYTES_PER_TOKEN  {a}  ({doc['ACTIVE_BYTES_PER_TOKEN']['kind']})")
    print(f"  DRAM_BYTES_PER_TOKEN    {d}  ({doc['DRAM_BYTES_PER_TOKEN']['kind']})")
    print(f"  GPU_NS warm             {g}  spread_pct={doc['warm']['gpu_ns']['spread']['spread_pct']:.2f}")
    print(f"  TOKEN_NS warm           {w}")
    print(f"  dispatches/CBs          {doc['production_shape']['dispatches_per_token']}/"
          f"{doc['production_shape']['command_buffers_per_token']}")
    print(f"  Q80 verdict             {doc['q80_anchor']['verdict']}")
    print(f"  doctor ranking          {doc['doctor_objective']['ranking_quantity']}")
    print(f"  did_not_load_second_27b {doc['did_not_load_second_27b']}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
