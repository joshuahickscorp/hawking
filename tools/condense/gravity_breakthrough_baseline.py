#!/usr/bin/env python3.12
"""Matched baseline on REAL GLM-5.2 artifacts (mandate sections 0 and 3).

Every comparison goes through gravity_bench_lab, which refuses unmatched specs.  Every
tensor comes from gravity_real_fixtures, read-only, off a safe-age shard.  Activations are
SYNTHETIC and every result says so.

Run:  python gravity_breakthrough_baseline.py [--layer N] [--reps 30]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_bench_lab as lab          # noqa: E402
import gravity_decode                    # noqa: E402
import gravity_forge as forge            # noqa: E402
import gravity_metal                     # noqa: E402
import gravity_real_fixtures as grf      # noqa: E402

SEED = 20260722
REPORT_DIR = HERE.parents[1] / "reports" / "condense" / "breakthrough"


# ------------------------------------------------------------------ fixtures / geometry

def pick_expert(layer: int | None) -> tuple[int, int, dict[str, grf.Fixture]]:
    """One real expert's gate/up/down from a layer the index reports complete."""
    index = grf.layer_index()
    candidates = [layer] if layer is not None else [
        l for l, e in index.items() if e["complete_expert_count"] >= 1]
    for cand in candidates:
        entry = index.get(cand)
        if not entry or not entry["complete_experts"]:
            continue
        expert = entry["complete_experts"][0]
        shards = entry["experts"][str(expert)]
        fixtures = {
            p: grf._fixture(grf.ARTIFACT_DIR / shards[p],
                            f"model.layers.{cand}.mlp.experts.{expert}.{p}_proj.weight")
            for p in grf.PROJECTIONS
        }
        return cand, expert, fixtures
    raise SystemExit("no layer with a complete expert on a safe shard")


def artifact_of(codes: dict) -> forge.PackedArtifact:
    ledger = forge.ByteLedger()
    ledger.add("indices", codes["indices"].size * 7)
    return forge.PackedArtifact(
        "product_quant", np.empty((0,), dtype=np.float32),
        codes["rows"] * codes["cols"], ledger, ledger.total_bits(), 0, {"pq_codes": codes})


def dense_from_codes(codes: dict) -> np.ndarray:
    """The dense weight the artifact encodes.  Built once, outside every timed region."""
    book = np.asarray(codes["codebooks"][0], dtype=np.float32)
    idx = np.asarray(codes["indices"])[:, 0]
    return book[idx].reshape(int(codes["rows"]), int(codes["cols"]))


def geometry_check(fixture: grf.Fixture, header: dict) -> dict:
    c = fixture.codes
    book = np.asarray(c["codebooks"][0])
    return {
        "tensor": fixture.tensor,
        "shape": list(fixture.shape),
        "D": int(c["D"]), "S": int(c["S"]), "sub": int(c["sub"]),
        "k": int(book.shape[0]), "codebook_shape": list(book.shape),
        "rotate": bool(c["rotate"]), "nchunk": int(c["nchunk"]),
        "rows": int(c["rows"]), "cols": int(c["cols"]),
        "descriptor_rung": fixture.descriptor.get("rung"),
        "descriptor_bpw": fixture.descriptor.get("bpw"),
        "header_production_rung": header["compression"].get("production_rung"),
        "header_packed_bpw": header["compression"].get("packed_bpw"),
        "matches_R0_header": (int(c["D"]) == 8 and int(c["S"]) == 1
                              and int(book.shape[0]) == 128 and not c["rotate"]
                              and header["compression"].get("production_rung") == "R0"),
    }


# ------------------------------------------------------------------ parity

def parity(fixture: grf.Fixture, dec) -> dict:
    codes = fixture.codes
    x = fixture.activation(seed=SEED)
    ref = forge.pq_execute(artifact_of(codes), x)
    got = dec.matvec(codes, x, key=gravity_metal.content_key(codes))
    diff = ref - got
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-30))
    cos = float(ref @ got / ((np.linalg.norm(ref) * np.linalg.norm(got)) + 1e-30))
    book = np.asarray(codes["codebooks"][0], dtype=np.float32)
    fp16_lossless = bool(np.array_equal(book, book.astype(np.float16).astype(np.float32)))
    return {
        "codebook_fp16_roundtrip_lossless": fp16_lossless,
        "codebook_fp16_roundtrip_max_abs_delta": float(
            np.abs(book - book.astype(np.float16).astype(np.float32)).max()),
        "tensor": fixture.tensor,
        "authority": "gravity_forge.pq_execute",
        "candidate": f"gravity_metal v{gravity_metal.KERNEL_VERSION}",
        "relative_l2": rel_l2,
        "max_abs_error": float(np.abs(diff).max()),
        "cosine": cos,
        "reference_abs_max": float(np.abs(ref).max()),
        "relative_max_gap": float(np.abs(diff).max() / (np.abs(ref).max() + 1e-30)),
        "gate": 2e-3,
        "gate_source": "gravity_decode.parity tolerance; 1e-6 is NOT the gate -- "
                       "gravity_metal casts the codebook to fp16, so 2.1e-4 is the fp16 floor",
        "within_gate": bool(np.abs(diff).max() / (np.abs(ref).max() + 1e-30) < 2e-3),
        "finite": bool(np.isfinite(got).all()),
        "activation_source": fixture.activation_source,
    }


def fp16_cast_control(dec) -> dict:
    """Why real-artifact parity is ~1e-6 while the synthetic path measured 2.1e-4.

    Same rung geometry, synthetic weights.  A synthetic pack keeps its codebook in fp32,
    so gravity_metal's cast to fp16 is lossy; a real artifact's codebook came off disk as
    fp16 already, so the same cast is a no-op.  Run both and let the numbers say it.
    """
    rng = np.random.default_rng(0)
    w = rng.standard_normal((512, 256)).astype(np.float32)
    art = forge.pack_product_quant(w, dim=8, subspaces=1, k=128, seed=0, iters=4)
    codes = art.config["pq_codes"]
    x = rng.standard_normal(256).astype(np.float32)
    ref = forge.pq_execute(art, x)
    got = dec.matvec(codes, x, key=gravity_metal.content_key(codes))
    book = np.asarray(codes["codebooks"][0], dtype=np.float32)
    diff = ref - got
    return {
        "purpose": "control for the 2.1e-4 parity figure; synthetic weights, R0 geometry, "
                   "512x256 (the size gravity_metal.selftest uses)",
        "synthetic_codebook_fp16_roundtrip_lossless": bool(
            np.array_equal(book, book.astype(np.float16).astype(np.float32))),
        "relative_max_gap": float(np.abs(diff).max() / (np.abs(ref).max() + 1e-30)),
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-30)),
    }


# ------------------------------------------------------------------ metal internals

def metal_bits(dec, codes: dict, key: str):
    """The cached upload plus a queue that will not deadlock when batching."""
    import Metal
    entry = dec._cache_tensor(codes, key)
    queue = dec.device.newCommandQueueWithMaxCommandBufferCount_(1024)
    return Metal, entry, queue


def encode(Metal, dec, enc, entry, n: int) -> None:
    enc.setComputePipelineState_(dec.pipeline)
    for slot, buf in enumerate((entry["idx"], entry["book"], entry["x"],
                                entry["y"], entry["dims"])):
        enc.setBuffer_offset_atIndex_(buf, 0, slot)
    enc.setThreadgroupMemoryLength_atIndex_(entry["scratch"], 0)
    groups = (entry["rows"] + gravity_metal.THREADS - 1) // gravity_metal.THREADS
    size_g = Metal.MTLSizeMake(groups, 1, 1)
    size_t = Metal.MTLSizeMake(gravity_metal.THREADS, 1, 1)
    for _ in range(n):
        enc.dispatchThreadgroups_threadsPerThreadgroup_(size_g, size_t)


def one_call_decomposed(Metal, objc, dec, entry, queue, xv: bytes) -> dict:
    """One dispatch, one command buffer, split into upload / encode / gpu / residual."""
    with objc.autorelease_pool():
        t0 = time.perf_counter_ns()
        entry["x"].contents().as_buffer(entry["x_bytes"])[:] = xv
        t1 = time.perf_counter_ns()
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        encode(Metal, dec, enc, entry, 1)
        enc.endEncoding()
        t2 = time.perf_counter_ns()
        cb.commit()
        cb.waitUntilCompleted()
        t3 = time.perf_counter_ns()
        gpu = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3          # s -> ms
        try:
            kernel = (cb.kernelEndTime() - cb.kernelStartTime()) * 1e3
        except Exception:                                           # noqa: BLE001
            kernel = None
    wall = (t3 - t0) / 1e6
    upload = (t1 - t0) / 1e6
    host_encode = (t2 - t1) / 1e6
    return {"wall_ms": wall, "x_upload_ms": upload, "host_encode_ms": host_encode,
            "gpu_execution_ms": gpu, "driver_kernel_ms": kernel,
            "command_buffer_ms": wall - upload - host_encode - gpu}


def batched(Metal, objc, dec, entry, queue, n: int) -> tuple[float, float]:
    """n dispatches in ONE command buffer.  Returns (wall_ms, gpu_ms) for the whole buffer."""
    with objc.autorelease_pool():
        t0 = time.perf_counter_ns()
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        encode(Metal, dec, enc, entry, n)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        wall = (time.perf_counter_ns() - t0) / 1e6
        gpu = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
    return wall, gpu


def min_ratio(baseline: lab.BenchResult, candidate: lab.BenchResult, *, label: str) -> dict:
    """The same matched pair, read at min instead of median.

    Not a second claim: identical BenchSpecs, identical timed region, still routed through
    require_matched and the refuted-value guard.  It exists because on a box with a live
    campaign on it min is the least contended order statistic, and the median speedup and
    the min speedup disagreeing is itself information.
    """
    lab.require_matched(baseline.spec, candidate.spec,
                        left=f"baseline:{baseline.baseline}", right=f"candidate:{label}")
    base_source, base_stats = baseline.timings.primary()
    cand_source, cand_stats = candidate.timings.primary()
    if base_source != cand_source:
        raise lab.MatchedBenchmarkError("component mismatch")
    lab.assert_not_refuted(kind="milliseconds", value=base_stats.min_ms)
    lab.assert_not_refuted(kind="milliseconds", value=cand_stats.min_ms)
    ratio = base_stats.min_ms / cand_stats.min_ms
    lab.assert_not_refuted(kind="ratio", value=ratio)
    return {
        "baseline": baseline.baseline, "candidate": label, "specs_matched": True,
        "spec_fingerprint": baseline.spec.fingerprint, "timing_source": base_source,
        "statistic": "min_ms",
        "baseline_min_ms": base_stats.min_ms, "candidate_min_ms": cand_stats.min_ms,
        "ratio": ratio, "slower_than_baseline": ratio < 1.0,
    }


# ------------------------------------------------------------------ one geometry

def run_geometry(fixture: grf.Fixture, dec, *, reps: int, warmup: int,
                 batch_n: int) -> dict:
    import objc
    import torch

    codes = fixture.codes
    rows, cols = int(codes["rows"]), int(codes["cols"])
    key = gravity_metal.content_key(codes)
    art = artifact_of(codes)

    spec = lab.BenchSpec(
        rows=rows, cols=cols, batch=1, input_seed=SEED,
        input_dtype="float32", output_dtype="float32",
        warmup=warmup, reps=reps,
        sync_boundary="per_call_host_sync",
        dependency_shape="independent_calls",
        pack_in_timed_region=False, unpack_in_timed_region=True,
    )
    x = fixture.activation(seed=SEED)                    # SYNTHETIC, labelled by the fixture

    # ---- (a) dense fp16 torch MPS.  Reconstruction happens HERE, outside the timed region.
    dense = dense_from_codes(codes)
    w16 = torch.from_numpy(dense.astype(np.float16)).to("mps")
    x16 = torch.from_numpy(x.astype(np.float16)).to("mps")
    torch.mps.synchronize()

    def dense_call():
        y = w16 @ x16
        torch.mps.synchronize()
        return y

    t_dense = lab.measure(dense_call, spec)

    # ---- (b) torch/MPS compact path
    t_compact = lab.measure(lambda: gravity_decode.decode_matvec_mps(codes, x), spec)

    # ---- (c) custom kernel, one call including waitUntilCompleted
    t_custom = lab.measure(lambda: dec.matvec(codes, x, key=key), spec)

    # ---- (e) CPU authority
    t_cpu = lab.measure(lambda: forge.pq_execute(art, x), spec)

    del w16, x16
    torch.mps.empty_cache()

    # ---- decomposition of the one-call wall
    Metal, entry, queue = metal_bits(dec, codes, key)
    xv = np.ascontiguousarray(x.ravel()).tobytes()
    for _ in range(warmup):
        one_call_decomposed(Metal, objc, dec, entry, queue, xv)
    parts = [one_call_decomposed(Metal, objc, dec, entry, queue, xv) for _ in range(reps)]

    def stats(field: str) -> lab.TimingStats:
        return lab.TimingStats(tuple(p[field] for p in parts))

    decomp_wall = stats("wall_ms")
    decomp = {
        "raw_samples": parts,
        "components": {
            f: {k: v for k, v in stats(f).to_json().items() if k != "raw_samples_ms"}
            for f in ("wall_ms", "x_upload_ms", "host_encode_ms", "gpu_execution_ms",
                      "command_buffer_ms")
        },
        "percent_of_wall_median": {
            f: 100.0 * stats(f).median_ms / decomp_wall.median_ms
            for f in ("x_upload_ms", "host_encode_ms", "gpu_execution_ms",
                      "command_buffer_ms")
        },
        "command_buffer_definition": "residual = wall - x_upload - host_encode - gpu_execution "
                                     "(commit, scheduling, waitUntilCompleted, readback)",
        "driver_kernel_ms_median": (
            lab.TimingStats(tuple(p["driver_kernel_ms"] for p in parts)).median_ms
            if parts[0]["driver_kernel_ms"] is not None else lab.UNMEASURED),
    }

    # ---- (d) N dispatches in ONE command buffer
    for _ in range(2):
        batched(Metal, objc, dec, entry, queue, batch_n)
    bat = [batched(Metal, objc, dec, entry, queue, batch_n) for _ in range(max(5, reps // 3))]
    bat_wall = lab.TimingStats(tuple(w / batch_n for w, _ in bat))
    bat_gpu = lab.TimingStats(tuple(g / batch_n for _, g in bat))

    # ---- roofline bytes
    traffic = gravity_metal.matvec_bytes(codes, threadgroup_memory_limit=dec.threadgroup_memory_limit)
    flops = 2 * rows * cols
    dense_bytes = rows * cols * 2 + cols * 2 + rows * 2
    compact_bytes = (rows * int(codes["nchunk"]) * 8       # int64 index upload
                     + 2 * rows * cols * 4                 # decoded fp32 written then read
                     + cols * 4 + rows * 4)

    results = {
        "dense": lab.BenchResult(
            baseline="dense_fp16_mps", spec=spec,
            timings=lab.ComponentTimings(end_to_end=t_dense),
            bytes_moved=dense_bytes, flops=flops,
            notes="torch fp16 matvec on MPS, torch.mps.synchronize() inside the timed region; "
                  "the dense weight was reconstructed from the artifact ONCE, outside it"),
        "compact": lab.BenchResult(
            baseline="torch_mps_compact", spec=spec,
            timings=lab.ComponentTimings(end_to_end=t_compact),
            bytes_moved=compact_bytes, flops=flops,
            notes="gravity_decode.decode_matvec_mps; bytes are an ANALYTIC model of the "
                  "materialized [rows,nchunk,sub] decode, not a counter reading"),
        "custom_1call": lab.BenchResult(
            baseline="custom_v2", spec=spec,
            timings=lab.ComponentTimings(end_to_end=t_custom),
            bytes_moved=traffic["executed_total_bytes"], flops=flops,
            notes="gravity_metal one call including waitUntilCompleted; bytes from "
                  "gravity_metal.matvec_bytes executed model (per-threadgroup re-reads billed)"),
        "cpu": lab.BenchResult(
            baseline="cpu_authority", spec=spec,
            timings=lab.ComponentTimings(end_to_end=t_cpu),
            bytes_moved=None, flops=flops,
            notes="gravity_forge.pq_execute on CPU, the parity authority"),
    }

    batch_spec = lab.BenchSpec(
        rows=rows, cols=cols, batch=1, input_seed=SEED,
        input_dtype="float32", output_dtype="float32",
        warmup=2, reps=len(bat),
        sync_boundary="per_command_buffer_host_sync",
        dependency_shape=f"{batch_n}_serial_dispatches_one_command_buffer",
        pack_in_timed_region=False, unpack_in_timed_region=True,
    )
    results["custom_batched"] = lab.BenchResult(
        baseline="custom_v2", spec=batch_spec,
        timings=lab.ComponentTimings(end_to_end=bat_wall, gpu_execution=bat_gpu),
        bytes_moved=traffic["executed_total_bytes"], flops=flops,
        notes=f"{batch_n} dispatches in ONE command buffer, per-dispatch amortized; "
              "queue built with maxCommandBufferCount=1024 and an autorelease pool per rep. "
              "NOT matched to the single-call spec, so no speedup is formed against it")

    pairs = (("compact", results["compact"]), ("custom_1call", results["custom_1call"]),
             ("cpu", results["cpu"]))
    speedups = [lab.speedup(results["dense"], cand) for _, cand in pairs]
    min_ratios = [min_ratio(results["dense"], cand, label=name) for name, cand in pairs]

    return {
        "fixture": fixture.as_json(),
        "geometry": {"rows": rows, "cols": cols, "nchunk": int(codes["nchunk"]),
                     "D": int(codes["D"]), "k": int(np.asarray(codes["codebooks"][0]).shape[0])},
        "traffic_model": traffic,
        "flops_per_matvec": flops,
        "results": {name: r.to_json() for name, r in results.items()},
        "matched_speedups": speedups,
        "matched_min_ratios": min_ratios,
        "one_call_decomposition": decomp,
        "batched": {"dispatches_per_command_buffer": batch_n,
                    "per_dispatch_wall_ms": bat_wall.to_json(),
                    "per_dispatch_gpu_ms": bat_gpu.to_json()},
        "roofline_gpu_billed_custom": {
            "seconds_used": bat_gpu.median_ms / 1e3,
            "achieved_gb_s": traffic["executed_total_bytes"] / (bat_gpu.median_ms / 1e3) / 1e9,
            "fraction_of_bandwidth_roof": (traffic["executed_total_bytes"]
                                           / (bat_gpu.median_ms / 1e3) / 1e9
                                           / lab.BANDWIDTH_ROOF_GB_S),
            "achieved_gflop_s": flops / (bat_gpu.median_ms / 1e3) / 1e9,
            "fraction_of_compute_roof": (flops / (bat_gpu.median_ms / 1e3) / 1e9
                                         / lab.COMPUTE_ROOF_GFLOP_S),
            "note": "billed against the batched per-dispatch GPU time, which is the only "
                    "measurement free of command-buffer cost",
        },
    }


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-n", type=int, default=32)
    args = ap.parse_args()

    started = time.time()
    import gravity_format

    layer, expert, fixtures = pick_expert(args.layer)
    headers = {p: gravity_format.read_header(Path(f.shard_path))
               for p, f in fixtures.items()}
    dec = gravity_metal.decoder()

    geometry = {p: geometry_check(f, headers[p]) for p, f in fixtures.items()}
    parities = {p: parity(f, dec) for p, f in fixtures.items()}
    distributions = {p: {k: v for k, v in grf.index_distribution(f.codes).items()
                         if k != "histogram"}
                     for p, f in fixtures.items()}

    geometries = {
        "gate_up": run_geometry(fixtures["gate"], dec, reps=args.reps,
                                warmup=args.warmup, batch_n=args.batch_n),
        "down": run_geometry(fixtures["down"], dec, reps=args.reps,
                             warmup=args.warmup, batch_n=args.batch_n),
    }

    honest = {
        name: {
            "geometry": [g["geometry"]["rows"], g["geometry"]["cols"]],
            "dense_fp16_mps_median_ms": g["results"]["dense"]["timings"]["end_to_end"]["median_ms"],
            "custom_v2_1call_median_ms": g["results"]["custom_1call"]["timings"]["end_to_end"]["median_ms"],
            "matched_speedup_custom_vs_dense": next(
                s["speedup"] for s in g["matched_speedups"]
                if s["candidate"] == "custom_v2"),
            "matched_min_ratio_custom_vs_dense": next(
                s["ratio"] for s in g["matched_min_ratios"]
                if s["candidate"] == "custom_1call"),
            "verdict": ("CUSTOM_SLOWER_THAN_DENSE" if next(
                s["speedup"] for s in g["matched_speedups"]
                if s["candidate"] == "custom_v2") < 1.0 else "CUSTOM_FASTER"),
        }
        for name, g in geometries.items()
    }

    report = {
        "schema": "hawking.glm52.breakthrough_baseline.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": "GLM-5.2 matched baseline on REAL artifacts",
        "activation_source": grf.SYNTHETIC,
        "activation_status": grf.teacher_activation_status(),
        "machine": dict(lab.MACHINE_FACTS, observed_platform=platform.platform(),
                        metal_device=str(dec.device.name()),
                        threadgroup_memory_limit=dec.threadgroup_memory_limit),
        "refuted_claims": [{"name": c.name, "kind": c.kind, "value": c.value,
                            "reason": c.reason} for c in lab.REFUTED_CLAIMS],
        "provenance": {
            "layer": layer, "expert": expert,
            "router_present": grf.layer_index()[layer]["router_present"],
            "expert_selection": "FIXED LIST: model.layers.N.mlp.gate.weight is absent from "
                                "every shard, so no routing decision was made or claimed",
            "artifact_root": str(grf.ARTIFACT_DIR),
            "safety_age_s": grf.SAFETY_AGE_SECONDS,
            "fixtures": {p: f.as_json() for p, f in fixtures.items()},
            "shard_headers": {p: {"shard": Path(f.shard_path).name,
                                  "production_rung": headers[p]["compression"].get("production_rung"),
                                  "packed_bpw": headers[p]["compression"].get("packed_bpw"),
                                  "tensor_count": headers[p]["integrity"]["tensor_count"]}
                              for p, f in fixtures.items()},
        },
        "geometry_check": geometry,
        "parity": parities,
        "fp16_cast_control": fp16_cast_control(dec),
        "index_distribution": distributions,
        "benchmarks": geometries,
        "honest_matched_speedup": honest,
        "elapsed_s": time.time() - started,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "GLM52_BREAKTHROUGH_BASELINE.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"wrote": str(out), "layer": layer, "expert": expert,
                      "honest": honest, "elapsed_s": report["elapsed_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
