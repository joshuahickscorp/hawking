#!/usr/bin/env python3
"""MLX quantized-matvec reference on this box. Measure only; no Hawking kernels.

Times are HOST_WALL after mx.synchronize(). That is not
MTLCommandBuffer.GPUEndTime-GPUStartTime. Host wall is an upper bound on GPU
time, so the implied GB/s is a lower bound on kernel GB/s.

Paired A/B are two independent quantized weights, alternated A,B,A,B,A,B.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "receipts" / "ascent-2026-08-16" / "matvec-mlx-reference.json"

HAWKING = {
    "q80_q4_gate_split_gbps": [2.6463467933491684, 2.476729089971367],
    "q80_q4_gate_bytes": 557056,
    "q80_q4_gate_gpu_ns_A": [212209, 209916, 210500],
    "q80_mixed_binary_gate_gbps": [2.4678828451882846, 2.4123680981595093],
    "dram_row_probe_seq_gbps": 560.2520996243217,
    "dram_row_probe_stride_gbps": 647.0,
    "source": "receipts/ascent-2026-08-16/dram-row-locality.json",
}

SHAPES = [
    {"name": "q80_512x2048", "rows": 512, "cols": 2048, "organ": "Q80 expert proj"},
    {"name": "q80_2048x512", "rows": 2048, "cols": 512, "organ": "Q80 expert proj T"},
    {"name": "qwen38_5120x17408", "rows": 5120, "cols": 17408, "organ": "Qwen3.8 MLP"},
    {"name": "qwen38_17408x5120", "rows": 17408, "cols": 5120, "organ": "Qwen3.8 MLP T"},
]

CODECS = [
    {"name": "affine_q4_gs64", "mode": "affine", "bits": 4, "group_size": 64},
    {"name": "affine_q2_gs64", "mode": "affine", "bits": 2, "group_size": 64},
    {"name": "mxfp4_gs32", "mode": "mxfp4", "bits": 4, "group_size": 32},
]


def _pkg_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("mlx")
    except Exception:
        return "unknown"


def _device_info() -> dict:
    fn = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
    try:
        return dict(fn()) if fn else {}
    except Exception as e:
        return {"error": str(e)}


def _quantize(w, codec):
    out = mx.quantize(
        w, group_size=codec["group_size"], bits=codec["bits"], mode=codec["mode"]
    )
    if len(out) == 3:
        return out[0], out[1], out[2]
    return out[0], out[1], None


def _qmm(x, wq, scales, biases, codec):
    kw = dict(
        transpose=True,
        group_size=codec["group_size"],
        bits=codec["bits"],
        mode=codec["mode"],
    )
    if biases is None:
        return mx.quantized_matmul(x, wq, scales, **kw)
    return mx.quantized_matmul(x, wq, scales, biases, **kw)


def _bytes(wq, scales, biases) -> int:
    n = int(wq.nbytes) + int(scales.nbytes)
    if biases is not None:
        n += int(biases.nbytes)
    return n


def _kernel_route(rows: int, cols: int, bits: int) -> dict:
    # M=1, transpose=True → dispatch_qmv → qmv (not wide, K not 64/128).
    pack = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    packs_per_thread = 1 if bits == 2 else 2
    k_align = pack * packs_per_thread * 32
    fast = (rows % 8 == 0) and (cols % k_align == 0)
    values_per_thread = pack * packs_per_thread
    return {
        "eval_gpu": "dispatch_qmv → qmv_fast" if fast else "dispatch_qmv → qmv",
        "threadgroup": [32, 2, 1],
        "threads_per_tg": 64,
        "simdgroups_per_tg": 2,
        "simdgroup_size": 32,
        "results_per_simdgroup": 4,
        "output_rows_per_tg": 8,
        "values_per_thread": values_per_thread,
        "k_block": values_per_thread * 32,
        "grid": [1, (rows + 7) // 8, 1],
        "qmv_fast": fast,
        "source": "mlx 0.32 quantized.h qmv_fast_impl + quantized.cpp qmv()/dispatch_qmv()",
    }


def _correctness(x, w, wq, scales, biases, codec) -> dict:
    y = _qmm(x, wq, scales, biases, codec)
    deq_kw = dict(
        group_size=codec["group_size"], bits=codec["bits"], mode=codec["mode"]
    )
    if biases is None:
        w_hat = mx.dequantize(wq, scales, **deq_kw)
    else:
        w_hat = mx.dequantize(wq, scales, biases, **deq_kw)
    # fp32 ref: fp16 matmul accum disagrees with qdot at ~0.1 abs on O(100) outputs.
    y_ref = x.astype(mx.float32) @ mx.transpose(w_hat.astype(mx.float32))
    mx.eval(y, y_ref)
    y32 = y.astype(mx.float32)
    diff = mx.abs(y32 - y_ref)
    mx.eval(diff)
    max_abs = float(mx.max(diff).item())
    peak = float(mx.max(mx.abs(y_ref)).item()) + 1e-6
    max_rel = max_abs / peak
    # Stated numeric gate: relative, not abs. Affine q4 on these shapes
    # measured ~8e-4 peak-relative vs dequant+fp32 matmul.
    gate = 5e-3
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "peak_abs_ref": peak,
        "gate_max_rel": gate,
        "passed": max_rel <= gate,
        "ref": "mx.dequantize + fp32 matmul (x @ W_hat.T)",
    }


def _pick_iters(nbytes: int) -> int:
    # Target ~30 ms at a pessimistic 20 GB/s so a 400 GB/s kernel is still
    # a few milliseconds and a 2.5 GB/s kernel is not a 30 s hostage.
    target_s = 0.03
    guess = 20e9
    n = int(target_s * guess / max(nbytes, 1))
    return max(8, min(n, 400))


def _time_eval_loop(fn, iters: int) -> int:
    mx.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return time.perf_counter_ns() - t0


def _time_multix(xs, wq, scales, biases, codec) -> int:
    mx.synchronize()
    t0 = time.perf_counter_ns()
    ys = [_qmm(xi, wq, scales, biases, codec) for xi in xs]
    mx.eval(*ys)
    mx.synchronize()
    return time.perf_counter_ns() - t0


def _stats(ns_list, nbytes, iters) -> dict:
    per = [n / iters for n in ns_list]
    # bytes / ns == GB/s
    gbps = [nbytes / n for n in per]
    return {
        "host_wall_ns_per_call": per,
        "host_wall_gbps": gbps,
        "min_ns": min(per),
        "median_ns": statistics.median(per),
        "max_ns": max(per),
        "min_gbps": min(gbps),
        "median_gbps": statistics.median(gbps),
        "max_gbps": max(gbps),
        "time_authority": "HOST_WALL after mx.synchronize(); NOT MTLCommandBuffer GPU time. GB/s is a lower bound on kernel GB/s.",
    }


def _make_side(rows, cols, codec, seed):
    mx.random.seed(seed)
    w = mx.random.normal((rows, cols)).astype(mx.float16)
    x = mx.random.normal((1, cols)).astype(mx.float16)
    mx.eval(w, x)
    wq, scales, biases = _quantize(w, codec)
    mx.eval(wq, scales, *([biases] if biases is not None else []))
    return w, x, wq, scales, biases


def measure_one(shape, codec) -> dict:
    rows, cols = shape["rows"], shape["cols"]
    a = _make_side(rows, cols, codec, seed=1)
    b = _make_side(rows, cols, codec, seed=2)
    nbytes = _bytes(a[2], a[3], a[4])
    iters = _pick_iters(nbytes)

    def fn_a():
        return _qmm(a[1], a[2], a[3], a[4], codec)

    def fn_b():
        return _qmm(b[1], b[2], b[3], b[4], codec)

    for _ in range(6):
        mx.eval(fn_a())
        mx.eval(fn_b())
    mx.synchronize()

    corr = _correctness(a[1], a[0], a[2], a[3], a[4], codec)

    loop_ns = []
    order = []
    for i in range(6):
        if i % 2 == 0:
            loop_ns.append(_time_eval_loop(fn_a, iters))
            order.append("A")
        else:
            loop_ns.append(_time_eval_loop(fn_b, iters))
            order.append("B")

    # One-eval graph of independent x's: amortizes host submit vs a loop of
    # single evals. Same W, so this is still the warm-resident organ case.
    n_multi = max(8, min(iters, 32))
    xs_a = []
    for i in range(n_multi):
        mx.random.seed(100 + i)
        xs_a.append(mx.random.normal((1, cols)).astype(mx.float16))
    mx.eval(*xs_a)
    for _ in range(2):
        _time_multix(xs_a, a[2], a[3], a[4], codec)
    multi_ns = []
    multi_order = []
    xs_b = []
    for i in range(n_multi):
        mx.random.seed(200 + i)
        xs_b.append(mx.random.normal((1, cols)).astype(mx.float16))
    mx.eval(*xs_b)
    for i in range(6):
        if i % 2 == 0:
            multi_ns.append(_time_multix(xs_a, a[2], a[3], a[4], codec))
            multi_order.append("A")
        else:
            multi_ns.append(_time_multix(xs_b, b[2], b[3], b[4], codec))
            multi_order.append("B")

    return {
        "shape": shape,
        "codec": codec,
        "bytes_w_scales_biases": nbytes,
        "dtype": "float16",
        "batch": 1,
        "kernel": _kernel_route(rows, cols, codec["bits"]),
        "correctness": corr,
        "eval_loop": {
            "iters_per_rep": iters,
            "pair_order": order,
            **_stats(loop_ns, nbytes, iters),
        },
        "multi_x_one_eval": {
            "xs_per_rep": n_multi,
            "pair_order": multi_order,
            **_stats(multi_ns, nbytes, n_multi),
        },
        "measurement_label": "DIRTY_ENGINEERING",
    }


def _make_xs(n: int, cols: int, seed0: int):
    xs = []
    for i in range(n):
        mx.random.seed(seed0 + i)
        xs.append(mx.random.normal((1, cols)).astype(mx.float16))
    mx.eval(*xs)
    return xs


def occupancy_sweep() -> dict:
    """Same W, N independent x, one mx.eval. Amortizes host submit.

    If isolated 512x2048 is ~2.4 GB/s because of launch, GB/s should rise
    with N. If the kernel itself is 2.4 GB/s, it stays flat.
    """
    codec = CODECS[0]
    rows, cols = 512, 2048
    a = _make_side(rows, cols, codec, seed=11)
    b = _make_side(rows, cols, codec, seed=12)
    nbytes = _bytes(a[2], a[3], a[4])
    # warmup kernels
    mx.eval(_qmm(a[1], a[2], a[3], a[4], codec))
    mx.eval(_qmm(b[1], b[2], b[3], b[4], codec))
    mx.synchronize()

    levels = [1, 4, 8, 32, 128, 256]
    out = []
    for n in levels:
        xs_a = _make_xs(n, cols, 3000 + n * 100)
        xs_b = _make_xs(n, cols, 8000 + n * 100)
        for _ in range(2):
            _time_multix(xs_a, a[2], a[3], a[4], codec)
        ns = []
        order = []
        for i in range(6):
            if i % 2 == 0:
                ns.append(_time_multix(xs_a, a[2], a[3], a[4], codec))
                order.append("A")
            else:
                ns.append(_time_multix(xs_b, b[2], b[3], b[4], codec))
                order.append("B")
        stats = _stats(ns, nbytes, n)
        print(
            f"  occupancy N={n:3d} {stats['min_gbps']:.2f}/"
            f"{stats['median_gbps']:.2f}/{stats['max_gbps']:.2f} GB/s "
            f"median {stats['median_ns']:.0f} ns/call",
            flush=True,
        )
        out.append({"n": n, "pair_order": order, "bytes_per_call": nbytes, **stats})
    return {
        "shape": "q80_512x2048",
        "codec": codec,
        "note": (
            "Same W reused across N x's in one eval. Counts N*bytes against "
            "host wall of that eval. Not a cold-DRAM stream. DIRTY_ENGINEERING."
        ),
        "levels": out,
        "measurement_label": "DIRTY_ENGINEERING",
    }


def _parse_gputrace(path: Path) -> dict:
    info = {"path": str(path), "exists": path.exists(), "files": [], "hints": []}
    if not path.exists():
        return info
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                sz = fp.stat().st_size
            except OSError:
                continue
            info["files"].append({"rel": str(fp.relative_to(path)), "bytes": sz})
    # Best-effort: look for numeric timing tokens in small text-ish files.
    keys = (
        b"GPUEndTime",
        b"GPUStartTime",
        b"gpuTime",
        b"gpu_time",
        b"duration",
        b"timeStamp",
    )
    for rec in info["files"]:
        fp = path / rec["rel"]
        if rec["bytes"] > 8_000_000 or rec["bytes"] < 16:
            continue
        try:
            blob = fp.read_bytes()
        except OSError:
            continue
        if any(k in blob for k in keys):
            info["hints"].append(rec["rel"])
    return info


def metal_captures() -> dict:
    """One-shot Metal captures. Parse is best-effort; traces stay in /tmp."""
    codec = CODECS[0]
    captures = []
    jobs = [
        ("q80_512x2048_single", 512, 2048, 1),
        ("q80_512x2048_x32", 512, 2048, 32),
        ("qwen38_5120x17408_single", 5120, 17408, 1),
    ]
    tmp = Path("/tmp/hawking-mlx-gputrace")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, rows, cols, n in jobs:
        side = _make_side(rows, cols, codec, seed=99)
        xs = _make_xs(n, cols, 5000)
        # warmup so the capture is not a shader compile
        mx.eval(*[_qmm(xi, side[2], side[3], side[4], codec) for xi in xs])
        mx.synchronize()
        dest = tmp / f"{name}.gputrace"
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        try:
            mx.metal.start_capture(str(dest))
            mx.eval(*[_qmm(xi, side[2], side[3], side[4], codec) for xi in xs])
            mx.synchronize()
            mx.metal.stop_capture()
            parsed = _parse_gputrace(dest)
            parsed["ok"] = True
            parsed["error"] = None
        except Exception as e:
            try:
                mx.metal.stop_capture()
            except Exception:
                pass
            parsed = {"ok": False, "error": str(e), "path": str(dest)}
        parsed["name"] = name
        parsed["n"] = n
        parsed["rows"] = rows
        parsed["cols"] = cols
        captures.append(parsed)
        print(f"  capture {name} ok={parsed.get('ok')} files={len(parsed.get('files', []))}", flush=True)
    return {
        "note": "gputrace is not committed. GPU timestamps only if parse found them.",
        "captures": captures,
        "measurement_label": "DIRTY_ENGINEERING",
    }


def _recompute_stats_inplace(results: list[dict]) -> None:
    for r in results:
        for key in ("eval_loop", "multi_x_one_eval"):
            block = r[key]
            iters = block.get("iters_per_rep") or block.get("xs_per_rep")
            per = block["host_wall_ns_per_call"]
            nbytes = r["bytes_w_scales_biases"]
            gbps = [nbytes / n for n in per]
            block["host_wall_gbps"] = gbps
            block["min_ns"] = min(per)
            block["median_ns"] = statistics.median(per)
            block["max_ns"] = max(per)
            block["min_gbps"] = min(gbps)
            block["median_gbps"] = statistics.median(gbps)
            block["max_gbps"] = max(gbps)


def _verdict(results: list[dict], sweep: dict | None) -> dict:
    q80 = [r for r in results if r["shape"]["name"] == "q80_512x2048" and r["codec"]["name"] == "affine_q4_gs64"]
    qwen = [r for r in results if r["shape"]["name"] == "qwen38_5120x17408" and r["codec"]["name"] == "affine_q4_gs64"]
    q80_r = q80[0] if q80 else None
    qwen_r = qwen[0] if qwen else None
    sweep_hi = None
    if sweep and sweep.get("levels"):
        last = sweep["levels"][-1]
        sweep_hi = {
            "n": last["n"],
            "min_gbps": last["min_gbps"],
            "median_gbps": last["median_gbps"],
            "max_gbps": last["max_gbps"],
            "ns": last["host_wall_ns_per_call"],
        }

    qwen_loop = qwen_r["eval_loop"] if qwen_r else None
    q80_loop = q80_r["eval_loop"] if q80_r else None
    qwen_min = qwen_loop["min_gbps"] if qwen_loop else 0.0
    q80_med = q80_loop["median_gbps"] if q80_loop else 0.0

    # Shape-split verdict. Do not average a 0.6 MB organ with a 50 MB organ.
    if qwen_min >= 100.0 and q80_med < 10.0:
        kind = "SHAPE_SPLIT"
        text = (
            f"SPLIT. Qwen3.8 5120x17408 affine-4 isolated eval is "
            f"{qwen_loop['min_gbps']:.1f}-{qwen_loop['max_gbps']:.1f} GB/s "
            f"(median {qwen_loop['median_gbps']:.1f}) HOST_WALL — that is the "
            f"100-400 GB/s band, so a quantized matvec CAN stream on this box "
            f"when the organ is ~50 MiB. Q80 512x2048 affine-4 isolated eval is "
            f"{q80_loop['min_gbps']:.2f}-{q80_loop['max_gbps']:.2f} GB/s "
            f"(median {q80_med:.2f}) HOST_WALL, same ballpark as Hawking's "
            f"2.47-2.65 GB/s GPU timestamps on that exact organ. Comparing a "
            f"0.59 MiB batch-1 qmv to a 64 MiB DRAM row probe is a category "
            f"error; the 230x framing is WRONG at Q80 organ size and RIGHT as "
            f"a kernel-quality gap at Qwen3.8 size."
        )
    elif qwen_min >= 100.0:
        kind = "KERNELS_ARE_THE_GAP"
        text = (
            f"MLX Qwen3.8 affine-4 isolated eval {qwen_loop['min_gbps']:.1f}-"
            f"{qwen_loop['max_gbps']:.1f} GB/s. Quantized matvec can stream. "
            f"Hawking 2.47-2.65 is not a hardware ceiling."
        )
    elif (sweep_hi and sweep_hi["min_gbps"] < 10.0) and q80_med < 10.0:
        kind = "FRAMING_WRONG_LATENCY_BOUND"
        text = (
            "MLX stays single-digit GB/s on Q80 512x2048 even after amortizing "
            "N independent x into one eval. Batch-1 quantized matvec at that "
            "shape is latency/occupancy bound. The 230x vs the 64 MiB row "
            "probe is the wrong comparison."
        )
    else:
        kind = "IN_BETWEEN"
        text = "See per-shape numbers. Do not collapse 0.6 MiB and 50 MiB organs."

    if sweep_hi:
        text += (
            f" Occupancy sweep 512x2048 affine-4 N={sweep_hi['n']}: "
            f"{sweep_hi['min_gbps']:.1f}-{sweep_hi['max_gbps']:.1f} GB/s "
            f"(median {sweep_hi['median_gbps']:.1f}) HOST_WALL."
        )

    return {
        "kind": kind,
        "text": text,
        "q80_512x2048_affine_q4_isolated": q80_loop,
        "qwen38_5120x17408_affine_q4_isolated": qwen_loop,
        "occupancy_top": sweep_hi,
        "hawking_packed_gbps": "2.47-2.65 (q80_q4_gate, GPU timestamps, 557056 B)",
        "row_probe_gbps": "560-647 (64 MiB read-reduce, GPU timestamps)",
        "label": "DIRTY_ENGINEERING",
        "time_authority": "HOST_WALL lower bound unless a capture parse says otherwise",
    }


def main() -> int:
    info = _device_info()
    print(
        f"mlx {_pkg_version()} device={mx.default_device()} info={info}",
        flush=True,
    )
    reuse = "--reuse" in sys.argv
    results = []
    if reuse and OUT.exists():
        prev = json.loads(OUT.read_text())
        results = prev.get("results") or []
        if results:
            _recompute_stats_inplace(results)
            for r in results:
                c = r.get("correctness") or {}
                if "max_rel" in c:
                    c["gate_max_rel"] = 5e-3
                    c["passed"] = c["max_rel"] <= 5e-3
                    c["ref"] = c.get("ref", "") + " [gate restated as max_rel<=5e-3]"
            print(f"reused {len(results)} configs from {OUT}", flush=True)
    if not results:
        for shape in SHAPES:
            for codec in CODECS:
                print(f"measuring {shape['name']} {codec['name']} ...", flush=True)
                r = measure_one(shape, codec)
                el = r["eval_loop"]
                mx_ = r["multi_x_one_eval"]
                print(
                    f"  bytes={r['bytes_w_scales_biases']} "
                    f"loop {el['min_gbps']:.2f}/{el['median_gbps']:.2f}/{el['max_gbps']:.2f} GB/s "
                    f"({el['median_ns']:.0f} ns) "
                    f"multix {mx_['min_gbps']:.2f}/{mx_['median_gbps']:.2f}/{mx_['max_gbps']:.2f} GB/s "
                    f"corr={r['correctness']['passed']} rel={r['correctness']['max_rel']:.3e}",
                    flush=True,
                )
                results.append(r)

    print("occupancy sweep 512x2048 affine_q4 ...", flush=True)
    sweep = occupancy_sweep()
    print("metal captures ...", flush=True)
    captures = metal_captures()

    receipt = {
        "schema": "hawking.ascent.matvec_mlx_reference.v2",
        "lane": "matvec-mlx-reference",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_label": "DIRTY_ENGINEERING",
        "question": (
            "Does Apple MLX quantized matvec at our organ shapes approach "
            "100-400 GB/s on this box, or does it also land in single-digit "
            "GB/s like our packed kernels (2.47-2.65) against a 560-647 GB/s "
            "DRAM row probe?"
        ),
        "mlx": {
            "version": _pkg_version(),
            "python": sys.executable,
            "device": str(mx.default_device()),
            "device_info": info,
        },
        "comparators": HAWKING,
        "time_authority": (
            "HOST_WALL after mx.synchronize(). Not GPUEndTime-GPUStartTime. "
            "Host wall >= GPU time, so reported GB/s is a lower bound."
        ),
        "byte_accounting": (
            "w.nbytes + scales.nbytes + biases.nbytes. MLX affine-4 gs64 "
            "512x2048 is 589824 B (fp16 scale+bias). Hawking q80_q4_gate "
            "counted 557056 B on the same geometry."
        ),
        "geometry": {
            "batch1_transpose_true": (
                "QuantizedMatmul::eval_gpu: M=1 < qmv_batch_limit → dispatch_qmv. "
                "K not in {64,128} so not qmv_quad. M==1 so not qmv_wide. "
                "qmv() uses qmv_fast when N%8==0 and K%(pack*packs_per_thread*32)==0."
            ),
            "qmv_fast": {
                "threadgroup": [32, 2, 1],
                "threads": 64,
                "simdgroups": 2,
                "simd_size": 32,
                "results_per_simdgroup": 4,
                "output_rows_per_tg": 8,
                "affine_q4_values_per_thread": 16,
                "affine_q4_k_block": 512,
                "vector_width": "uint32 packed weights, 8x4-bit per pack; qdot unrolled by 4 (uint16 loads)",
                "source_headers": [
                    "site-packages/mlx/include/mlx/backend/metal/kernels/quantized.h",
                    "ml-explore/mlx mlx/backend/metal/quantized.cpp (v0.32 line of qmv/dispatch_qmv)",
                ],
            },
        },
        "results": results,
        "occupancy_sweep_q80_512x2048": sweep,
        "metal_captures": captures,
        "verdict": _verdict(results, sweep),
        "fallbacks": 0,
        "pid": os.getpid(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print("WROTE", OUT, flush=True)
    print("VERDICT", receipt["verdict"]["kind"], receipt["verdict"]["text"], flush=True)
    return 0 if all(r["correctness"]["passed"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
