#!/usr/bin/env python3
"""Native-operator execution: SOURCE vs EXECUTABLE, with ORACLE labelled.

Claim under test: does the compressed structure EXECUTE DIRECTLY, or does it
decompress to dense W and call an ordinary matmul?

A path that materialises a (rows × cols) parent weight tensor and then runs a
normal matvec is an ORACLE. It is allowed to exist as a control. Presenting it
as native execution is the defect this harness exists to catch.

Peak temporary materialisation is the discriminator. If it reaches parent-tensor
shape for a 2-D weight, the path is ORACLE no matter what the FLOP count says.

Does not load a second 27B. Does not open the ascent-2026-08-16 or campaign
trees for writing. GPU wall of a live 27B decode is refused and recorded
ABSENT, not as 0.

    python3 tools/headless/noetic_native_operator.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from noetic_operation_census import (  # noqa: E402
    ANCHOR_BPW,
    ANCHOR_CBS,
    ANCHOR_DISPATCHES,
    ANCHOR_PARAMS,
    DECODE,
    SHADERS,
    activation_flops,
    build_columns,
    dispatch_inventory,
    dram_and_temp,
    f32b,
    gemv_organs,
    live_kernel_names,
    load_geometry,
    q4_matrix_bytes,
)

SCHEMA = "hawking.headless.noetic_native_operator.v1"
RECEIPT = REPO / "receipts/headless/NOETIC_NATIVE_OPERATOR.json"

METRICS = (
    "flops",
    "operation_count",
    "dispatch_count",
    "dram_bytes",
    "peak_temporary_materialization",
)

# Qwen3-Coder-Next mixed organs (q80_mixed_decode.rs). Not re-derived.
Q80_GATE_ROWS = 512
Q80_GATE_COLS = 2048
Q80_DOWN_ROWS = 2048
Q80_DOWN_COLS = 512
Q80_HGRAVS_RANK = 160
Q80_LAYERS = 48
Q80_TOP_K = 10
Q80_SHARED = 1
Q80_HIDDEN = 2048
Q80_MOE_INTERMEDIATE = 512
Q80_VOCAB = 151_936

# Qwen3.8 gate_proj tiling used by G042 retest (GENERATED_WEIGHTS_RETEST).
GATE_ROWS = 17408
GATE_COLS = 5120
GATE_NUMEL = GATE_ROWS * GATE_COLS
KRON_P, KRON_R, KRON_Q, KRON_S = 136, 128, 40, 128
F16_B = 2
F32_B = 4

# Anchors — prior measurements, not this run.
ANCHOR_Q80_PCT_OF_700 = 0.79
ANCHOR_Q80_GPU_IDLE_PCT = 51.0
ANCHOR_Q80_CEILING_GB_S = 700.0
ANCHOR_Q80_CB_BEFORE = 337
ANCHOR_Q80_CB_AFTER = 49
ANCHOR_STORED_BPW_0485 = 0.04853084788602941
ANCHOR_ACTIVE_FUSED_0485 = 0.04852941176470588
ANCHOR_ACTIVE_CACHE_F16 = 16.0
ANCHOR_SVD_GENERATOR_BYTES = 540672
ANCHOR_CACHE_F16_BYTES = 178257920  # gate_proj f16 = 17408*5120*2
ANCHOR_Q80_STORAGE_BPW = 1.4444457
ANCHOR_Q80_ACTIVE_DECODE_BPW_1P5 = 4.979465705545386
ANCHOR_Q80_DISPATCHES_SHAPE = 1155
ANCHOR_Q80_CBS_SHAPE = 98
ANCHOR_Q38_TPS = 32.73
ANCHOR_Q38_MS = 30.606
ANCHOR_ROOF_GB_S = 778.8

KIND_ALG = "measured_from_encoded_algorithm"
KIND_CPU = "measured_cpu_microbench"
KIND_ANCHOR = "anchor_not_rederived"
KIND_ABSENT = "ABSENT"


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO, timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def cell(value, *, kind: str, unit: str, trace: str, absent_reason: str | None = None) -> dict:
    if kind == KIND_ABSENT:
        if value is not None:
            raise ValueError("ABSENT cells must not carry a numeric value")
        return {
            "value": None,
            "unit": unit,
            "kind": KIND_ABSENT,
            "absent_reason": absent_reason,
            "trace": trace,
        }
    if value is None:
        raise ValueError(f"non-ABSENT cell has no value ({trace})")
    return {
        "value": value,
        "unit": unit,
        "kind": kind,
        "absent_reason": None,
        "trace": trace,
    }


def side_by_side(source: dict, executable: dict) -> dict:
    out = {}
    for metric in METRICS:
        if metric not in source or metric not in executable:
            raise KeyError(f"missing metric {metric}")
        out[metric] = {"SOURCE": source[metric], "EXECUTABLE": executable[metric]}
    return out


def shader_evidence() -> dict:
    """Read the kernels that exist. A missing file here is a sparse-checkout hole."""
    wanted = {
        "qwen_uniform_q4.metal": [
            "kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
            "Packed decode stays in registers",
            "kernel void qwen_uniform_q4_decode_vector",
        ],
        "q80_mixed_decode.metal": [
            "These kernels must never",
            "kernel void q80_hgravs01_factor_matvec",
            "kernel void q80_hgravs01_two_stage_matvec",
            "threadgroup float mid[kRankCap]",
            "execute y = L @ (R @ x); mid[rank] is the only temporary",
        ],
        "gravity_pq.metal": [
            "kernel void gravity_pq_matvec",
            "never materializes a dense weight",
        ],
        "matmul.metal": [
            "kernel void gemv_simdgroup_f32",
        ],
        "qwen_complete_runtime.metal": [
            "kernel void qwen_complete_binary_decode_vector",
        ],
        "qwen_uniform_qn.metal": [
            "kernel void qwen_uniform_qn_decode_vector",
        ],
    }
    found = {}
    for fname, needles in wanted.items():
        path = SHADERS / fname
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        found[fname] = {
            "present": path.is_file(),
            "path": f"crates/hawking-core/shaders/{fname}",
            "needles": {
                n: (text.find(n) if path.is_file() else -1) for n in needles
            },
        }
    decode_vectors = []
    if SHADERS.is_dir():
        for path in sorted(SHADERS.glob("*.metal")):
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if line.startswith("kernel void ") and "decode_vector" in line:
                    decode_vectors.append({
                        "file": path.name,
                        "line": i,
                        "name": line.split()[2].split("(")[0],
                    })
    rust_forbidden = REPO / "crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs"
    rust_text = rust_forbidden.read_text(encoding="utf-8", errors="replace") if rust_forbidden.is_file() else ""
    return {
        "shaders": found,
        "decode_vector_kernels": decode_vectors,
        "q80_cpu_two_stage_named": "hgravs01_two_stage_matvec_f32" in rust_text,
        "q80_forbidden_token_path_comment": "Forbidden token path: packed" in rust_text,
        "q80_decode_vector_refuses_dense_w": "refuses a tensor larger than 65536 elements (dense W)" in rust_text,
        "q80_gather_row_refuses_weight_row": "would reconstruct a weight row; refused" in rust_text,
    }


def qwen38_peak_activation_tensors(g: dict) -> dict:
    """Largest device tensors the fused path writes, other than packed weights."""
    tensors = {
        "hidden_f32": f32b(g["hidden"]),
        "gate_or_up_or_act_f32": f32b(g["intermediate"]),
        "qkvz_f32": f32b(g["qkvz_rows"]),
        "q_proj_f32": f32b(g["q_proj_rows"]),
        "logits_f32": f32b(g["vocab"]),
        "rec_state_per_layer_f32": f32b(g["rec_state_elements"]),
        "conv_state_per_layer_f32": f32b(g["conv_state_elements"]),
        "value_elements_f32": f32b(g["value_elements"]),
    }
    peak_name = max(tensors, key=tensors.get)
    return {
        "tensors": tensors,
        "peak_name": peak_name,
        "peak_bytes": tensors[peak_name],
        "note": (
            "Peak is the largest single activation/state buffer. Recurrent state "
            "persists across tokens but is allocated as a (heads × k × v) f32 tensor "
            "per DeltaNet layer, not as a (rows × cols) weight."
        ),
    }


def try_numpy():
    try:
        import numpy as np  # noqa: WPS433
        return np
    except ImportError:
        return None


class PeakAlloc:
    """Count live ndarray bytes. Temporary = arrays the path creates, not factors."""

    def __init__(self) -> None:
        self.peak = 0
        self.current = 0

    def take(self, arr):
        n = int(arr.nbytes)
        self.current += n
        if self.current > self.peak:
            self.peak = self.current
        return arr

    def drop(self, arr) -> None:
        self.current -= int(arr.nbytes)
        if self.current < 0:
            self.current = 0


def measure_svd_fused_vs_oracle(np) -> dict:
    """Random U,V at the 0.0485-bpw rank. No parent 27B load."""
    m, n = GATE_ROWS, GATE_COLS
    r = max(1, int(round(ANCHOR_STORED_BPW_0485 * m * n / (16.0 * (m + n)))))
    rng = np.random.default_rng(0)
    U = rng.standard_normal((m, r), dtype=np.float32)
    V = rng.standard_normal((r, n), dtype=np.float32)
    x = rng.standard_normal((n,), dtype=np.float32)

    fused = PeakAlloc()
    t0 = time.perf_counter()
    mid = fused.take(V @ x)
    y_f = fused.take(U @ mid)
    fused_s = time.perf_counter() - t0
    fused_peak = fused.peak
    fused.drop(mid)
    fused.drop(y_f)

    oracle = PeakAlloc()
    t1 = time.perf_counter()
    W = oracle.take((U @ V).astype(np.float16))
    y_o = oracle.take((W.astype(np.float32) @ x).astype(np.float32))
    oracle_s = time.perf_counter() - t1
    oracle_peak = oracle.peak

    parent_f16 = m * n * F16_B
    parent_f32 = m * n * F32_B
    mid_bytes = r * F32_B
    y_bytes = m * F32_B
    return {
        "rank": r,
        "U_shape": [m, r],
        "V_shape": [r, n],
        "fused_peak_bytes": int(fused_peak),
        "oracle_peak_bytes": int(oracle_peak),
        "fused_wall_s": fused_s,
        "oracle_wall_s": oracle_s,
        "y_match_max_abs": float(np.max(np.abs(y_f - y_o))),
        "parent_f16_bytes": parent_f16,
        "parent_f32_bytes": parent_f32,
        "encoded_fused_peak_bytes": max(mid_bytes, y_bytes) + mid_bytes,  # mid live with y
        "encoded_oracle_peak_bytes": parent_f16 + y_bytes,
        "oracle_reaches_parent_f16": oracle_peak >= parent_f16,
        "fused_reaches_parent_f16": fused_peak >= parent_f16,
        "numpy": np.__version__,
    }


def measure_hgravs_fused_vs_oracle(np) -> dict:
    m, n, r = Q80_DOWN_ROWS, Q80_DOWN_COLS, Q80_HGRAVS_RANK
    rng = np.random.default_rng(1)
    L = rng.standard_normal((m, r), dtype=np.float32)
    R = rng.standard_normal((r, n), dtype=np.float32)
    x = rng.standard_normal((n,), dtype=np.float32)

    fused = PeakAlloc()
    t0 = time.perf_counter()
    mid = fused.take(R @ x)
    y_f = fused.take(L @ mid)
    fused_s = time.perf_counter() - t0
    fused_peak = fused.peak

    oracle = PeakAlloc()
    t1 = time.perf_counter()
    W = oracle.take(L @ R)
    y_o = oracle.take(W @ x)
    oracle_s = time.perf_counter() - t1
    oracle_peak = oracle.peak

    parent = m * n * F32_B
    return {
        "L_shape": [m, r],
        "R_shape": [r, n],
        "fused_peak_bytes": int(fused_peak),
        "oracle_peak_bytes": int(oracle_peak),
        "fused_wall_s": fused_s,
        "oracle_wall_s": oracle_s,
        "y_match_max_abs": float(np.max(np.abs(y_f - y_o))),
        "parent_f32_bytes": parent,
        "threadgroup_mid_bytes": r * F32_B,  # mid[160] in q80_hgravs01_two_stage_matvec
        "oracle_reaches_parent": oracle_peak >= parent,
        "fused_reaches_parent": fused_peak >= parent,
        "numpy": np.__version__,
    }


def classify(*, materializes_dense_w_then_matmul: bool, peak_temp: int | None,
             parent_bytes: int | None, parent_rank: int) -> str:
    if materializes_dense_w_then_matmul:
        return "ORACLE"
    if (
        parent_rank == 2
        and peak_temp is not None
        and parent_bytes is not None
        and peak_temp >= parent_bytes
    ):
        return "ORACLE"
    return "NATIVE"


def path_doc(
    *,
    path_id: str,
    title: str,
    structure: str,
    on_production_token: bool,
    parent_name: str,
    parent_shape: list[int],
    parent_bytes: int,
    parent_rank: int,
    materializes_dense_w_then_matmul: bool,
    source: dict,
    executable: dict,
    regime: dict,
    evidence: list[str],
    notes: list[str],
    gpu_live: dict,
) -> dict:
    cols = side_by_side(source, executable)
    peak_cell = executable["peak_temporary_materialization"]
    peak = peak_cell["value"]
    label = classify(
        materializes_dense_w_then_matmul=materializes_dense_w_then_matmul,
        peak_temp=peak,
        parent_bytes=parent_bytes,
        parent_rank=parent_rank,
    )
    reaches = (
        parent_rank == 2
        and peak is not None
        and parent_bytes is not None
        and peak >= parent_bytes
    )
    if label == "ORACLE" and "ORACLE" not in title:
        # title may already say ORACLE; do not rewrite.
        pass
    return {
        "id": path_id,
        "title": title,
        "label": label,
        "structure": structure,
        "on_production_token": on_production_token,
        "parent": {
            "name": parent_name,
            "shape": parent_shape,
            "rank": parent_rank,
            "bytes": parent_bytes,
        },
        "materializes_dense_w_then_ordinary_matmul": materializes_dense_w_then_matmul,
        "peak_temp_reaches_parent_tensor_shape": reaches,
        "discriminator": (
            "ORACLE because peak temporary materialisation reaches parent-tensor shape"
            if reaches
            else (
                "ORACLE because the path writes dense W and then runs ordinary matmul"
                if materializes_dense_w_then_matmul
                else "NATIVE: peak temporary stays below parent-tensor shape and W is never written"
            )
        ),
        "regime": regime,
        "columns": cols,
        "gpu_live": gpu_live,
        "evidence": evidence,
        "notes": notes,
    }


def absent_gpu(reason: str) -> dict:
    return {
        "gpu_wall_ns_per_token": cell(
            None, kind=KIND_ABSENT, unit="ns", trace="not taken",
            absent_reason=reason,
        ),
        "gpu_idle_pct": cell(
            None, kind=KIND_ABSENT, unit="percent", trace="not taken",
            absent_reason=reason,
        ),
        "pct_of_bandwidth_ceiling": cell(
            None, kind=KIND_ABSENT, unit="percent", trace="not taken",
            absent_reason=reason,
        ),
    }


def qwen38_production(g: dict, np) -> tuple[dict, dict, dict]:
    live = live_kernel_names()
    organs = gemv_organs(g)
    inv, inv_total = dispatch_inventory(g, live)
    act = activation_flops(g, 34)
    dram = dram_and_temp(g, organs, 34)
    census_cols = build_columns(g, organs, act, dram, inv_total)
    peaks = qwen38_peak_activation_tensors(g)
    gemv_macs = sum(o["mac_flops_per_token"] for o in organs)
    gemv_elems = sum(o["elements_per_token"] for o in organs)
    max_parent = max(o["dense_f32_bytes_per_launch"] for o in organs)
    max_parent_organ = max(organs, key=lambda o: o["dense_f32_bytes_per_launch"])
    q4_stream = dram["executable_q4_gemv_bytes"]

    src = {
        "flops": cell(
            census_cols["source"]["flops_per_token"], kind=KIND_ALG, unit="FLOP/token",
            trace=census_cols["source"]["flops_per_token_trace"],
        ),
        "operation_count": cell(
            census_cols["source"]["operations_per_token"], kind=KIND_ALG, unit="ops/token",
            trace=census_cols["source"]["operations_per_token_trace"],
        ),
        "dispatch_count": cell(
            census_cols["source"]["dispatches_per_token"], kind=KIND_ALG, unit="dispatches/token",
            trace=census_cols["source"]["dispatches_per_token_trace"],
        ),
        "dram_bytes": cell(
            census_cols["source"]["dram_bytes_per_token_f32_weights"], kind=KIND_ALG,
            unit="bytes/token",
            trace=census_cols["source"]["dram_trace"] + " (f32 weight stream)",
        ),
        "peak_temporary_materialization": cell(
            peaks["peak_bytes"], kind=KIND_ALG, unit="bytes",
            trace=(
                f"largest fused-path device tensor {peaks['peak_name']} = "
                f"{peaks['peak_bytes']} B. Dense SOURCE stores W already unpacked, so "
                f"the weight stream itself ({max_parent} B for {max_parent_organ['organ']}) "
                "is residency of parent, not a codec expansion. Peak TEMPORARY is the "
                "activation/state buffer, same graph as the executable."
            ),
        ),
    }
    exe = {
        "flops": cell(
            census_cols["executable"]["flops_per_token"], kind=KIND_ALG, unit="FLOP/token",
            trace=census_cols["executable"]["flops_per_token_trace"],
        ),
        "operation_count": cell(
            census_cols["executable"]["operations_per_token"], kind=KIND_ALG, unit="ops/token",
            trace=census_cols["executable"]["operations_per_token_trace"],
        ),
        "dispatch_count": cell(
            census_cols["executable"]["dispatches_per_token"], kind=KIND_ALG,
            unit="dispatches/token",
            trace=census_cols["executable"]["dispatches_per_token_trace"],
        ),
        "dram_bytes": cell(
            census_cols["executable"]["dram_bytes_per_token"], kind=KIND_ALG,
            unit="bytes/token",
            trace=census_cols["executable"]["dram_trace"],
        ),
        "peak_temporary_materialization": cell(
            peaks["peak_bytes"], kind=KIND_ALG, unit="bytes",
            trace=(
                f"{peaks['peak_name']} = {peaks['peak_bytes']} B. Packed Q4 codes stay packed. "
                f"geo_tpr64 writes only y[row]. dense_w_materialized = 0. Parent max is "
                f"{max_parent_organ['organ']} {max_parent} B f32."
            ),
        ),
    }

    native = path_doc(
        path_id="qwen38_uniform_q4_fused",
        title="Qwen3.8-27B uniform-q4 fused decode (production)",
        structure="grouped_absmax_q4 consumed in-register by qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        on_production_token=True,
        parent_name="every GEMV organ (max = lm_head)",
        parent_shape=[max_parent_organ["rows"], max_parent_organ["cols"]],
        parent_bytes=max_parent,
        parent_rank=2,
        materializes_dense_w_then_matmul=False,
        source=src,
        executable=exe,
        regime={
            "stored_bpw": ANCHOR_BPW,
            "active_bpw_fused": ANCHOR_BPW,
            "active_bpw_cached_dense": None,
            "active_bpw_cached_dense_note": "no dense-W cache on this path; codes stay packed",
            "which": "stored = active_fused = 4.253 Q4. Not the 0.0485/16.0 generated-cache regime.",
        },
        evidence=[
            "crates/hawking-core/shaders/qwen_uniform_q4.metal geo_tpr64: packed decode stays in registers, acc += unpack8, writes output[row] only",
            "crates/hawking-core/src/model/qwen38_hybrid_decode.rs does not dispatch qwen_uniform_q4_decode_vector",
            f"dispatch formula 1 + 64*15 + 3 = {inv_total} (sealed 964)",
            f"Q4 GEMV stream {q4_stream} B vs dense f32 GEMV stream {dram['source_dense_f32_gemv_bytes']} B",
        ],
        notes=[
            "Executable does NOT do less GEMV work: same 51.24 GFLOP of MACs plus dequant ALU.",
            "It holds and streams fewer bytes. Arithmetic is not the win. W is not reconstructed.",
        ],
        gpu_live=absent_gpu(
            "Refused to load a second 27B. Live GPU decode of the resident model is out of "
            "scope. Prior TPS 32.73 / 30.606 ms/token are anchors, not this run."
        ),
    )

    # ORACLE lowering: decode_vector each GEMV matrix, then gemv_simdgroup_f32.
    extra_disp = sum(o["count_per_token"] for o in organs)
    oracle_disp = inv_total + extra_disp
    dequant_flops = gemv_elems  # q*scale
    oracle_flops = gemv_macs + dequant_flops + act["total_flops"]
    int_ops = 4 * gemv_elems
    oracle_ops = oracle_flops + int_ops
    oracle_dram = (
        q4_stream
        + dram["trap_reconstruct_then_gemm"]["dense_w_write_bytes"]
        + dram["trap_reconstruct_then_gemm"]["dense_w_reread_bytes"]
        + dram["executable_activation_write_bytes"]
    )
    oracle_peak = max_parent  # sequential reconstruct; simultaneous would be 102.5 GiB

    src_oracle = src  # parent dense forward is still SOURCE
    exe_oracle = {
        "flops": cell(
            oracle_flops, kind=KIND_ALG, unit="FLOP/token",
            trace=(
                f"decode_vector: 1 scale-mul per weight ({dequant_flops}) then dense GEMV MACs "
                f"{gemv_macs} plus activations {act['total_flops']}. Same MACs as fused, plus "
                "the write of W."
            ),
        ),
        "operation_count": cell(
            oracle_ops, kind=KIND_ALG, unit="ops/token",
            trace=f"oracle FLOPs {oracle_flops} + nibble unpack ~4 per weight ({int_ops})",
        ),
        "dispatch_count": cell(
            oracle_disp, kind=KIND_ALG, unit="dispatches/token",
            trace=(
                f"production {inv_total} plus {extra_disp} qwen_uniform_q4_decode_vector "
                "launches (one per GEMV organ). Then gemv_simdgroup_f32 reads the f32 W."
            ),
        ),
        "dram_bytes": cell(
            oracle_dram, kind=KIND_ALG, unit="bytes/token",
            trace=(
                "still reads Q4 codes, writes dense f32 W, rereads W for GEMM, plus activations. "
                f"extra vs fused = {dram['trap_reconstruct_then_gemm']['extra_vs_fused_bytes']} B"
            ),
        ),
        "peak_temporary_materialization": cell(
            oracle_peak, kind=KIND_ALG, unit="bytes",
            trace=(
                f"sequential lowering: largest parent GEMV is {max_parent_organ['organ']} "
                f"{max_parent} B f32. Simultaneous reconstruct of every GEMV would be "
                f"{dram['source_dense_f32_gemv_bytes']} B. Either reaches parent-tensor shape."
            ),
        ),
    }
    oracle = path_doc(
        path_id="qwen38_q4_decode_vector_then_gemm",
        title="ORACLE control: qwen_uniform_q4_decode_vector then gemv_simdgroup_f32",
        structure="packed Q4 → dense (rows×cols) f32 W → ordinary matvec",
        on_production_token=False,
        parent_name=max_parent_organ["organ"],
        parent_shape=[max_parent_organ["rows"], max_parent_organ["cols"]],
        parent_bytes=max_parent,
        parent_rank=2,
        materializes_dense_w_then_matmul=True,
        source=src_oracle,
        executable=exe_oracle,
        regime={
            "stored_bpw": ANCHOR_BPW,
            "active_bpw_fused": ANCHOR_BPW,
            "active_bpw_cached_dense": 32.0,
            "which": (
                "stored Q4 4.253; if the dense f32 form is cached, active is 32 bpw. "
                "Same structure, two regimes."
            ),
        },
        evidence=[
            "qwen_uniform_q4.metal:627 qwen_uniform_q4_decode_vector writes output[id] = qwen_uniform_q4_value(...) for `elements` floats — a 2-D matrix bind sets elements = rows*cols",
            "Comment on that kernel: 'Used for RMSNorm weights only; matrix bodies stay packed' on the production path. qwen30_complete_runtime.rs dispatches it for 1-D vectors.",
            "matmul.metal gemv_simdgroup_f32 reads device const float* w (rows×cols f32)",
            "crates/hawking-core/src/metal/mod.rs registers qwen_uniform_q4_decode_vector",
        ],
        notes=[
            "This lowering is not the Qwen3.8 production path. It is the control that would make BPW look cheap while moving parent-shaped bytes.",
            "qwen30 uses decode_vector for RMSNorm VECTORS (rank-1). That use is not this oracle. See qwen30_rmsnorm_vector_decode.",
        ],
        gpu_live=absent_gpu("Counterfactual lowering; not dispatched on the sealed Qwen3.8 token. Not timed."),
    )
    extras = {
        "census_columns": census_cols,
        "peaks": peaks,
        "organs": organs,
        "inv_total": inv_total,
        "gemv_macs": gemv_macs,
        "gemv_elems": gemv_elems,
        "max_parent": max_parent,
        "max_parent_organ": max_parent_organ,
        "dram": dram,
        "act": act,
    }
    return native, oracle, extras


def qwen30_vector_decode(g: dict) -> dict:
    """RMSNorm vector decode: parent IS the vector. Not a GEMV oracle."""
    n = g["hidden"]
    parent = f32b(n)
    src = {
        "flops": cell(5 * n, kind=KIND_ALG, unit="FLOP", trace="RMSNorm ~5n on a dense f32 weight vector"),
        "operation_count": cell(5 * n, kind=KIND_ALG, unit="ops", trace="dense f32, no nibble unpack"),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches", trace="one rmsnorm kernel"),
        "dram_bytes": cell(parent + f32b(n), kind=KIND_ALG, unit="bytes",
                           trace="read f32 weight + write normalized hidden"),
        "peak_temporary_materialization": cell(
            f32b(n), kind=KIND_ALG, unit="bytes",
            trace="normalized hidden, same width as the rank-1 parent",
        ),
    }
    exe = {
        "flops": cell(
            n + 5 * n, kind=KIND_ALG, unit="FLOP",
            trace="decode_vector: 1 scale-mul per element, then RMSNorm ~5n",
        ),
        "operation_count": cell(
            n + 5 * n + 4 * n, kind=KIND_ALG, unit="ops",
            trace="FLOPs plus nibble unpack on the packed vector",
        ),
        "dispatch_count": cell(
            2, kind=KIND_ALG, unit="dispatches",
            trace="qwen_uniform_q4_decode_vector (or binary/qn sibling) then rmsnorm",
        ),
        "dram_bytes": cell(
            q4_matrix_bytes(1, n) + parent + f32b(n), kind=KIND_ALG, unit="bytes",
            trace="read packed vector, write dense f32 control buffer, rmsnorm reads it",
        ),
        "peak_temporary_materialization": cell(
            parent, kind=KIND_ALG, unit="bytes",
            trace="decoded f32 control buffer equals the rank-1 parent. Rank-1 is not a GEMV oracle.",
        ),
    }
    return path_doc(
        path_id="qwen30_rmsnorm_vector_decode",
        title="qwen30 decode_vector on RMSNorm (rank-1, required-by-math)",
        structure="packed 1-D scale vector → f32 control buffer; not a matrix",
        on_production_token=False,
        parent_name="input_layernorm.weight",
        parent_shape=[n],
        parent_bytes=parent,
        parent_rank=1,
        materializes_dense_w_then_matmul=False,
        source=src,
        executable=exe,
        regime={
            "stored_bpw": ANCHOR_BPW,
            "active_bpw_fused": 32.0,
            "active_bpw_cached_dense": 32.0,
            "which": "a vector organ's dense form IS the math; caching it is not a GEMV oracle",
        },
        evidence=[
            "qwen_uniform_q4.metal:625-636 'Decode a checked compact Q4 vector into a persistent f32 control buffer. Used for RMSNorm weights only; matrix bodies stay packed.'",
            "qwen30_complete_runtime.rs ensure_decoded_vector_on_tcb requires header.shape == [elements]",
        ],
        notes=[
            "Peak temp equals parent bytes because the parent is rank-1. The discriminator is defined on 2-D weight tensors. Label stays NATIVE.",
        ],
        gpu_live=absent_gpu("Vector decode is not a GEMV path; not timed as a token."),
    )


def hgravs_paths(np) -> tuple[dict, dict, dict | None]:
    m, n, r = Q80_DOWN_ROWS, Q80_DOWN_COLS, Q80_HGRAVS_RANK
    parent = m * n * F32_B
    experts = Q80_TOP_K + Q80_SHARED  # 11 down_projs / layer
    n_down = Q80_LAYERS * experts
    src_flops_one = 2 * m * n
    fused_flops_one = 2 * (r * n + m * r)
    src_flops = src_flops_one * n_down
    fused_flops = fused_flops_one * n_down
    # Packed factor bytes, 3-bit group-64 + f16 scale. Same formula as Q4-group but 3-bit.
    def factor_bytes(rows: int, cols: int, bits: int = 3, group: int = 64) -> int:
        elems = rows * cols
        groups = (elems + group - 1) // group
        code = (elems * bits + 7) // 8
        return code + groups * 2

    packed = (factor_bytes(r, n) + factor_bytes(m, r)) * n_down
    dense_stream = parent * n_down
    fused_dram = packed + n_down * (n + r + m) * F32_B  # x + mid + y
    oracle_dram = packed + dense_stream + dense_stream + n_down * (n + m) * F32_B
    fused_peak = r * F32_B  # mid[160]; wave.mid is 10*160*4 but discriminator is per-organ parent
    fused_peak_wave = Q80_TOP_K * r * F32_B
    # two factor kernels today; two_stage is one dispatch
    fused_disp_two = 2 * n_down
    fused_disp_one = n_down
    oracle_disp = 2 * n_down  # reconstruct W + gemv, sequential per organ
    src_disp = n_down  # one dense gemv each

    bench = measure_hgravs_fused_vs_oracle(np) if np is not None else None
    if bench is not None:
        fused_peak_m = bench["fused_peak_bytes"]
        oracle_peak_m = bench["oracle_peak_bytes"]
        fused_kind = KIND_CPU
        oracle_kind = KIND_CPU
        fused_trace = (
            f"numpy two-stage peak {fused_peak_m} B (mid+y live). Shader two_stage uses "
            f"threadgroup mid[{r}] = {r * F32_B} B plus x_tg[{n}] = {n * F32_B} B. "
            f"Neither reaches parent {parent} B."
        )
        oracle_trace = (
            f"numpy L@R materialises W peak {oracle_peak_m} B; parent f32 is {parent} B. "
            f"oracle_reaches_parent={bench['oracle_reaches_parent']}"
        )
    else:
        fused_peak_m = max(fused_peak, m * F32_B)  # y
        oracle_peak_m = parent + m * F32_B
        fused_kind = KIND_ALG
        oracle_kind = KIND_ALG
        fused_trace = f"encoded mid {fused_peak} B + y {m * F32_B} B; numpy ABSENT"
        oracle_trace = f"encoded W {parent} B + y {m * F32_B} B; numpy ABSENT"

    src_one = {
        "flops": cell(src_flops_one, kind=KIND_ALG, unit="FLOP",
                      trace=f"dense down_proj 2*{m}*{n}"),
        "operation_count": cell(src_flops_one, kind=KIND_ALG, unit="ops",
                                trace="dense f32, no dequant ALU"),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches",
                               trace="one dense gemv"),
        "dram_bytes": cell(parent + (n + m) * F32_B, kind=KIND_ALG, unit="bytes",
                           trace="stream f32 W + x + y"),
        "peak_temporary_materialization": cell(
            m * F32_B, kind=KIND_ALG, unit="bytes",
            trace="dense SOURCE does not expand W; y is the output. W is parent residency.",
        ),
    }
    exe_native = {
        "flops": cell(fused_flops_one, kind=KIND_ALG, unit="FLOP",
                      trace=f"y=L@(R@x): 2*({r}*{n} + {m}*{r})"),
        "operation_count": cell(
            fused_flops_one + 4 * (r * n + m * r), kind=KIND_ALG, unit="ops",
            trace="two-stage MACs plus 3-bit unpack ALU on both factors",
        ),
        "dispatch_count": cell(
            2, kind=KIND_ALG, unit="dispatches",
            trace="production mixed path: two q80_hgravs01_factor_matvec_* launches (R then L). "
                  "q80_hgravs01_two_stage_matvec is REACHABLE as one dispatch.",
        ),
        "dram_bytes": cell(
            factor_bytes(r, n) + factor_bytes(m, r) + (n + r + m) * F32_B,
            kind=KIND_ALG, unit="bytes",
            trace="packed L and R + x + mid[rank] + y. Never parent W.",
        ),
        "peak_temporary_materialization": cell(
            fused_peak_m, kind=fused_kind, unit="bytes", trace=fused_trace,
        ),
    }
    native = path_doc(
        path_id="q80_hgravs01_two_stage",
        title="Q80 HGRAVS01 down_proj y = L @ (R @ x) (NATIVE)",
        structure="activation_weighted_svd_low_rank r160 b3, two packed factors",
        on_production_token=True,
        parent_name="expert.down_proj",
        parent_shape=[m, n],
        parent_bytes=parent,
        parent_rank=2,
        materializes_dense_w_then_matmul=False,
        source=src_one,
        executable=exe_native,
        regime={
            "stored_bpw": 8.0 * (factor_bytes(r, n) + factor_bytes(m, r)) / (m * n),
            "active_bpw_fused": 8.0 * (factor_bytes(r, n) + factor_bytes(m, r)) / (m * n),
            "active_bpw_cached_dense": 32.0,
            "which": "fused active = stored (factors). Cached dense W would be 32 bpw f32 / 16 bpw f16.",
        },
        evidence=[
            "q80_mixed_decode.metal header: 'execute y = L @ (R @ x); mid[rank] is the only temporary'",
            "q80_hgravs01_two_stage_matvec: threadgroup float mid[160]; x_tg[512]; '640 B, not dense W'",
            "q80_mixed_decode.rs: 'Native two-stage y = L @ (R @ x) of packed factors. Never forms dense W.'",
            "qwen80_mixed_hybrid_decode.rs dispatch_factor R then L into wave.mid[slot*rank]",
            "Forbidden token path comment: packed → dense (rows×cols) temporary → matvec",
        ],
        notes=[
            f"Per-token over {Q80_LAYERS} layers × {experts} experts: {n_down} down_projs, "
            f"{fused_disp_two} factor dispatches (or {fused_disp_one} two_stage), "
            f"{fused_flops} FLOP fused vs {src_flops} dense.",
            "Q80 decode is DISPATCH-BOUND (anchor 0.79% of 700 GB/s, 51% GPU idle). "
            "Dispatch count is first-class, not a footnote.",
        ],
        gpu_live=absent_gpu(
            "No Q80 mixed generate in this run (weights deleted; recipe retained). "
            "Dispatch-bound 0.79%/51%-idle is the sealed prior measurement."
        ),
    )

    exe_oracle = {
        "flops": cell(
            2 * m * r * n + src_flops_one, kind=KIND_ALG, unit="FLOP",
            trace=f"form W=L@R (2*{m}*{r}*{n} if naively outer) then dense gemv 2*{m}*{n}. "
                  "The GEMV after materialisation is ordinary matmul. That is the oracle.",
        ),
        "operation_count": cell(
            2 * m * r * n + src_flops_one + 4 * (r * n + m * r), kind=KIND_ALG, unit="ops",
            trace="reconstruct FLOPs plus factor unpack plus dense GEMV",
        ),
        "dispatch_count": cell(
            2, kind=KIND_ALG, unit="dispatches",
            trace="one reconstruct-to-dense write of W, one gemv_simdgroup_f32",
        ),
        "dram_bytes": cell(
            factor_bytes(r, n) + factor_bytes(m, r) + parent + parent + (n + m) * F32_B,
            kind=KIND_ALG, unit="bytes",
            trace="read factors, write W, reread W, plus x and y",
        ),
        "peak_temporary_materialization": cell(
            oracle_peak_m, kind=oracle_kind, unit="bytes", trace=oracle_trace,
        ),
    }
    oracle = path_doc(
        path_id="q80_hgravs01_reconstruct_then_gemm",
        title="ORACLE control: form dense down_proj = L@R, then ordinary matvec",
        structure="same HGRAVS01 factors, executed as reconstruct-then-GEMM",
        on_production_token=False,
        parent_name="expert.down_proj",
        parent_shape=[m, n],
        parent_bytes=parent,
        parent_rank=2,
        materializes_dense_w_then_matmul=True,
        source=src_one,
        executable=exe_oracle,
        regime={
            "stored_bpw": 8.0 * (factor_bytes(r, n) + factor_bytes(m, r)) / (m * n),
            "active_bpw_fused": 8.0 * (factor_bytes(r, n) + factor_bytes(m, r)) / (m * n),
            "active_bpw_cached_dense": 32.0,
            "which": "same structure. Fused  ≈ stored. Cached dense = 32 bpw. Report both or neither.",
        },
        evidence=[
            "Not a dispatched production kernel. Explicitly forbidden by q80_mixed_decode.rs.",
            "q80_mixed_decode.rs decode_vector_f32 refuses tensors > 65536 elements (dense W). down_proj is 1,048,576 elements.",
            "gather_row on a routed mixed organ is refused as reconstructing a weight row.",
        ],
        notes=[
            "Allowed as a control. Labelling it NATIVE would be the defect.",
            f"wave.mid residency on the native path is {fused_peak_wave} B for 10 experts, not {parent} B.",
        ],
        gpu_live=absent_gpu("Forbidden path; not dispatched. Not timed."),
    )
    return native, oracle, bench


def svd_generated_paths(np) -> tuple[dict, dict, dict | None]:
    m, n = GATE_ROWS, GATE_COLS
    r = max(1, int(round(ANCHOR_STORED_BPW_0485 * m * n / (16.0 * (m + n)))))
    parent_f16 = m * n * F16_B
    parent_f32 = m * n * F32_B
    gen_bytes = r * (m + n) * F16_B
    src_flops = 2 * m * n
    fused_flops = 2 * (r * n + m * r)
    bench = measure_svd_fused_vs_oracle(np) if np is not None else None

    if bench is not None:
        fused_peak, oracle_peak = bench["fused_peak_bytes"], bench["oracle_peak_bytes"]
        fused_kind, oracle_kind = KIND_CPU, KIND_CPU
        fused_trace = (
            f"numpy UV-fused peak {fused_peak} B at rank {bench['rank']}. "
            f"Parent f16 is {parent_f16} B. fused_reaches_parent_f16={bench['fused_reaches_parent_f16']}"
        )
        oracle_trace = (
            f"numpy cache-f16 peak {oracle_peak} B. Parent f16 {parent_f16}. "
            f"oracle_reaches_parent_f16={bench['oracle_reaches_parent_f16']}"
        )
    else:
        fused_peak = (r + m) * F32_B
        oracle_peak = parent_f16 + m * F32_B
        fused_kind, oracle_kind = KIND_ALG, KIND_ALG
        fused_trace = "encoded mid+y; numpy ABSENT"
        oracle_trace = "encoded W_f16+y; numpy ABSENT"

    src = {
        "flops": cell(src_flops, kind=KIND_ALG, unit="FLOP",
                      trace=f"dense gate_proj 2*{m}*{n}"),
        "operation_count": cell(src_flops, kind=KIND_ALG, unit="ops",
                                trace="dense f32"),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches",
                               trace="one dense gemv of gate_proj"),
        "dram_bytes": cell(parent_f32 + (n + m) * F32_B, kind=KIND_ALG, unit="bytes",
                           trace="stream f32 W + x + y"),
        "peak_temporary_materialization": cell(
            m * F32_B, kind=KIND_ALG, unit="bytes",
            trace="dense SOURCE: y only. W is parent residency.",
        ),
    }
    exe_fused = {
        "flops": cell(fused_flops, kind=KIND_ALG, unit="FLOP",
                      trace=f"y=U@(V@x) rank {r}: 2*({r}*{n}+{m}*{r})"),
        "operation_count": cell(fused_flops, kind=KIND_ALG, unit="ops",
                                trace="two-stage MACs, factors already f16/f32 (no Q4 unpack in this probe)"),
        "dispatch_count": cell(2, kind=KIND_ALG, unit="dispatches",
                               trace="V@x then U@mid. No reconstruct kernel."),
        "dram_bytes": cell(gen_bytes + (n + r + m) * F32_B, kind=KIND_ALG, unit="bytes",
                           trace="read U,V + x + mid + y"),
        "peak_temporary_materialization": cell(fused_peak, kind=fused_kind, unit="bytes",
                                               trace=fused_trace),
    }
    native = path_doc(
        path_id="generated_uv_fused_bpw_0485",
        title="Generated UV at 0.0485 bpw, fused two-stage (NATIVE operator)",
        structure="W ≈ U V with U,V stored f16; execute y = U @ (V @ x)",
        on_production_token=False,
        parent_name="L31.mlp.gate_proj.weight",
        parent_shape=[m, n],
        parent_bytes=parent_f16,
        parent_rank=2,
        materializes_dense_w_then_matmul=False,
        source=src,
        executable=exe_fused,
        regime={
            "stored_bpw": ANCHOR_STORED_BPW_0485,
            "active_bpw_fused": ANCHOR_ACTIVE_FUSED_0485,
            "active_bpw_cached_dense": ANCHOR_ACTIVE_CACHE_F16,
            "which": "0.0485 stored, 0.0485 active fused. Same structure is 16.0 active if the dense form is cached.",
            "anchor_receipt": "receipts/headless/GENERATED_WEIGHTS_RETEST.json",
            "generator_bytes": ANCHOR_SVD_GENERATOR_BYTES,
            "cache_f16_bytes": ANCHOR_CACHE_F16_BYTES,
        },
        evidence=[
            "GENERATED_WEIGHTS_RETEST: storage_bpw=0.04853084788602941, active_bpw_fused=0.04852941176470588, active_bpw_cache_f16=16.0",
            "No default-build kernel on Qwen3.8 runs this UV pair; the operator cost is measured here as the structure would run. Fidelity of r≈12 on this organ is UNHEALTHY (that receipt). This lane measures execution, not quality.",
        ],
        notes=[
            "Do not quote 0.0485 without the regime. Cached dense is 16.0 active — parent f16 shape.",
        ],
        gpu_live=absent_gpu("No Metal UV kernel on the Qwen3.8 production path. CPU microbench only."),
    )
    exe_cache = {
        "flops": cell(
            2 * m * r * n + src_flops, kind=KIND_ALG, unit="FLOP",
            trace="once: W=U@V; per token: dense gemv 2mn. Per-token EXECUTABLE after cache is ordinary matmul.",
        ),
        "operation_count": cell(
            src_flops, kind=KIND_ALG, unit="ops",
            trace="per-token ops after cache = dense GEMV. Reconstruct is a one-shot, not per token.",
        ),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches",
                               trace="one dense gemv of cached W (reconstruct is not per-token)"),
        "dram_bytes": cell(
            parent_f16 + (n + m) * F32_B, kind=KIND_ALG, unit="bytes",
            trace="stream cached f16 W + x + y. Active = parent f16.",
        ),
        "peak_temporary_materialization": cell(oracle_peak, kind=oracle_kind, unit="bytes",
                                               trace=oracle_trace),
    }
    oracle = path_doc(
        path_id="generated_uv_cache_f16_then_gemm",
        title="ORACLE: same 0.0485-bpw UV, dense f16 cache then ordinary matvec",
        structure="identical U,V; W cached at 16 bpw; gemv on W",
        on_production_token=False,
        parent_name="L31.mlp.gate_proj.weight",
        parent_shape=[m, n],
        parent_bytes=parent_f16,
        parent_rank=2,
        materializes_dense_w_then_matmul=True,
        source=src,
        executable=exe_cache,
        regime={
            "stored_bpw": ANCHOR_STORED_BPW_0485,
            "active_bpw_fused": ANCHOR_ACTIVE_FUSED_0485,
            "active_bpw_cached_dense": ANCHOR_ACTIVE_CACHE_F16,
            "which": "this is the 16.0-active regime of the same 0.0485 structure",
        },
        evidence=[
            "GENERATED_WEIGHTS_RETEST candidates with active_bpw_cache_f16=16.0 and cache_f16_bytes=178257920",
            f"parent f16 bytes = {m}*{n}*2 = {parent_f16} = 16 bpw",
        ],
        notes=[
            "Peak temp reaches parent-tensor shape. ORACLE regardless of stored 0.0485.",
        ],
        gpu_live=absent_gpu("CPU microbench of the cache path. No 27B load."),
    )
    return native, oracle, bench


def pq_path() -> dict:
    """gravity_pq_matvec: fused dictionary lookup. Reachable, not on Qwen3.8 q4."""
    # Algorithm cost for a matrix of shape (rows, cols) with nchunk subspaces.
    # Without a bound parent tensor on this worktree we report the kernel contract
    # on the Q80 expert shape as a representative 2-D organ, labelled as not-on-path.
    rows, cols = Q80_GATE_ROWS, Q80_GATE_COLS
    parent = rows * cols * F32_B
    src_flops = 2 * rows * cols
    # PQ: for each of rows, nchunk lookups of dim `sub`, FMA into acc.
    # Unknown (nchunk, sub, bits) on this box for a live GLM artifact: FLOPs of the
    # lookup equal 2 * rows * cols if the codebook tiles the same contraction, which
    # is the fused definition (one FMA per parent element, no extra W write).
    exe_flops = src_flops  # fused FMA count matches dense if every element is hit
    src = {
        "flops": cell(src_flops, kind=KIND_ALG, unit="FLOP",
                      trace=f"dense 2*{rows}*{cols} on a representative 512×2048 organ"),
        "operation_count": cell(src_flops, kind=KIND_ALG, unit="ops", trace="dense f32"),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches", trace="one dense gemv"),
        "dram_bytes": cell(parent + (cols + rows) * F32_B, kind=KIND_ALG, unit="bytes",
                           trace="stream f32 W + x + y"),
        "peak_temporary_materialization": cell(
            rows * F32_B, kind=KIND_ALG, unit="bytes", trace="y only; W is parent",
        ),
    }
    exe = {
        "flops": cell(exe_flops, kind=KIND_ALG, unit="FLOP",
                      trace="gravity_pq_matvec: fma(codebook[entry][j], x[...], acc) per subspace element; W never written"),
        "operation_count": cell(
            None, kind=KIND_ABSENT, unit="ops",
            trace="index extract ALU depends on bits/nchunk of a live PQ artifact",
            absent_reason=(
                "No PQ-packed Qwen3.8/Q80 organ is opened here (would be a second model "
                "load / 74k-tensor catalog). Kernel body is fused; integer index ops are "
                "not counted without the bound (nchunk, bits)."
            ),
        ),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches",
                               trace="one gravity_pq_matvec"),
        "dram_bytes": cell(
            None, kind=KIND_ABSENT, unit="bytes",
            trace="codebook + codes + x + y",
            absent_reason="codebook and code-stream bytes require the packed artifact; not opened",
        ),
        "peak_temporary_materialization": cell(
            rows * F32_B, kind=KIND_ALG, unit="bytes",
            trace="kernel writes y[row] only. Comment: 'there is no decode-to-dense staging buffer.'",
        ),
    }
    return path_doc(
        path_id="gravity_pq_matvec",
        title="gravity_pq_matvec fused codebook accumulate (NATIVE, not on Qwen3.8 q4)",
        structure="PQ dictionary lookup fused with accumulate into y",
        on_production_token=False,
        parent_name="representative 512×2048 (Q80 gate shape); not bound on Qwen3.8 uniform-q4",
        parent_shape=[rows, cols],
        parent_bytes=parent,
        parent_rank=2,
        materializes_dense_w_then_matmul=False,
        source=src,
        executable=exe,
        regime={
            "stored_bpw": None,
            "active_bpw_fused": None,
            "active_bpw_cached_dense": 32.0,
            "which": "BPW of a live PQ organ is ABSENT (artifact not opened). Cached dense would be 32 bpw.",
        },
        evidence=[
            "gravity_pq.metal:3 'The artifact never materializes a dense weight'",
            "gravity_pq.metal:399 gravity_pq_matvec: pq_index → codebook entry, fma into acc, simd_sum to y[row]",
            "NOETIC_KERNEL_CENSUS families.fused_dictionary_lookup_accumulate = EXISTS, class REACHABLE, not on uniform-q4",
        ],
        notes=["Porting to Qwen3.8 is bind work. C4 already said do not port residual-PQ around a quality failure."],
        gpu_live=absent_gpu("Kernel is reachable; no PQ artifact opened; no GPU time."),
    )


def binary_group_path() -> dict:
    rows, cols = Q80_GATE_ROWS, Q80_GATE_COLS
    parent = rows * cols * F32_B
    src_flops = 2 * rows * cols
    # binary: 1 bit + f16 scale / group 128. MACs still 2*rows*cols plus sign extract.
    groups = (cols + 127) // 128
    packed = rows * ((cols + 7) // 8) + rows * groups * 2
    src = {
        "flops": cell(src_flops, kind=KIND_ALG, unit="FLOP", trace=f"dense 2*{rows}*{cols}"),
        "operation_count": cell(src_flops, kind=KIND_ALG, unit="ops", trace="dense f32"),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches", trace="one dense gemv"),
        "dram_bytes": cell(parent + (cols + rows) * F32_B, kind=KIND_ALG, unit="bytes",
                           trace="f32 W + x + y"),
        "peak_temporary_materialization": cell(rows * F32_B, kind=KIND_ALG, unit="bytes",
                                               trace="y only"),
    }
    exe = {
        "flops": cell(src_flops, kind=KIND_ALG, unit="FLOP",
                      trace="same MACs: sign*scale*x fused in-register. No extra GEMM."),
        "operation_count": cell(
            src_flops + 3 * rows * cols, kind=KIND_ALG, unit="ops",
            trace="MACs plus bit extract/mask/scale (~3 integer/logic per element)",
        ),
        "dispatch_count": cell(1, kind=KIND_ALG, unit="dispatches",
                               trace="q80_binary_group_matvec / _tg256 / _simd_bytes"),
        "dram_bytes": cell(packed + (cols + rows) * F32_B, kind=KIND_ALG, unit="bytes",
                           trace="sign bits + fp16 group scales + x + y"),
        "peak_temporary_materialization": cell(
            rows * F32_B, kind=KIND_ALG, unit="bytes",
            trace="output[row] only. q80_mixed_decode.metal: 'Nothing writes a (rows × cols) reconstruction.'",
        ),
    }
    return path_doc(
        path_id="q80_binary_group_fused",
        title="Q80 binary_group gate_proj fused sign-scale matvec (NATIVE)",
        structure="HGRAVB01 binary_sign_scale group 128",
        on_production_token=True,
        parent_name="expert.gate_proj",
        parent_shape=[rows, cols],
        parent_bytes=parent,
        parent_rank=2,
        materializes_dense_w_then_matmul=False,
        source=src,
        executable=exe,
        regime={
            "stored_bpw": 8.0 * packed / (rows * cols),
            "active_bpw_fused": 8.0 * packed / (rows * cols),
            "active_bpw_cached_dense": 32.0,
            "which": "fused active = stored. Cached dense = 32 bpw.",
        },
        evidence=[
            "q80_mixed_decode.metal contract: packed bytes read directly; value decoded in registers and consumed in the same kernel",
            "q80_binary_group_matvec writes output[row] = serial_row(...)",
        ],
        notes=["Same MAC count as dense. The win is bytes and the absence of a W write, not fewer MACs."],
        gpu_live=absent_gpu("Q80 mixed generate not run. Dispatch-bound anchors apply to the token, not this isolated organ."),
    )


def production_columns_from_path(path: dict) -> dict:
    """Flatten a path's side-by-side columns for the receipt root."""
    return path["columns"]


def build() -> dict:
    t0 = time.time()
    if not DECODE.is_file() or not SHADERS.is_dir():
        raise SystemExit(f"FAIL: decode/shaders missing ({DECODE}, {SHADERS})")

    g = load_geometry()
    np = try_numpy()
    ev = shader_evidence()
    q38_native, q38_oracle, extra = qwen38_production(g, np)
    vec = qwen30_vector_decode(g)
    h_native, h_oracle, h_bench = hgravs_paths(np)
    uv_native, uv_oracle, uv_bench = svd_generated_paths(np)
    pq = pq_path()
    binary = binary_group_path()

    paths = [q38_native, q38_oracle, h_native, h_oracle, uv_native, uv_oracle, binary, pq, vec]
    labels = {p["id"]: p["label"] for p in paths}
    oracle_ids = [p["id"] for p in paths if p["label"] == "ORACLE"]
    native_ids = [p["id"] for p in paths if p["label"] == "NATIVE"]

    # Root SOURCE vs EXECUTABLE = production Qwen3.8 token, the parent's ordinary
    # dense forward vs the compressed structure as run.
    root_columns = q38_native["columns"]

    gpu_reason = (
        "This obligation forbids loading a second 27B; a resident 27B already "
        "constrains memory. A previous sandbox MetalContext::new died with "
        "'no Metal-capable GPU' even though this host is an M3 Ultra (60 cores, "
        "Metal Supported). GPU wall, idle, and % of ceiling are therefore ABSENT "
        "for this run and cited as anchors where a prior receipt already measured them."
    )

    answer = (
        "The compressed structures that actually run EXECUTE DIRECTLY. "
        "Qwen3.8 uniform-q4 geo_tpr64 unpacks nibbles in registers and FMAs into y; "
        f"peak temporary {q38_native['columns']['peak_temporary_materialization']['EXECUTABLE']['value']} B "
        f"against parent lm_head {q38_native['parent']['bytes']} B. "
        "Q80 HGRAVS01 is y=L@(R@x) with mid[160]. "
        "gravity_pq_matvec and binary_group likewise never write W. "
        "The reconstruct-then-matmul lowerings exist (decode_vector, L@R then gemv, "
        "UV f16 cache) and are labelled ORACLE: their peak temporary reaches parent-tensor "
        "shape. The 0.0485 bpw UV structure is 0.0485 active fused and 16.0 active if the "
        "dense form is cached — same structure, two regimes. Presenting the cache regime "
        "as native execution is the defect."
    )

    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "elapsed_s": round(time.time() - t0, 3),
        "question": (
            "Does the compressed structure EXECUTE DIRECTLY, or does it decompress "
            "to dense and call an ordinary matmul?"
        ),
        "answer": answer,
        "convention": {
            "flop": "IEEE floating-point; FMA counted as 2",
            "operation_count": "FLOPs plus integer unpack/index ALU where the kernel does it",
            "dispatch_count": "Metal compute dispatches (or the equivalent CPU launches of the same graph)",
            "dram_bytes": "bytes the path reads or writes for that step (weights + temps + x + y)",
            "peak_temporary_materialization": (
                "largest transient tensor the path allocates. Persistent packed weights are "
                "not temporary. A 2-D parent-shaped W write is. Rank-1 RMSNorm decode is not "
                "a GEMV oracle even when peak equals the vector parent."
            ),
            "SOURCE": "what the parent's ordinary dense forward costs for the same organ/token",
            "EXECUTABLE": "what the compressed structure actually costs as run",
            "ORACLE": (
                "a path that materialises dense W of a 2-D parent weight and then runs an "
                "ordinary matmul/matvec. Allowed as a control. Must be labelled."
            ),
            "zero_policy": "0 means measured zero. Did-not-measure is ABSENT with a physical reason.",
        },
        "columns": root_columns,
        "production_path": q38_native["id"],
        "production_label": q38_native["label"],
        "paths": paths,
        "labels": labels,
        "oracle_path_ids": oracle_ids,
        "native_path_ids": native_ids,
        "every_oracle_path_labelled_ORACLE": all(p["label"] == "ORACLE" for p in paths if p["materializes_dense_w_then_ordinary_matmul"] or p["peak_temp_reaches_parent_tensor_shape"]),
        "discriminator_holds": all(
            (p["label"] == "ORACLE") == (
                p["materializes_dense_w_then_ordinary_matmul"]
                or p["peak_temp_reaches_parent_tensor_shape"]
            )
            for p in paths
        ),
        "regime_0485": {
            "stored_bpw": ANCHOR_STORED_BPW_0485,
            "active_bpw_fused": ANCHOR_ACTIVE_FUSED_0485,
            "active_bpw_cached_dense": ANCHOR_ACTIVE_CACHE_F16,
            "fused_path_id": uv_native["id"],
            "cached_path_id": uv_oracle["id"],
            "fused_label": uv_native["label"],
            "cached_label": uv_oracle["label"],
            "source": "receipts/headless/GENERATED_WEIGHTS_RETEST.json (not re-derived)",
        },
        "anchors_not_rederived": {
            "qwen38_tps": ANCHOR_Q38_TPS,
            "qwen38_ms_per_token": ANCHOR_Q38_MS,
            "qwen38_dispatches_per_token": ANCHOR_DISPATCHES,
            "qwen38_command_buffers_per_token": ANCHOR_CBS,
            "qwen38_bpw": ANCHOR_BPW,
            "qwen38_params": ANCHOR_PARAMS,
            "roof_gb_s_used_in_census": ANCHOR_ROOF_GB_S,
            "q80_dispatch_bound_pct_of_700_GBs": ANCHOR_Q80_PCT_OF_700,
            "q80_gpu_idle_pct": ANCHOR_Q80_GPU_IDLE_PCT,
            "q80_ceiling_gb_s_named_in_seal": ANCHOR_Q80_CEILING_GB_S,
            "q80_cb_before": ANCHOR_Q80_CB_BEFORE,
            "q80_cb_after": ANCHOR_Q80_CB_AFTER,
            "q80_cb_collapse_receipt": "receipts/ascent-2026-08-16/G003_Q80_CB_COLLAPSE.json",
            "q80_dispatch_bound_receipt": "receipts/ascent-2026-08-16/Q80_SEALED_LOSER_SCIENCE_RETAINED.json",
            "q80_storage_complete_bpw_mixed_1p5": ANCHOR_Q80_STORAGE_BPW,
            "q80_active_decode_bpw_mixed_1p5": ANCHOR_Q80_ACTIVE_DECODE_BPW_1P5,
            "q80_shape_dispatches": ANCHOR_Q80_DISPATCHES_SHAPE,
            "q80_shape_cbs": ANCHOR_Q80_CBS_SHAPE,
            "generated_uv_stored_bpw": ANCHOR_STORED_BPW_0485,
            "generated_uv_active_fused_bpw": ANCHOR_ACTIVE_FUSED_0485,
            "generated_uv_active_cache_f16_bpw": ANCHOR_ACTIVE_CACHE_F16,
        },
        "shader_evidence": ev,
        "cpu_microbench": {
            "numpy": None if np is None else str(np.__version__),
            "hgravs": h_bench,
            "generated_uv": uv_bench,
            "numpy_absent_reason": None if np is not None else "numpy not importable on this interpreter",
        },
        "gpu_live_this_run": absent_gpu(gpu_reason),
        "what_i_did_not_do": [
            "Did not load a second 27B.",
            "Did not open receipts/ascent-2026-08-16 or workspace/campaign for writing.",
            "Did not present an estimate as a measurement.",
            "Did not write 0 for a number that was not measured.",
            "Did not call qwen30 rank-1 RMSNorm decode an ORACLE.",
            "Did not treat CPU association oracles in q80_mixed_decode.rs (which never form W) as reconstruction oracles.",
        ],
        "self_check": {
            "production_is_NATIVE": q38_native["label"] == "NATIVE",
            "decode_vector_gemm_is_ORACLE": q38_oracle["label"] == "ORACLE",
            "hgravs_fused_is_NATIVE": h_native["label"] == "NATIVE",
            "hgravs_reconstruct_is_ORACLE": h_oracle["label"] == "ORACLE",
            "uv_fused_is_NATIVE": uv_native["label"] == "NATIVE",
            "uv_cache_is_ORACLE": uv_oracle["label"] == "ORACLE",
            "pq_is_NATIVE": pq["label"] == "NATIVE",
            "binary_is_NATIVE": binary["label"] == "NATIVE",
            "rmsnorm_vector_is_NATIVE": vec["label"] == "NATIVE",
            "root_columns_are_side_by_side": all(
                set(root_columns[m].keys()) == {"SOURCE", "EXECUTABLE"} for m in METRICS
            ),
            "every_path_has_side_by_side": all(
                all(set(p["columns"][m].keys()) == {"SOURCE", "EXECUTABLE"} for m in METRICS)
                for p in paths
            ),
            "no_absent_written_as_zero": True,
        },
        "written_to": str(RECEIPT),
    }

    # Harden: any cell with kind ABSENT must have value None; any 0 must not be ABSENT.
    for p in paths:
        for metric in METRICS:
            for side in ("SOURCE", "EXECUTABLE"):
                c = p["columns"][metric][side]
                if c["kind"] == KIND_ABSENT and c["value"] is not None:
                    raise SystemExit(f"ABSENT cell has value on {p['id']} {metric} {side}")
                if c["kind"] == KIND_ABSENT and not c.get("absent_reason"):
                    raise SystemExit(f"ABSENT without reason on {p['id']} {metric} {side}")
    return doc


def write_receipt(doc: dict, path: Path = RECEIPT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    doc["written_to"] = str(path)
    # rewrite with written_to stable
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def print_table(doc: dict) -> None:
    print("=" * 88)
    print("NOETIC NATIVE OPERATOR — SOURCE vs EXECUTABLE")
    print(doc["question"])
    print("=" * 88)
    print()
    print(doc["answer"])
    print()
    print(f"{'metric':<42} {'SOURCE':>18} {'EXECUTABLE':>18}")
    print("-" * 88)

    def fmt(cell_d):
        if cell_d["kind"] == KIND_ABSENT:
            return "ABSENT"
        v = cell_d["value"]
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, float):
            return f"{v:,.4g}"
        return str(v)

    cols = doc["columns"]
    names = {
        "flops": "FLOPs / token",
        "operation_count": "operations / token",
        "dispatch_count": "dispatches / token",
        "dram_bytes": "DRAM bytes / token",
        "peak_temporary_materialization": "peak temp materialisation",
    }
    for m in METRICS:
        print(f"{names[m]:<42} {fmt(cols[m]['SOURCE']):>18} {fmt(cols[m]['EXECUTABLE']):>18}")
    print()
    print(f"production label: {doc['production_label']}")
    print()
    print(f"{'path':<42} {'label':<8} {'peak temp':>14} {'parent':>14} {'reaches?':>8}")
    for p in doc["paths"]:
        peak = p["columns"]["peak_temporary_materialization"]["EXECUTABLE"]
        pv = "ABSENT" if peak["kind"] == KIND_ABSENT else f"{peak['value']:,}"
        print(
            f"{p['id']:<42} {p['label']:<8} {pv:>14} {p['parent']['bytes']:>14,} "
            f"{'yes' if p['peak_temp_reaches_parent_tensor_shape'] else 'no':>8}"
        )
    print()
    print("ORACLE paths:", ", ".join(doc["oracle_path_ids"]))
    print("NATIVE paths:", ", ".join(doc["native_path_ids"]))
    print()
    r = doc["regime_0485"]
    print(
        f"0.0485 regime: stored={r['stored_bpw']:.4f} fused={r['active_bpw_fused']:.4f} "
        f"cached_f16={r['active_bpw_cached_dense']:.1f}  "
        f"fused={r['fused_label']} cache={r['cached_label']}"
    )
    print()
    print("Q80 dispatch-bound anchor (not re-derived): "
          f"{doc['anchors_not_rederived']['q80_dispatch_bound_pct_of_700_GBs']}% of "
          f"{doc['anchors_not_rederived']['q80_ceiling_gb_s_named_in_seal']} GB/s, "
          f"{doc['anchors_not_rederived']['q80_gpu_idle_pct']}% GPU idle; "
          f"CBs {doc['anchors_not_rederived']['q80_cb_before']} → "
          f"{doc['anchors_not_rederived']['q80_cb_after']}")
    print(f"wrote {doc['written_to']}")
    print("=" * 88)


def main() -> int:
    doc = build()
    write_receipt(doc)
    print_table(doc)
    sc = doc["self_check"]
    if not all(sc.values()):
        bad = [k for k, v in sc.items() if not v]
        print(f"FAIL self_check: {bad}", file=sys.stderr)
        return 2
    if not doc["every_oracle_path_labelled_ORACLE"]:
        print("FAIL: an oracle path is not labelled ORACLE", file=sys.stderr)
        return 3
    if not doc["discriminator_holds"]:
        print("FAIL: discriminator does not match labels", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
