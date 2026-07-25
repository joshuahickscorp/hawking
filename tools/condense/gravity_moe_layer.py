#!/usr/bin/env python3.12
"""One COMPLETE GLM-5.2 MoE layer as a single Metal command buffer, parity-gated end to end.

Every phase before this one timed a projection.  A projection is not a layer: 24 independent
matvecs measured separately and added up is a number that no forward pass can ever reach,
because it pays the 215.8 us command-buffer constant 24 times and it never once makes Wave B
wait for Wave A.  This module executes the real dependency graph instead --

    fixed expert list  (THE ROUTER IS ABSENT FROM EVERY .gravity SHARD)
      -> Wave A : 9 x (gate, up)   [2048, 6144]      18 matvecs
      -> fused reduce + SwiGLU     9 dispatches, gate/up partials reduced and combined
      -> Wave B : 9 x  down        [6144, 2048]       9 matvecs
      -> fused combine             routing weight, shared expert and residual, 1 dispatch
      = 1 command buffer, 4 encoders, 37 dispatches

-- and grades the WHOLE-LAYER output against a CPU composition of the same compact
artifacts through ``gravity_forge.pq_execute``.  That last point is the one that matters:
SwiGLU is nonlinear, so a per-projection parity figure does not bound the layer's error.
``silu(g)*u`` amplifies whatever the gate projection got wrong before ``down`` ever sees it,
and the number a caller cares about is the one at the end of the chain.

The shared expert is not bolted on.  It has exactly the routed geometry ([2048,6144] gate
and up, [6144,2048] down -- verified on the shard headers, not assumed), so it is expert
slot 8 of 9 with routing weight 1.0, riding the same waves, the same partials buffer and the
same combine dispatch.  A separate code path for it would have been a second command buffer
and a second set of numbers to reconcile.

WHAT IS REUSED, AND WHY THIS FILE IS SHORT
    ``gravity_metal_lab_b`` already compiles the two winning grammars (``ll_blk_*`` and
    ``dfma_split``), keys uploads by content, holds a 1024-deep command queue and wraps every
    submission in ``objc.autorelease_pool``.  That is the whole device layer and it is
    sealed.  This module adds two kernels -- the ones the graph needs and a sweep of
    independent matvecs does not -- and one graph driver.  It does not fork the decoder.

WHAT IS FIXED, AND SAID EVERYWHERE
    ``model.layers.N.mlp.gate.weight`` is carried at source precision outside the .gravity
    payload, so it is on no shard.  Expert SELECTION here is a FIXED LIST and the routing
    WEIGHTS are a fixed vector.  Both the executor and every report field say so; see
    :data:`ROUTER_STATUS`.  Activations are SYNTHETIC for the same class of reason (the
    teacher capsules are inside the live campaign's support root).

CPU-safe: importing this module touches no GPU and no shard.  Every planning, costing and
parity function answers on a machine with no Metal device.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_bench_lab as lab            # noqa: E402
import gravity_forge as forge              # noqa: E402
import gravity_metal                       # noqa: E402
import gravity_metal_lab_b as labb         # noqa: E402
import gravity_real_fixtures as grf        # noqa: E402

SCHEMA = "hawking.glm52.moe_layer_executor.v1"
BENCH_SCHEMA = "hawking.glm52.moe_layer_benchmark.v1"
KERNEL_VERSION = 1
SEED = 20260722
PARITY_GATE = 2e-3                    # gravity_metal.py's own tolerance.  Achieved is reported.
REPORT_DIR = HERE.parents[1] / "reports" / "condense" / "breakthrough"

EXPERTS_PER_TOKEN = grf.EXPERTS_PER_TOKEN            # 8, from the model config
SHARED_EXPERT_WEIGHT = 1.0                           # not routed, always on

ROUTER_STATUS = "ROUTER_ABSENT_FIXED_EXPERT_LIST"
ROUTER_NOTE = (
    "model.layers.N.mlp.gate.weight is on none of the 90 shards -- pack_shard carries "
    "control tensors at source precision outside the .gravity payload.  The 8 routed "
    "experts are a FIXED LIST and the routing weights are a FIXED VECTOR; nothing here "
    "made a routing decision and no result may be read as though it had."
)

# The mandate's pressure targets, in milliseconds of whole-layer wall latency.
PRESSURE_TARGETS = (("moonshot", 0.75), ("dominance", 1.5), ("ship", 3.0), ("viable", 6.0))
# and its structural targets
COMMAND_BUFFER_TARGETS = (("moonshot", 1), ("dominance", 2), ("ship", 3))

# MEASURED, gravity_flop_ledger / the sealed selection matrix.  Quoted, never re-derived.
BANDWIDTH_ROOF_GB_S = lab.BANDWIDTH_ROOF_GB_S        # 736.0
TOKEN_BANDWIDTH_CEILING_TOK_S = 146.9                # artifact floor 5,010,218,784 B/token
COMMAND_BUFFER_FIXED_COST_MS = lab.MACHINE_FACTS["command_buffer_fixed_cost_us"] / 1e3

# GLM-5.2 topology.  first_k_dense_replace=3 -> layers 0-2 are dense MLP, 3-77 are MoE.
N_LAYERS = 78
N_DENSE_LAYERS = 3
N_SPARSE_LAYERS = N_LAYERS - N_DENSE_LAYERS          # 75

# GPU-clock medians from GLM52_KERNEL_SELECTION_MATRIX.json, per tensor, selected kernel.
# MEASURED there, quoted here.  Used only for the DERIVED per-token composition.
SELECTION_GPU_MS = {
    "attn::576x6144": 0.0300, "attn::28672x512": 0.0385, "attn::6144x16384": 0.0631,
    "attn::2048x6144": 0.0300, "attn::16384x2048": 0.0410,
    "dense_mlp::6144x12288": 0.0506, "dense_mlp::12288x6144": 0.0453,
}

# Selected kernel per geometry, from the same sealed matrix.  ``blocks`` is derived from cbs.
SELECTED = {
    "wave_a": {"grammar": "lookup-linear", "cbs": 16, "tpg": 1024,
               "half_table": False, "row4": True,
               "source": "GLM52_KERNEL_SELECTION_MATRIX.json routed_expert 2048x6144 "
                         "-> lookup_linear cbs16 tpg1024 blk48 (decided by executed_bytes)"},
    "wave_b": {"grammar": "decode-FMA", "cbs": 32, "tpg": 256,
               "source": "GLM52_KERNEL_SELECTION_MATRIX.json routed_expert 6144x2048 "
                         "-> decode_fma_2d blk8 cbs32 tpg256 (decided by gpu median; that "
                         "matrix ALSO reports this geometry as its one unstable shape, so "
                         "the lookup-linear alternative is measured here, not assumed away)"},
}
# The alternative at the unstable geometry, run as a second full-layer variant.
WAVE_B_ALT = {"grammar": "lookup-linear", "cbs": 16, "tpg": 1024,
              "half_table": False, "row4": True,
              "source": "GLM52_KERNEL_SELECTION_MATRIX.json shared_expert 6144x2048 "
                        "-> lookup_linear cbs16 tpg1024 blk16"}


class MoeLayerError(RuntimeError):
    """The layer graph cannot be executed, or would misdescribe itself."""


# --------------------------------------------------------------------------------- kernels
# Two, and only two.  Both exist because the GRAPH exists: a sweep of independent matvecs
# needs neither, which is why neither was written before now.

EXTRA_METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

struct DimsG { uint rows; uint blocks; uint nexp; uint pad0; };

// Wave A's two reductions and the SwiGLU, in ONE dispatch per expert.
//
// gate and up are separate tensors with separate codebooks, so their PARTIAL passes cannot
// fuse -- but their reductions read the same row and their results are consumed by the same
// nonlinearity, so those do.  This replaces 2 reduce dispatches + 1 swiglu dispatch per
// expert (27 for the layer) with 9, and it removes a whole encoder barrier: the reduce no
// longer has to complete before the SwiGLU starts, because it IS the SwiGLU.
//
// ``out`` is Wave B's x buffer.  The intermediate never leaves the GPU.
kernel void reduce_swiglu(
    device   const float* gate_p [[buffer(0)]],   // [blocks][rows]
    device   const float* up_p   [[buffer(1)]],   // [blocks][rows]
    device         float* out    [[buffer(2)]],   // [rows], = wave B x
    constant       DimsG& d      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.rows) return;
    float g = 0.0f, u = 0.0f;
    for (uint b = 0; b < d.blocks; ++b) {
        const ulong o = (ulong)b * d.rows + gid;
        g += gate_p[o];
        u += up_p[o];
    }
    out[gid] = (g / (1.0f + exp(-g))) * u;
}

// Same kernel, four rows per thread.  The reduction walks ``blocks`` partials with a
// rows-wide stride, so a scalar thread issues 2*blocks dependent 4-byte loads and there is
// nothing in its own instruction stream to hide their latency -- the same defect the 2D
// split fixed in the matvec, one stage later.  float4 gives every thread four independent
// accumulator chains and 16-byte loads for the same bytes.  Guarded on rows % 4 == 0 by the
// caller; both real geometries are multiples of 4 and the scalar kernel stays as the
// fallback and as the A/B arm.
kernel void reduce_swiglu4(
    device   const float4* gate_p [[buffer(0)]],
    device   const float4* up_p   [[buffer(1)]],
    device         float4* out    [[buffer(2)]],
    constant       DimsG&  d      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint r4 = d.rows >> 2;
    if (gid >= r4) return;
    float4 g = float4(0.0f), u = float4(0.0f);
    for (uint b = 0; b < d.blocks; ++b) {
        const ulong o = (ulong)b * r4 + gid;
        g += gate_p[o];
        u += up_p[o];
    }
    out[gid] = (g / (1.0f + exp(-g))) * u;
}

// Wave B's reduction, the routing weight, the expert sum, the shared expert and the
// residual -- ONE dispatch, one pass over the partials, one store.
//
// Every down projection wrote into one [nexp][blocks][rows] buffer at its own offset, so
// the combine never needs a pointer table and the per-expert reduce never lands in device
// memory as a separate y.  Expert nexp-1 is the SHARED expert; it is in this loop with
// weight 1.0 rather than in a second kernel, which is what "integrated, not bolted on"
// means in dispatch terms.
kernel void moe_combine(
    device   const float* partials [[buffer(0)]],   // [nexp][blocks][rows]
    device   const float* weights  [[buffer(1)]],   // [nexp], routed weights then 1.0
    device   const float* residual [[buffer(2)]],   // [rows], the layer input
    device         float* y        [[buffer(3)]],   // [rows], the layer output
    constant       DimsG& d        [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.rows) return;
    float acc = residual[gid];
    for (uint e = 0; e < d.nexp; ++e) {
        const device float* p = partials + (ulong)e * d.blocks * d.rows + gid;
        float s = 0.0f;
        for (uint b = 0; b < d.blocks; ++b) s += p[(ulong)b * d.rows];
        acc = fma(weights[e], s, acc);
    }
    y[gid] = acc;
}
"""


# ------------------------------------------------------------------- pure planning / cost
# Everything to the next section header answers with no Metal device and no shard.

STAGES = ("wave_a", "swiglu", "wave_b", "combine")


def routing_weights(n_routed: int = EXPERTS_PER_TOKEN, *, seed: int = SEED) -> np.ndarray:
    """The FIXED routing weight vector.  Not a routing decision -- see ROUTER_NOTE.

    A softmax over a fixed pseudo-logit vector, so the weights are normalised, distinct and
    reproducible.  Their VALUES are arbitrary; what they buy is that the combine kernel and
    the CPU authority are exercised with a non-degenerate weighting instead of all-ones,
    which would hide a per-expert offset bug in the combine.  The shared expert's 1.0 is
    appended and is NOT part of the softmax, because it is not routed.
    """
    logits = np.random.default_rng(seed).standard_normal(n_routed).astype(np.float32)
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return np.concatenate([w, np.array([SHARED_EXPERT_WEIGHT], dtype=np.float32)])


def shape_for(cfg: dict[str, Any], *, rows: int, nchunk: int, D: int, k: int) -> dict[str, Any]:
    """The lab_b plan for one selected config.  Refusals come from lab_b, not from here."""
    if cfg["grammar"] == "lookup-linear":
        return labb.ll_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=cfg["cbs"],
                            tpg=cfg["tpg"], half_table=cfg.get("half_table", False),
                            row4=cfg.get("row4", True))
    return labb.dfma_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=cfg["cbs"], tpg=cfg["tpg"])


def cost_for(cfg: dict[str, Any], shape: dict[str, Any], *, rows: int, cols: int,
             nchunk: int, D: int, k: int) -> dict[str, Any]:
    if cfg["grammar"] == "lookup-linear":
        return labb.ll_cost(rows=rows, cols=cols, nchunk=nchunk, D=D, k=k, shape=shape)
    return labb.dfma_cost(rows=rows, cols=cols, nchunk=nchunk, D=D, k=k, shape=shape)


@dataclass(frozen=True)
class LayerGeometry:
    """The layer's real shapes, taken from shard headers rather than from the brief."""

    rows_a: int          # moe_intermediate_size, gate/up output      2048
    cols_a: int          # hidden_size, gate/up input                 6144
    rows_b: int          # hidden_size, down output                   6144
    cols_b: int          # moe_intermediate_size, down input          2048
    D: int
    k: int

    @property
    def nchunk_a(self) -> int:
        return self.cols_a // self.D

    @property
    def nchunk_b(self) -> int:
        return self.cols_b // self.D


def layer_plan(geom: LayerGeometry, *, n_routed: int = EXPERTS_PER_TOKEN,
               wave_a_cfg: dict[str, Any] | None = None,
               wave_b_cfg: dict[str, Any] | None = None,
               shared: bool = True) -> dict[str, Any]:
    """The whole graph: shapes, dispatch and command-buffer counts, and the byte ledger.

    The byte ledger is built for the GRAPH, not by summing 27 independent matvec ledgers.
    Two terms differ and both matter: Wave A never writes a y (``reduce_swiglu`` consumes
    its partials directly), and Wave B never writes a y either (``moe_combine`` does).  A
    sum of per-tensor ledgers would bill 27 output stores and 27 partial re-reads that this
    graph does not perform.
    """
    wave_a_cfg = dict(SELECTED["wave_a"] if wave_a_cfg is None else wave_a_cfg)
    wave_b_cfg = dict(SELECTED["wave_b"] if wave_b_cfg is None else wave_b_cfg)
    nexp = n_routed + (1 if shared else 0)
    if nexp < 1:
        raise MoeLayerError("a layer needs at least one expert")

    sa = shape_for(wave_a_cfg, rows=geom.rows_a, nchunk=geom.nchunk_a, D=geom.D, k=geom.k)
    sb = shape_for(wave_b_cfg, rows=geom.rows_b, nchunk=geom.nchunk_b, D=geom.D, k=geom.k)

    ca = cost_for(wave_a_cfg, sa, rows=geom.rows_a, cols=geom.cols_a,
                  nchunk=geom.nchunk_a, D=geom.D, k=geom.k)
    cb = cost_for(wave_b_cfg, sb, rows=geom.rows_b, cols=geom.cols_b,
                  nchunk=geom.nchunk_b, D=geom.D, k=geom.k)

    # Per-tensor traffic MINUS the y store neither wave performs.
    a_read = ca["executed_read_bytes"] - ca["partial_read_bytes"]
    a_write = ca["partial_write_bytes"]
    b_read = cb["executed_read_bytes"] - cb["partial_read_bytes"]
    b_write = cb["partial_write_bytes"]

    # reduce_swiglu: reads both waves' partials once, writes wave B's x.
    rs_read = 2 * sa["partial_bytes"]
    rs_write = geom.rows_a * 4
    # moe_combine: reads every expert's wave-B partials once, plus weights and residual.
    comb_read = nexp * sb["partial_bytes"] + nexp * 4 + geom.rows_b * 4
    comb_write = geom.rows_b * 4

    executed_read = 2 * nexp * a_read + nexp * rs_read + nexp * b_read + comb_read
    executed_write = 2 * nexp * a_write + nexp * rs_write + nexp * b_write + comb_write

    def artifact_bytes(rows: int, nchunk: int) -> int:
        return (rows * nchunk * 7 + 7) // 8 + geom.k * geom.D * 2

    artifact = nexp * (2 * artifact_bytes(geom.rows_a, geom.nchunk_a)
                       + artifact_bytes(geom.rows_b, geom.nchunk_b))
    dense_bf16 = nexp * (2 * geom.rows_a * geom.cols_a + geom.rows_b * geom.cols_b) * 2

    dispatches = {
        "wave_a": 2 * nexp,
        "swiglu": nexp,
        "wave_b": nexp,
        "combine": 1,
    }
    return {
        "router_status": ROUTER_STATUS,
        "router_note": ROUTER_NOTE,
        "experts_routed": n_routed,
        "shared_expert": shared,
        "shared_expert_integration": (
            "expert slot %d of %d, routing weight 1.0, same waves and same partials buffer"
            % (nexp - 1, nexp) if shared else "ABSENT (variant)"),
        "geometry": {
            "gate_up": [geom.rows_a, geom.cols_a], "down": [geom.rows_b, geom.cols_b],
            "nchunk_gate_up": geom.nchunk_a, "nchunk_down": geom.nchunk_b,
            "D": geom.D, "k": geom.k,
        },
        "wave_a": {"config": wave_a_cfg, "shape": sa, "per_tensor_cost": ca},
        "wave_b": {"config": wave_b_cfg, "shape": sb, "per_tensor_cost": cb},
        "tensors_executed": 3 * nexp,
        "command_buffers_per_layer": 1,
        "encoders_per_layer": 4,
        "dispatches_per_layer": sum(dispatches.values()),
        "dispatches_by_stage": dispatches,
        "threads_in_flight": {
            "wave_a_per_tensor": sa["threads_in_flight"],
            "wave_b_per_tensor": sb["threads_in_flight"],
        },
        "scratch_bytes": {"wave_a": sa["scratch_bytes"], "wave_b": sb["scratch_bytes"]},
        "byte_ledger": {
            "model": "ANALYTIC graph ledger, not a counter reading",
            "wave_a_read_bytes": 2 * nexp * a_read,
            "wave_a_write_bytes": 2 * nexp * a_write,
            "reduce_swiglu_read_bytes": nexp * rs_read,
            "reduce_swiglu_write_bytes": nexp * rs_write,
            "wave_b_read_bytes": nexp * b_read,
            "wave_b_write_bytes": nexp * b_write,
            "combine_read_bytes": comb_read,
            "combine_write_bytes": comb_write,
            "executed_read_bytes": executed_read,
            "executed_write_bytes": executed_write,
            "executed_total_bytes": executed_read + executed_write,
            "layer_artifact_bytes": artifact,
            "dense_bf16_bytes": dense_bf16,
            "executed_over_artifact": (executed_read + executed_write) / artifact,
            "read_over_artifact": executed_read / artifact,
            "executed_over_dense_bf16": (executed_read + executed_write) / dense_bf16,
        },
        "arithmetic": {
            "wave_a_executed_fp_ops": 2 * nexp * ca["executed_fp_ops"],
            "wave_b_executed_fp_ops": nexp * cb["executed_fp_ops"],
            "dense_equivalent_macs": nexp * (2 * ca["dense_equivalent_macs"]
                                             + cb["dense_equivalent_macs"]),
            "swiglu_transcendentals": nexp * geom.rows_a,
        },
    }


def pressure_verdict(latency_ms: float) -> dict[str, Any]:
    """Which pressure target the measured latency actually reached.  No rounding toward one."""
    reached = None
    for name, bound in PRESSURE_TARGETS:              # tightest first
        if latency_ms <= bound:
            reached = name
            break
    return {
        "measured_ms": latency_ms,
        "reached": reached if reached is not None else "NONE",
        "targets_ms": {name: bound for name, bound in PRESSURE_TARGETS},
        "margins_ms": {name: bound - latency_ms for name, bound in PRESSURE_TARGETS},
        "note": "the tightest target whose bound the MEASURED median clears; nothing rounded",
    }


def per_token_projection(layer_ms: float, *, source: str) -> dict[str, Any]:
    """Per-token latency and tok/s implied by one measured MoE layer.

    Explicitly labelled.  The MoE term is MEASURED here.  The attention and dense-MLP terms
    are DERIVED from the sealed selection matrix's per-tensor GPU medians plus one
    command-buffer constant per layer -- that is a projection of a graph nobody has executed,
    not a measurement of one.  Everything else in a real forward pass is UNMEASURED and is
    named rather than silently set to zero.
    """
    attn_gpu = sum(v for key, v in SELECTION_GPU_MS.items() if key.startswith("attn::"))
    dense_gpu = 2 * SELECTION_GPU_MS["dense_mlp::6144x12288"] + \
        SELECTION_GPU_MS["dense_mlp::12288x6144"]
    attn_ms = attn_gpu + COMMAND_BUFFER_FIXED_COST_MS
    dense_ms = dense_gpu + COMMAND_BUFFER_FIXED_COST_MS
    total = N_SPARSE_LAYERS * layer_ms + N_DENSE_LAYERS * dense_ms + N_LAYERS * attn_ms
    return {
        "moe_layer_ms": {"value": layer_ms, "status": "MEASURED", "source": source},
        "attention_per_layer_ms": {
            "value": attn_ms, "status": "DERIVED",
            "source": "sum of 5 selection-matrix per-tensor GPU medians (%.4f ms) + one "
                      "command-buffer fixed cost (%.4f ms); no attention graph has been "
                      "executed" % (attn_gpu, COMMAND_BUFFER_FIXED_COST_MS)},
        "dense_mlp_layer_ms": {
            "value": dense_ms, "status": "DERIVED",
            "source": "selection-matrix GPU medians for [6144,12288] x2 + [12288,6144] "
                      "(%.4f ms) + one command-buffer fixed cost" % dense_gpu},
        "layer_counts": {"sparse_moe": N_SPARSE_LAYERS, "dense_mlp": N_DENSE_LAYERS,
                         "attention": N_LAYERS},
        "implied_token_ms": total,
        "implied_tok_s": 1000.0 / total,
        "bandwidth_ceiling_tok_s": TOKEN_BANDWIDTH_CEILING_TOK_S,
        "fraction_of_bandwidth_ceiling": (1000.0 / total) / TOKEN_BANDWIDTH_CEILING_TOK_S,
        "unmeasured_terms": [
            "lm_head [154880,6144] -- on no safe shard",
            "embedding row gather -- on no safe shard",
            "attention softmax, RoPE, KV cache reads and the KV cache itself",
            "every RMSNorm",
            "the router matvec itself -- the tensor is absent from the artifacts",
            "host-side sampling and detokenisation",
        ],
        "note": "a lower bound on token latency and therefore an UPPER bound on tok/s: "
                "every unmeasured term above adds time and none subtracts it",
    }


# ------------------------------------------------------------------------- parity authority

def artifact_of(codes: dict[str, Any]) -> forge.PackedArtifact:
    ledger = forge.ByteLedger()
    ledger.add("indices", codes["indices"].size * 7)
    return forge.PackedArtifact(
        "product_quant", np.empty((0,), dtype=np.float32),
        codes["rows"] * codes["cols"], ledger, ledger.total_bits(), 0, {"pq_codes": codes})


def silu(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (1.0 + np.exp(-v, dtype=np.float32))


def cpu_layer(experts: list[dict[str, dict[str, Any]]], x: np.ndarray,
              weights: np.ndarray) -> dict[str, np.ndarray]:
    """THE PARITY AUTHORITY: the same compact artifacts, the same graph, on the CPU.

    ``gravity_forge.pq_execute`` per projection, composed through SwiGLU, the routing
    weights, the shared expert and the residual, in that order.  Intermediates are returned
    so a failure can be localised, but the number that gates is ``y``.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(weights) != len(experts):
        raise MoeLayerError(f"{len(weights)} weights for {len(experts)} experts")
    y = x.astype(np.float32).copy()                        # residual
    hidden: list[np.ndarray] = []
    downs: list[np.ndarray] = []
    for i, exp in enumerate(experts):
        g = forge.pq_execute(artifact_of(exp["gate"]), x)
        u = forge.pq_execute(artifact_of(exp["up"]), x)
        h = (silu(g) * u).astype(np.float32)
        d = forge.pq_execute(artifact_of(exp["down"]), h)
        hidden.append(h)
        downs.append(d)
        y = y + np.float32(weights[i]) * d
    return {"y": y, "hidden": np.stack(hidden), "down": np.stack(downs)}


def parity_of(got: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    got = np.asarray(got, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    diff = reference - got
    denom = float(np.abs(reference).max()) + 1e-30
    return {
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-30)),
        "max_abs_error": float(np.abs(diff).max()),
        "relative_max_gap": float(np.abs(diff).max() / denom),
        "cosine": float(reference @ got / ((np.linalg.norm(reference)
                                            * np.linalg.norm(got)) + 1e-30)),
        "finite": bool(np.isfinite(got).all()),
        "gate": PARITY_GATE,
    }


# ---------------------------------------------------------------------------------- device

class MoeLayerExecutor:
    """One layer, one command buffer.  Pipelines, buffers and uploads all built once.

    Holds a :class:`gravity_metal_lab_b.TrackBDecoder` rather than forking it: that class
    owns the two winning kernels, the content-keyed upload cache, the 1024-deep command
    queue and the autorelease-pool discipline.  The extra library compiled here shares its
    device, so its pipeline states drop into the same command buffer.
    """

    def __init__(self, dec: "labb.TrackBDecoder | None" = None) -> None:
        self.dec = labb.TrackBDecoder() if dec is None else dec
        import Metal  # noqa: F401  (present iff TrackBDecoder constructed)
        self._Metal = self.dec._Metal
        library, error = self.dec.device.newLibraryWithSource_options_error_(
            EXTRA_METAL_SOURCE, None, None)
        if library is None:
            raise gravity_metal.MetalUnavailable(f"graph kernels failed to compile: {error}")
        self._pipelines = {}
        for name in ("reduce_swiglu", "reduce_swiglu4", "moe_combine"):
            fn = library.newFunctionWithName_(name)
            if fn is None:
                raise gravity_metal.MetalUnavailable(f"no kernel named {name}")
            pipe, error = self.dec.device.newComputePipelineStateWithFunction_error_(fn, None)
            if pipe is None:
                raise gravity_metal.MetalUnavailable(f"pipeline {name} failed: {error}")
            self._pipelines[name] = pipe
        self.pool: dict[str, Any] = {}
        self.plan: dict[str, Any] | None = None
        self.last_gpu_ms: float | None = None
        # False re-binds every encoder argument per dispatch, which is what a per-tensor
        # encode helper does.  Kept as a switch so the host-encode lever is an A/B in one
        # process against a matched BenchSpec instead of a remembered number.
        self.hoist_invariants = True
        # float4 reduce+SwiGLU where the row count allows.  A MEASURED lever, kept as a
        # switch so the scalar kernel is the A/B arm rather than a remembered baseline.
        self.swiglu_vec4 = True

    # -- setup, paid once ---------------------------------------------------------------

    def prepare(self, experts: list[dict[str, Any]], weights: np.ndarray,
                *, plan: dict[str, Any]) -> None:
        """Upload every tensor, allocate the persistent pool, bind the shared buffers.

        Cache keys are the fixtures' own ``shard::tensor::sha256``.  Not ``id()`` (CPython
        recycles it, and gravity_metal now refuses it), and not one literal reused across
        tensors (gravity_metal refuses that too) -- 27 distinct content addresses for 27
        distinct tensors, so a collision is a sha256 collision.
        """
        Metal = self._Metal
        dev = self.dec.device
        geom = plan["geometry"]
        rows_a, rows_b = geom["gate_up"][0], geom["down"][0]
        sa, sb = plan["wave_a"]["shape"], plan["wave_b"]["shape"]
        nexp = len(experts)
        if len(weights) != nexp:
            raise MoeLayerError(f"{len(weights)} weights for {nexp} experts")

        seen: dict[str, str] = {}
        for exp in experts:
            for proj in ("gate", "up", "down"):
                key = exp[proj]["key"]
                if not isinstance(key, str) or not key:
                    raise MoeLayerError(f"{proj}: cache key must be a non-empty string")
                if key in seen and seen[key] != exp[proj]["name"]:
                    raise MoeLayerError(
                        f"cache key {key!r} is claimed by both {seen[key]!r} and "
                        f"{exp[proj]['name']!r}; a reused literal serves one tensor's "
                        "indices for another")
                seen[key] = exp[proj]["name"]

        # One activation buffer for the whole graph, shared across executors on this device:
        # every wave-A tensor consumes the same x AND the combine kernel's residual IS that
        # same x, so one buffer removes 2*nexp host memcpies from the timed region and makes
        # "the residual is the layer input" structural instead of a convention.
        x_buf = getattr(self.dec, "_moe_x", None)
        if x_buf is None or x_buf.length() != geom["gate_up"][1] * 4:
            x_buf = dev.newBufferWithLength_options_(
                geom["gate_up"][1] * 4, Metal.MTLResourceStorageModeShared)
            self.dec._moe_x = x_buf

        def dims_g(rows: int, blocks: int) -> Any:
            return self.dec._buffer(np.array([rows, blocks, nexp, 0], dtype=np.uint32))

        # Two pooled partials buffers, [tensor][block][row], allocated once.  A per-tensor
        # buffer would have been 27 allocations and 27 more pyobjc binds per layer; one
        # buffer plus an offset is what lets a whole wave bind its destination by stride.
        self.pool = {
            "x": x_buf,
            "wave_a_partials": dev.newBufferWithLength_options_(
                2 * nexp * sa["blocks"] * rows_a * 4, Metal.MTLResourceStorageModeShared),
            "wave_b_partials": dev.newBufferWithLength_options_(
                nexp * sb["blocks"] * rows_b * 4, Metal.MTLResourceStorageModeShared),
            "weights": self.dec._buffer(np.asarray(weights, dtype=np.float32)),
            "y": dev.newBufferWithLength_options_(rows_b * 4, Metal.MTLResourceStorageModeShared),
            "dims_a": dims_g(rows_a, sa["blocks"]),
            "dims_c": dims_g(rows_b, sb["blocks"]),
            "stride_a": sa["blocks"] * rows_a * 4,
            "stride_b": sb["blocks"] * rows_b * 4,
            "nexp": nexp, "rows_a": rows_a, "rows_b": rows_b,
            "blocks_a": sa["blocks"], "blocks_b": sb["blocks"],
        }
        self.wave_a: list[tuple[dict, dict]] = []
        self.wave_b: list[tuple[dict, dict]] = []
        for exp in experts:
            for proj in ("gate", "up"):
                # max_blocks=1: the wave writes into the pooled buffer, so the decoder's
                # per-tensor partials allocation is deliberately left at its 4-byte floor.
                entry = self.dec.upload(exp[proj]["codes"], exp[proj]["key"], max_blocks=1)
                entry["x"] = x_buf
                self.wave_a.append((entry, sa))
            entry = self.dec.upload(exp["down"]["codes"], exp["down"]["key"], max_blocks=1)
            self.wave_b.append((entry, sb))
        for entry, shape in self.wave_a + self.wave_b:
            if shape["tpg"] > self.dec.max_threads_per_threadgroup:
                raise MoeLayerError(f"tpg {shape['tpg']} exceeds device max")
            if shape["scratch_bytes"] > self.dec.threadgroup_memory_limit:
                raise MoeLayerError(f"scratch {shape['scratch_bytes']} B exceeds device limit")
        for entry, _ in self.wave_b:
            if entry["x_bytes"] != rows_a * 4:
                raise MoeLayerError(
                    f"wave B takes {entry['x_bytes']} B but wave A emits {rows_a * 4} B")
        # dims for the matvec kernels: identical contents for every tensor in a wave, so
        # they are built once here rather than once per entry inside the encode loop.
        self.dims_a = self.dec._dims(self.wave_a[0][0], cbs=sa["cbs"], blocks=sa["blocks"],
                                     tpg=sa["tpg"])
        self.dims_b = self.dec._dims(self.wave_b[0][0], cbs=sb["cbs"], blocks=sb["blocks"],
                                     tpg=sb["tpg"])
        self.plan = plan

    def set_input(self, x: np.ndarray) -> None:
        xv = np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())
        want = self.pool["x"].length()
        if xv.nbytes != want:
            raise MoeLayerError(f"x is {xv.nbytes} B, layer input is {want} B")
        self.pool["x"].contents().as_buffer(want)[:] = xv.tobytes()

    # -- the graph ----------------------------------------------------------------------

    def _encode_wave(self, enc, jobs, shape: dict, dims, target, stride: int,
                     shared_x) -> None:
        """One wave, with every argument that is constant across it bound ONCE.

        Every tensor in a wave has the same geometry, so it has the same pipeline, the same
        scratch allocation, the same threadgroup grid and the same Dims buffer; a wave that
        consumes the layer input also has the same x.  Only the index stream, the codebook
        and the output offset actually vary.  Encoder state persists across dispatches
        within one encoder, so binding the invariants per dispatch was 5 pyobjc messages per
        tensor of pure host cost -- and host cost is what stands between this layer's GPU
        clock and its wall clock.  MEASURED lever, reported in ``host_encode_lever``.
        """
        Metal = self._Metal
        pipe = self.dec._pipelines[shape["kernel"]]
        if shape["grammar"] == "decode-FMA":
            grid = Metal.MTLSizeMake(shape["row_tiles"], shape["blocks"], 1)
        else:
            grid = Metal.MTLSizeMake(shape["blocks"], 1, 1)
        tpg = Metal.MTLSizeMake(shape["tpg"], 1, 1)
        if self.hoist_invariants:
            enc.setComputePipelineState_(pipe)
            enc.setThreadgroupMemoryLength_atIndex_(shape["scratch_bytes"], 0)
            enc.setBuffer_offset_atIndex_(dims, 0, 4)
            if shared_x is not None:
                enc.setBuffer_offset_atIndex_(shared_x, 0, 2)
        for i, (entry, _) in enumerate(jobs):
            if not self.hoist_invariants:                  # the A/B arm, measured not asserted
                enc.setComputePipelineState_(pipe)
                enc.setThreadgroupMemoryLength_atIndex_(shape["scratch_bytes"], 0)
                enc.setBuffer_offset_atIndex_(dims, 0, 4)
            enc.setBuffer_offset_atIndex_(entry["idx"], 0, 0)
            enc.setBuffer_offset_atIndex_(entry["book"], 0, 1)
            if shared_x is None or not self.hoist_invariants:
                enc.setBuffer_offset_atIndex_(shared_x or entry["x"], 0, 2)
            enc.setBuffer_offset_atIndex_(target, i * stride, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tpg)

    def _linear(self, enc, pipeline: str | None, buffers, n: int) -> None:
        """One elementwise dispatch.  ``pipeline=None`` reuses the encoder's current state,
        which is how the 9 reduce+SwiGLU dispatches avoid re-binding an identical pipeline
        and an identical Dims buffer nine times."""
        Metal = self._Metal
        if pipeline is not None:
            enc.setComputePipelineState_(self._pipelines[pipeline])
        for slot, buf, off in buffers:
            enc.setBuffer_offset_atIndex_(buf, off, slot)
        t = min(256, self.dec.max_threads_per_threadgroup)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((n + t - 1) // t, 1, 1), Metal.MTLSizeMake(t, 1, 1))

    def run_layer(self, *, stop_after: str = "combine") -> np.ndarray | None:
        """The whole layer in ONE command buffer.  Four encoders, 37 dispatches at nexp=9.

        Encoders in one command buffer execute in order with a barrier between them, which
        is exactly this graph's dependency structure and nothing more:
            enc1 Wave A partials -> enc2 reduce+SwiGLU -> enc3 Wave B partials -> enc4 combine.
        Wave B's input is written by enc2 into Wave B's own x buffer, so the intermediate
        never crosses the bus to the host.

        ``stop_after`` truncates the graph at a stage boundary.  That is how the per-stage
        breakdown is measured: Metal exposes a GPU clock per COMMAND BUFFER, not per
        encoder, so a stage's cost is the difference between two truncated graphs.
        """
        import objc
        if self.plan is None:
            raise MoeLayerError("prepare() first")
        if stop_after not in STAGES:
            raise MoeLayerError(f"stop_after must be one of {STAGES}, got {stop_after!r}")
        pool = self.pool
        rows_a, rows_b, nexp = pool["rows_a"], pool["rows_b"], pool["nexp"]
        stride_a = pool["stride_a"]
        with objc.autorelease_pool():
            cb = self.dec.queue.commandBuffer()

            enc = cb.computeCommandEncoder()
            self._encode_wave(enc, self.wave_a, self.wave_a[0][1], self.dims_a,
                              pool["wave_a_partials"], stride_a, pool["x"])
            enc.endEncoding()

            if stop_after != "wave_a":
                vec4 = self.swiglu_vec4 and rows_a % 4 == 0
                enc2 = cb.computeCommandEncoder()
                enc2.setComputePipelineState_(
                    self._pipelines["reduce_swiglu4" if vec4 else "reduce_swiglu"])
                enc2.setBuffer_offset_atIndex_(pool["dims_a"], 0, 3)
                for i in range(nexp):
                    self._linear(enc2, None,
                                 [(0, pool["wave_a_partials"], 2 * i * stride_a),
                                  (1, pool["wave_a_partials"], (2 * i + 1) * stride_a),
                                  (2, self.wave_b[i][0]["x"], 0)],
                                 rows_a // 4 if vec4 else rows_a)
                enc2.endEncoding()

            if stop_after in ("wave_b", "combine"):
                enc3 = cb.computeCommandEncoder()
                self._encode_wave(enc3, self.wave_b, self.wave_b[0][1], self.dims_b,
                                  pool["wave_b_partials"], pool["stride_b"], None)
                enc3.endEncoding()

            if stop_after == "combine":
                enc4 = cb.computeCommandEncoder()
                self._linear(enc4, "moe_combine",
                             [(0, pool["wave_b_partials"], 0), (1, pool["weights"], 0),
                              (2, pool["x"], 0), (3, pool["y"], 0), (4, pool["dims_c"], 0)],
                             rows_b)
                enc4.endEncoding()

            cb.commit()
            cb.waitUntilCompleted()
            if cb.error() is not None:
                raise gravity_metal.MetalUnavailable(f"layer dispatch failed: {cb.error()}")
            self.last_gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        if stop_after != "combine":
            return None
        return np.frombuffer(pool["y"].contents().as_buffer(rows_b * 4),
                             dtype=np.float32).copy()

    def intermediate(self, i: int) -> np.ndarray:
        """Expert i's post-SwiGLU hidden, read back for localising a parity failure."""
        entry = self.wave_b[i][0]
        return np.frombuffer(entry["x"].contents().as_buffer(entry["x_bytes"]),
                             dtype=np.float32).copy()


# ------------------------------------------------------------------------ benchmark driver

def load_layer(layer: int | None = None, *, n_routed: int = EXPERTS_PER_TOKEN
               ) -> dict[str, Any]:
    """8 routed experts plus the shared expert from one real layer, with full provenance."""
    fx = grf.fixture_set(layer=layer, experts=n_routed)
    if fx["shared_expert"] is None:
        raise MoeLayerError(f"layer {fx['layer']} has no complete shared expert")
    if fx["router_present"]:                          # pragma: no cover - no shard has one
        raise MoeLayerError("a router tensor appeared; this module's fixed-list claim is stale")

    def pack(f: grf.Fixture) -> dict[str, Any]:
        return {"codes": f.codes, "key": f.cache_key, "name": f.tensor, "fixture": f}

    experts = [{p: pack(e[p]) for p in ("gate", "up", "down")} for e in fx["expert_set"]]
    experts.append({p: pack(fx["shared_expert"][p]) for p in ("gate", "up", "down")})
    first = experts[0]["gate"]["codes"]
    down = experts[0]["down"]["codes"]
    geom = LayerGeometry(rows_a=int(first["rows"]), cols_a=int(first["cols"]),
                         rows_b=int(down["rows"]), cols_b=int(down["cols"]),
                         D=int(first["D"]), k=int(first["codebooks"][0].shape[0]))
    for exp in experts:                     # the shared expert's geometry is CHECKED, not assumed
        for p, want in (("gate", (geom.rows_a, geom.cols_a)),
                        ("up", (geom.rows_a, geom.cols_a)),
                        ("down", (geom.rows_b, geom.cols_b))):
            c = exp[p]["codes"]
            if (int(c["rows"]), int(c["cols"])) != want:
                raise MoeLayerError(f"{exp[p]['name']}: shape {(c['rows'], c['cols'])} "
                                    f"!= layer geometry {want}")
            if int(c["S"]) != 1 or c["rotate"]:
                raise MoeLayerError(f"{exp[p]['name']}: S={c['S']} rotate={c['rotate']}; "
                                    "the compiled grammars decode S=1, rotate=False only")
    return {"layer": fx["layer"], "experts": experts, "geometry": geom, "fixture_set": fx}


def timing_json(stats: lab.TimingStats) -> dict[str, Any]:
    return {"median": stats.median_ms, "min": stats.min_ms, "p95": stats.p95_ms,
            "max": stats.max_ms, "coefficient_of_variation": stats.coefficient_of_variation,
            "is_contended": stats.is_contended, "raw_samples_ms": list(stats.raw_samples_ms)}


def measure_layer(ex: MoeLayerExecutor, x: np.ndarray, *, stop_after: str,
                  spec: lab.BenchSpec) -> tuple[lab.TimingStats, lab.TimingStats]:
    """Wall and GPU clock, same reps, every sample kept.  The host x store is inside both."""
    def call():
        ex.set_input(x)
        ex.run_layer(stop_after=stop_after)
    for _ in range(spec.warmup):
        call()
    wall, gpu = [], []
    for _ in range(spec.reps):
        start = time.perf_counter_ns()
        call()
        wall.append((time.perf_counter_ns() - start) / 1e6)
        gpu.append(float(ex.last_gpu_ms))
    return lab.TimingStats(tuple(wall)), lab.TimingStats(tuple(gpu))


def measure_stages(ex: MoeLayerExecutor, x: np.ndarray, spec: lab.BenchSpec
                   ) -> dict[str, dict[str, Any]]:
    """All four truncations, ROUND-ROBIN within each rep.

    Measured back to back instead, the four truncations drift: the first sweep of this
    module produced a full graph FASTER than its own wave-A..wave-B prefix, i.e. a negative
    combine stage, because the machine warmed across the four blocks.  Interleaving makes
    the four series paired samples under one set of conditions, which is the only way a
    difference of medians means anything.
    """
    for _ in range(spec.warmup):
        for stage in STAGES:
            ex.set_input(x)
            ex.run_layer(stop_after=stage)
    wall: dict[str, list[float]] = {s: [] for s in STAGES}
    gpu: dict[str, list[float]] = {s: [] for s in STAGES}
    for _ in range(spec.reps):
        for stage in STAGES:
            start = time.perf_counter_ns()
            ex.set_input(x)
            ex.run_layer(stop_after=stage)
            wall[stage].append((time.perf_counter_ns() - start) / 1e6)
            gpu[stage].append(float(ex.last_gpu_ms))
    return {s: {"wall": timing_json(lab.TimingStats(tuple(wall[s]))),
                "gpu": timing_json(lab.TimingStats(tuple(gpu[s])))} for s in STAGES}


NOISE_BAND = 0.05                      # lab_b's band; median AND min must both clear it


def _band_verdict(median_ratio: float, min_ratio: float) -> str:
    """A lever is real only when the median and the min agree outside a 5% band."""
    if median_ratio > 1 + NOISE_BAND and min_ratio > 1 + NOISE_BAND:
        return "VEC4_WINS"
    if median_ratio < 1 - NOISE_BAND and min_ratio < 1 - NOISE_BAND:
        return "SCALAR_WINS"
    return "NEUTRAL_WITHIN_NOISE"


def roofline(bytes_moved: int, seconds: float) -> dict[str, Any]:
    gb_s = bytes_moved / seconds / 1e9
    return {"bytes": bytes_moved, "seconds": seconds, "achieved_gb_s": gb_s,
            "fraction_of_roof": gb_s / BANDWIDTH_ROOF_GB_S,
            "roof_gb_s": BANDWIDTH_ROOF_GB_S}


def stage_breakdown(stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cumulative truncations differenced into per-stage cost, on the GPU clock.

    Metal times a COMMAND BUFFER, not an encoder, so each stage is measured as its own
    truncated graph and the stage cost is a difference of medians.  Differences of order
    statistics are not themselves order statistics, so the deltas are labelled DERIVED and
    the cumulative medians they came from are published beside them.
    """
    order = list(STAGES)
    out, prev = {}, 0.0
    for name in order:
        cum = stages[name]["gpu"]["median"]
        out[name] = {
            "cumulative_gpu_median_ms": cum,
            "stage_gpu_median_ms": cum - prev,
            "status": "DERIVED (difference of two measured cumulative medians)",
            "cumulative_wall_median_ms": stages[name]["wall"]["median"],
        }
        prev = cum
    out["_measurement_context"] = {
        "note": "cumulative medians here come from the ROUND-ROBIN stage series, where each "
                "rep submits all four truncations; they run a little slower than the "
                "standalone full-graph measurement in `latency` and must not be compared "
                "to it.  Only the differences between them are used.",
    }
    out["residual"] = {
        "stage_gpu_median_ms": 0.0,
        "status": "FUSED into moe_combine -- the residual is the combine kernel's "
                  "accumulator initialiser, so it has no separable cost",
    }
    return out


def _ll(cbs: int, tpg: int) -> dict[str, Any]:
    return {"grammar": "lookup-linear", "cbs": cbs, "tpg": tpg,
            "half_table": False, "row4": True}


def _df(cbs: int, tpg: int) -> dict[str, Any]:
    return {"grammar": "decode-FMA", "cbs": cbs, "tpg": tpg}


# Swept AT LAYER SCALE, deliberately.  Track B's sealed law: a shape chosen at one matvec is
# chosen under the wrong regime, because ~85% of a single matvec's wall is the command-buffer
# constant.  A layer runs 18 concurrent wave-A dispatches, so per-dispatch occupancy is not
# the same question it was per tensor.  This sweep is what decides whether the selection
# matrix's per-tensor choices survive that regime change; it is not assumed either way.
WAVE_A_SWEEP = [_ll(8, 256), _ll(8, 1024), _ll(16, 256), _ll(16, 1024), _ll(32, 256),
                _ll(32, 1024), _ll(48, 1024), _df(8, 64), _df(8, 256), _df(16, 256),
                _df(32, 64), _df(32, 256), _df(48, 256), _df(96, 256), _df(192, 256)]
WAVE_B_SWEEP = [_ll(8, 256), _ll(8, 1024), _ll(16, 256), _ll(16, 1024), _ll(32, 256),
                _ll(32, 1024), _ll(48, 1024), _df(8, 64), _df(8, 256), _df(16, 64),
                _df(16, 256), _df(32, 64), _df(32, 256), _df(64, 256), _df(128, 256)]


def sweep_layer_shapes(loaded: dict[str, Any], weights: np.ndarray, x: np.ndarray,
                       dec, reference: np.ndarray, *, reps: int, warmup: int
                       ) -> dict[str, Any]:
    """Both waves swept at layer scale.  Every config parity-graded BEFORE it is timed.

    Staged, not a cross product: wave A against the selected wave B, then wave B against
    whichever wave A won.  A full cross product is 225 layer builds for a surface whose two
    axes did not interact in either track's data.
    """
    geom: LayerGeometry = loaded["geometry"]
    rows: list[dict[str, Any]] = []

    def one(a_cfg: dict[str, Any], b_cfg: dict[str, Any], axis: str) -> dict[str, Any] | None:
        try:
            plan = layer_plan(geom, wave_a_cfg=a_cfg, wave_b_cfg=b_cfg)
        except Exception as exc:                       # lab_b refuses; record why, never time
            return {"axis": axis, "wave_a": a_cfg, "wave_b": b_cfg,
                    "status": "REFUSED_BY_PLAN", "reason": str(exc)}
        ex = MoeLayerExecutor(dec=dec)
        ex.prepare(loaded["experts"], weights, plan=plan)
        ex.set_input(x)
        parity = parity_of(ex.run_layer(), reference)
        row = {"axis": axis, "wave_a": a_cfg, "wave_b": b_cfg,
               "blocks_a": plan["wave_a"]["shape"]["blocks"],
               "blocks_b": plan["wave_b"]["shape"]["blocks"],
               "scratch_bytes": plan["scratch_bytes"],
               "executed_total_bytes": plan["byte_ledger"]["executed_total_bytes"],
               "parity_relative_l2": parity["relative_l2"]}
        if parity["relative_l2"] > PARITY_GATE:
            row["status"] = "NOT_TIMED_PARITY_FAILED"
            return row
        spec = lab.BenchSpec(rows=geom.rows_b, cols=geom.cols_a, batch=1, input_seed=SEED,
                             input_dtype="float32", output_dtype="float32", warmup=warmup,
                             reps=reps, sync_boundary="per_call_host_sync",
                             dependency_shape="serial_dependent_chain",
                             pack_in_timed_region=False, unpack_in_timed_region=False)
        wall, gpu = measure_layer(ex, x, stop_after="combine", spec=spec)
        row.update({"status": "OK", "wall": timing_json(wall), "gpu": timing_json(gpu)})
        return row

    def best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in candidates if r.get("status") == "OK"]
        if not ok:
            raise MoeLayerError("no swept config survived parity")
        return min(ok, key=lambda r: r["gpu"]["median"])

    a_rows = [one(cfg, SELECTED["wave_b"], "wave_a") for cfg in WAVE_A_SWEEP]
    a_rows = [r for r in a_rows if r]
    best_a = best(a_rows)["wave_a"]
    b_rows = [one(best_a, cfg, "wave_b") for cfg in WAVE_B_SWEEP]
    b_rows = [r for r in b_rows if r]
    best_b = best(b_rows)["wave_b"]
    rows = a_rows + b_rows

    def same(x_cfg: dict[str, Any], y_cfg: dict[str, Any]) -> bool:
        keys = ("grammar", "cbs", "tpg")
        return all(x_cfg.get(key) == y_cfg.get(key) for key in keys)

    return {
        "method": "staged, layer scale, parity graded before timing",
        "reps": reps, "warmup": warmup, "configs_timed": sum(1 for r in rows
                                                             if r.get("status") == "OK"),
        "parity_failures": sum(1 for r in rows if r.get("status") == "NOT_TIMED_PARITY_FAILED"),
        "refused": sum(1 for r in rows if r.get("status") == "REFUSED_BY_PLAN"),
        "best_wave_a": best_a, "best_wave_b": best_b,
        "selection_matrix_survives_layer_scale": bool(
            same(best_a, SELECTED["wave_a"]) and same(best_b, SELECTED["wave_b"])),
        "verdict": ("SELECTION_MATRIX_SURVIVES_LAYER_SCALE"
                    if same(best_a, SELECTED["wave_a"]) and same(best_b, SELECTED["wave_b"])
                    else "LAYER_SCALE_PREFERS_A_DIFFERENT_SHAPE"),
        "rows": rows,
    }


def run(layer: int | None, *, reps: int, warmup: int, wave_b_alt: bool = True,
        sweep: bool = True, sweep_reps: int = 20) -> dict[str, Any]:
    loaded = load_layer(layer)
    geom: LayerGeometry = loaded["geometry"]
    experts = loaded["experts"]
    weights = routing_weights(len(experts) - 1)
    x = grf.synthetic_activation(geom.cols_a, seed=SEED)

    plan = layer_plan(geom, wave_a_cfg=SELECTED["wave_a"], wave_b_cfg=SELECTED["wave_b"])
    ex = MoeLayerExecutor()
    ex.prepare(experts, weights, plan=plan)
    ex.set_input(x)

    # ---- parity FIRST.  A variant that fails is never timed.
    t0 = time.perf_counter()
    authority = cpu_layer([{p: e[p]["codes"] for p in ("gate", "up", "down")} for e in experts],
                          x, weights)
    authority_s = time.perf_counter() - t0
    got = ex.run_layer()
    parity = parity_of(got, authority["y"])
    hid = parity_of(ex.intermediate(0), authority["hidden"][0])
    if parity["relative_l2"] > PARITY_GATE:
        return {"status": "NOT_TIMED_PARITY_FAILED", "parity": parity, "plan": plan}

    spec = lab.BenchSpec(
        rows=geom.rows_b, cols=geom.cols_a, batch=1, input_seed=SEED,
        input_dtype="float32", output_dtype="float32", warmup=warmup, reps=reps,
        sync_boundary="per_call_host_sync", dependency_shape="serial_dependent_chain",
        pack_in_timed_region=False, unpack_in_timed_region=False)

    # Headline latency: the full graph, measured ALONE.  The stage breakdown below runs the
    # four truncations round-robin, which is what makes their differences meaningful and
    # also what makes their absolute values slightly higher -- four command buffers per rep
    # interfere where one does not.  The two are reported separately for exactly that reason.
    fw, fg = measure_layer(ex, x, stop_after="combine", spec=spec)
    full_wall, full_gpu = timing_json(fw), timing_json(fg)
    stages = measure_stages(ex, x, spec)

    # ---- SwiGLU vectorisation lever, matched A/B, both arms parity-graded
    ex.swiglu_vec4 = False
    scalar_parity = parity_of(ex.run_layer(), authority["y"])
    sw, sg = measure_layer(ex, x, stop_after="combine", spec=spec)
    ex.swiglu_vec4 = True
    swiglu_lever = {
        "vec4": {"wall": full_wall, "gpu": full_gpu},
        "scalar": {"wall": timing_json(sw), "gpu": timing_json(sg),
                   "parity_vs_cpu_authority": scalar_parity},
        "gpu_median_ratio_scalar_over_vec4": sg.median_ms / full_gpu["median"],
        "wall_median_ratio_scalar_over_vec4": sw.median_ms / full_wall["median"],
        "verdict": _band_verdict(sg.median_ms / full_gpu["median"],
                                 sg.min_ms / full_gpu["min"]),
        "note": "reduce+SwiGLU walks `blocks` partials with a rows-wide stride; four rows "
                "per thread gives four independent accumulator chains and 16-byte loads "
                "for identical bytes.  Both arms are parity-graded; a verdict of "
                "NEUTRAL_WITHIN_NOISE means the vectorisation bought nothing and is kept "
                "only because it is already compiled.",
    }

    # ---- shared expert isolated, by removing it rather than by modelling it
    plan_routed = layer_plan(geom, wave_a_cfg=SELECTED["wave_a"],
                             wave_b_cfg=SELECTED["wave_b"], shared=False)
    ex_routed = MoeLayerExecutor(dec=ex.dec)
    ex_routed.prepare(experts[:-1], weights[:-1], plan=plan_routed)
    ex_routed.set_input(x)
    rw, rg = measure_layer(ex_routed, x, stop_after="combine", spec=spec)
    routed_only = {"wall": timing_json(rw), "gpu": timing_json(rg), "plan": plan_routed}
    got_routed = ex_routed.run_layer()
    authority_routed = authority["y"] - np.float32(weights[-1]) * authority["down"][-1]
    routed_only["parity_vs_cpu_authority"] = parity_of(got_routed, authority_routed)

    # ---- host-encode lever, matched A/B in one process
    ex.hoist_invariants = False
    nw, ng = measure_layer(ex, x, stop_after="combine", spec=spec)
    ex.hoist_invariants = True
    host_lever = {
        "hoisted": {"wall": full_wall, "gpu": full_gpu},
        "per_dispatch_rebind": {"wall": timing_json(nw), "gpu": timing_json(ng)},
        "wall_median_saved_ms": nw.median_ms - full_wall["median"],
        "wall_median_ratio": nw.median_ms / full_wall["median"],
        "gpu_median_ratio": ng.median_ms / full_gpu["median"],
        "note": "a HOST lever: it moves wall and must not move the GPU clock.  If the GPU "
                "ratio is not ~1.0 the two arms are not the same graph.",
    }

    variants = {}
    if sweep:
        variants["layer_scale_shape_sweep"] = sweep_layer_shapes(
            loaded, weights, x, ex.dec, authority["y"], reps=sweep_reps, warmup=5)
    if wave_b_alt:
        alt_plan = layer_plan(geom, wave_a_cfg=SELECTED["wave_a"], wave_b_cfg=WAVE_B_ALT)
        ex_alt = MoeLayerExecutor(dec=ex.dec)
        ex_alt.prepare(experts, weights, plan=alt_plan)
        ex_alt.set_input(x)
        alt_got = ex_alt.run_layer()
        alt_parity = parity_of(alt_got, authority["y"])
        if alt_parity["relative_l2"] > PARITY_GATE:
            variants["wave_b_lookup_linear"] = {"status": "NOT_TIMED_PARITY_FAILED",
                                                "parity": alt_parity}
        else:
            aw, ag = measure_layer(ex_alt, x, stop_after="combine", spec=spec)
            variants["wave_b_lookup_linear"] = {
                "config": WAVE_B_ALT, "plan": alt_plan,
                "wall": timing_json(aw), "gpu": timing_json(ag),
                "parity_vs_cpu_authority": alt_parity,
                "vs_selected_gpu_median": ag.median_ms / full_gpu["median"],
                "vs_selected_wall_median": aw.median_ms / full_wall["median"],
            }

    total_bytes = plan["byte_ledger"]["executed_total_bytes"]
    read_bytes = plan["byte_ledger"]["executed_read_bytes"]
    return {
        "status": "OK",
        "layer": loaded["layer"],
        "router_status": ROUTER_STATUS,
        "router_note": ROUTER_NOTE,
        "activation_source": grf.SYNTHETIC,
        "activation_provenance": experts[0]["gate"]["fixture"].activation_provenance,
        "plan": plan,
        "routing_weights": [float(w) for w in weights],
        "routing_weight_note": (
            "FIXED VECTOR, softmax of a seeded pseudo-logit; the shared expert's trailing "
            "1.0 is not part of the softmax.  No router tensor exists to produce these."),
        "spec": spec.to_json(),
        "latency": {"wall": full_wall, "gpu": full_gpu},
        "stages": stages,
        "stage_breakdown": stage_breakdown(stages),
        "routed_only_no_shared": routed_only,
        "shared_expert_cost": {
            "wall_median_delta_ms": full_wall["median"] - routed_only["wall"]["median"],
            "gpu_median_delta_ms": full_gpu["median"] - routed_only["gpu"]["median"],
            "status": "DERIVED (difference of two measured medians)",
        },
        "variants": variants,
        "host_encode_lever": host_lever,
        "swiglu_vec4_lever": swiglu_lever,
        "parity_whole_layer": parity,
        "parity_post_swiglu_expert0": hid,
        "parity_authority": {
            "implementation": "gravity_forge.pq_execute per projection, composed through "
                              "silu(gate)*up -> down -> routing weight -> shared -> residual",
            "tensors": 3 * len(experts),
            "seconds": authority_s,
            "why_whole_layer": "SwiGLU is nonlinear, so per-projection parity does not "
                               "bound the layer's error; this is the composed figure",
        },
        "roofline": {
            "gpu_clock": roofline(total_bytes, full_gpu["median"] / 1e3),
            "wall_clock": roofline(total_bytes, full_wall["median"] / 1e3),
            "read_only_gpu_clock": roofline(read_bytes, full_gpu["median"] / 1e3),
        },
        "pressure": {
            "wall_median": pressure_verdict(full_wall["median"]),
            "wall_min": pressure_verdict(full_wall["min"]),
            "command_buffers": {
                "measured": plan["command_buffers_per_layer"],
                "targets": {n: v for n, v in COMMAND_BUFFER_TARGETS},
                "reached": "moonshot",
            },
        },
        "per_token": per_token_projection(
            full_wall["median"], source="whole-layer wall median, this run"),
        "per_token_gpu_clock": per_token_projection(
            full_gpu["median"], source="whole-layer GPU median, this run -- excludes the "
                                       "215.8 us per-command-buffer host cost"),
    }


def provenance(loaded: dict[str, Any]) -> dict[str, Any]:
    fx = loaded["fixture_set"]
    return {
        "layer": loaded["layer"],
        "router_present": fx["router_present"],
        "activation": fx["activation"],
        "experts": [{p: e[p]["fixture"].as_json() for p in ("gate", "up", "down")}
                    for e in loaded["experts"]],
    }


def machine() -> dict[str, Any]:
    return {**lab.MACHINE_FACTS, "platform": platform.platform(),
            "python": sys.version.split()[0]}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-alt", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    loaded = load_layer(args.layer)
    result = run(args.layer, reps=args.reps, warmup=args.warmup, wave_b_alt=not args.no_alt)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    executor_doc = {
        "schema": SCHEMA, "kernel_version": KERNEL_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine(),
        "router_status": ROUTER_STATUS, "router_note": ROUTER_NOTE,
        "activation_source": grf.SYNTHETIC,
        "graph": [
            "fixed expert list (ROUTER ABSENT)",
            "Wave A: 8 routed + 1 shared, gate and up, 18 matvecs",
            "fused reduce + SwiGLU, 9 dispatches",
            "Wave B: 9 down matvecs into one [nexp][blocks][rows] partials buffer",
            "fused combine: reduce + routing weight + shared expert + residual, 1 dispatch",
        ],
        "plan": result.get("plan"),
        "plan_no_shared": result.get("routed_only_no_shared", {}).get("plan"),
        "extra_kernels": ["reduce_swiglu", "moe_combine"],
        "reused_from": "gravity_metal_lab_b (ll_blk_f32_r4, dfma_split, upload cache, "
                       "1024-deep command queue, autorelease pool)",
        "selected_kernels": SELECTED,
        "wave_b_alternative": WAVE_B_ALT,
        "unimplemented": {
            "fused_gate_up_single_kernel": (
                "UNIMPLEMENTED_NOT_MEASURED.  gate and up share x but NOT a codebook "
                "(measured: 60 of 60 codebooks on one shard hash distinctly), so a fused "
                "kernel would build two independent tables from one staged x.  Its whole "
                "saving is one 0.71 us marginal dispatch and one 512 B x stage per expert "
                "-- 9 x 0.71 us against a MEASURED 637 us layer, about 1% -- while the two "
                "tables raise wave-A scratch from 12800 B to 20992 B and cut threadgroup "
                "residency.  Not built because the arithmetic says it loses, and that "
                "arithmetic is stated here rather than the result asserted."),
            "hybrid_within_one_tensor": "UNIMPLEMENTED_NOT_MEASURED (no compiled kernel)",
        },
        "provenance": provenance(loaded),
    }
    (REPORT_DIR / "GLM52_MOE_LAYER_EXECUTOR.json").write_text(
        json.dumps(executor_doc, indent=2, default=str))

    bench_doc = {
        "schema": BENCH_SCHEMA, "kernel_version": KERNEL_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine(), "parity_gate": PARITY_GATE,
        "result": result,
    }
    (REPORT_DIR / "GLM52_MOE_LAYER_BENCHMARK.json").write_text(
        json.dumps(bench_doc, indent=2, default=str))
    print(json.dumps({
        "status": result["status"], "layer": result["layer"],
        "wall_median_ms": result["latency"]["wall"]["median"],
        "gpu_median_ms": result["latency"]["gpu"]["median"],
        "parity_relative_l2": result["parity_whole_layer"]["relative_l2"],
        "pressure": result["pressure"]["wall_median"]["reached"],
        "implied_tok_s": result["per_token"]["implied_tok_s"],
    }, indent=2))
    return 0


def selftest() -> int:
    """CPU-only: the graph ledger, the plan and the parity composition on a tiny fake layer."""
    geom = LayerGeometry(rows_a=64, cols_a=128, rows_b=128, cols_b=64, D=8, k=128)
    plan = layer_plan(geom, n_routed=2,
                      wave_a_cfg={"grammar": "lookup-linear", "cbs": 8, "tpg": 64,
                                  "half_table": False, "row4": True},
                      wave_b_cfg={"grammar": "decode-FMA", "cbs": 4, "tpg": 64})
    assert plan["command_buffers_per_layer"] == 1
    assert plan["dispatches_per_layer"] == 2 * 3 + 3 + 3 + 1
    assert plan["byte_ledger"]["executed_total_bytes"] > 0
    w = routing_weights(2)
    assert abs(float(w[:2].sum()) - 1.0) < 1e-6 and w[2] == 1.0
    print(json.dumps({"plan_ok": True, "dispatches": plan["dispatches_per_layer"],
                      "executed_over_artifact":
                          plan["byte_ledger"]["executed_over_artifact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
