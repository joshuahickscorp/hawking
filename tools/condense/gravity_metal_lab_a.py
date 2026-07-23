#!/usr/bin/env python3.12
"""Track A: 2D split-chunk decode-FMA kernel, and the ledger that decides whether it wins.

The production kernel (:mod:`gravity_metal`, v2) writes ``const uint row = gid``.  That
pins the total thread count at ``rows``, so THREADS only repackages a fixed grid instead of
widening it, and every thread then walks an ``nchunk``-deep serially dependent fma chain
with nothing to hide the latency behind.  Measured: 2048 threads 0.6434 ms, 65536 threads
0.0418 ms -- 15.2x, and core count explains none of it.

This module keeps the decode-in-registers idea and changes only the grid:

  * threadgroups are 2D, ``(row_tile, chunk_block)``.  A threadgroup owns ``tpg`` rows and
    one contiguous block of chunks, accumulates a partial in a register, and writes it to
    ``partials[block * rows + row]``.  That layout is the reason the second pass coalesces:
    adjacent threads (adjacent rows) touch adjacent floats in both passes.  Threads in
    flight become ``rows * chunk_blocks`` instead of ``rows``.
  * a second dispatch reduces ``chunk_blocks`` partials per row.  It re-associates the fp32
    sum, so parity is reported against the CPU authority AND against the production kernel.
    ``chunk_blocks == 1`` binds ``y`` as the partial buffer and skips the reduce entirely,
    which makes it an honest no-split control rather than a control with a tax on it.
  * the inner loop over D is float4 when D % 4 == 0: two vector loads instead of eight
    scalar ones.  Compiled as a separate entry point, not a runtime branch.
  * ``stage_x`` is a POLICY over the chunk BLOCK's slice, not over the whole tensor.  The
    production kernel stages x only when ``nchunk*D*4`` fits 32 KB, which fails for the
    [6144,16384] and [6144,12288] attention geometries -- the 81 tensors that stream the
    whole 64 KB x per thread and carry 33.3 GB of the 38.6 GB/token executed traffic.  A
    chunk block's slice is ``cbs*D*4``, which fits at any geometry once the block is small
    enough, so the split fixes the grid underfill and the x blowup with one change.
  * indices are read either as the pre-unpacked uint8 stream the production kernel uploads
    or, natively, as the artifact's own 7-bit packed stream unpacked in-kernel.  The 7-bit
    path saves 14.3% of the index traffic and costs a shift-mask per lookup; which one wins
    is measured per geometry, never asserted.

Everything here is additive.  ``gravity_metal`` is not imported for anything but its
content-address helper and the production comparison, and is not modified.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack  # noqa: E402
import gravity_bench_lab as lab  # noqa: E402
import gravity_forge as forge  # noqa: E402
import gravity_metal  # noqa: E402
import gravity_real_fixtures as grf  # noqa: E402

SCHEMA = "hawking.glm52.track_a_benchmark.v1"
KERNEL_VERSION = 3
SEED = 20260722
PARITY_GATE = 2e-3
REPORT_DIR = HERE.parents[1] / "reports" / "condense" / "breakthrough"
DEFAULT_THREADGROUP_MEMORY = gravity_metal.DEFAULT_THREADGROUP_MEMORY

METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

struct DimsA {
    uint rows;     // output rows
    uint nchunk;   // chunks per row
    uint D;        // subvector width
    uint k;        // codebook entries
    uint cbs;      // chunks per chunk block
    uint blocks;   // number of chunk blocks
    uint pad0;
    uint pad1;
};

// The artifact's own 7-bit stream, MSB-first, exactly as glm52_pack.pack_indices lays it
// out.  Index i occupies stream bits [7i, 7i+7); a 16-bit big-endian window starting at
// byte 7i/8 always contains all of them, so one shift and one mask recover the codeword.
inline uint gravity_idx7(device const uchar* p, uint i)
{
    uint bit = i * 7u;
    uint byte = bit >> 3;
    uint shift = bit & 7u;
    uint window = (uint(p[byte]) << 8) | uint(p[byte + 1u]);
    return (window >> (9u - shift)) & 127u;
}

#define GRAVITY_PARTIAL(NAME, BITS7, STAGEX, VEC4)                                        \
kernel void NAME(                                                                         \
    device   const uchar* indices  [[buffer(0)]],                                         \
    device   const half*  codebook [[buffer(1)]],                                         \
    device   const float* x        [[buffer(2)]],                                          \
    device         float* partials [[buffer(3)]],                                          \
    constant       DimsA& dims     [[buffer(4)]],                                          \
    threadgroup    float* scratch  [[threadgroup(0)]],                                     \
    uint2 tgpos [[threadgroup_position_in_grid]],                                          \
    uint2 tidv  [[thread_position_in_threadgroup]],                                        \
    uint2 tszv  [[threads_per_threadgroup]])                                               \
{                                                                                          \
    const uint tid = tidv.x, tsz = tszv.x;                                                 \
    const uint D = dims.D, k = dims.k, rows = dims.rows;                                   \
    const uint c0 = tgpos.y * dims.cbs;                                                    \
    const uint c1 = min(dims.nchunk, c0 + dims.cbs);                                       \
    threadgroup float* book = scratch;                                                     \
    threadgroup float* xs   = scratch + k * D;                                             \
    for (uint i = tid; i < k * D; i += tsz) book[i] = float(codebook[i]);                  \
    if (STAGEX && c1 > c0) {                                                               \
        const uint span = (c1 - c0) * D;                                                   \
        for (uint i = tid; i < span; i += tsz) xs[i] = x[c0 * D + i];                      \
    }                                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
    const uint row = tgpos.x * tsz + tid;                                                  \
    if (row >= rows) return;                                                               \
    float acc = 0.0f;                                                                      \
    for (uint c = c0; c < c1; ++c) {                                                       \
        const uint slot = c * rows + row;                                                  \
        const uint code = BITS7 ? gravity_idx7(indices, slot) : uint(indices[slot]);       \
        threadgroup const float* w = book + code * D;                                      \
        if (STAGEX) {                                                                      \
            threadgroup const float* xv = xs + (c - c0) * D;                               \
            if (VEC4) {                                                                    \
                threadgroup const float4* w4 = (threadgroup const float4*)w;               \
                threadgroup const float4* x4 = (threadgroup const float4*)xv;              \
                for (uint j = 0; j < D / 4u; ++j) acc += dot(w4[j], x4[j]);                \
            } else {                                                                       \
                for (uint j = 0; j < D; ++j) acc = fma(w[j], xv[j], acc);                  \
            }                                                                              \
        } else {                                                                           \
            device const float* xv = x + c * D;                                            \
            if (VEC4) {                                                                    \
                threadgroup const float4* w4 = (threadgroup const float4*)w;               \
                device const float4* x4 = (device const float4*)xv;                        \
                for (uint j = 0; j < D / 4u; ++j) acc += dot(w4[j], x4[j]);                \
            } else {                                                                       \
                for (uint j = 0; j < D; ++j) acc = fma(w[j], xv[j], acc);                  \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    partials[tgpos.y * rows + row] = acc;                                                  \
}

GRAVITY_PARTIAL(gp_u8_nox_scalar, 0, 0, 0)
GRAVITY_PARTIAL(gp_u8_nox_vec4,   0, 0, 1)
GRAVITY_PARTIAL(gp_u8_stx_scalar, 0, 1, 0)
GRAVITY_PARTIAL(gp_u8_stx_vec4,   0, 1, 1)
GRAVITY_PARTIAL(gp_b7_nox_scalar, 1, 0, 0)
GRAVITY_PARTIAL(gp_b7_nox_vec4,   1, 0, 1)
GRAVITY_PARTIAL(gp_b7_stx_scalar, 1, 1, 0)
GRAVITY_PARTIAL(gp_b7_stx_vec4,   1, 1, 1)

// Coalesced by construction: thread `row` walks partials[b*rows + row], so neighbouring
// threads read neighbouring floats on every one of the `blocks` strides.
kernel void gravity_pq_reduce(
    device   const float* partials [[buffer(0)]],
    device         float* y        [[buffer(1)]],
    constant       DimsA& dims     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint rows = dims.rows;
    if (gid >= rows) return;
    float acc = 0.0f;
    for (uint b = 0; b < dims.blocks; ++b) acc += partials[b * rows + gid];
    y[gid] = acc;
}
"""


class TrackAError(RuntimeError):
    """A Track A configuration cannot be run, or would misdescribe itself."""


# ------------------------------------------------------------------ pure planning / accounting
# Everything below this line answers on a machine with no Metal device, which is what makes
# the tests runnable without a GPU.

def kernel_name(*, bits7: bool, stage_x: bool, vec4: bool) -> str:
    return (f"gp_{'b7' if bits7 else 'u8'}_{'stx' if stage_x else 'nox'}"
            f"_{'vec4' if vec4 else 'scalar'}")


def plan(*, rows: int, nchunk: int, D: int, k: int, tpg: int, blocks: int,
         stage_x_policy: bool = True, vec4: bool = True, bits7: bool = False,
         threadgroup_memory_limit: int = DEFAULT_THREADGROUP_MEMORY) -> dict[str, Any]:
    """The whole shape of one variant: grid, staging decision, scratch, threads in flight.

    ``stage_x_policy`` asks for staging; whether it happens is decided here, against the
    CHUNK BLOCK's slice rather than the whole activation.  A caller cannot end up believing
    x was staged when it was not, because the answer comes back in the same dict.
    """
    if rows < 1 or nchunk < 1 or D < 1 or k < 1:
        raise TrackAError("rows, nchunk, D and k must all be >= 1")
    if tpg < 1 or blocks < 1:
        raise TrackAError("tpg and blocks must both be >= 1")
    cbs = (nchunk + blocks - 1) // blocks
    row_tiles = (rows + tpg - 1) // tpg
    vec4 = bool(vec4) and D % 4 == 0
    want = stage_x_policy
    scratch_staged = (k * D + cbs * D) * 4
    stage_x = bool(want) and scratch_staged <= threadgroup_memory_limit
    scratch = scratch_staged if stage_x else k * D * 4
    if scratch > threadgroup_memory_limit:
        raise TrackAError(
            f"codebook alone needs {scratch} B of threadgroup memory, limit is "
            f"{threadgroup_memory_limit} B")
    return {
        "tpg": tpg, "blocks": blocks, "chunks_per_block": cbs, "row_tiles": row_tiles,
        "threadgroups": row_tiles * blocks,
        "threads_in_flight": rows * blocks,
        "dispatches": 1 if blocks == 1 else 2,
        "command_buffers": 1,
        "scratch_bytes": scratch,
        "stage_x_requested": bool(want),
        "stage_x": stage_x,
        "stage_x_refused_reason": (
            None if stage_x or not want
            else f"chunk-block slice {scratch_staged} B exceeds {threadgroup_memory_limit} B"),
        "vec4": vec4,
        "bits7": bool(bits7),
        "kernel": kernel_name(bits7=bool(bits7), stage_x=stage_x, vec4=vec4),
        "partial_bytes": 0 if blocks == 1 else rows * blocks * 4,
    }


def executed_bytes(*, rows: int, cols: int, nchunk: int, D: int, k: int,
                   shape: dict[str, Any]) -> dict[str, Any]:
    """Traffic this variant really moves, split the same way gravity_metal.matvec_bytes does.

    Threadgroup memory is not shared between threadgroups, so the codebook is re-read once
    per threadgroup and, when staged, so is the chunk block's x slice.  When x is NOT staged
    every thread streams its own chunk range, which is where the attention geometries lose
    33.3 GB/token.  Both are billed; neither is hidden inside a single integer.
    """
    tgs = shape["threadgroups"]
    index = (rows * nchunk * 7 + 7) // 8 if shape["bits7"] else rows * nchunk
    book = tgs * k * D * 2
    if shape["stage_x"]:
        activation = tgs * shape["chunks_per_block"] * D * 4
    else:
        activation = rows * shape["blocks"] * shape["chunks_per_block"] * D * 4
    partial_w = shape["partial_bytes"]
    partial_r = shape["partial_bytes"]
    out = rows * 4
    read = index + book + activation + partial_r
    write = partial_w + out
    # The re-read terms above are an UPPER bound: they assume every threadgroup's (or, when
    # x is not staged, every thread's) re-read of the codebook and of x reaches DRAM.  The
    # lower bound counts each distinct byte once, which is what a perfect cache would move.
    # Reporting only the upper bound produced a >roof achieved bandwidth on the attention
    # geometry, which is not a result -- it is the model being wrong.  Both are carried.
    unique_read = index + k * D * 2 + nchunk * D * 4 + partial_r
    return {
        "model": "ANALYTIC, not a counter reading",
        "index_bytes": index,
        "index_bits_executed": 7 if shape["bits7"] else 8,
        "codebook_bytes": book,
        "activation_bytes": activation,
        "partial_write_bytes": partial_w,
        "partial_read_bytes": partial_r,
        "output_bytes": out,
        "executed_read_bytes": read,
        "executed_total_bytes": read + write,
        "unique_read_bytes": unique_read,
        "unique_total_bytes": unique_read + partial_w + out,
        "logical_artifact_bytes": (rows * nchunk * 7 + 7) // 8 + k * D * 2,
        "dense_bf16_bytes": rows * cols * 2,
        "executed_read_bpw": read * 8 / (rows * cols),
        "unique_read_bpw": unique_read * 8 / (rows * cols),
    }


def unpack7_reference(raw: bytes, count: int) -> np.ndarray:
    """The in-kernel unpack, in numpy, so the MSL can be checked without a GPU."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    bit = np.arange(count, dtype=np.int64) * 7
    byte = bit >> 3
    shift = bit & 7
    window = (buf[byte].astype(np.uint32) << 8) | buf[byte + 1].astype(np.uint32)
    return ((window >> (9 - shift)) & 127).astype(np.uint8)


def split_reference(codes: dict, x: np.ndarray, blocks: int) -> np.ndarray:
    """What the 2D split computes, in numpy, including the reassociated fp32 reduce."""
    book = np.asarray(codes["codebooks"][0], dtype=np.float32).astype(np.float16).astype(np.float32)
    rows, nchunk, D = int(codes["rows"]), int(codes["nchunk"]), int(codes["D"])
    idx = np.asarray(codes["indices"])[:, 0].reshape(rows, nchunk)
    xc = np.ascontiguousarray(x, dtype=np.float32).reshape(nchunk, D)
    cbs = (nchunk + blocks - 1) // blocks
    partials = np.zeros((blocks, rows), dtype=np.float32)
    for b in range(blocks):
        lo, hi = b * cbs, min(nchunk, (b + 1) * cbs)
        if hi <= lo:
            continue
        partials[b] = np.einsum("rcj,cj->r", book[idx[:, lo:hi]], xc[lo:hi],
                                optimize=True).astype(np.float32)
    return partials.sum(axis=0, dtype=np.float32)


# ------------------------------------------------------------------ device

class TrackADecoder:
    """Compiles the eight partial variants plus the reduce once, reuses uploads."""

    def __init__(self) -> None:
        try:
            import Metal
        except ImportError as exc:  # noqa: BLE001
            raise gravity_metal.MetalUnavailable("pyobjc-framework-Metal is not installed") from exc
        self._Metal = Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise gravity_metal.MetalUnavailable("no default Metal device")
        library, error = self.device.newLibraryWithSource_options_error_(METAL_SOURCE, None, None)
        if library is None:
            raise gravity_metal.MetalUnavailable(f"kernel failed to compile: {error}")
        self._pipelines: dict[str, Any] = {}
        for bits7 in (False, True):
            for stage_x in (False, True):
                for vec4 in (False, True):
                    self._pipelines[kernel_name(bits7=bits7, stage_x=stage_x, vec4=vec4)] = None
        self._pipelines["gravity_pq_reduce"] = None
        for name in list(self._pipelines):
            fn = library.newFunctionWithName_(name)
            if fn is None:
                raise gravity_metal.MetalUnavailable(f"no kernel named {name}")
            pipe, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if pipe is None:
                raise gravity_metal.MetalUnavailable(f"pipeline {name} failed: {error}")
            self._pipelines[name] = pipe
        # The queue defaults to 64 in-flight command buffers and pyobjc never drains the
        # autorelease pool, so anything that commits without waiting deadlocks at 64.
        self.queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
        self.threadgroup_memory_limit = int(self.device.maxThreadgroupMemoryLength())
        self.max_threads_per_threadgroup = int(self.device.maxThreadsPerThreadgroup().width)
        self._uploads: dict[str, dict] = {}

    def _buffer(self, array: np.ndarray):
        data = np.ascontiguousarray(array)
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, self._Metal.MTLResourceStorageModeShared)

    def upload(self, codes: dict, key: str, *, max_blocks: int) -> dict:
        """Both index streams, the codebook, x, partials and y.  Once per tensor."""
        hit = self._uploads.get(key)
        if hit is not None and hit["max_blocks"] >= max_blocks:
            return hit
        rows, nchunk, D = int(codes["rows"]), int(codes["nchunk"]), int(codes["D"])
        book = np.ascontiguousarray(codes["codebooks"][0], dtype=np.float16)
        k = int(book.shape[0])
        if int(book.shape[1]) != D:
            raise TrackAError(f"codebook subvector {int(book.shape[1])} != D {D}")
        flat = np.ascontiguousarray(np.asarray(codes["indices"]).ravel())
        if flat.size != rows * nchunk:
            raise TrackAError(f"{flat.size} indices, geometry says {rows*nchunk}")
        if flat.size and int(flat.max()) >= k:
            raise TrackAError(f"index {int(flat.max())} out of range for k={k}")
        transposed = np.ascontiguousarray(
            flat.reshape(rows, nchunk).T.ravel().astype(np.uint8))          # [nchunk][rows]
        packed = glm52_pack.pack_indices(transposed, 7)
        # +2 bytes so the 16-bit window of the last index cannot read past the allocation.
        packed_padded = np.frombuffer(packed + b"\x00\x00", dtype=np.uint8)
        entry = {
            "idx_u8": self._buffer(transposed),
            "idx_b7": self._buffer(packed_padded),
            "book": self._buffer(book),
            "x": self.device.newBufferWithLength_options_(
                nchunk * D * 4, self._Metal.MTLResourceStorageModeShared),
            "y": self.device.newBufferWithLength_options_(
                rows * 4, self._Metal.MTLResourceStorageModeShared),
            "partials": self.device.newBufferWithLength_options_(
                rows * max_blocks * 4, self._Metal.MTLResourceStorageModeShared),
            "rows": rows, "nchunk": nchunk, "D": D, "k": k,
            "x_bytes": nchunk * D * 4, "max_blocks": max_blocks,
            "packed_index_bytes": len(packed),
            "u8_index_bytes": int(transposed.nbytes),
            "dims": {},
        }
        self._uploads[key] = entry
        return entry

    def _dims(self, entry: dict, shape: dict):
        cache_key = (shape["chunks_per_block"], shape["blocks"])
        buf = entry["dims"].get(cache_key)
        if buf is None:
            dims = np.array([entry["rows"], entry["nchunk"], entry["D"], entry["k"],
                             shape["chunks_per_block"], shape["blocks"], 0, 0], dtype=np.uint32)
            buf = self._buffer(dims)
            entry["dims"][cache_key] = buf
        return buf

    def matvec(self, entry: dict, x: np.ndarray, shape: dict) -> np.ndarray:
        """One command buffer: partial dispatch, then reduce unless blocks == 1."""
        import objc
        Metal = self._Metal
        if shape["blocks"] > entry["max_blocks"]:
            raise TrackAError("partial buffer was sized for fewer blocks")
        if shape["tpg"] > self.max_threads_per_threadgroup:
            raise TrackAError(f"tpg {shape['tpg']} exceeds device max")
        xv = np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())
        if xv.nbytes != entry["x_bytes"]:
            raise TrackAError(f"x is {xv.nbytes} B, geometry needs {entry['x_bytes']} B")
        split = shape["blocks"] > 1
        target = entry["partials"] if split else entry["y"]
        dims = self._dims(entry, shape)
        with objc.autorelease_pool():
            entry["x"].contents().as_buffer(entry["x_bytes"])[:] = xv.tobytes()
            cb = self.queue.commandBuffer()
            enc = cb.computeCommandEncoder()
            enc.setComputePipelineState_(self._pipelines[shape["kernel"]])
            idx = entry["idx_b7"] if shape["bits7"] else entry["idx_u8"]
            for slot, buf in enumerate((idx, entry["book"], entry["x"], target, dims)):
                enc.setBuffer_offset_atIndex_(buf, 0, slot)
            enc.setThreadgroupMemoryLength_atIndex_(shape["scratch_bytes"], 0)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(shape["row_tiles"], shape["blocks"], 1),
                Metal.MTLSizeMake(shape["tpg"], 1, 1))
            enc.endEncoding()
            if split:
                enc2 = cb.computeCommandEncoder()
                enc2.setComputePipelineState_(self._pipelines["gravity_pq_reduce"])
                for slot, buf in enumerate((entry["partials"], entry["y"], dims)):
                    enc2.setBuffer_offset_atIndex_(buf, 0, slot)
                red_t = min(256, self.max_threads_per_threadgroup)
                enc2.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((entry["rows"] + red_t - 1) // red_t, 1, 1),
                    Metal.MTLSizeMake(red_t, 1, 1))
                enc2.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.error() is not None:
                raise gravity_metal.MetalUnavailable(f"dispatch failed: {cb.error()}")
            gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        out = np.frombuffer(entry["y"].contents().as_buffer(entry["rows"] * 4),
                            dtype=np.float32).copy()
        self.last_gpu_ms = gpu_ms
        return out


# ------------------------------------------------------------------ benchmark driver

def artifact_of(codes: dict) -> forge.PackedArtifact:
    ledger = forge.ByteLedger()
    ledger.add("indices", codes["indices"].size * 7)
    return forge.PackedArtifact(
        "product_quant", np.empty((0,), dtype=np.float32),
        codes["rows"] * codes["cols"], ledger, ledger.total_bits(), 0, {"pq_codes": codes})


def dense_from_codes(codes: dict) -> np.ndarray:
    book = np.asarray(codes["codebooks"][0], dtype=np.float32)
    idx = np.asarray(codes["indices"])[:, 0]
    return book[idx].reshape(int(codes["rows"]), int(codes["cols"]))


def parity_of(got: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    diff = reference - got
    denom = float(np.abs(reference).max()) + 1e-30
    return {
        "relative_l2": float(np.linalg.norm(diff) / (np.linalg.norm(reference) + 1e-30)),
        "max_abs_error": float(np.abs(diff).max()),
        "relative_max_gap": float(np.abs(diff).max() / denom),
        "cosine": float(reference @ got / ((np.linalg.norm(reference)
                                            * np.linalg.norm(got)) + 1e-30)),
        "finite": bool(np.isfinite(got).all()),
    }


B7_NOISE_BAND = 0.03


def _b7_verdict(ratios: list[float]) -> str:
    """Whether the in-kernel 7-bit unpack pays, with a noise band around 1.0.

    The unpack saves exactly 1/8 of the index stream and costs a shift and a mask per
    lookup.  Repeating the whole sweep moved the median ratio across 1.0 in both
    directions, so anything inside +/-3% is reported as neutral rather than as a win.
    """
    if not ratios:
        return "UNMEASURED"
    median = sorted(ratios)[len(ratios) // 2]
    if median < 1.0 - B7_NOISE_BAND:
        return ("NATIVE_7BIT_WINS: the in-kernel unpack costs less than the index traffic "
                f"it removes (median b7/u8 time ratio {median:.4f})")
    if median > 1.0 + B7_NOISE_BAND:
        return ("NATIVE_7BIT_LOSES: the in-kernel unpack costs more than the 12.5% of the "
                f"index stream it removes (median b7/u8 time ratio {median:.4f})")
    return ("NATIVE_7BIT_NEUTRAL_WITHIN_NOISE: median b7/u8 time ratio "
            f"{median:.4f}, inside the +/-{B7_NOISE_BAND:.0%} run-to-run band. The 12.5% "
            "index saving is real but index traffic is not the binding term at these "
            "geometries, so it buys no measurable time")


def sweep_configs(*, tpgs: tuple[int, ...], block_counts: tuple[int, ...]) -> list[dict]:
    configs = []
    for tpg in tpgs:
        for blocks in block_counts:
            for vec4 in (False, True):
                for bits7 in (False, True):
                    for stage in (False, True):
                        configs.append({"tpg": tpg, "blocks": blocks, "vec4": vec4,
                                        "bits7": bits7, "stage_x_policy": stage})
    return configs


def run_geometry(fixture: grf.Fixture, dec: TrackADecoder, prod, *, reps: int, warmup: int,
                 tpgs: tuple[int, ...], block_counts: tuple[int, ...]) -> dict[str, Any]:
    import torch

    codes = fixture.codes
    rows, cols = int(codes["rows"]), int(codes["cols"])
    nchunk, D = int(codes["nchunk"]), int(codes["D"])
    k = int(np.asarray(codes["codebooks"][0]).shape[0])
    x = fixture.activation(seed=SEED)                     # SYNTHETIC, labelled by the fixture
    flops = 2 * rows * cols

    spec = lab.BenchSpec(
        rows=rows, cols=cols, batch=1, input_seed=SEED,
        input_dtype="float32", output_dtype="float32",
        warmup=warmup, reps=reps,
        sync_boundary="per_call_host_sync",
        dependency_shape="independent_calls",
        # Neither side unpacks inside the clock: indices are unpacked and transposed during
        # upload, for the lab grammars and for the production control alike.  The flag was
        # True and wrong for both, which is symmetric and moved no ratio, but a spec that
        # misdescribes its own timed region is not a spec.
        pack_in_timed_region=False, unpack_in_timed_region=False)

    # ---- authority + the two reference outputs every variant is graded against
    t_cpu0 = time.perf_counter()
    reference = forge.pq_execute(artifact_of(codes), x)
    cpu_authority_s = time.perf_counter() - t_cpu0

    prod_key = gravity_metal.content_key(codes)
    prod_out = prod.matvec(codes, x, key=prod_key)
    prod_parity = parity_of(prod_out, reference)
    t_prod = lab.measure(lambda: prod.matvec(codes, x, key=prod_key), spec)
    prod_traffic = gravity_metal.matvec_bytes(
        codes, threadgroup_memory_limit=prod.threadgroup_memory_limit)

    # ---- dense fp16 MPS, the honest speed baseline.  Reconstruction is OUTSIDE the timed region.
    dense = dense_from_codes(codes)
    w16 = torch.from_numpy(dense.astype(np.float16)).to("mps")
    x16 = torch.from_numpy(x.astype(np.float16)).to("mps")
    torch.mps.synchronize()

    def dense_call():
        y = w16 @ x16
        torch.mps.synchronize()
        return y

    t_dense = lab.measure(dense_call, spec)
    del dense, w16, x16
    torch.mps.empty_cache()

    dense_result = lab.BenchResult(
        baseline="dense_fp16_mps", spec=spec,
        timings=lab.ComponentTimings(end_to_end=t_dense),
        bytes_moved=rows * cols * 2 + cols * 2 + rows * 2, flops=flops,
        notes="torch fp16 matvec on MPS; the dense weight was rebuilt from the artifact "
              "ONCE, outside the timed region")
    prod_result = lab.BenchResult(
        baseline="custom_v2", spec=spec,
        timings=lab.ComponentTimings(end_to_end=t_prod),
        bytes_moved=prod_traffic["executed_total_bytes"], flops=flops,
        notes="gravity_metal v2, one thread per row, unmodified")

    configs = sweep_configs(tpgs=tpgs, block_counts=block_counts)
    max_blocks = max(block_counts)
    entry = dec.upload(codes, fixture.cache_key, max_blocks=max_blocks)

    ledger: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for cfg in configs:
        try:
            shape = plan(rows=rows, nchunk=nchunk, D=D, k=k,
                         threadgroup_memory_limit=dec.threadgroup_memory_limit, **cfg)
        except TrackAError as exc:
            ledger.append({"config": cfg, "verdict": "REFUSED", "reason": str(exc)})
            continue
        signature = (shape["kernel"], shape["tpg"], shape["blocks"])
        if signature in seen:
            continue                                  # stage_x policy collapsed onto an existing run
        seen.add(signature)

        got = dec.matvec(entry, x, shape)
        vs_authority = parity_of(got, reference)
        vs_production = parity_of(got, prod_out)
        passed = vs_authority["finite"] and vs_authority["relative_max_gap"] < PARITY_GATE
        row: dict[str, Any] = {
            "variant": f"{shape['kernel']}/tpg{shape['tpg']}/blk{shape['blocks']}",
            "config": cfg,
            "shape": shape,
            "parity_vs_cpu_authority": vs_authority,
            "parity_vs_production_kernel": vs_production,
            "parity_gate": PARITY_GATE,
            "parity_passed": bool(passed),
        }
        if not passed:
            row["verdict"] = "FAILED_PARITY"
            row["timing"] = "NOT_TIMED: a variant that fails parity has no speed"
            ledger.append(row)
            continue

        stats = lab.measure(lambda s=shape: dec.matvec(entry, x, s), spec)
        traffic = executed_bytes(rows=rows, cols=cols, nchunk=nchunk, D=D, k=k, shape=shape)
        result = lab.BenchResult(
            baseline="track_a", spec=spec,
            timings=lab.ComponentTimings(end_to_end=stats),
            bytes_moved=traffic["executed_total_bytes"], flops=flops,
            notes="Track A 2D split-chunk; wall includes x upload, both dispatches, "
                  "waitUntilCompleted and readback")
        # Bill bandwidth on the GPU clock, not on wall.  A single matvec's wall is 70 to 90
        # percent the 215.8 us command-buffer constant, so dividing bytes by wall answers
        # "how fast is submission" and calls it bandwidth; it understates the kernel by
        # roughly 6.5x at gate.  Wall-billed rates are kept alongside, named for what they
        # are, because the wall is still what a caller pays.
        gpu_seconds = (dec.last_gpu_ms / 1e3) if dec.last_gpu_ms else None
        wall_seconds = stats.median_ms / 1e3
        gb_s = (traffic["executed_total_bytes"] / gpu_seconds / 1e9) if gpu_seconds else None
        unique_gb_s = (traffic["unique_total_bytes"] / gpu_seconds / 1e9) if gpu_seconds else None
        gb_s_wall = traffic["executed_total_bytes"] / wall_seconds / 1e9
        unique_gb_s_wall = traffic["unique_total_bytes"] / wall_seconds / 1e9
        row.update({
            "traffic": traffic,
            "timing_ms": {"median": stats.median_ms, "min": stats.min_ms,
                          "p95": stats.p95_ms, "max": stats.max_ms,
                          "coefficient_of_variation": stats.coefficient_of_variation,
                          "is_contended": stats.is_contended,
                          "raw_samples_ms": list(stats.raw_samples_ms)},
            "last_gpu_ms": dec.last_gpu_ms,
            "achieved_gb_s": gb_s,
            "fraction_of_bandwidth_roof": (gb_s / lab.BANDWIDTH_ROOF_GB_S) if gb_s else None,
            "achieved_gb_s_unique_bytes": unique_gb_s,
            "fraction_of_bandwidth_roof_unique_bytes": (
                unique_gb_s / lab.BANDWIDTH_ROOF_GB_S) if unique_gb_s else None,
            "rate_billed_on": "gpu_clock" if gpu_seconds else "UNAVAILABLE_NO_GPU_CLOCK",
            "achieved_gb_s_wall": gb_s_wall,
            "achieved_gb_s_unique_bytes_wall": unique_gb_s_wall,
            "wall_rate_note": "wall includes the 215.8 us command-buffer constant, which is "
                              "70 to 90 percent of a single matvec; these are submission "
                              "rates, not kernel bandwidth",
            "byte_model_verdict": (
                "UNGRADED_NO_GPU_CLOCK" if not gb_s else
                "ANALYTIC_REREAD_MODEL_REFUTED_BY_MEASUREMENT: the upper-bound model bills "
                "more DRAM traffic than the 736 GB/s roof can deliver in the measured time, "
                "so the caches are serving the per-thread re-reads of x"
                if gb_s > lab.BANDWIDTH_ROOF_GB_S else
                "upper and lower byte models both fit under the roof"),
            "achieved_gflop_s": (flops / gpu_seconds / 1e9) if gpu_seconds else None,
            "fraction_of_compute_roof": (
                flops / gpu_seconds / 1e9 / lab.COMPUTE_ROOF_GFLOP_S) if gpu_seconds else None,
            "achieved_gflop_s_wall": flops / wall_seconds / 1e9,
            # A review found the labs read a [nchunk][rows] transpose prepared outside the
            # timed region and concluded gravity_metal.matvec reads the on-disk layout, so
            # vs_production would not be a drop-in claim.  Measured on real gate and down
            # tensors, that is wrong: gravity_metal.py:394 performs the identical transpose
            # in _cache_tensor, the two buffers are byte-identical, and both are built in a
            # cached upload outside the clock.  The comparison is symmetric.  The transpose
            # costs 1.67 ms per tensor once, on both sides.
            "vs_production_caveat": "NONE_INDEX_LAYOUT_IS_SYMMETRIC_MEASURED_BYTE_IDENTICAL",
            "vs_production_layout_note":
                "gravity_metal.py:394 and this lab both transpose to [nchunk][rows] in a "
                "cached upload; buffers verified byte-identical on real tensors; one-time "
                "cost 1.67 ms median per tensor, paid by both",
            "vs_production_median": lab.speedup(prod_result, result)["speedup"],
            "vs_production_min": t_prod.min_ms / stats.min_ms,
            "vs_dense_fp16_median": lab.speedup(dense_result, result)["speedup"],
            "vs_dense_fp16_min": t_dense.min_ms / stats.min_ms,
        })
        row["verdict"] = ("FASTER_THAN_DENSE" if row["vs_dense_fp16_median"] >= 1.0
                          else "FASTER_THAN_PRODUCTION_STILL_SLOWER_THAN_DENSE"
                          if row["vs_production_median"] >= 1.0
                          else "SLOWER_THAN_PRODUCTION")
        ledger.append(row)

    timed = [r for r in ledger if r.get("parity_passed") and "timing_ms" in r]
    best = min(timed, key=lambda r: r["timing_ms"]["median"]) if timed else None

    # ---- the head-to-head mandate 1.5 asks for, at the best (tpg, blocks, vec4, stage) shape
    unpack_pairs = []
    for row in timed:
        if row["config"]["bits7"]:
            continue
        mate = next((r for r in timed
                     if r["config"]["bits7"]
                     and r["shape"]["tpg"] == row["shape"]["tpg"]
                     and r["shape"]["blocks"] == row["shape"]["blocks"]
                     and r["shape"]["vec4"] == row["shape"]["vec4"]
                     and r["shape"]["stage_x"] == row["shape"]["stage_x"]), None)
        if mate is None:
            continue
        saved = row["traffic"]["index_bytes"] - mate["traffic"]["index_bytes"]
        unpack_pairs.append({
            "shape": {kk: row["shape"][kk] for kk in ("tpg", "blocks", "vec4", "stage_x")},
            "u8_variant": row["variant"], "b7_variant": mate["variant"],
            "u8_median_ms": row["timing_ms"]["median"], "b7_median_ms": mate["timing_ms"]["median"],
            "u8_min_ms": row["timing_ms"]["min"], "b7_min_ms": mate["timing_ms"]["min"],
            "index_bytes_saved": saved,
            "index_traffic_saving_fraction": saved / max(1, row["traffic"]["index_bytes"]),
            "total_traffic_saving_fraction": saved / max(1, row["traffic"]["executed_total_bytes"]),
            "b7_over_u8_median": mate["timing_ms"]["median"] / row["timing_ms"]["median"],
            "b7_over_u8_min": mate["timing_ms"]["min"] / row["timing_ms"]["min"],
            "b7_parity_vs_u8_max_abs": float(np.abs(
                np.array(mate["parity_vs_cpu_authority"]["max_abs_error"])
                - np.array(row["parity_vs_cpu_authority"]["max_abs_error"]))),
            "verdict": ("B7_WINS" if mate["timing_ms"]["median"] < row["timing_ms"]["median"]
                        else "B7_COSTS_MORE_THAN_IT_SAVES"),
        })

    def ablation(field: str, on: bool, off: bool) -> list[dict]:
        pairs = []
        for row in timed:
            if row["shape"][field] != on:
                continue
            mate = next((r for r in timed
                         if r["shape"][field] == off
                         and all(r["shape"][kk] == row["shape"][kk]
                                 for kk in ("tpg", "blocks", "vec4", "stage_x", "bits7")
                                 if kk != field)), None)
            if mate is None:
                continue
            pairs.append({
                "on_variant": row["variant"], "off_variant": mate["variant"],
                "on_median_ms": row["timing_ms"]["median"],
                "off_median_ms": mate["timing_ms"]["median"],
                "on_over_off_median": row["timing_ms"]["median"] / mate["timing_ms"]["median"],
                "speedup_of_on": mate["timing_ms"]["median"] / row["timing_ms"]["median"],
            })
        return pairs

    def band(values: list[float]) -> Any:
        if not values:
            return lab.UNMEASURED
        ordered = sorted(values)
        return {"n": len(ordered), "min": ordered[0],
                "median": ordered[len(ordered) // 2], "max": ordered[-1]}

    vec4_pairs = ablation("vec4", True, False)
    stage_pairs = ablation("stage_x", True, False)
    return {
        "fixture": fixture.as_json(),
        "geometry": {"rows": rows, "cols": cols, "nchunk": nchunk, "D": D, "k": k,
                     "S": int(codes["S"]), "sub": int(codes["sub"]),
                     "rotate": bool(codes["rotate"])},
        "index_distribution": {kk: v for kk, v in
                               grf.index_distribution(codes).items() if kk != "histogram"},
        "router": "ABSENT from every shard; expert selection here is a FIXED LIST, "
                  "not a routing decision",
        "cpu_authority_seconds": cpu_authority_s,
        "production_baseline": {
            "result": prod_result.to_json(),
            "parity_vs_cpu_authority": prod_parity,
            "traffic": prod_traffic,
        },
        "dense_baseline": dense_result.to_json(),
        "ledger": ledger,
        "variants_timed": len(timed),
        "variants_failed_parity": len([r for r in ledger if r.get("verdict") == "FAILED_PARITY"]),
        "worst_timed_variant": None if not timed else max(
            timed, key=lambda r: r["timing_ms"]["median"])["variant"],
        "best": None if best is None else {
            "variant": best["variant"], "shape": best["shape"],
            "median_ms": best["timing_ms"]["median"], "min_ms": best["timing_ms"]["min"],
            "p95_ms": best["timing_ms"]["p95"],
            "vs_production_median": best["vs_production_median"],
            "vs_production_min": best["vs_production_min"],
            "vs_dense_fp16_median": best["vs_dense_fp16_median"],
            "vs_dense_fp16_min": best["vs_dense_fp16_min"],
            "achieved_gb_s": best["achieved_gb_s"],
            "fraction_of_bandwidth_roof": best["fraction_of_bandwidth_roof"],
            "achieved_gb_s_unique_bytes": best["achieved_gb_s_unique_bytes"],
            "fraction_of_bandwidth_roof_unique_bytes":
                best["fraction_of_bandwidth_roof_unique_bytes"],
            "byte_model_verdict": best["byte_model_verdict"],
            "parity_vs_cpu_authority": best["parity_vs_cpu_authority"],
            "parity_vs_production_kernel": best["parity_vs_production_kernel"],
            "verdict": best["verdict"],
        },
        "native_7bit_head_to_head": unpack_pairs,
        "ablation_vec4": vec4_pairs,
        "ablation_stage_x": stage_pairs,
        "lever_summary": {
            "vec4_speedup": band([p["speedup_of_on"] for p in vec4_pairs]),
            "stage_x_speedup": band([p["speedup_of_on"] for p in stage_pairs]),
            "b7_over_u8_time_ratio": band([p["b7_over_u8_median"] for p in unpack_pairs]),
            "b7_index_traffic_saving_fraction": band(
                [p["index_traffic_saving_fraction"] for p in unpack_pairs]),
            "b7_total_traffic_saving_fraction": band(
                [p["total_traffic_saving_fraction"] for p in unpack_pairs]),
            # +/-3% is inside the run-to-run spread this contended box shows on a repeated
            # sweep, so a ratio inside that band is not a win in either direction.
            "b7_verdict": _b7_verdict([p["b7_over_u8_median"] for p in unpack_pairs]),
        },
    }


def pick_fixtures(layer: int | None) -> dict[str, grf.Fixture]:
    """gate_proj (skewed), down_proj (near-uniform), o_proj (the 64 KB-x attention geometry)."""
    index = grf.layer_index()
    candidates = [layer] if layer is not None else [
        l for l, e in index.items() if e["complete_expert_count"] >= 1 and e["attention"]]
    for cand in candidates:
        entry = index.get(cand)
        if not entry or not entry["complete_experts"] or not entry["attention"]:
            continue
        expert = entry["complete_experts"][0]
        shards = entry["experts"][str(expert)]
        out = {
            name: grf._fixture(grf.ARTIFACT_DIR / shards[proj],
                               f"model.layers.{cand}.mlp.experts.{expert}.{proj}_proj.weight")
            for name, proj in (("gate_proj_skewed", "gate"), ("down_proj_uniform", "down"))
        }
        attn = next((n for n in entry["attention"] if n.endswith("o_proj.weight")), None)
        if attn is None:
            continue
        out["attention_o_proj"] = grf._fixture(grf.ARTIFACT_DIR / entry["attention"][attn], attn)
        return out
    raise SystemExit("no safe layer carries a complete expert plus an o_proj")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tpgs", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--only", nargs="*", default=None, help="fixture names to run")
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "GLM52_TRACK_A_BENCHMARK.json")
    args = ap.parse_args()

    dec = TrackADecoder()
    prod = gravity_metal.decoder()
    fixtures = pick_fixtures(args.layer)
    if args.only:
        fixtures = {k: v for k, v in fixtures.items() if k in args.only}

    t0 = time.perf_counter()
    geometries = {}
    for name, fixture in fixtures.items():
        geometries[name] = run_geometry(
            fixture, dec, prod, reps=args.reps, warmup=args.warmup,
            tpgs=tuple(args.tpgs), block_counts=tuple(args.blocks))

    report = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "A: 2D split-chunk decode-FMA",
        "kernel_version": KERNEL_VERSION,
        "module": str(Path(__file__).resolve()),
        "production_kernel_untouched": "tools/condense/gravity_metal.py is imported, "
                                       "never modified",
        "machine": {
            "platform": platform.platform(),
            "device": str(dec.device.name()),
            "threadgroup_memory_limit": dec.threadgroup_memory_limit,
            "max_threads_per_threadgroup": dec.max_threads_per_threadgroup,
            "bandwidth_roof_gb_s": lab.BANDWIDTH_ROOF_GB_S,
            "compute_roof_gflop_s": lab.COMPUTE_ROOF_GFLOP_S,
        },
        "activation_source": grf.SYNTHETIC,
        "activation_status": grf.teacher_activation_status(),
        "parity_authority": "gravity_forge.pq_execute on the same compact artifact",
        "parity_gate": PARITY_GATE,
        "parity_note": "2e-3 is the module's own tolerance; the fp16 codebook floor is "
                       "~2.1e-4 on synthetic packs and ~1e-6 on real ones (real codebooks "
                       "are already fp16 on disk). A 2D split adds fp32 reassociation, "
                       "so every row also reports its delta against the production kernel.",
        "sweep": {"tpgs": args.tpgs, "block_counts": args.blocks,
                  "vec4": [False, True], "bits7": [False, True],
                  "stage_x_policy": [False, True],
                  "reps": args.reps, "warmup": args.warmup},
        "wall_seconds": time.perf_counter() - t0,
        "headline": {
            name: {
                "best_variant": g["best"] and g["best"]["variant"],
                "best_median_ms": g["best"] and g["best"]["median_ms"],
                "production_median_ms":
                    g["production_baseline"]["result"]["timings"]["end_to_end"]["median_ms"],
                "dense_fp16_median_ms":
                    g["dense_baseline"]["timings"]["end_to_end"]["median_ms"],
                "vs_production_median": g["best"] and g["best"]["vs_production_median"],
                "vs_dense_fp16_median": g["best"] and g["best"]["vs_dense_fp16_median"],
                "variants_timed": g["variants_timed"],
                "variants_failed_parity": g["variants_failed_parity"],
                "verdict": g["best"] and g["best"]["verdict"],
            } for name, g in geometries.items()
        },
        "geometries": geometries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"written": str(args.out), "wall_seconds": report["wall_seconds"],
                      "best": {k: v.get("best") for k, v in geometries.items()}},
                     indent=2, default=str))
    return 0


def selftest() -> int:
    """CPU-only checks of the planner, the byte model and the 7-bit unpack."""
    rng = np.random.default_rng(0)
    values = rng.integers(0, 128, size=1000, dtype=np.uint64)
    raw = glm52_pack.pack_indices(values, 7) + b"\x00\x00"
    assert np.array_equal(unpack7_reference(raw, values.size), values.astype(np.uint8))

    shape = plan(rows=2048, nchunk=768, D=8, k=128, tpg=256, blocks=32)
    assert shape["threads_in_flight"] == 2048 * 32
    assert shape["stage_x"] and shape["chunks_per_block"] == 24
    attn = plan(rows=6144, nchunk=2048, D=8, k=128, tpg=256, blocks=1)
    assert not attn["stage_x"], "whole-tensor x cannot fit 32 KB; that is the defect"
    attn8 = plan(rows=6144, nchunk=2048, D=8, k=128, tpg=256, blocks=8)
    assert attn8["stage_x"], "a chunk block's slice is what makes attention stageable"
    print(json.dumps({"selftest": "PASS", "kernel_version": KERNEL_VERSION}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    raise SystemExit(main())
