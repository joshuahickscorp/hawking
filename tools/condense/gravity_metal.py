#!/usr/bin/env python3.12
"""Hand-written Metal kernel for Gravity: decode inside the accumulation, never in memory.

Roadmap Phase 2, the real one.  The torch path in :mod:`gravity_decode` expresses decode
as ``codebook[indices]``, which for the ``subspaces=1`` geometry the whole production
ladder uses produces a ``[rows, nchunk, sub]`` tensor -- that is the *entire dense weight*,
materialized in fp32, larger than the BF16 it replaced.  It reads 1.4 MB of compressed
weights and then writes and re-reads 50 MB of decoded ones, which is why it lands ~100x
off the bandwidth bound.  Compression bought nothing there.

This kernel never writes a decoded weight anywhere.  Each threadgroup owns one output row
and walks that row's codeword indices; for every chunk it reads a one-byte index, points
at the codeword already sitting in threadgroup memory, and accumulates the dot product in
registers.  The decoded value exists only in a register, for the duration of one multiply.
The bytes that cross the memory bus are the indices and nothing else, so the traffic
really is the compressed size.

The codebook is the reason this works on Apple silicon: a whole tensor's codebook is
k*D halves -- 8 KB at k=256/D=16 -- so it is staged once into the 32 KB of threadgroup
memory and every one of the row's lookups hits on-chip storage instead of device memory.
Codebook staging is what turns a gather-bound kernel into a bandwidth-bound one.

Scope of this version, stated plainly: ``subspaces == 1`` (the entire production ladder),
one-byte indices (k <= 256, covering every ladder rung), and no in-kernel bit unpacking, so
a k=128 rung uploads 8-bit indices where the file stores 7-bit.  Reported traffic is what
the kernel actually reads, never the billed ideal.  Correctness is gated against
``gravity_forge.pq_execute`` exactly like every other backend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402

KERNEL_VERSION = 2
THREADS = 256

# v1 gave each threadgroup one output row: every thread did a single chunk and then paid a
# 128-step tree reduction, so the reduction was the kernel.  v2 gives every THREAD a whole
# row -- the accumulator lives in a register for the entire contraction, and there is no
# reduction and no barrier after staging.  Two further changes make that shape pay:
#
#   * indices are uploaded transposed to [nchunk][rows], so adjacent threads (adjacent
#     rows) read adjacent bytes and the index stream coalesces.  In [rows][nchunk] order
#     neighbouring threads were nchunk bytes apart, which is the worst case.
#   * x is staged in threadgroup memory alongside the codebook when it fits, so the inner
#     loop touches device memory only for the one index byte per chunk.
METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

struct Dims { uint rows; uint nchunk; uint D; uint k; uint stage_x; uint pad0; uint pad1; uint pad2; };

// One THREAD per output row.  Accumulator stays in a register; no reduction, no barriers
// past the cooperative staging.  Indices arrive transposed so this reads them coalesced.
kernel void gravity_pq_matvec_rows(
    device   const uchar*  indices   [[buffer(0)]],   // [nchunk][rows], transposed
    device   const half*   codebook  [[buffer(1)]],   // [k * D]
    device   const float*  x         [[buffer(2)]],   // [nchunk * D]
    device         float*  y         [[buffer(3)]],   // [rows]
    constant       Dims&   dims      [[buffer(4)]],
    threadgroup    float*  scratch   [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsz [[threads_per_threadgroup]])
{
    const uint D = dims.D, k = dims.k, nchunk = dims.nchunk, rows = dims.rows;

    threadgroup float* book = scratch;
    threadgroup float* xs   = scratch + k * D;

    for (uint i = tid; i < k * D; i += tsz) book[i] = float(codebook[i]);
    if (dims.stage_x) {
        for (uint i = tid; i < nchunk * D; i += tsz) xs[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = gid;
    if (row >= rows) return;

    float acc = 0.0f;
    if (dims.stage_x) {
        for (uint c = 0; c < nchunk; ++c) {
            threadgroup const float* w  = book + (uint)indices[c * rows + row] * D;
            threadgroup const float* xv = xs + c * D;
            for (uint j = 0; j < D; ++j) acc = fma(w[j], xv[j], acc);
        }
    } else {
        for (uint c = 0; c < nchunk; ++c) {
            threadgroup const float* w = book + (uint)indices[c * rows + row] * D;
            device const float* xv = x + c * D;
            for (uint j = 0; j < D; ++j) acc = fma(w[j], xv[j], acc);
        }
    }
    y[row] = acc;
}

struct DimsV1 { uint rows; uint nchunk; uint D; uint k; };

// One threadgroup per output row.  The codebook is staged in threadgroup memory once and
// then every codeword lookup is on-chip; decoded weights live only in registers.
kernel void gravity_pq_matvec(
    device   const uchar*  indices   [[buffer(0)]],   // [rows * nchunk], one index per chunk
    device   const half*   codebook  [[buffer(1)]],   // [k * D]
    device   const float*  x         [[buffer(2)]],   // [nchunk * D]
    device         float*  y         [[buffer(3)]],   // [rows]
    constant       DimsV1& dims      [[buffer(4)]],
    threadgroup    float*  scratch   [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tsz  [[threads_per_threadgroup]])
{
    const uint D = dims.D;
    const uint k = dims.k;
    const uint nchunk = dims.nchunk;

    threadgroup float* book = scratch;              // k * D floats
    threadgroup float* partial = scratch + k * D;   // tsz floats, for the reduction

    for (uint i = tid; i < k * D; i += tsz) {
        book[i] = float(codebook[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tgid;
    float acc = 0.0f;
    if (row < dims.rows) {
        device const uchar* row_idx = indices + (ulong)row * nchunk;
        for (uint c = tid; c < nchunk; c += tsz) {
            threadgroup const float* w = book + (uint)row_idx[c] * D;
            device const float* xv = x + c * D;
            float s = 0.0f;
            for (uint j = 0; j < D; ++j) {
                s = fma(w[j], xv[j], s);            // decoded weight stays in a register
            }
            acc += s;
        }
    }

    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && row < dims.rows) y[row] = partial[0];
}
"""


class MetalUnavailable(RuntimeError):
    """The Metal framework or a suitable device is not usable from this process."""


class GravityMetalDecoder:
    """Compiles the kernel once and reuses the pipeline across calls."""

    def __init__(self) -> None:
        try:
            import Metal
        except ImportError as exc:  # noqa: BLE001
            raise MetalUnavailable("pyobjc-framework-Metal is not installed") from exc
        self._Metal = Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise MetalUnavailable("no default Metal device")
        library, error = self.device.newLibraryWithSource_options_error_(
            METAL_SOURCE, None, None)
        if library is None:
            raise MetalUnavailable(f"kernel failed to compile: {error}")
        function = library.newFunctionWithName_("gravity_pq_matvec_rows")
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None)
        if pipeline is None:
            raise MetalUnavailable(f"pipeline failed: {error}")
        self.pipeline = pipeline
        self.queue = self.device.newCommandQueue()
        self._cache: dict = {}
        self.threadgroup_memory_limit = int(self.device.maxThreadgroupMemoryLength())

    def _buffer(self, array: np.ndarray):
        data = np.ascontiguousarray(array)
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, self._Metal.MTLResourceStorageModeShared)

    def _cache_tensor(self, codes: dict, key: str):
        """Upload the immutable parts once and keep them.  Per-call allocation was pure overhead."""
        entry = self._cache.get(key)
        if entry is not None:
            return entry
        D = int(codes["D"])
        rows, nchunk = int(codes["rows"]), int(codes["nchunk"])
        book = np.ascontiguousarray(codes["codebooks"][0], dtype=np.float16)
        k = int(book.shape[0])
        # transpose to [nchunk][rows] so adjacent threads read adjacent index bytes
        indices = np.ascontiguousarray(
            np.ascontiguousarray(codes["indices"], dtype=np.uint8).reshape(rows, nchunk).T)
        stage_x = (k * D + nchunk * D) * 4 <= self.threadgroup_memory_limit
        scratch = (k * D + (nchunk * D if stage_x else 0)) * 4
        entry = {
            "idx": self._buffer(indices), "book": self._buffer(book),
            "dims": self._buffer(np.array([rows, nchunk, D, k, int(stage_x), 0, 0, 0],
                                          dtype=np.uint32)),
            "y": self.device.newBufferWithLength_options_(
                rows * 4, self._Metal.MTLResourceStorageModeShared),
            "x": self.device.newBufferWithLength_options_(
                nchunk * D * 4, self._Metal.MTLResourceStorageModeShared),
            "rows": rows, "scratch": scratch, "stage_x": stage_x,
        }
        self._cache[key] = entry
        return entry

    def matvec(self, codes: dict, x: np.ndarray, *, key: str | None = None) -> np.ndarray:
        """y = W_gravity @ x, decoding in registers.  One thread per row, no reduction."""
        if int(codes["S"]) != 1:
            raise MetalUnavailable("this kernel handles subspaces == 1 only")
        if codes["rotate"]:
            raise MetalUnavailable("rotated geometry is not wired into this kernel yet")
        if int(codes["codebooks"][0].shape[0]) > 256:
            raise MetalUnavailable("k > 256 exceeds the one-byte index this kernel uses")

        entry = self._cache_tensor(codes, key or f"{id(codes):x}")
        if entry["scratch"] > self.threadgroup_memory_limit:
            raise MetalUnavailable("codebook exceeds threadgroup memory")

        rows = entry["rows"]
        xv = np.ascontiguousarray(np.asarray(x, dtype=np.float32).ravel())
        # write straight into the persistent shared buffer instead of allocating a new one
        entry["x"].contents().as_buffer(xv.nbytes)[:] = xv.tobytes()

        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        for slot, buffer in enumerate((entry["idx"], entry["book"], entry["x"],
                                       entry["y"], entry["dims"])):
            encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
        encoder.setThreadgroupMemoryLength_atIndex_(entry["scratch"], 0)
        groups = (rows + THREADS - 1) // THREADS
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self._Metal.MTLSizeMake(groups, 1, 1), self._Metal.MTLSizeMake(THREADS, 1, 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

        if command.error() is not None:
            raise MetalUnavailable(f"dispatch failed: {command.error()}")
        return np.frombuffer(entry["y"].contents().as_buffer(rows * 4), dtype=np.float32).copy()

    def bytes_read_per_matvec(self, codes: dict) -> int:
        """What the kernel actually streams: indices plus one codebook.  No decoded weights."""
        rows, nchunk = int(codes["rows"]), int(codes["nchunk"])
        book = codes["codebooks"][0]
        return rows * nchunk + book.size * 2


_DECODER: GravityMetalDecoder | None = None


def decoder() -> GravityMetalDecoder:
    global _DECODER
    if _DECODER is None:
        _DECODER = GravityMetalDecoder()
    return _DECODER


def selftest() -> int:
    """Parity against the CPU authority, plus the traffic the kernel really moves."""
    rng = np.random.default_rng(0)
    try:
        gpu = decoder()
    except MetalUnavailable as exc:
        print(json.dumps({"selftest": "SKIPPED", "reason": str(exc)}, indent=2))
        return 0

    checks = []
    for shape, dim, k in (((256, 128), 8, 16), ((1024, 512), 16, 256), ((512, 256), 8, 128)):
        weights = rng.standard_normal(shape).astype(np.float32)
        artifact = forge.pack_product_quant(weights, dim=dim, subspaces=1, k=k, seed=0, iters=4)
        codes = artifact.config["pq_codes"]
        probe = rng.standard_normal(shape[1]).astype(np.float32)

        reference = forge.pq_execute(artifact, probe)
        got = gpu.matvec(codes, probe)
        gap = float(np.abs(reference - got).max() / (np.abs(reference).max() + 1e-12))
        assert np.isfinite(got).all(), (shape, dim, k)
        assert gap < 2e-3, f"{shape} dim={dim} k={k} gap={gap}"
        checks.append({"shape": list(shape), "dim": dim, "k": k, "relative_max_gap": gap})

    print(json.dumps({"selftest": "PASS", "kernel_version": KERNEL_VERSION,
                      "device": str(gpu.device.name()),
                      "threadgroup_memory_limit": gpu.threadgroup_memory_limit,
                      "parity": checks}, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    sys.stderr.write("import this module; only `selftest` runs standalone\n")
    raise SystemExit(2)
