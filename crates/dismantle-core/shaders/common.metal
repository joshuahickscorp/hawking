// common.metal — shared primitives across attention, MoE, and the
// per-token residual path.
//
// Kernels:
//   rmsnorm               — RMS normalization. fp32 reduction, fp16 mul.
//                           [Phase 0]
//   silu_mul              — SwiGLU activation (silu(a) * b) for the
//                           gate-up projection.
//                           [Phase 0]
//   rope_inplace          — Rotary position embedding, applied in-place
//                           to Q and K projections.
//                           [Phase 0]
//   embed_lookup          — input-token embedding lookup with optional
//                           tied LM head.
//                           [Phase 0]
//   add_inplace           — element-wise residual add: a[i] += b[i].
//                           [Phase 4 Wedge 4a]

#include <metal_stdlib>
using namespace metal;

// One workgroup normalizes one (hidden,) row.
kernel void rmsnorm(
    device const half*  x        [[buffer(0)]],
    device const half*  weight   [[buffer(1)]],
    device       half*  out      [[buffer(2)]],
    constant     uint&  hidden   [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    threadgroup  float* shmem    [[threadgroup(0)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                tg_size  [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = (float)x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / float(hidden) + eps);
    float inv = 1.0f / rms;
    for (uint i = tid; i < hidden; i += tg_size) {
        out[i] = half((float)x[i] * inv * (float)weight[i]);
    }
}

kernel void silu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device       half* out  [[buffer(2)]],
    constant     uint& n    [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    if (id >= n) return;
    float g = (float)gate[id];
    float s = g / (1.0f + exp(-g));
    out[id] = half(s * (float)up[id]);
}

kernel void rope_inplace(
    device       half* x        [[buffer(0)]],
    constant     uint& head_dim [[buffer(1)]],
    constant     uint& pos      [[buffer(2)]],
    constant     float& base    [[buffer(3)]],
    uint id                      [[thread_position_in_grid]])
{
    uint half_dim = head_dim / 2;
    if (id >= half_dim) return;
    float theta = (float)pos / pow(base, 2.0f * float(id) / float(head_dim));
    float c = cos(theta);
    float s = sin(theta);
    float x0 = (float)x[2 * id];
    float x1 = (float)x[2 * id + 1];
    x[2 * id]     = half(x0 * c - x1 * s);
    x[2 * id + 1] = half(x0 * s + x1 * c);
}

kernel void embed_lookup(
    device const half* embed  [[buffer(0)]],
    device       half* out    [[buffer(1)]],
    constant     uint& hidden [[buffer(2)]],
    constant     uint& token  [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    out[id] = embed[token * hidden + id];
}

// G1.2 — fp16-weight × fp32-vec → fp32 GEMV (LM-head shape).
//
// One workgroup per output row, tg_size threads per group, threadgroup
// reduction across the inner-product accumulator (same pattern as
// rmsnorm above). Output is fp32 because rows like the LM head reach
// magnitudes ~√cols, where fp16 precision (~1 part in 1024) is too
// coarse for the parity tolerance.
kernel void gemv_f16(
    device const half*  w      [[buffer(0)]],   // (rows, cols) row-major fp16
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint                tid        [[thread_position_in_threadgroup]],
    uint                gid        [[threadgroup_position_in_grid]],
    uint                tg_size    [[threads_per_threadgroup]])
{
    if (gid >= rows) return;
    device const half* row = w + (uint64_t)gid * (uint64_t)cols;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += (float)row[c] * x[c];
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) y[gid] = shmem[0];
}

// Phase 4 Wedge 4a — element-wise residual add.
// Computes a[i] += b[i] for i in [0, n).
// One thread per element; grid (n, 1, 1), threadgroup (TG_SIZE, 1, 1).
kernel void add_inplace(
    device       float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    constant     uint&  n    [[buffer(2)]],
    uint                gid  [[thread_position_in_grid]])
{
    if (gid >= n) return;
    a[gid] += b[gid];
}
