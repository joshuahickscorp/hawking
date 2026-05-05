// common.metal — shared primitives across attention, MoE, and the
// per-token residual path.
//
// Kernels:
//   rmsnorm               — RMS normalization. fp32 reduction, fp16 mul.
//                           [Phase 0]
//   rmsnorm_f32           — RMS normalization, full fp32 I/O. Used by the
//                           Wedge B TCB path (f32 residual stream).
//                           [v1.0.0-B]
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

// v1.0.0-B Wedge B — full fp32 rmsnorm for the f32 residual stream TCB path.
// Same math as rmsnorm above; operates on f32 x, f32 weight, f32 out.
// Threadgroup reduction accumulates variance in f32 (no precision loss).
kernel void rmsnorm_f32(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant     uint&  hidden  [[buffer(3)]],
    constant     float& eps     [[buffer(4)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / (float)hidden + eps);
    float inv = 1.0f / rms;
    for (uint i = tid; i < hidden; i += tg_size) {
        out[i] = x[i] * inv * weight[i];
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

// Phase 7 Wedge 7b — fp16 rmsnorm.
// Reads f16 input, computes variance in f32 (sensitive to overflow at
// large activations), writes f16 output. Weight is f32.
//
// Threadgroup size 256 (parallel reduction; must be power of two ≤ 1024).
kernel void rmsnorm_f16(
    device const half*  x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    constant     float& eps     [[buffer(2)]],
    constant     uint&  hidden  [[buffer(3)]],
    device       half*  out     [[buffer(4)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = (float)x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shmem[0] / (float)hidden;
    float scale = rsqrt(mean + eps);

    for (uint i = tid; i < hidden; i += tg_size) {
        float v = (float)x[i];
        out[i] = (half)(v * scale * weight[i]);
    }
}

// Phase 7 Wedge 7d-prep — fp16 silu_mul.
// Computes out[i] = silu(gate[i]) * up[i] reading f16, writing f16.
// Internal sigmoid + multiply in f32 (silu's exp is sensitive).
kernel void silu_mul_f16(
    device const half*  gate   [[buffer(0)]],
    device const half*  up     [[buffer(1)]],
    device       half*  out    [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = (float)gate[gid];
    float u = (float)up[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = (half)(silu_g * u);
}

// v0.5.9-C — fp16 residual add: a[i] += b[i], both f16.
kernel void add_inplace_f16(
    device       half*  a   [[buffer(0)]],
    device const half*  b   [[buffer(1)]],
    constant     uint&  n   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    a[gid] = (half)((float)a[gid] + (float)b[gid]);
}

// v0.5.9-E — standalone f16 softmax.
// Single-threadgroup kernel: reads n f16 logits, writes n f16 probabilities.
// Max and exp-sum computed in f32. Grid: (TG_SIZE, 1, 1), TG: (TG_SIZE, 1, 1).
kernel void softmax_f16(
    device const half*  x     [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  n     [[buffer(2)]],
    threadgroup  float* shmem [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    // Phase 1: parallel max reduction.
    float local_max = -INFINITY;
    for (uint i = tid; i < n; i += tg_size) {
        float v = (float)x[i];
        if (v > local_max) local_max = v;
    }
    shmem[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shmem[0];

    // Phase 2: parallel sum of exp(x - max).
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local_sum += exp((float)x[i] - max_val);
    }
    shmem[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shmem[0];

    // Phase 3: write normalized probabilities.
    for (uint i = tid; i < n; i += tg_size) {
        out[i] = (half)(exp((float)x[i] - max_val) * inv_sum);
    }
}

// v0.5.9-F — f16 layer normalization (mean-centering + variance + bias).
// Like rmsnorm_f16 but subtracts mean first and adds a bias term.
// Single-threadgroup kernel. Grid: (TG_SIZE, 1, 1), TG: (TG_SIZE, 1, 1).
kernel void layer_norm_f16(
    device const half*  x       [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const half*  bias    [[buffer(2)]],
    constant     float& eps     [[buffer(3)]],
    constant     uint&  n       [[buffer(4)]],
    device       half*  out     [[buffer(5)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    // Phase 1: mean.
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local_sum += (float)x[i];
    }
    shmem[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shmem[0] / float(n);

    // Phase 2: variance.
    float local_var = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float v = (float)x[i] - mean;
        local_var += v * v;
    }
    shmem[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shmem[0] / float(n) + eps);

    // Phase 3: normalize, scale, bias.
    for (uint i = tid; i < n; i += tg_size) {
        float v = ((float)x[i] - mean) * inv_std;
        out[i] = (half)(v * (float)weight[i] + (float)bias[i]);
    }
}
