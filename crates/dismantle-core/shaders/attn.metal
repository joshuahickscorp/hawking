// attn.metal — attention kernels.
//
// Kernels:
//   attn_mha_qkv          — standard multi-head attention (Qwen3-MoE).
//                           Flash-style tiling, fp16 mma via simdgroup
//                           matrix.
//                           [Phase 0 reference; Phase 3 tuned]
//   attn_mla_compress     — DeepSeek MLA: compresses K/V into the
//                           latent on prefill.
//                           [Phase 3]
//   attn_mla_decompress   — MLA: decompresses on read inside the
//                           attention kernel; KV cache stays compressed.
//                           [Phase 3]
//   attn_kv_append        — fused KV-cache append (no separate copy
//                           pass).
//                           [Phase 3]

#include <metal_stdlib>
using namespace metal;

// Stub kernels — Phase 0 runs attention on the host CPU. Replaced by
// flash-attention-style tiled kernels in Phase 3.
kernel void attn_mha_qkv_stub(
    device const half* q   [[buffer(0)]],
    device const half* k   [[buffer(1)]],
    device const half* v   [[buffer(2)]],
    device       half* out [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    (void)id;
}

kernel void attn_mla_compress_stub(
    device const half* x   [[buffer(0)]],
    device       half* c   [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    (void)id;
}

kernel void attn_mla_decompress_stub(
    device const half* c   [[buffer(0)]],
    device       half* kv  [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    (void)id;
}

kernel void attn_kv_append_stub(
    device const half* k_new [[buffer(0)]],
    device       half* k_buf [[buffer(1)]],
    constant     uint& pos   [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    (void)id;
}

// G1.3 — fp32 GEMV for attention's o_proj.
// One workgroup per output row; threadgroup reduction; same shape as
// gemv_f16 but with f32 weights (lazy-dequant scratch from the host).
kernel void gemv_f32_attn(
    device const float* w     [[buffer(0)]],   // (rows, cols) row-major fp32
    device const float* x     [[buffer(1)]],   // (cols,)
    device       float* y     [[buffer(2)]],   // (rows,)
    constant     uint&  rows  [[buffer(3)]],
    constant     uint&  cols  [[buffer(4)]],
    threadgroup  float* shmem [[threadgroup(0)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                tg_size   [[threads_per_threadgroup]])
{
    if (gid >= rows) return;
    device const float* row = w + (uint64_t)gid * (uint64_t)cols;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += row[c] * x[c];
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) y[gid] = shmem[0];
}
