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

// Wedge 1 — Metal MLA decode kernel.
//
// Replaces the CPU `mla_decode_step` path for DeepSeek-V2-family
// models. Operates on the COMPRESSED KV cache (c_kv + k_pe) rather
// than the expanded form. The key optimisations vs a naive port:
//
//   1. q_nope_proj = w_uk^T @ q_nope precomputed once per head in TG
//      memory, then used for all seq positions (avoids O(qk_nope ×
//      kv_lora_rank) per token in the inner loop).
//
//   2. c_kv_weighted = Σ_t scores[t] × c_kv[t] accumulated in TG
//      memory, then w_uv @ c_kv_weighted completes the output (avoids
//      O(v_head_dim × seq_len × kv_lora_rank) → just
//      O(seq_len × kv_lora_rank + v_head_dim × kv_lora_rank)).
//
// Buffer layout:
//   0  q          (n_heads × q_head_dim) f32        read-only
//   1  c_kv       (seq_len × kv_lora_rank) f32      read-only
//   2  k_pe       (seq_len × qk_rope_head_dim) f32  read-only
//   3  kv_b_proj  (n_heads × (qk_nope+v_head_dim) × kv_lora_rank) f32  read-only
//   4  out        (n_heads × v_head_dim) f32         write
//   5..11  scalars: n_heads, qk_nope_head_dim, qk_rope_head_dim,
//                   v_head_dim, kv_lora_rank, seq_len, scale
//
// Threadgroup slots (host sets sizes at dispatch):
//   0  q_nope_proj   kv_lora_rank floats
//   1  scores        seq_len floats
//   2  c_kv_weighted kv_lora_rank floats
//
// Dispatch: grid=(n_heads×TG_SIZE,1,1), tg=(TG_SIZE,1,1).
// gid = workgroup/head index.
kernel void mla_decode_kernel(
    device const float* q          [[buffer(0)]],
    device const float* c_kv       [[buffer(1)]],
    device const float* k_pe       [[buffer(2)]],
    device const float* kv_b_proj  [[buffer(3)]],
    device       float* out        [[buffer(4)]],
    constant     uint&  n_heads             [[buffer(5)]],
    constant     uint&  qk_nope_head_dim    [[buffer(6)]],
    constant     uint&  qk_rope_head_dim    [[buffer(7)]],
    constant     uint&  v_head_dim          [[buffer(8)]],
    constant     uint&  kv_lora_rank        [[buffer(9)]],
    constant     uint&  seq_len             [[buffer(10)]],
    constant     float& scale               [[buffer(11)]],
    threadgroup  float* q_nope_proj         [[threadgroup(0)]],
    threadgroup  float* scores              [[threadgroup(1)]],
    threadgroup  float* c_kv_wt             [[threadgroup(2)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]],
    uint                tg_size [[threads_per_threadgroup]])
{
    if (gid >= n_heads) return;

    const uint head      = gid;
    const uint q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    // Pointers into the query for this head.
    device const float* q_nope = q + head * q_head_dim;
    device const float* q_rope = q_nope + qk_nope_head_dim;

    // kv_b_proj layout per head: [w_uk rows, then w_uv rows]
    //   w_uk: (qk_nope_head_dim, kv_lora_rank) row-major
    //   w_uv: (v_head_dim,       kv_lora_rank) row-major
    const uint kv_b_per_head = (qk_nope_head_dim + v_head_dim) * kv_lora_rank;
    device const float* w_uk = kv_b_proj + (uint64_t)head * kv_b_per_head;
    device const float* w_uv = w_uk + (uint64_t)qk_nope_head_dim * kv_lora_rank;

    // ------------------------------------------------------------------
    // Phase 0: q_nope_proj[r] = Σ_i w_uk[i,r] × q_nope[i]
    //   r distributed across threads.
    // ------------------------------------------------------------------
    for (uint r = tid; r < kv_lora_rank; r += tg_size) {
        float acc = 0.0f;
        for (uint i = 0; i < qk_nope_head_dim; i++) {
            acc += w_uk[i * kv_lora_rank + r] * q_nope[i];
        }
        q_nope_proj[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Phase 1: scores[t] = (q_nope_proj · c_kv[t] + q_rope · k_pe[t]) × scale
    //   t distributed across threads.
    // ------------------------------------------------------------------
    for (uint t = tid; t < seq_len; t += tg_size) {
        device const float* c_kv_t = c_kv + (uint64_t)t * kv_lora_rank;
        device const float* k_pe_t = k_pe + (uint64_t)t * qk_rope_head_dim;

        float s = 0.0f;
        for (uint r = 0; r < kv_lora_rank; r++)      s += q_nope_proj[r] * c_kv_t[r];
        for (uint r = 0; r < qk_rope_head_dim; r++)  s += q_rope[r] * k_pe_t[r];
        scores[t] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Phase 2: softmax (thread 0 serial — fine for seq_len ≤ 4096)
    // ------------------------------------------------------------------
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint t = 0; t < seq_len; t++) if (scores[t] > mx) mx = scores[t];
        float sum = 0.0f;
        for (uint t = 0; t < seq_len; t++) { scores[t] = exp(scores[t] - mx); sum += scores[t]; }
        float inv = 1.0f / sum;
        for (uint t = 0; t < seq_len; t++) scores[t] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Phase 3: c_kv_wt[r] = Σ_t scores[t] × c_kv[t, r]
    //   r distributed across threads.
    // ------------------------------------------------------------------
    for (uint r = tid; r < kv_lora_rank; r += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < seq_len; t++) acc += scores[t] * c_kv[(uint64_t)t * kv_lora_rank + r];
        c_kv_wt[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Phase 4: out[head, vi] = w_uv[vi, :] · c_kv_wt
    //   vi distributed across threads.
    // ------------------------------------------------------------------
    for (uint vi = tid; vi < v_head_dim; vi += tg_size) {
        device const float* w_uv_row = w_uv + (uint64_t)vi * kv_lora_rank;
        float acc = 0.0f;
        for (uint r = 0; r < kv_lora_rank; r++) acc += w_uv_row[r] * c_kv_wt[r];
        out[head * v_head_dim + vi] = acc;
    }
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
