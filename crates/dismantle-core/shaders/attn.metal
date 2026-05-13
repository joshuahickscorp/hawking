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
//   rmsnorm_gemv_f32_attn_pinned — fused rmsnorm + gemv_f32_attn.
//                           Reads x once; computes variance in-register,
//                           then runs GEMV with normalized x.
//                           One threadgroup per output row, TG_SIZE=256.
//                           [v0.5.8]
//   rmsnorm_gemv_q4k_pair — fused rmsnorm + Q4_K_M GEMV pair (gate+up).
//                           Reads x once; grid = 2×rows threadgroups.
//                           gid < rows → gate; gid >= rows → up.
//                           [v0.5.8]
//   rmsnorm_gemv_f16_attn_pinned — f16-input bridge variant of
//                           rmsnorm_gemv_f32_attn_pinned. Reads half*
//                           from x; variance accumulation stays f32.
//                           [v0.8.1 Phase 7 session 1]
//   rmsnorm_gemv_q4k_pair_f16 — f16-input bridge variant of
//                           rmsnorm_gemv_q4k_pair. Reads half* from x.
//                           [v0.8.2 Phase 7 session 1]

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

// Phase A Wedge A2 — batched MLA decode kernel (M=1..8 tokens, M=4 default).
//
// Grid: (n_heads * TG_SIZE, M, 1) with (TG_SIZE, 1, 1) threads per group.
// threadgroup_position_in_grid.x = head index (0..n_heads-1)
// threadgroup_position_in_grid.y = token index m (0..M-1)
//
// q_batch layout: [M][n_heads][head_dim_q]  (token-major)
// c_kv layout:   [total_seq][kv_lora_rank]  (all tokens: prefix + batch)
// k_pe layout:   [total_seq][qk_rope_head_dim]
// out_batch:     [M][n_heads][v_head_dim]   (token-major)
//
// Causal mask: token m attends to entries 0..max(base_seq_len + m, 1) - 1.
// The M new tokens' KVs are pre-appended at slots base_seq_len..base_seq_len+M-1.
kernel void mla_decode_kernel_batched(
    device const float* q_batch     [[buffer(0)]],
    device const float* c_kv        [[buffer(1)]],
    device const float* k_pe        [[buffer(2)]],
    device const float* kv_b_proj   [[buffer(3)]],
    device       float* out_batch   [[buffer(4)]],
    constant     uint&  n_heads             [[buffer(5)]],
    constant     uint&  qk_nope_head_dim    [[buffer(6)]],
    constant     uint&  qk_rope_head_dim    [[buffer(7)]],
    constant     uint&  v_head_dim          [[buffer(8)]],
    constant     uint&  kv_lora_rank        [[buffer(9)]],
    constant     uint&  base_seq_len        [[buffer(10)]],
    constant     float& scale               [[buffer(11)]],
    threadgroup  float* q_nope_proj         [[threadgroup(0)]],
    threadgroup  float* scores              [[threadgroup(1)]],
    threadgroup  float* c_kv_wt             [[threadgroup(2)]],
    uint2               gid     [[threadgroup_position_in_grid]],
    uint2               tid_v   [[thread_position_in_threadgroup]],
    uint2               tg_size_v [[threads_per_threadgroup]])
{
    const uint head = gid.x;
    const uint m    = gid.y;  // token index in batch

    if (head >= n_heads) return;

    // Causal seq_len for token m: prefix + m tokens from this batch.
    // max(base_seq_len + m, 1u) handles the first-token edge case.
    const uint seq_len_m = (base_seq_len + m > 0u) ? (base_seq_len + m) : 1u;

    const uint q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    // Q for token m, head h.
    device const float* q_base = q_batch + (uint64_t)m * n_heads * q_head_dim;
    device const float* q_nope = q_base + (uint64_t)head * q_head_dim;
    device const float* q_rope = q_nope + qk_nope_head_dim;

    // kv_b_proj layout per head: [w_uk rows (qk_nope_head_dim × kv_lora_rank),
    //                              w_uv rows (v_head_dim × kv_lora_rank)]
    const uint kv_b_per_head = (qk_nope_head_dim + v_head_dim) * kv_lora_rank;
    device const float* w_uk = kv_b_proj + (uint64_t)head * kv_b_per_head;
    device const float* w_uv = w_uk + (uint64_t)qk_nope_head_dim * kv_lora_rank;

    // Phase 0: q_nope_proj[r] = Σ_i w_uk[i,r] × q_nope[i]
    for (uint r = tid_v.x; r < kv_lora_rank; r += tg_size_v.x) {
        float acc = 0.0f;
        for (uint i = 0; i < qk_nope_head_dim; i++) {
            acc += w_uk[i * kv_lora_rank + r] * q_nope[i];
        }
        q_nope_proj[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: scores[t] = (q_nope_proj · c_kv[t] + q_rope · k_pe[t]) × scale
    for (uint t = tid_v.x; t < seq_len_m; t += tg_size_v.x) {
        device const float* c_kv_t = c_kv + (uint64_t)t * kv_lora_rank;
        device const float* k_pe_t = k_pe + (uint64_t)t * qk_rope_head_dim;

        float s = 0.0f;
        for (uint r = 0; r < kv_lora_rank; r++)      s += q_nope_proj[r] * c_kv_t[r];
        for (uint r = 0; r < qk_rope_head_dim; r++)  s += q_rope[r] * k_pe_t[r];
        scores[t] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax over attended entries (serial in thread 0).
    if (tid_v.x == 0) {
        float mx = -INFINITY;
        for (uint t = 0; t < seq_len_m; t++) if (scores[t] > mx) mx = scores[t];
        float sum = 0.0f;
        for (uint t = 0; t < seq_len_m; t++) { scores[t] = exp(scores[t] - mx); sum += scores[t]; }
        float inv = 1.0f / sum;
        for (uint t = 0; t < seq_len_m; t++) scores[t] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: c_kv_wt[r] = Σ_t scores[t] × c_kv[t, r]
    for (uint r = tid_v.x; r < kv_lora_rank; r += tg_size_v.x) {
        float acc = 0.0f;
        for (uint t = 0; t < seq_len_m; t++) acc += scores[t] * c_kv[(uint64_t)t * kv_lora_rank + r];
        c_kv_wt[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out_batch[m, head, vi] = w_uv[vi, :] · c_kv_wt
    device float* out_m = out_batch + ((uint64_t)m * n_heads + head) * v_head_dim;
    for (uint vi = tid_v.x; vi < v_head_dim; vi += tg_size_v.x) {
        device const float* w_uv_row = w_uv + (uint64_t)vi * kv_lora_rank;
        float acc = 0.0f;
        for (uint r = 0; r < kv_lora_rank; r++) acc += w_uv_row[r] * c_kv_wt[r];
        out_m[vi] = acc;
    }
}

kernel void mla_decode_kernel_batched_slots(
    device const float* q_batch     [[buffer(0)]],
    device const float* c_kv        [[buffer(1)]],
    device const float* k_pe        [[buffer(2)]],
    device const float* kv_b_proj   [[buffer(3)]],
    device       float* out_batch   [[buffer(4)]],
    device const uint*  slot_offsets        [[buffer(5)]],
    device const uint*  seq_lens            [[buffer(6)]],
    constant     uint&  n_heads             [[buffer(7)]],
    constant     uint&  qk_nope_head_dim    [[buffer(8)]],
    constant     uint&  qk_rope_head_dim    [[buffer(9)]],
    constant     uint&  v_head_dim          [[buffer(10)]],
    constant     uint&  kv_lora_rank        [[buffer(11)]],
    constant     float& scale               [[buffer(12)]],
    threadgroup  float* q_nope_proj         [[threadgroup(0)]],
    threadgroup  float* scores              [[threadgroup(1)]],
    threadgroup  float* c_kv_wt             [[threadgroup(2)]],
    uint2               gid     [[threadgroup_position_in_grid]],
    uint2               tid_v   [[thread_position_in_threadgroup]],
    uint2               tg_size_v [[threads_per_threadgroup]])
{
    const uint head = gid.x;
    const uint m    = gid.y;

    if (head >= n_heads) return;

    const uint seq_len_m = seq_lens[m];
    device float* out_m = out_batch + ((uint64_t)m * n_heads + head) * v_head_dim;
    if (seq_len_m == 0u) {
        for (uint vi = tid_v.x; vi < v_head_dim; vi += tg_size_v.x) out_m[vi] = 0.0f;
        return;
    }

    const uint slot_base = slot_offsets[m];
    const uint q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    device const float* q_base = q_batch + (uint64_t)m * n_heads * q_head_dim;
    device const float* q_nope = q_base + (uint64_t)head * q_head_dim;
    device const float* q_rope = q_nope + qk_nope_head_dim;

    const uint kv_b_per_head = (qk_nope_head_dim + v_head_dim) * kv_lora_rank;
    device const float* w_uk = kv_b_proj + (uint64_t)head * kv_b_per_head;
    device const float* w_uv = w_uk + (uint64_t)qk_nope_head_dim * kv_lora_rank;

    for (uint r = tid_v.x; r < kv_lora_rank; r += tg_size_v.x) {
        float acc = 0.0f;
        for (uint i = 0; i < qk_nope_head_dim; i++) {
            acc += w_uk[i * kv_lora_rank + r] * q_nope[i];
        }
        q_nope_proj[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid_v.x; t < seq_len_m; t += tg_size_v.x) {
        const uint kv_index = slot_base + t;
        device const float* c_kv_t = c_kv + (uint64_t)kv_index * kv_lora_rank;
        device const float* k_pe_t = k_pe + (uint64_t)kv_index * qk_rope_head_dim;

        float s = 0.0f;
        for (uint r = 0; r < kv_lora_rank; r++)      s += q_nope_proj[r] * c_kv_t[r];
        for (uint r = 0; r < qk_rope_head_dim; r++)  s += q_rope[r] * k_pe_t[r];
        scores[t] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_v.x == 0) {
        float mx = -INFINITY;
        for (uint t = 0; t < seq_len_m; t++) if (scores[t] > mx) mx = scores[t];
        float sum = 0.0f;
        for (uint t = 0; t < seq_len_m; t++) { scores[t] = exp(scores[t] - mx); sum += scores[t]; }
        float inv = 1.0f / sum;
        for (uint t = 0; t < seq_len_m; t++) scores[t] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = tid_v.x; r < kv_lora_rank; r += tg_size_v.x) {
        float acc = 0.0f;
        for (uint t = 0; t < seq_len_m; t++) {
            acc += scores[t] * c_kv[(uint64_t)(slot_base + t) * kv_lora_rank + r];
        }
        c_kv_wt[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint vi = tid_v.x; vi < v_head_dim; vi += tg_size_v.x) {
        device const float* w_uv_row = w_uv + (uint64_t)vi * kv_lora_rank;
        float acc = 0.0f;
        for (uint r = 0; r < kv_lora_rank; r++) acc += w_uv_row[r] * c_kv_wt[r];
        out_m[vi] = acc;
    }
}

// G1.3 — fp32 GEMV for attention's o_proj.
// One workgroup per output row; threadgroup reduction; same shape as
// gemv_f16 but with f32 weights (lazy-dequant scratch from the host).
kernel void gemv_f32_attn(
    device const float* w     [[buffer(0)]],   // (rows, cols) row-major fp32
    device const float* x     [[buffer(1)]],   // (cols,)
    device       float* y     [[buffer(2)]],   // (rows,)
    constant ArgbufRowsCols& args [[buffer(3)]],
    threadgroup  float* shmem [[threadgroup(0)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                tg_size   [[threads_per_threadgroup]])
{
    if (gid >= args.rows) return;
    device const float* row = w + (uint64_t)gid * (uint64_t)args.cols;

    float partial = 0.0f;
    for (uint c = tid; c < args.cols; c += tg_size) {
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

// v0.5.9-A — fp16 activation variant: gemv_f32_attn with f16 x and f16 y.
// Same threadgroup structure as gemv_f32_attn. Internal MAC in f32.
kernel void gemv_f32_attn_f16(
    device const float* w     [[buffer(0)]],   // (rows, cols) row-major fp32
    device const half*  x     [[buffer(1)]],   // (cols,) fp16
    device       half*  y     [[buffer(2)]],   // (rows,) fp16
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
        partial += row[c] * (float)x[c];
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) y[gid] = (half)shmem[0];
}

// v0.5.8-A — fused RMSNorm + gemv_f32_attn.
//
// Eliminates the intermediate normalized-x buffer and halves DRAM reads
// for x. Each threadgroup (one per output row) independently computes the
// RMS norm variance (x is small → stays in L2 across threadgroups), then
// runs the GEMV with the normalized activation.
//
// Binding scheme:
//   0  w       (rows × cols) f32   pinned weight matrix
//   1  x       (cols,)        f32   residual stream
//   2  weight  (cols,)        f32   rmsnorm learnable scale
//   3  eps     constant float
//   4  out     (rows,)        f32
//   5  rows    constant uint
//   6  cols    constant uint
//   threadgroup(0): shmem (TG_SIZE × f32) — reused for variance then dot-product
//
// Grid:  (rows, 1, 1) threadgroups; each of size TG_SIZE (256).
kernel void rmsnorm_gemv_f32_attn_pinned(
    device const float* w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant     float& eps     [[buffer(3)]],
    device       float* out     [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]],
    uint                tg_size [[threads_per_threadgroup]])
{
    if (gid >= rows) return;

    // Phase 1: parallel variance reduction over x.
    float partial_sq = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        float v = x[c];
        partial_sq += v * v;
    }
    shmem[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shmem[0] / float(cols) + eps);

    // Phase 2: GEMV with rmsnorm-scaled x.
    device const float* row = w + (uint64_t)gid * (uint64_t)cols;
    float partial_dot = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial_dot += row[c] * (x[c] * inv_rms * weight[c]);
    }
    shmem[tid] = partial_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[gid] = shmem[0];
}

// f16-weight variant of rmsnorm_gemv_f32_attn_pinned.
// Identical binding scheme and logic; only the weight buffer dtype changes.
// Halves weight DRAM bandwidth for q_a_proj and kv_a_proj_with_mqa vs f32.
//   0  w       (rows × cols) f16   pinned weight matrix
//   1  x       (cols,)        f32   residual stream
//   2  weight  (cols,)        f32   rmsnorm learnable scale
//   3  eps     constant float
//   4  out     (rows,)        f32
//   5  rows    constant uint
//   6  cols    constant uint
kernel void rmsnorm_gemv_f16w_attn_pinned(
    device const half*  w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant     float& eps     [[buffer(3)]],
    device       float* out     [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]],
    uint                tg_size [[threads_per_threadgroup]])
{
    if (gid >= rows) return;
    float partial_sq = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        float v = x[c]; partial_sq += v * v;
    }
    shmem[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shmem[0] / float(cols) + eps);
    device const half* row = w + (uint64_t)gid * (uint64_t)cols;
    float partial_dot = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial_dot += (float)row[c] * (x[c] * inv_rms * weight[c]);
    }
    shmem[tid] = partial_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[gid] = shmem[0];
}

// Append one KV entry to the persistent GPU KV cache.
// Used by the merged Phase-1+Wedge-N TCB to avoid a CPU round-trip for kv_append.
// Writes c_kv_normed (kv_lora_rank f32) and k_pe (qk_rope_head_dim f32) at seq_slot.
//   0  src_c_kv_normed  (kv_lora_rank,)         f32  — c_kv_normed_buf
//   1  src_kv_a_out     (kv_a_dim,)              f32  — kv_a_out_buf (k_pe at [kv_lora_rank..])
//   2  dst_c_kv         (max_seq × kv_lora_rank) f32  — persistent GPU KV latent
//   3  dst_k_pe         (max_seq × rope_dim)     f32  — persistent GPU RoPE K
//   4  seq_slot         constant uint             — position index (= kv.seq_len before append)
//   5  kv_lora_rank     constant uint
//   6  qk_rope_head_dim constant uint
kernel void kv_append_f32(
    device const float* src_c_kv_normed  [[buffer(0)]],
    device const float* src_kv_a_out     [[buffer(1)]],
    device       float* dst_c_kv         [[buffer(2)]],
    device       float* dst_k_pe         [[buffer(3)]],
    constant ArgbufKvAppend& args        [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint64_t c_base  = (uint64_t)args.seq_slot * (uint64_t)args.kv_lora_rank;
    uint64_t pe_base = (uint64_t)args.seq_slot * (uint64_t)args.qk_rope_head_dim;
    if (tid < args.kv_lora_rank) {
        dst_c_kv[c_base + tid] = src_c_kv_normed[tid];
    }
    if (tid < args.qk_rope_head_dim) {
        dst_k_pe[pe_base + tid] = src_kv_a_out[(uint64_t)args.kv_lora_rank + tid];
    }
}

// v0.5.8-B — fused RMSNorm + Q4_K_M GEMV pair (gate and up projections).
//
// Reads x once, computes rmsnorm, then runs two Q4_K_M GEMVs in parallel
// using a grid of 2×rows threadgroups: gid < rows → gate; gid >= rows → up.
// The variance computation is redundant across the two halves of the grid,
// but x is in L2 by the time the second half reads it.
//
// Binding scheme (matches manifest):
//   0  weight   (cols,) f16   rmsnorm learnable scale
//   1  eps      constant float
//   2  w_gate   (rows, cols) Q4_K_M bytes — gate projection
//   3  w_up     (rows, cols) Q4_K_M bytes — up projection
//   4  gate_out (rows,) f32
//   5  up_out   (rows,) f32
//   6  x        (cols,) f32   residual stream
//   7  rows     constant uint
//   8  cols     constant uint
//   threadgroup(0): shmem (TG_SIZE × f32) — variance then dot-product
//
// Grid:  (2 × rows, 1, 1); tg_size must be 256 (Q4_K_M super-block).
// Precondition: cols % 256 == 0.
kernel void rmsnorm_gemv_q4k_pair(
    device const half*  weight   [[buffer(0)]],
    constant     float& eps      [[buffer(1)]],
    device const uchar* w_gate   [[buffer(2)]],
    device const uchar* w_up     [[buffer(3)]],
    device       float* gate_out [[buffer(4)]],
    device       float* up_out   [[buffer(5)]],
    device const float* x        [[buffer(6)]],
    constant     uint&  rows     [[buffer(7)]],
    constant     uint&  cols     [[buffer(8)]],
    threadgroup  float* shmem    [[threadgroup(0)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[threadgroup_position_in_grid]],
    uint                tg_size  [[threads_per_threadgroup]])
{
    bool is_up = gid >= rows;
    uint row_idx = is_up ? (gid - rows) : gid;
    if (row_idx >= rows) return;

    device const uchar* w_q4 = is_up ? w_up : w_gate;
    device float*       out_ptr = is_up ? up_out : gate_out;

    // Phase 1: parallel variance reduction over x.
    float partial_sq = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        float v = x[c];
        partial_sq += v * v;
    }
    shmem[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shmem[0] / float(cols) + eps);

    // Phase 2: Q4_K_M GEMV with normalized x.
    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)row_idx * (uint64_t)blocks_per_row * 144ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;

        ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        uint sub = tid >> 5;
        uchar s_byte, m_byte;
        if (sub < 4u) {
            s_byte = w_q4[bo + 4u + sub]      & 0x3F;
            m_byte = w_q4[bo + 4u + 4u + sub] & 0x3F;
        } else {
            uint j = sub - 4u;
            s_byte = (w_q4[bo + 4u + 8u + j] & 0x0F)
                   | ((w_q4[bo + 4u + j]      >> 6) << 4);
            m_byte = (w_q4[bo + 4u + 8u + j] >> 4)
                   | ((w_q4[bo + 4u + 4u + j] >> 6) << 4);
        }

        uint pair = sub >> 1;
        bool upper = (sub & 1u) != 0u;
        uint i = tid & 31u;
        uchar q = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
        uint nib = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
        float w_val = d * (float)s_byte * (float)nib - dmin * (float)m_byte;

        uint global_col = b * 256u + tid;
        float xv = x[global_col] * inv_rms * (float)weight[global_col];
        partial += w_val * xv;
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out_ptr[row_idx] = shmem[0];
}

// v0.8.1 — f16-input bridge: rmsnorm + f32-weight GEMV (attention path).
//
// Mirrors rmsnorm_gemv_f32_attn_pinned exactly; only the input dtype
// changes from float to half. Variance accumulation stays f32.
//
// Binding scheme (matches rmsnorm_gemv_f32_attn_pinned):
//   0  w       (rows × cols) f32   pinned weight matrix
//   1  x       (cols,)       f16   residual stream (bridge input)
//   2  weight  (cols,)       f32   rmsnorm learnable scale
//   3  eps     constant float
//   4  out     (rows,)       f32   output
//   5  rows    constant uint
//   6  cols    constant uint
//   threadgroup(0): shmem (TG_SIZE × f32)
//
// Grid: (rows, 1, 1) threadgroups; TG_SIZE 256.
kernel void rmsnorm_gemv_f16_attn_pinned(
    device const float* w       [[buffer(0)]],
    device const half*  x       [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant     float& eps     [[buffer(3)]],
    device       float* out     [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]],
    uint                tg_size [[threads_per_threadgroup]])
{
    if (gid >= rows) return;

    // Phase 1: variance reduction over x (f16 → f32 accumulation).
    float partial_sq = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        float v = (float)x[c];
        partial_sq += v * v;
    }
    shmem[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shmem[0] / float(cols) + eps);

    // Phase 2: GEMV with rmsnorm-scaled x.
    device const float* row = w + (uint64_t)gid * (uint64_t)cols;
    float partial_dot = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial_dot += row[c] * ((float)x[c] * inv_rms * weight[c]);
    }
    shmem[tid] = partial_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[gid] = shmem[0];
}

// v0.8.2 — f16-input bridge: rmsnorm + Q4_K_M GEMV pair (gate+up).
//
// Mirrors rmsnorm_gemv_q4k_pair exactly; only x dtype changes to half.
//
// Binding scheme (matches rmsnorm_gemv_q4k_pair):
//   0  weight   (cols,) f16   rmsnorm learnable scale
//   1  eps      constant float
//   2  w_gate   Q4_K_M bytes
//   3  w_up     Q4_K_M bytes
//   4  gate_out (rows,) f32
//   5  up_out   (rows,) f32
//   6  x        (cols,) f16   residual stream (bridge input)
//   7  rows     constant uint
//   8  cols     constant uint
//   threadgroup(0): shmem (TG_SIZE × f32)
//
// Grid: (2 × rows, 1, 1); TG_SIZE 256. cols % 256 == 0 required.
kernel void rmsnorm_gemv_q4k_pair_f16(
    device const half*  weight   [[buffer(0)]],
    constant     float& eps      [[buffer(1)]],
    device const uchar* w_gate   [[buffer(2)]],
    device const uchar* w_up     [[buffer(3)]],
    device       float* gate_out [[buffer(4)]],
    device       float* up_out   [[buffer(5)]],
    device const half*  x        [[buffer(6)]],
    constant     uint&  rows     [[buffer(7)]],
    constant     uint&  cols     [[buffer(8)]],
    threadgroup  float* shmem    [[threadgroup(0)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[threadgroup_position_in_grid]],
    uint                tg_size  [[threads_per_threadgroup]])
{
    bool is_up = gid >= rows;
    uint row_idx = is_up ? (gid - rows) : gid;
    if (row_idx >= rows) return;

    device const uchar* w_q4 = is_up ? w_up : w_gate;
    device float*       out_ptr = is_up ? up_out : gate_out;

    // Phase 1: variance reduction over x (f16 → f32).
    float partial_sq = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        float v = (float)x[c];
        partial_sq += v * v;
    }
    shmem[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shmem[0] / float(cols) + eps);

    // Phase 2: Q4_K_M GEMV with normalized x.
    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)row_idx * (uint64_t)blocks_per_row * 144ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;

        ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        uint sub = tid >> 5;
        uchar s_byte, m_byte;
        if (sub < 4u) {
            s_byte = w_q4[bo + 4u + sub]      & 0x3F;
            m_byte = w_q4[bo + 4u + 4u + sub] & 0x3F;
        } else {
            uint j = sub - 4u;
            s_byte = (w_q4[bo + 4u + 8u + j] & 0x0F)
                   | ((w_q4[bo + 4u + j]      >> 6) << 4);
            m_byte = (w_q4[bo + 4u + 8u + j] >> 4)
                   | ((w_q4[bo + 4u + 4u + j] >> 6) << 4);
        }

        uint pair = sub >> 1;
        bool upper = (sub & 1u) != 0u;
        uint i = tid & 31u;
        uchar q = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
        uint nib = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
        float w_val = d * (float)s_byte * (float)nib - dmin * (float)m_byte;

        uint global_col = b * 256u + tid;
        float xv = (float)x[global_col] * inv_rms * (float)weight[global_col];
        partial += w_val * xv;
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out_ptr[row_idx] = shmem[0];
}


// ── flash_attn_decode_kernel ──────────────────────────────────────────────────
// Wedge L — Flash attention decode for DeepSeek MLA compressed KV cache.
//
// Fuses Phases 1-3 of mla_decode_kernel into a tiled online-softmax pass.
// Eliminates the materialized scores[seq_len] threadgroup buffer — 4× smaller
// TG memory footprint → more concurrent TGs per shader core → better GPU
// utilization on long-sequence decode.
//
// Algorithm (Flash Attention v2 online softmax for single-token decode):
//   Phase 0: q_nope_proj = w_uk^T × q_nope  (same as mla_decode_kernel)
//   Flash loop (tiles of FLASH_TG tokens each):
//     - Each thread computes 1 attention score
//     - Online max: simd_max → thread-0 tree reduce → m_new
//     - Online softmax correction: acc *= exp(m_old - m_new)
//     - Weighted accumulation: acc += exp(s - m_new) * c_kv (each thread r-slice)
//     - Update l_running (thread 0 serial, O(FLASH_TG) per tile)
//   Normalize: acc /= l_running
//   Phase 4: out = w_uv × acc  (same as mla_decode_kernel)
//
// Shmem layout (host sets sizes at dispatch):
//   slot 0 — q_nope_proj:  kv_lora_rank floats
//   slot 1 — acc:          kv_lora_rank floats
//   slot 2 — scores_tile:  FLASH_TG floats  (current tile scores)
//   slot 3 — state[8]:     {m_run, l_run, correction, m_tile, simd[0..3]_max}
//
// Grid: (n_heads * FLASH_TG, 1, 1)   TG: (FLASH_TG, 1, 1)
// FLASH_TG = 128 (4 simdgroups × 32 threads).
// Buffer layout: identical to mla_decode_kernel (buffers 0..11).

#define FLASH_TG   128u
#define FLASH_NSG  4u

kernel void flash_attn_decode_kernel(
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
    threadgroup  float* q_nope_proj [[threadgroup(0)]],
    threadgroup  float* acc         [[threadgroup(1)]],
    threadgroup  float* scores_tile [[threadgroup(2)]],
    threadgroup  float* state       [[threadgroup(3)]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                gid         [[threadgroup_position_in_grid]],
    uint                simd_lane   [[thread_index_in_simdgroup]],
    uint                simd_id     [[simdgroup_index_in_threadgroup]])
{
    if (gid >= n_heads) return;

    const uint head       = gid;
    const uint q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    device const float* q_nope = q + head * q_head_dim;
    device const float* q_rope = q_nope + qk_nope_head_dim;

    const uint kv_b_per_head = (qk_nope_head_dim + v_head_dim) * kv_lora_rank;
    device const float* w_uk = kv_b_proj + (uint64_t)head * kv_b_per_head;
    device const float* w_uv = w_uk + (uint64_t)qk_nope_head_dim * kv_lora_rank;

    // Phase 0: q_nope_proj[r] = w_uk^T x q_nope
    for (uint r = tid; r < kv_lora_rank; r += FLASH_TG) {
        float dot = 0.0f;
        for (uint i = 0; i < qk_nope_head_dim; i++)
            dot += w_uk[i * kv_lora_rank + r] * q_nope[i];
        q_nope_proj[r] = dot;
    }
    for (uint r = tid; r < kv_lora_rank; r += FLASH_TG) acc[r] = 0.0f;
    if (tid == 0u) { state[0] = -INFINITY; state[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flash loop
    const uint n_tiles = (seq_len + FLASH_TG - 1u) / FLASH_TG;

    for (uint tile = 0u; tile < n_tiles; ++tile) {
        const uint t_base = tile * FLASH_TG;
        const uint t_len  = min(FLASH_TG, seq_len - t_base);
        const uint t      = t_base + tid;

        // 1. Score
        float s_local = -INFINITY;
        if (tid < t_len) {
            float s = 0.0f;
            device const float* c_kv_t = c_kv + (uint64_t)t * kv_lora_rank;
            device const float* k_pe_t = k_pe + (uint64_t)t * qk_rope_head_dim;
            for (uint r = 0u; r < kv_lora_rank; r++) s += q_nope_proj[r] * c_kv_t[r];
            for (uint r = 0u; r < qk_rope_head_dim; r++) s += q_rope[r] * k_pe_t[r];
            s_local = s * scale;
        }
        scores_tile[tid] = s_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Parallel max
        float simd_mx = simd_max(s_local);
        if (simd_lane == 0u) state[4u + simd_id] = simd_mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Thread 0: online softmax state update
        if (tid == 0u) {
            float tile_max = max(max(state[4], state[5]), max(state[6], state[7]));
            float m_old    = state[0];
            float m_new    = max(m_old, tile_max);
            float corr     = exp(m_old - m_new);
            float tile_sum = 0.0f;
            for (uint ti = 0u; ti < t_len; ++ti)
                tile_sum += exp(scores_tile[ti] - m_new);
            state[0] = m_new;
            state[1] = state[1] * corr + tile_sum;
            state[2] = corr;
            state[3] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Scale acc and accumulate weighted c_kv
        const float corr_bc = state[2];
        const float m_bc    = state[3];
        for (uint r = tid; r < kv_lora_rank; r += FLASH_TG) {
            float a = acc[r] * corr_bc;
            for (uint ti = 0u; ti < t_len; ++ti) {
                float w = exp(scores_tile[ti] - m_bc);
                a += w * c_kv[(uint64_t)(t_base + ti) * kv_lora_rank + r];
            }
            acc[r] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize
    float inv_l = 1.0f / state[1];
    for (uint r = tid; r < kv_lora_rank; r += FLASH_TG) acc[r] *= inv_l;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out = w_uv x acc
    for (uint vi = tid; vi < v_head_dim; vi += FLASH_TG) {
        device const float* w_uv_row = w_uv + (uint64_t)vi * kv_lora_rank;
        float dot = 0.0f;
        for (uint r = 0u; r < kv_lora_rank; r++) dot += w_uv_row[r] * acc[r];
        out[head * v_head_dim + vi] = dot;
    }
}
