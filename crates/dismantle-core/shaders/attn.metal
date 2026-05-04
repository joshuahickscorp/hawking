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
