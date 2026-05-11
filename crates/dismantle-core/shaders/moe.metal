// moe.metal — the moat. Wedges 1 + 2.
//
// Kernels:
//   moe_topk_gate         — top-K softmax gate over routed-expert logits.
//                           Builds a (token, expert, weight) work queue.
//                           [Phase 1]
//   moe_grouped_gemm_q4   — per-expert grouped GEMM with Q4_K_M dequant
//                           fused inside the FMA loop in threadgroup
//                           memory. DRAM ships 4-bit weights only.
//                           [Phase 1, wedge 2]
//   moe_block_fused       — single-grid replacement that subsumes gate +
//                           dispatch + grouped GEMM + gather into one
//                           launch. Threadgroups pull (expert, token-tile)
//                           work items from the queue.
//                           [Phase 2, wedge 1]
//   moe_gather_combine    — weighted gather of expert outputs back to
//                           per-token activations.
//                           [Phase 1]

#include <metal_stdlib>
using namespace metal;

static inline int signed_u8(uchar v)
{
    int x = (int)v;
    return x >= 128 ? x - 256 : x;
}

static inline float fp16_at(device const uchar* p, uint64_t off)
{
    ushort bits = (ushort)p[off] | ((ushort)p[off + 1] << 8);
    return (float)as_type<half>(bits);
}

static inline float q4_k_value(device const uchar* w_q4, uint64_t bo, uint tid)
{
    float d = fp16_at(w_q4, bo);
    float dmin = fp16_at(w_q4, bo + 2);

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
    return d * (float)s_byte * (float)nib - dmin * (float)m_byte;
}

static inline float q8_0_value(device const uchar* w_q8, uint64_t row_byte_off, uint c)
{
    uint block = c >> 5;
    uint i = c & 31u;
    uint64_t bo = row_byte_off + (uint64_t)block * 34ul;
    float d = fp16_at(w_q8, bo);
    int q = signed_u8(w_q8[bo + 2ul + (uint64_t)i]);
    return d * (float)q;
}

static inline float q5_0_value(device const uchar* w_q5, uint64_t row_byte_off, uint c)
{
    uint block = c >> 5;
    uint i = c & 31u;
    uint64_t bo = row_byte_off + (uint64_t)block * 22ul;
    float d = fp16_at(w_q5, bo);
    uint qh = ((uint)w_q5[bo + 2ul])
            | ((uint)w_q5[bo + 3ul] << 8)
            | ((uint)w_q5[bo + 4ul] << 16)
            | ((uint)w_q5[bo + 5ul] << 24);
    uchar packed = w_q5[bo + 6ul + (uint64_t)(i & 15u)];
    uint low = i < 16u ? ((uint)packed & 0x0Fu) : (((uint)packed >> 4) & 0x0Fu);
    uint high = (qh >> i) & 0x01u;
    int q = (int)(low | (high << 4)) - 16;
    return d * (float)q;
}

static inline float q6_k_value(device const uchar* w_q6, uint64_t bo, uint tid)
{
    float d = fp16_at(w_q6, bo + 208ul);
    uint half_idx = tid >> 7;
    uint local = tid & 127u;
    uint l = local & 31u;
    uint group = local >> 5;

    uint64_t ql_base = bo + (uint64_t)half_idx * 64ul;
    uint64_t qh_base = bo + 128ul + (uint64_t)half_idx * 32ul;
    uchar qhi = w_q6[qh_base + (uint64_t)l];
    uint q;
    if (group == 0u) {
        q = ((uint)w_q6[ql_base + (uint64_t)l] & 0x0Fu)
          | (((uint)(qhi >> 0) & 0x03u) << 4);
    } else if (group == 1u) {
        q = ((uint)w_q6[ql_base + 32ul + (uint64_t)l] & 0x0Fu)
          | (((uint)(qhi >> 2) & 0x03u) << 4);
    } else if (group == 2u) {
        q = ((uint)(w_q6[ql_base + (uint64_t)l] >> 4))
          | (((uint)(qhi >> 4) & 0x03u) << 4);
    } else {
        q = ((uint)(w_q6[ql_base + 32ul + (uint64_t)l] >> 4))
          | (((uint)(qhi >> 6) & 0x03u) << 4);
    }

    int scale = signed_u8(w_q6[bo + 192ul + (uint64_t)half_idx * 8ul
                              + (uint64_t)(l >> 4) + (uint64_t)group * 2ul]);
    return d * (float)scale * (float)((int)q - 32);
}

// H2.1 — top-K softmax gate over routed-expert logits.
// One workgroup per token. n_experts is small (64 for DeepSeek-V2-Lite),
// so we use a single-thread reduction inside the workgroup; the rest of
// the threads idle. Perf isn't the wedge here — the gate is tiny vs the
// grouped-gemm — and a serial 64-element softmax+top-k is far faster
// than its dispatch overhead anyway.
//
// Input/output are fp32: top-K selection compares softmax probabilities
// for *integer* expert-id tie-breaking, so any precision loss on input
// can flip ordering for two close experts. The upstream kernel
// (`gemv_f32_moe`) already produces fp32 logits, so f32 here is also
// the natural shape.
kernel void moe_topk_gate(
    device const float* logits    [[buffer(0)]],   // (n_tokens, n_experts) row-major fp32
    device       uint*  expert_ids[[buffer(1)]],   // (n_tokens, top_k)
    device       float* weights   [[buffer(2)]],   // (n_tokens, top_k) raw softmax probs
    constant     uint&  n_experts [[buffer(3)]],
    constant     uint&  top_k     [[buffer(4)]],
    threadgroup  float* shmem     [[threadgroup(0)]],   // n_experts floats
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],   // token index
    uint                tg_size   [[threads_per_threadgroup]])
{
    // Cooperative load — pure fp32 copy.
    for (uint i = tid; i < n_experts; i += tg_size) {
        shmem[i] = logits[(uint64_t)gid * n_experts + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // Stable softmax: subtract max before exp.
        float m = -INFINITY;
        for (uint i = 0; i < n_experts; ++i) if (shmem[i] > m) m = shmem[i];

        float sum = 0.0f;
        for (uint i = 0; i < n_experts; ++i) {
            shmem[i] = exp(shmem[i] - m);
            sum += shmem[i];
        }
        float inv = 1.0f / sum;
        for (uint i = 0; i < n_experts; ++i) shmem[i] *= inv;

        // Top-K via masked selection (k passes; n_experts small).
        for (uint k = 0; k < top_k; ++k) {
            uint best_idx = 0;
            float best_val = -INFINITY;
            for (uint i = 0; i < n_experts; ++i) {
                if (shmem[i] > best_val) { best_val = shmem[i]; best_idx = i; }
            }
            expert_ids[(uint64_t)gid * top_k + k] = best_idx;
            weights[(uint64_t)gid * top_k + k]    = best_val;
            shmem[best_idx] = -INFINITY;   // mask for next pass
        }
    }
}

// H2.2 — fp32 GEMV with Q4_K_M weights, dequant fused inside the FMA loop.
// One workgroup per output row; tg_size MUST be 256 (matches the
// Q4_K_M super-block size). Each thread tid ∈ [0, 256) processes one
// element of the current super-block: dequantizes its 4-bit nibble and
// multiplies it into the running dot product. All 256 threads then
// tree-reduce across the threadgroup to produce y[row].
//
// This is the wedge-2 win: weights stay 4-bit in DRAM, only the dequant
// arithmetic is materialized in the FMA. ~2× weight bandwidth vs the
// Phase 0 dequant-then-gemv path.
//
// Q4_K_M block layout (144 bytes per 256 elements):
//   off+0..2:    fp16 d
//   off+2..4:    fp16 dmin
//   off+4..16:   12 bytes packed (scale, min) pairs (8 of each, 6-bit)
//   off+16..144: 128 bytes of 4-bit quants
//
// Indexing within a block (matches `decode_q_k_scale_min` /
// `dequant_q4_k` in quant/mod.rs):
//   sub = tid / 32          (which 32-elem sub-block, 0..7)
//   i   = tid % 32          (element within the sub-block)
//   pair  = sub / 2         (which 32-byte qs pair, 0..3)
//   upper = (sub & 1) == 1  (low nibble for sub=2k, high for sub=2k+1)
//   q  = qs[pair*32 + i]
//   nib = upper ? (q >> 4) & 0x0F : q & 0x0F
//   dst = sub*32 + i = tid  (every thread covers exactly one elem)
kernel void moe_grouped_gemm_q4(
    device const uchar* w_q4   [[buffer(0)]],   // (rows, cols) Q4_K_M
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    threadgroup  float* shmem  [[threadgroup(0)]],   // 256 floats
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                tg_size   [[threads_per_threadgroup]])
{
    if (gid >= rows) return;

    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)gid * (uint64_t)blocks_per_row * 144ul;

    // Per-thread scalar accumulator across all blocks in this row.
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;

        // Block scales: fp16 d, dmin. Each thread reads (small constant;
        // a broadcast via shmem would add a barrier without saving work).
        ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        // Decode this thread's (scale, min) for its sub-block (8 sub-blocks
        // per 256-elem block). Layout matches `decode_q_k_scale_min`:
        //   sub<4: low 6 bits of bytes [4..8] / [8..12]
        //   sub≥4: low 4 bits of bytes [12..16] OR'd with high 2 bits
        //          of bytes [4..8] / [8..12].
        uint sub = tid >> 5;            // tid / 32
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

        // Read this thread's quantized nibble.
        uint pair = sub >> 1;           // sub / 2
        bool upper = (sub & 1u) != 0u;
        uint i = tid & 31u;             // tid % 32
        uchar q = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
        uint nib = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);

        // Fused dequant: w_val = d * scale * nib - dmin * min.
        float w_val = d * (float)s_byte * (float)nib
                    - dmin * (float)m_byte;

        // Activation index. Within block: dst = sub*32 + i = tid.
        float xv = x[(uint64_t)b * 256ul + (uint64_t)tid];
        partial += w_val * xv;
    }

    // Threadgroup reduction (canonical pairwise; tg_size must be power of two).
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[gid] = shmem[0];
}

// Phase 2 — batched expert GEMV. The host packs only the selected
// route matrices contiguously, so route `r` owns one full row-major
// matrix at `r * per_matrix_bytes`.
kernel void moe_batched_gemm_q4(
    device const uchar* w_q4   [[buffer(0)]],   // (routes, rows, cols) Q4_K
    device const float* x      [[buffer(1)]],   // (cols,) shared across routes
    device       float* y      [[buffer(2)]],   // (routes, rows)
    constant     uint&  routes [[buffer(3)]],
    constant     uint&  rows   [[buffer(4)]],
    constant     uint&  cols   [[buffer(5)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off = (uint64_t)route * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 144ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;
        partial += q4_k_value(w_q4, bo, tid)
                 * x[(uint64_t)b * 256ul + (uint64_t)tid];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

// No-pack variant: routed experts are selected by route_ids from the
// full fused GGUF tensor. w_all is the whole GGUF mmap, and base_offset
// points at the first byte of the fused tensor inside that file.
kernel void moe_batched_gemm_q4_indexed(
    device const uchar* w_all     [[buffer(0)]],
    device const uint*  route_ids [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device       float* y         [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes    [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    threadgroup  float* shmem     [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 144ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;
        partial += q4_k_value(w_all, bo, tid)
                 * x[(uint64_t)b * 256ul + (uint64_t)tid];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

// v2: multi-row TG + simd_sum, zero inner-loop barriers.
// Grid: (ceil(rows/8), routes, 1), TG: (256, 1, 1), 8 simdgroups per TG.
kernel void moe_batched_gemm_q4_indexed_v2(
    device const uchar* w_all     [[buffer(0)]],
    device const uint*  route_ids [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device       float* y         [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes    [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = tgp.x * 8u + simd_id;
    uint route    = tgp.y;
    if (route >= routes) return;
    if (base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;
        for (uint k = 0; k < 8u; ++k) {
            uint elem = k * 32u + simd_lane;
            partial += q4_k_value(w_all, bo, elem)
                     * x[(uint64_t)b * 256ul + (uint64_t)elem];
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = partial;
    }
}

// v2s: v2 geometry (256 threads/TG, 8 simdgroups × 1 row each) + sumy trick.
// Loads d/dmin/s_byte/m_byte once per sub-block; accumulates dmin correction as
// dm * simd_sum(x_slice) per sub-block instead of dm * x per element.
// ~23% fewer ops per element vs v2; same register footprint (~7 floats/thread).
kernel void moe_batched_gemm_q4_indexed_v2s(
    device const uchar* w_all     [[buffer(0)]],
    device const uint*  route_ids [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device       float* y         [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes    [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = tgp.x * 8u + simd_id;
    uint route    = tgp.y;
    if (route >= routes) return;
    if (base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    float partial    = 0.0f;
    float total_corr = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;
        float d    = fp16_at(w_all, bo);
        float dmin = fp16_at(w_all, bo + 2ul);

        for (uint k = 0; k < 8u; ++k) {
            uchar s_byte, m_byte;
            if (k < 4u) {
                s_byte = w_all[bo + 4u + k]      & 0x3F;
                m_byte = w_all[bo + 4u + 4u + k] & 0x3F;
            } else {
                uint j = k - 4u;
                s_byte = (w_all[bo + 4u + 8u + j] & 0x0F)
                       | ((w_all[bo + 4u + j]      >> 6) << 4);
                m_byte = (w_all[bo + 4u + 8u + j] >> 4)
                       | ((w_all[bo + 4u + 4u + j] >> 6) << 4);
            }
            float ds = d    * (float)s_byte;
            float dm = dmin * (float)m_byte;

            uint elem = k * 32u + simd_lane;
            uint pair = k >> 1u;
            uchar q   = w_all[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)simd_lane];
            uint  nib = (k & 1u) ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
            float xi  = x[(uint64_t)b * 256ul + (uint64_t)elem];

            partial    += ds * (float)nib * xi;
            total_corr += dm * xi;
        }
    }

    partial    = simd_sum(partial)    - simd_sum(total_corr);
    if (simd_lane == 0u) {
        y[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = partial;
    }
}

// v2t: v2s geometry + threadgroup x-preload.
// All 256 threads cooperatively load x (≤8KB for cols≤2048) into threadgroup SRAM once
// per TG before the dot-product loop. The 8 simdgroups then read x from fast SRAM
// instead of independently fetching from L1/DRAM. One extra barrier at start.
// Grid/TG same as v2/v2s: (ceil(rows/8)*256, routes, 1), TG (256,1,1).
kernel void moe_batched_gemm_q4_indexed_v2t(
    device const uchar* w_all       [[buffer(0)]],
    device const uint*  route_ids   [[buffer(1)]],
    device const float* x           [[buffer(2)]],
    device       float* y           [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes      [[buffer(5)]],
    constant     uint&  rows        [[buffer(6)]],
    constant     uint&  cols        [[buffer(7)]],
    threadgroup  float* x_cache     [[threadgroup(0)]],  // cols floats
    uint2               tid2        [[thread_position_in_threadgroup]],
    uint2               tgp         [[threadgroup_position_in_grid]],
    uint                simd_lane   [[thread_index_in_simdgroup]],
    uint                simd_id     [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid2.x;
    // Cooperative x preload into threadgroup SRAM (256 threads, each loads cols/256 elements)
    for (uint i = tid; i < cols; i += 256u) {
        x_cache[i] = x[(uint64_t)i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_row = tgp.x * 8u + simd_id;
    uint route    = tgp.y;
    if (route >= routes || base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    float partial    = 0.0f;
    float total_corr = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;
        float d    = fp16_at(w_all, bo);
        float dmin = fp16_at(w_all, bo + 2ul);

        for (uint k = 0; k < 8u; ++k) {
            uchar s_byte, m_byte;
            if (k < 4u) {
                s_byte = w_all[bo + 4u + k]      & 0x3F;
                m_byte = w_all[bo + 4u + 4u + k] & 0x3F;
            } else {
                uint j = k - 4u;
                s_byte = (w_all[bo + 4u + 8u + j] & 0x0F)
                       | ((w_all[bo + 4u + j]      >> 6) << 4);
                m_byte = (w_all[bo + 4u + 8u + j] >> 4)
                       | ((w_all[bo + 4u + 4u + j] >> 6) << 4);
            }
            float ds = d    * (float)s_byte;
            float dm = dmin * (float)m_byte;

            uint elem = k * 32u + simd_lane;
            uint pair = k >> 1u;
            uchar q   = w_all[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)simd_lane];
            uint  nib = (k & 1u) ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
            float xi  = x_cache[(uint64_t)b * 256ul + (uint64_t)elem];

            partial    += ds * (float)nib * xi;
            total_corr += dm * xi;
        }
    }

    partial = simd_sum(partial) - simd_sum(total_corr);
    if (simd_lane == 0u) {
        y[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = partial;
    }
}

kernel void moe_batched_gemm_q8_0(
    device const uchar* w_q8   [[buffer(0)]],   // (routes, rows, cols) Q8_0
    device const float* x      [[buffer(1)]],   // (routes, cols)
    device       float* y      [[buffer(2)]],   // (routes, rows)
    constant     uint&  routes [[buffer(3)]],
    constant     uint&  rows   [[buffer(4)]],
    constant     uint&  cols   [[buffer(5)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint blocks_per_row = cols / 32u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 34ul;
    uint64_t row_byte_off = (uint64_t)route * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 34ul;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += q8_0_value(w_q8, row_byte_off, c)
                 * x[(uint64_t)route * cols + c];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

kernel void moe_batched_gemm_q8_0_indexed(
    device const uchar* w_all     [[buffer(0)]],
    device const uint*  route_ids [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device       float* y         [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes    [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    threadgroup  float* shmem     [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 32u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 34ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 34ul;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += q8_0_value(w_all, row_byte_off, c)
                 * x[(uint64_t)route * cols + c];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

kernel void moe_batched_gemm_q5_0_indexed(
    device const uchar* w_all     [[buffer(0)]],
    device const uint*  route_ids [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device       float* y         [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes    [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    threadgroup  float* shmem     [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 32u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 22ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 22ul;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += q5_0_value(w_all, row_byte_off, c)
                 * x[(uint64_t)route * cols + c];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

kernel void moe_batched_gemm_q6_k(
    device const uchar* w_q6   [[buffer(0)]],   // (routes, rows, cols) Q6_K
    device const float* x      [[buffer(1)]],   // (routes, cols)
    device       float* y      [[buffer(2)]],   // (routes, rows)
    constant     uint&  routes [[buffer(3)]],
    constant     uint&  rows   [[buffer(4)]],
    constant     uint&  cols   [[buffer(5)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 210ul;
    uint64_t row_byte_off = (uint64_t)route * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 210ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 210ul;
        partial += q6_k_value(w_q6, bo, tid)
                 * x[(uint64_t)route * cols + (uint64_t)b * 256ul + (uint64_t)tid];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

kernel void moe_batched_gemm_q6_k_indexed(
    device const uchar* w_all     [[buffer(0)]],
    device const uint*  route_ids [[buffer(1)]],
    device const float* x         [[buffer(2)]],
    device       float* y         [[buffer(3)]],
    constant     ulong& base_offset [[buffer(4)]],
    constant     uint&  routes    [[buffer(5)]],
    constant     uint&  rows      [[buffer(6)]],
    constant     uint&  cols      [[buffer(7)]],
    threadgroup  float* shmem     [[threadgroup(0)]],
    uint2               tid2      [[thread_position_in_threadgroup]],
    uint2               tgp       [[threadgroup_position_in_grid]],
    uint2               tg_size2  [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint row = tgp.x;
    uint route = tgp.y;
    if (row >= rows || route >= routes) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 210ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)row * (uint64_t)blocks_per_row * 210ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 210ul;
        partial += q6_k_value(w_all, bo, tid)
                 * x[(uint64_t)route * cols + (uint64_t)b * 256ul + (uint64_t)tid];
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[(uint64_t)route * rows + row] = shmem[0];
}

kernel void moe_batched_silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant     uint& n     [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    if (id >= n) return;
    float g = gate[id];
    out[id] = (g / (1.0f + exp(-g))) * up[id];
}

kernel void moe_route_accumulate(
    device const float* routed_out  [[buffer(0)]],   // (routes, hidden)
    device const float* weights     [[buffer(1)]],   // (routes)
    device const float* shared_out  [[buffer(2)]],   // (hidden) when has_shared=1
    device       float* out         [[buffer(3)]],   // (hidden)
    constant     uint&  hidden      [[buffer(4)]],
    constant     uint&  routes      [[buffer(5)]],
    constant     uint&  has_shared  [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    float acc = has_shared != 0u ? shared_out[id] : 0.0f;
    for (uint r = 0; r < routes; ++r) {
        acc += weights[r] * routed_out[(uint64_t)r * hidden + id];
    }
    out[id] = acc;
}

// H2.3 — weighted gather of per-(token, expert) outputs back into
// per-token activations. One thread per (token, hidden) pair.
//
//   token_out[t, h] = Σ_k weights[t, k] * expert_out[t, k, h]
//
// fp32 throughout: token activations stay f32 in the residual path.
kernel void moe_gather_combine(
    device const float* expert_out  [[buffer(0)]],   // (n_tokens, top_k, hidden)
    device const float* weights     [[buffer(1)]],   // (n_tokens, top_k)
    device       float* token_out   [[buffer(2)]],   // (n_tokens, hidden)
    constant     uint&  hidden      [[buffer(3)]],
    constant     uint&  top_k       [[buffer(4)]],
    uint2               gid         [[thread_position_in_grid]])
{
    uint h = gid.x;
    uint t = gid.y;
    if (h >= hidden) return;

    float acc = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        float w = weights[(uint64_t)t * top_k + k];
        float v = expert_out[((uint64_t)t * top_k + k) * hidden + h];
        acc += w * v;
    }
    token_out[(uint64_t)t * hidden + h] = acc;
}

// Wedge 1: single-launch MoE block. Workgroups pull (expert, tile)
// work items off a queue built by the gate. Stub for Phase 2.
kernel void moe_block_fused_stub(
    device const uchar* weights    [[buffer(0)]],
    device const half*  x          [[buffer(1)]],
    device       half*  y          [[buffer(2)]],
    device       uint*  work_queue [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    (void)id;
}

// ---------------------------------------------------------------------
// Stage 1b — strict single-launch fused MoE block, TOP-K variant.
//
// Same workgroup-per-output-row design as Stage 1a, but each workgroup
// iterates K experts and accumulates their weighted contributions into
// the output element. The intermediate buffer in threadgroup memory is
// reused across experts (recomputed per-expert; we never need K
// intermediates simultaneously).
//
// Indexed pattern: gate_w / up_w / down_w are the full fused-expert
// tensors; expert_ids[k] selects which expert's slab to use for the
// k-th iteration. Mirrors the no-pack convention used by the batched
// MoE wedge.
//
// Shapes:
//   gate_w  (n_experts, mid, hidden) Q4_K
//   up_w    (n_experts, mid, hidden) Q4_K
//   down_w  (n_experts, hidden, mid) Q4_K
//   expert_ids (top_k,)
//   weights    (top_k,)
//   x       (hidden,)     fp32
//   y       (hidden,)     fp32
kernel void moe_block_fused_q4_topk(
    device const uchar* gate_w     [[buffer(0)]],
    device const uchar* up_w       [[buffer(1)]],
    device const uchar* down_w     [[buffer(2)]],
    device const uint*  expert_ids [[buffer(3)]],
    device const float* weights    [[buffer(4)]],
    device const float* x          [[buffer(5)]],
    device       float* y          [[buffer(6)]],
    constant     uint&  hidden     [[buffer(7)]],
    constant     uint&  mid        [[buffer(8)]],
    constant     uint&  top_k      [[buffer(9)]],
    threadgroup  float* intermed   [[threadgroup(0)]],
    threadgroup  float* shmem      [[threadgroup(1)]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                gid         [[threadgroup_position_in_grid]],
    uint                tg_size     [[threads_per_threadgroup]])
{
    uint out_row = gid;
    if (out_row >= hidden) return;

    uint hidden_blocks = hidden / 256u;
    uint mid_blocks    = mid / 256u;
    uint64_t per_mid_row_bytes = (uint64_t)hidden_blocks * 144ul;
    uint64_t per_out_row_bytes = (uint64_t)mid_blocks    * 144ul;
    uint64_t per_expert_gate_up = (uint64_t)mid    * per_mid_row_bytes;
    uint64_t per_expert_down    = (uint64_t)hidden * per_out_row_bytes;

    // Per-thread accumulator across all K experts for this output row.
    // Only thread 0 actually writes the final y[out_row]; other threads'
    // accumulators are unused. Keeping them as locals (rather than
    // gating on tid == 0) saves a barrier at each K iteration.
    float acc = 0.0f;

    for (uint k = 0; k < top_k; ++k) {
        uint  expert   = expert_ids[k];
        float w_expert = weights[k];
        uint64_t gate_base = (uint64_t)expert * per_expert_gate_up;
        uint64_t up_base   = (uint64_t)expert * per_expert_gate_up;
        uint64_t down_base = (uint64_t)expert * per_expert_down;

        // Stage 1: compute this expert's intermediate vector.
        for (uint tile_base = 0u; tile_base < mid; tile_base += tg_size) {
            uint mid_row = tile_base + tid;
            if (mid_row < mid) {
                uint64_t row_off = (uint64_t)mid_row * per_mid_row_bytes;
                float gate_dot = 0.0f;
                float up_dot   = 0.0f;
                for (uint b = 0; b < hidden_blocks; ++b) {
                    uint64_t bo_g = gate_base + row_off + (uint64_t)b * 144ul;
                    uint64_t bo_u = up_base   + row_off + (uint64_t)b * 144ul;
                    for (uint elem = 0; elem < 256u; ++elem) {
                        float w_g = q4_k_value(gate_w, bo_g, elem);
                        float w_u = q4_k_value(up_w,   bo_u, elem);
                        float xv  = x[(uint64_t)b * 256ul + (uint64_t)elem];
                        gate_dot += w_g * xv;
                        up_dot   += w_u * xv;
                    }
                }
                float silu_g = gate_dot / (1.0f + exp(-gate_dot));
                intermed[mid_row] = silu_g * up_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Stage 2: this expert's contribution to out[out_row].
        uint64_t out_row_off = down_base + (uint64_t)out_row * per_out_row_bytes;
        float partial = 0.0f;
        for (uint b = 0; b < mid_blocks; ++b) {
            uint64_t bo = out_row_off + (uint64_t)b * 144ul;
            for (uint c = tid; c < 256u; c += tg_size) {
                float w     = q4_k_value(down_w, bo, c);
                float i_val = intermed[(uint64_t)b * 256ul + (uint64_t)c];
                partial    += w * i_val;
            }
        }
        shmem[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) shmem[tid] += shmem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // shmem[0] now holds this expert's down-dot. Thread 0
        // accumulates the weighted contribution; barrier guarantees
        // intermediate is safe to overwrite for the next expert.
        if (tid == 0u) {
            acc += w_expert * shmem[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[out_row] = acc;
}

// ---------------------------------------------------------------------
// Stage 1a — strict single-launch fused MoE block, ONE-EXPERT variant.
//
// Grid: `hidden` workgroups, TG_SIZE=256 threads each. Each workgroup
// computes ONE element of the output. The intermediate vector
// (silu(gate·x) * (up·x), `mid` floats) is computed once per workgroup
// in threadgroup memory, then consumed by the down-projection.
//
// All three weight matrices are Q4_K_M-quantized in this variant
// (synthetic test fixture). Production hauls add Q8_0 down + Q4_K
// shared-expert paths once parity holds.
//
// Trade-off (decode mode, n_tokens=1):
//   - Compute is REDUNDANT: each workgroup recomputes the full
//     intermediate vector. Total compute per token = `hidden` × O(mid)
//     vs the batched path's O(mid × hidden) for one-expert.
//   - Dispatch is COLLAPSED: one launch replaces 4 (gate + up + silu +
//     down). Saves the wait/encode overhead.
// On Apple Silicon the dispatch saving dominates only when GPU
// parallelism hides the redundant compute. The Stage-1a parity test
// proves correctness; the Stage-B6 bench tells us if it's worth it.
//
// Shapes:
//   gate_w  (mid, hidden) Q4_K  — `mid * hidden_blocks * 144` bytes
//   up_w    (mid, hidden) Q4_K
//   down_w  (hidden, mid) Q4_K  — `hidden * mid_blocks * 144` bytes
//   x       (hidden,)     fp32
//   y       (hidden,)     fp32
// Threadgroup memory:
//   intermed[mid]  — fits 1408 floats × 4 = 5.6 KB (M3 Pro tg = 32 KB)
//   shmem[TG_SIZE] — reduction scratch
kernel void moe_block_fused_q4_one(
    device const uchar* gate_w   [[buffer(0)]],
    device const uchar* up_w     [[buffer(1)]],
    device const uchar* down_w   [[buffer(2)]],
    device const float* x        [[buffer(3)]],
    device       float* y        [[buffer(4)]],
    constant     uint&  hidden   [[buffer(5)]],
    constant     uint&  mid      [[buffer(6)]],
    threadgroup  float* intermed [[threadgroup(0)]],
    threadgroup  float* shmem    [[threadgroup(1)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                tg_size   [[threads_per_threadgroup]])
{
    uint out_row = gid;
    if (out_row >= hidden) return;

    uint hidden_blocks = hidden / 256u;          // gate/up cols / 256
    uint mid_blocks    = mid / 256u;             // down  cols / 256
    uint64_t per_mid_row_bytes = (uint64_t)hidden_blocks * 144ul;
    uint64_t per_out_row_bytes = (uint64_t)mid_blocks    * 144ul;

    // ---- Stage 1: compute full intermediate vector. ----
    // Each thread covers ceil(mid / tg_size) intermediate rows in tiles.
    // For each row, the thread serially walks all (256 elem × hidden_blocks)
    // q4 nibbles of gate AND up rows. SwiGLU collapses to one float that
    // lands in `intermed[mid_row]`.
    for (uint tile_base = 0u; tile_base < mid; tile_base += tg_size) {
        uint mid_row = tile_base + tid;
        if (mid_row < mid) {
            uint64_t row_off = (uint64_t)mid_row * per_mid_row_bytes;
            float gate_dot = 0.0f;
            float up_dot   = 0.0f;
            for (uint b = 0; b < hidden_blocks; ++b) {
                uint64_t bo = row_off + (uint64_t)b * 144ul;
                for (uint elem = 0; elem < 256u; ++elem) {
                    float w_g = q4_k_value(gate_w, bo, elem);
                    float w_u = q4_k_value(up_w,   bo, elem);
                    float xv  = x[(uint64_t)b * 256ul + (uint64_t)elem];
                    gate_dot += w_g * xv;
                    up_dot   += w_u * xv;
                }
            }
            // SwiGLU: silu(gate) * up = (gate / (1 + exp(-gate))) * up.
            float silu_g = gate_dot / (1.0f + exp(-gate_dot));
            intermed[mid_row] = silu_g * up_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Stage 2: out[out_row] = down_w[out_row, :] · intermed ----
    // Standard threadgroup gemv reduction across `mid` elements.
    uint64_t out_row_off = (uint64_t)out_row * per_out_row_bytes;
    float partial = 0.0f;
    for (uint b = 0; b < mid_blocks; ++b) {
        uint64_t bo = out_row_off + (uint64_t)b * 144ul;
        for (uint c = tid; c < 256u; c += tg_size) {
            float w     = q4_k_value(down_w, bo, c);
            float i_val = intermed[(uint64_t)b * 256ul + (uint64_t)c];
            partial    += w * i_val;
        }
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[out_row] = shmem[0];
}

// ---------------------------------------------------------------------
// Stage 1c — strict single-launch fused MoE block, DeepSeek-V2-Lite layout.
//
// Extends Stage 1b (top-K, all-Q4_K) with mixed quantization and a
// permanent shared expert:
//   - Routed experts: gate Q4_K, up Q4_K, down Q8_0
//   - Shared expert:  gate Q4_K, up Q4_K, down Q6_K
//
// Threadgroup memory layout:
//   slot 0 — intermed[max(routed_mid, shared_mid)] floats
//             Reused between routed and shared phases; the barrier at the
//             end of each routed-expert iteration makes the reuse safe.
//   slot 1 — shmem[TG_SIZE] floats (reduction scratch)
//
// Byte-stride facts embedded in the indexing:
//   Q4_K: 256 elems/block, 144 bytes/block
//   Q8_0:  32 elems/block,  34 bytes/block
//   Q6_K: 256 elems/block, 210 bytes/block
kernel void moe_block_fused_v2lite(
    device const uchar* routed_gate_w  [[buffer(0)]],   // (n_experts, routed_mid, hidden) Q4_K
    device const uchar* routed_up_w    [[buffer(1)]],   // (n_experts, routed_mid, hidden) Q4_K
    device const uchar* routed_down_w  [[buffer(2)]],   // (n_experts, hidden, routed_mid) Q8_0
    device const uchar* shared_gate_w  [[buffer(3)]],   // (1, shared_mid, hidden) Q4_K
    device const uchar* shared_up_w    [[buffer(4)]],   // (1, shared_mid, hidden) Q4_K
    device const uchar* shared_down_w  [[buffer(5)]],   // (1, hidden, shared_mid) Q6_K
    device const uint*  expert_ids     [[buffer(6)]],   // (top_k,)
    device const float* route_weights  [[buffer(7)]],   // (top_k,)
    device const float* x              [[buffer(8)]],   // (hidden,) fp32
    device       float* y              [[buffer(9)]],   // (hidden,) fp32 output
    constant     uint&  hidden         [[buffer(10)]],
    constant     uint&  routed_mid     [[buffer(11)]],
    constant     uint&  shared_mid     [[buffer(12)]],
    constant     uint&  top_k          [[buffer(13)]],
    threadgroup  float* intermed       [[threadgroup(0)]],
    threadgroup  float* shmem          [[threadgroup(1)]],
    uint                tid             [[thread_position_in_threadgroup]],
    uint                gid             [[threadgroup_position_in_grid]],
    uint                tg_size         [[threads_per_threadgroup]])
{
    uint out_row = gid;
    if (out_row >= hidden) return;

    uint hidden_blocks       = hidden / 256u;
    uint routed_mid_blks_q8  = routed_mid / 32u;     // Q8_0 down: 32 elems/block
    uint shared_mid_blks_q6  = shared_mid / 256u;    // Q6_K down: 256 elems/block

    uint64_t routed_gate_up_row_bytes = (uint64_t)hidden_blocks * 144ul;
    uint64_t routed_down_row_bytes    = (uint64_t)routed_mid_blks_q8 * 34ul;
    uint64_t per_expert_gate_up_bytes = (uint64_t)routed_mid * routed_gate_up_row_bytes;
    uint64_t per_expert_down_bytes    = (uint64_t)hidden * routed_down_row_bytes;

    uint64_t shared_gate_up_row_bytes = (uint64_t)hidden_blocks * 144ul;
    uint64_t shared_down_row_bytes    = (uint64_t)shared_mid_blks_q6 * 210ul;

    // Per-thread accumulator for weighted routed contributions.
    float acc = 0.0f;

    // ---- Phase A: routed experts ----
    for (uint k = 0u; k < top_k; ++k) {
        uint  expert   = expert_ids[k];
        float w_expert = route_weights[k];
        uint64_t gate_base = (uint64_t)expert * per_expert_gate_up_bytes;
        uint64_t up_base   = (uint64_t)expert * per_expert_gate_up_bytes;
        uint64_t down_base = (uint64_t)expert * per_expert_down_bytes;

        // Stage A1: gate Q4_K + up Q4_K → SwiGLU → intermed[routed_mid].
        for (uint tile_base = 0u; tile_base < routed_mid; tile_base += tg_size) {
            uint mid_row = tile_base + tid;
            if (mid_row < routed_mid) {
                uint64_t row_off = (uint64_t)mid_row * routed_gate_up_row_bytes;
                float gate_dot = 0.0f, up_dot = 0.0f;
                for (uint b = 0u; b < hidden_blocks; ++b) {
                    uint64_t bo_g = gate_base + row_off + (uint64_t)b * 144ul;
                    uint64_t bo_u = up_base   + row_off + (uint64_t)b * 144ul;
                    for (uint elem = 0u; elem < 256u; ++elem) {
                        float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];
                        gate_dot += q4_k_value(routed_gate_w, bo_g, elem) * xv;
                        up_dot   += q4_k_value(routed_up_w,   bo_u, elem) * xv;
                    }
                }
                float silu_g = gate_dot / (1.0f + exp(-gate_dot));
                intermed[mid_row] = silu_g * up_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Stage A2: down Q8_0 — dot(down_w[expert, out_row, :], intermed).
        uint64_t out_row_off = down_base + (uint64_t)out_row * routed_down_row_bytes;
        float partial = 0.0f;
        for (uint c = tid; c < routed_mid; c += tg_size) {
            partial += q8_0_value(routed_down_w, out_row_off, c) * intermed[c];
        }
        shmem[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) shmem[tid] += shmem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) acc += w_expert * shmem[0];
        // Barrier: intermed is safe to overwrite for next expert or shared phase.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Phase B: shared expert ----
    // Stage B1: gate Q4_K + up Q4_K → SwiGLU → intermed[shared_mid].
    for (uint tile_base = 0u; tile_base < shared_mid; tile_base += tg_size) {
        uint mid_row = tile_base + tid;
        if (mid_row < shared_mid) {
            uint64_t row_off = (uint64_t)mid_row * shared_gate_up_row_bytes;
            float gate_dot = 0.0f, up_dot = 0.0f;
            for (uint b = 0u; b < hidden_blocks; ++b) {
                uint64_t bo = row_off + (uint64_t)b * 144ul;
                for (uint elem = 0u; elem < 256u; ++elem) {
                    float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];
                    gate_dot += q4_k_value(shared_gate_w, bo, elem) * xv;
                    up_dot   += q4_k_value(shared_up_w,   bo, elem) * xv;
                }
            }
            float silu_g = gate_dot / (1.0f + exp(-gate_dot));
            intermed[mid_row] = silu_g * up_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage B2: down Q6_K — dot(shared_down_w[out_row, :], intermed).
    uint64_t shared_out_row_off = (uint64_t)out_row * shared_down_row_bytes;
    float s_partial = 0.0f;
    for (uint b = 0u; b < shared_mid_blks_q6; ++b) {
        uint64_t bo = shared_out_row_off + (uint64_t)b * 210ul;
        // tg_size == 256 == Q6_K block size, so c == tid for the one iteration.
        for (uint c = tid; c < 256u; c += tg_size) {
            s_partial += q6_k_value(shared_down_w, bo, c)
                       * intermed[(uint64_t)b * 256ul + (uint64_t)c];
        }
    }
    shmem[tid] = s_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[out_row] = acc + shmem[0];
}

// ---------------------------------------------------------------------
// Stage B.4 — indexed variant of moe_block_fused_v2lite. Same kernel
// logic but all six weight tensors live inside one large model buffer
// (the GGUF mmap already uploaded once at load). Six u64 offsets
// select each tensor within that buffer, matching the no-copy indexed
// convention of moe_block_batched_indexed.
kernel void moe_block_fused_v2lite_indexed(
    device const uchar* model          [[buffer(0)]],   // whole GGUF mmap
    device const uint*  expert_ids     [[buffer(1)]],   // (top_k,)
    device const float* route_weights  [[buffer(2)]],   // (top_k,)
    device const float* x              [[buffer(3)]],   // (hidden,) fp32
    device       float* y              [[buffer(4)]],   // (hidden,) fp32 output
    constant     uint&  hidden         [[buffer(5)]],
    constant     uint&  routed_mid     [[buffer(6)]],
    constant     uint&  shared_mid     [[buffer(7)]],
    constant     uint&  top_k          [[buffer(8)]],
    constant     ulong& routed_gate_off[[buffer(9)]],
    constant     ulong& routed_up_off  [[buffer(10)]],
    constant     ulong& routed_down_off[[buffer(11)]],
    constant     ulong& shared_gate_off[[buffer(12)]],
    constant     ulong& shared_up_off  [[buffer(13)]],
    constant     ulong& shared_down_off[[buffer(14)]],
    threadgroup  float* intermed       [[threadgroup(0)]],
    threadgroup  float* shmem          [[threadgroup(1)]],
    uint                tid             [[thread_position_in_threadgroup]],
    uint                gid             [[threadgroup_position_in_grid]],
    uint                tg_size         [[threads_per_threadgroup]])
{
    device const uchar* routed_gate_w = model + routed_gate_off;
    device const uchar* routed_up_w   = model + routed_up_off;
    device const uchar* routed_down_w = model + routed_down_off;
    device const uchar* shared_gate_w = model + shared_gate_off;
    device const uchar* shared_up_w   = model + shared_up_off;
    device const uchar* shared_down_w = model + shared_down_off;

    uint out_row = gid;
    if (out_row >= hidden) return;

    uint hidden_blocks      = hidden / 256u;
    uint routed_mid_blks_q8 = routed_mid / 32u;
    uint shared_mid_blks_q6 = shared_mid / 256u;

    uint64_t routed_gate_up_row_bytes = (uint64_t)hidden_blocks * 144ul;
    uint64_t routed_down_row_bytes    = (uint64_t)routed_mid_blks_q8 * 34ul;
    uint64_t per_expert_gate_up_bytes = (uint64_t)routed_mid * routed_gate_up_row_bytes;
    uint64_t per_expert_down_bytes    = (uint64_t)hidden * routed_down_row_bytes;

    uint64_t shared_gate_up_row_bytes = (uint64_t)hidden_blocks * 144ul;
    uint64_t shared_down_row_bytes    = (uint64_t)shared_mid_blks_q6 * 210ul;

    float acc = 0.0f;

    for (uint k = 0u; k < top_k; ++k) {
        uint  expert   = expert_ids[k];
        float w_expert = route_weights[k];
        uint64_t gate_base = (uint64_t)expert * per_expert_gate_up_bytes;
        uint64_t up_base   = (uint64_t)expert * per_expert_gate_up_bytes;
        uint64_t down_base = (uint64_t)expert * per_expert_down_bytes;

        for (uint tile_base = 0u; tile_base < routed_mid; tile_base += tg_size) {
            uint mid_row = tile_base + tid;
            if (mid_row < routed_mid) {
                uint64_t row_off = (uint64_t)mid_row * routed_gate_up_row_bytes;
                float gate_dot = 0.0f, up_dot = 0.0f;
                for (uint b = 0u; b < hidden_blocks; ++b) {
                    uint64_t bo_g = gate_base + row_off + (uint64_t)b * 144ul;
                    uint64_t bo_u = up_base   + row_off + (uint64_t)b * 144ul;
                    for (uint elem = 0u; elem < 256u; ++elem) {
                        float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];
                        gate_dot += q4_k_value(routed_gate_w, bo_g, elem) * xv;
                        up_dot   += q4_k_value(routed_up_w,   bo_u, elem) * xv;
                    }
                }
                float silu_g = gate_dot / (1.0f + exp(-gate_dot));
                intermed[mid_row] = silu_g * up_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint64_t out_row_off = down_base + (uint64_t)out_row * routed_down_row_bytes;
        float partial = 0.0f;
        for (uint c = tid; c < routed_mid; c += tg_size) {
            partial += q8_0_value(routed_down_w, out_row_off, c) * intermed[c];
        }
        shmem[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) shmem[tid] += shmem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) acc += w_expert * shmem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Shared expert
    for (uint tile_base = 0u; tile_base < shared_mid; tile_base += tg_size) {
        uint mid_row = tile_base + tid;
        if (mid_row < shared_mid) {
            uint64_t row_off = (uint64_t)mid_row * shared_gate_up_row_bytes;
            float gate_dot = 0.0f, up_dot = 0.0f;
            for (uint b = 0u; b < hidden_blocks; ++b) {
                uint64_t bo = row_off + (uint64_t)b * 144ul;
                for (uint elem = 0u; elem < 256u; ++elem) {
                    float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];
                    gate_dot += q4_k_value(shared_gate_w, bo, elem) * xv;
                    up_dot   += q4_k_value(shared_up_w,   bo, elem) * xv;
                }
            }
            float silu_g = gate_dot / (1.0f + exp(-gate_dot));
            intermed[mid_row] = silu_g * up_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint64_t shared_out_row_off = (uint64_t)out_row * shared_down_row_bytes;
    float s_partial = 0.0f;
    for (uint b = 0u; b < shared_mid_blks_q6; ++b) {
        uint64_t bo = shared_out_row_off + (uint64_t)b * 210ul;
        for (uint c = tid; c < 256u; c += tg_size) {
            s_partial += q6_k_value(shared_down_w, bo, c)
                       * intermed[(uint64_t)b * 256ul + (uint64_t)c];
        }
    }
    shmem[tid] = s_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[out_row] = acc + shmem[0];
}

// Wedge 2 — two-stage fused MoE.
//
// Replaces the single-kernel path (which serialized top_k experts inside
// each workgroup) with two kernels dispatched in one command buffer:
//
//   moe_block_two_stage_intermediate  — stage 1: gate+up+silu_mul for all
//     top_k routed experts AND all shared experts simultaneously.
//     One workgroup per (expert_slot, intermediate_row).
//     Writes (top_k*routed_mid + n_shared*shared_mid) floats to `intermed`.
//
//   moe_block_two_stage_output        — stage 2: down-project + accumulate.
//     One workgroup per output row. Reads stage-1 `intermed`.

// ---- Stage 1 -------------------------------------------------------
// Dispatch: ((top_k * routed_mid + n_shared * shared_mid) * TG_SIZE, 1, 1)
// tg:       (TG_SIZE, 1, 1)
kernel void moe_block_two_stage_intermediate(
    device const uchar* routed_gate_w  [[buffer(0)]],   // (n_experts, routed_mid, hidden) Q4_K
    device const uchar* routed_up_w    [[buffer(1)]],   // same layout
    device const uchar* shared_gate_w  [[buffer(2)]],   // (1, shared_mid, hidden) Q4_K
    device const uchar* shared_up_w    [[buffer(3)]],   // same
    device const uint*  expert_ids     [[buffer(4)]],   // (top_k,)
    device const float* x              [[buffer(5)]],   // (hidden,)
    device       float* intermed       [[buffer(6)]],   // (top_k*routed_mid + n_shared*shared_mid,)
    constant     uint&  hidden         [[buffer(7)]],
    constant     uint&  routed_mid     [[buffer(8)]],
    constant     uint&  shared_mid     [[buffer(9)]],
    constant     uint&  top_k          [[buffer(10)]],
    constant     uint&  n_shared       [[buffer(11)]],
    threadgroup  float* shmem          [[threadgroup(0)]],  // TG_SIZE floats
    uint                tid             [[thread_position_in_threadgroup]],
    uint                gid             [[threadgroup_position_in_grid]],
    uint                tg_size         [[threads_per_threadgroup]])
{
    uint routed_total = top_k * routed_mid;
    uint total        = routed_total + n_shared * shared_mid;
    if (gid >= total) return;

    uint hidden_blocks = hidden / 256u;
    uint64_t row_bytes = (uint64_t)hidden_blocks * 144ul;  // Q4_K: 144 bytes / 256-elem block

    device const uchar* gate_w;
    device const uchar* up_w;
    uint64_t base;
    uint     mid_row;

    if (gid < routed_total) {
        uint k   = gid / routed_mid;
        mid_row  = gid % routed_mid;
        uint eid = expert_ids[k];
        gate_w   = routed_gate_w;
        up_w     = routed_up_w;
        base     = (uint64_t)eid * (uint64_t)routed_mid * row_bytes;
    } else {
        // shared expert — shared_slot = (gid - routed_total)
        // For ≥ 1 shared experts, slot / shared_mid selects which one (always 0 here).
        mid_row = (gid - routed_total) % shared_mid;
        gate_w  = shared_gate_w;
        up_w    = shared_up_w;
        base    = 0ul;  // single shared expert → no per-expert stride
    }

    uint64_t row_off       = (uint64_t)mid_row * row_bytes;
    float    gate_partial  = 0.0f;
    float    up_partial    = 0.0f;

    // TG_SIZE == 256 == Q4_K block size, so tid == elem and each thread
    // handles exactly one element per block — no inner loop needed.
    for (uint b = 0u; b < hidden_blocks; ++b) {
        uint64_t bo = base + row_off + (uint64_t)b * 144ul;
        float xv    = x[b * 256u + tid];
        gate_partial += q4_k_value(gate_w, bo, tid) * xv;
        up_partial   += q4_k_value(up_w,   bo, tid) * xv;
    }

    // Reduce gate_partial across threadgroup.
    shmem[tid] = gate_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gate_dot = shmem[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce up_partial across threadgroup.
    shmem[tid] = up_partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float silu_g = gate_dot / (1.0f + exp(-gate_dot));
        intermed[gid] = silu_g * shmem[0];
    }
}

// ---- Stage 2 -------------------------------------------------------
// Dispatch: (hidden * TG_SIZE, 1, 1), tg: (TG_SIZE, 1, 1)
kernel void moe_block_two_stage_output(
    device const uchar* routed_down_w  [[buffer(0)]],   // (n_experts, hidden, routed_mid) Q8_0
    device const uchar* shared_down_w  [[buffer(1)]],   // (1, hidden, shared_mid) Q6_K
    device const uint*  expert_ids     [[buffer(2)]],   // (top_k,)
    device const float* route_weights  [[buffer(3)]],   // (top_k,)
    device const float* intermed       [[buffer(4)]],   // stage-1 output
    device       float* y              [[buffer(5)]],   // (hidden,) output
    constant     uint&  hidden         [[buffer(6)]],
    constant     uint&  routed_mid     [[buffer(7)]],
    constant     uint&  shared_mid     [[buffer(8)]],
    constant     uint&  top_k          [[buffer(9)]],
    constant     uint&  n_shared       [[buffer(10)]],
    threadgroup  float* shmem          [[threadgroup(0)]],  // TG_SIZE floats
    uint                tid             [[thread_position_in_threadgroup]],
    uint                gid             [[threadgroup_position_in_grid]],
    uint                tg_size         [[threads_per_threadgroup]])
{
    if (gid >= hidden) return;
    uint out_row = gid;

    uint64_t q8_row_bytes  = (uint64_t)(routed_mid / 32u) * 34ul;
    uint64_t per_expert_q8 = (uint64_t)hidden * q8_row_bytes;
    uint64_t q6_row_bytes  = (uint64_t)(shared_mid / 256u) * 210ul;

    float acc = 0.0f;

    // Routed experts: weighted dot product via Q8_0 down weights.
    for (uint k = 0u; k < top_k; ++k) {
        uint     eid     = expert_ids[k];
        uint64_t row_off = (uint64_t)eid * per_expert_q8 + (uint64_t)out_row * q8_row_bytes;
        uint     base_i  = k * routed_mid;

        float partial = 0.0f;
        for (uint c = tid; c < routed_mid; c += tg_size) {
            partial += q8_0_value(routed_down_w, row_off, c) * intermed[base_i + c];
        }
        shmem[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) shmem[tid] += shmem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) acc += route_weights[k] * shmem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Shared expert(s): Q6_K down weights.
    uint routed_total = top_k * routed_mid;
    for (uint sh = 0u; sh < n_shared; ++sh) {
        uint64_t row_off   = (uint64_t)out_row * q6_row_bytes;
        uint     blks      = shared_mid / 256u;
        uint     base_i    = routed_total + sh * shared_mid;

        float s_partial = 0.0f;
        for (uint b = 0u; b < blks; ++b) {
            uint64_t bo = row_off + (uint64_t)b * 210ul;
            // tg_size == 256 == Q6_K block size, c == tid (single iteration).
            s_partial += q6_k_value(shared_down_w, bo, tid)
                       * intermed[base_i + b * 256u + tid];
        }
        shmem[tid] = s_partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) shmem[tid] += shmem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) acc += shmem[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[out_row] = acc;
}

// G1.4 — fp32 GEMV for the MoE gate-logit projection (`ffn_gate_inp`).
// Tiny shape (n_routed_experts × hidden = 64 × 2048 for DeepSeek-V2-Lite)
// but proves MoE-shaped weight access. Same body as gemv_f32_attn; kept
// in its own file/kernel name per the manifest's gate split.
kernel void gemv_f32_moe(
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

// ── moe_grouped_gemm_q4_v2 ───────────────────────────────────────────────────
// Byte-identical body to gemm_q4_k_m_fused_v2 in quant.metal.
// Kept in moe.metal per the dense/MoE module split.
// Grid: (ceil(rows/8)*256, 1, 1)  threadgroup: (256, 1, 1)

kernel void moe_grouped_gemm_q4_v2(
    device const uchar* w_q4   [[buffer(0)]],   // (rows, cols) Q4_K_M
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[threadgroup_position_in_grid]],
    uint                simd_lane    [[thread_index_in_simdgroup]],
    uint                simd_id      [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;

        ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        for (uint k = 0; k < 8u; ++k) {
            uint elem = k * 32u + simd_lane;
            uint sub  = elem >> 5;
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
            uint pair  = sub >> 1;
            bool upper = (sub & 1u) != 0u;
            uint i     = elem & 31u;
            uchar q    = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
            uint nib   = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);

            float w_val = d * (float)s_byte * (float)nib - dmin * (float)m_byte;
            float xv    = x[(uint64_t)b * 256ul + (uint64_t)elem];
            partial    += w_val * xv;
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[base_row] = partial;
    }
}

// v0.5.9-B — fp16 activation variant: gemv_f32_moe with f16 x and f16 y.
// Same threadgroup structure as gemv_f32_moe. Internal MAC in f32.
kernel void gemv_f32_moe_f16(
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

// v0.5.10-B — fp16 activation variant of moe_grouped_gemm_q4.
// Q4_K_M weight (f32 scales), f16 x → f16 y. Internal MAC in f32.
// Grid: (rows, 1, 1), TG: (256, 1, 1), threadgroup_memory: 256 floats.
kernel void moe_grouped_gemm_q4_f16(
    device const uchar* w_q4   [[buffer(0)]],   // (rows, cols) Q4_K_M
    device const half*  x      [[buffer(1)]],   // (cols,) fp16
    device       half*  y      [[buffer(2)]],   // (rows,) fp16
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                tg_size   [[threads_per_threadgroup]])
{
    if (gid >= rows) return;

    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)gid * (uint64_t)blocks_per_row * 144ul;

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

        float xv = (float)x[(uint64_t)b * 256ul + (uint64_t)tid];
        partial += w_val * xv;
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) y[gid] = (half)shmem[0];
}
