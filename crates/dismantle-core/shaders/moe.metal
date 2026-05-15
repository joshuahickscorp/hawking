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
    constant ArgbufTopkGate& args [[buffer(3)]],
    threadgroup  float* shmem     [[threadgroup(0)]],   // n_experts floats
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],   // token index
    uint                tg_size   [[threads_per_threadgroup]])
{
    // Cooperative load — pure fp32 copy.
    for (uint i = tid; i < args.n_experts; i += tg_size) {
        shmem[i] = logits[(uint64_t)gid * args.n_experts + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // Stable softmax: subtract max before exp.
        float m = -INFINITY;
        for (uint i = 0; i < args.n_experts; ++i) if (shmem[i] > m) m = shmem[i];

        float sum = 0.0f;
        for (uint i = 0; i < args.n_experts; ++i) {
            shmem[i] = exp(shmem[i] - m);
            sum += shmem[i];
        }
        float inv = 1.0f / sum;
        for (uint i = 0; i < args.n_experts; ++i) shmem[i] *= inv;

        // Top-K via masked selection (k passes; n_experts small).
        for (uint k = 0; k < args.top_k; ++k) {
            uint best_idx = 0;
            float best_val = -INFINITY;
            for (uint i = 0; i < args.n_experts; ++i) {
                if (shmem[i] > best_val) { best_val = shmem[i]; best_idx = i; }
            }
            expert_ids[(uint64_t)gid * args.top_k + k] = best_idx;
            weights[(uint64_t)gid * args.top_k + k]    = best_val;
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

// v2t_gu: fused gate+up Q4_K GEMV with threadgroup x-preload and inline silu_mul.
// Replaces 3 dispatches (gate-v2t, up-v2t, silu_mul) with 1.
// Each simdgroup (1 row) computes gate[row] and up[row] in one pass over x_cache,
// applies silu(gate)*up inline, and writes the activation directly.
// Saves one full x_cache preload (cols floats, 8KB for cols=2048) and one kernel.
// Grid: (ceil(rows/8)*256, routes, 1), TG (256,1,1), shmem = cols*4 bytes.
kernel void moe_batched_gemm_q4_indexed_v2t_gu(
    device const uchar* w_all         [[buffer(0)]],
    device const uint*  route_ids     [[buffer(1)]],
    device const float* x             [[buffer(2)]],
    device       float* y_act         [[buffer(3)]],  // output: silu(gate) * up
    constant     ulong& gate_offset   [[buffer(4)]],
    constant     ulong& up_offset     [[buffer(5)]],
    constant     uint&  routes        [[buffer(6)]],
    constant     uint&  rows          [[buffer(7)]],
    constant     uint&  cols          [[buffer(8)]],
    threadgroup  float* x_cache       [[threadgroup(0)]],
    uint2               tid2          [[thread_position_in_threadgroup]],
    uint2               tgp           [[threadgroup_position_in_grid]],
    uint                simd_lane     [[thread_index_in_simdgroup]],
    uint                simd_id       [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid2.x;
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

    uint64_t gate_row_off = gate_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t up_row_off   = up_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    float gate_partial = 0.0f, gate_corr = 0.0f;
    float up_partial   = 0.0f, up_corr   = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo_g = gate_row_off + (uint64_t)b * 144ul;
        uint64_t bo_u = up_row_off   + (uint64_t)b * 144ul;

        float dg    = fp16_at(w_all, bo_g);
        float dming = fp16_at(w_all, bo_g + 2ul);
        float du    = fp16_at(w_all, bo_u);
        float dminu = fp16_at(w_all, bo_u + 2ul);

        for (uint k = 0; k < 8u; ++k) {
            uchar sg, mg, su, mu;
            if (k < 4u) {
                sg = w_all[bo_g + 4u + k]      & 0x3F;
                mg = w_all[bo_g + 4u + 4u + k] & 0x3F;
                su = w_all[bo_u + 4u + k]      & 0x3F;
                mu = w_all[bo_u + 4u + 4u + k] & 0x3F;
            } else {
                uint j = k - 4u;
                sg = (w_all[bo_g + 4u + 8u + j] & 0x0F) | ((w_all[bo_g + 4u + j] >> 6) << 4);
                mg = (w_all[bo_g + 4u + 8u + j] >> 4)   | ((w_all[bo_g + 4u + 4u + j] >> 6) << 4);
                su = (w_all[bo_u + 4u + 8u + j] & 0x0F) | ((w_all[bo_u + 4u + j] >> 6) << 4);
                mu = (w_all[bo_u + 4u + 8u + j] >> 4)   | ((w_all[bo_u + 4u + 4u + j] >> 6) << 4);
            }

            uint elem = k * 32u + simd_lane;
            uint pair = k >> 1u;
            uchar qg = w_all[bo_g + 16ul + (uint64_t)pair * 32ul + (uint64_t)simd_lane];
            uchar qu = w_all[bo_u + 16ul + (uint64_t)pair * 32ul + (uint64_t)simd_lane];
            uint nibg = (k & 1u) ? ((uint)(qg >> 4) & 0x0Fu) : ((uint)qg & 0x0Fu);
            uint nibu = (k & 1u) ? ((uint)(qu >> 4) & 0x0Fu) : ((uint)qu & 0x0Fu);

            float xi = x_cache[(uint64_t)b * 256ul + (uint64_t)elem];

            gate_partial += dg    * (float)sg * (float)nibg * xi;
            gate_corr    += dming * (float)mg * xi;
            up_partial   += du    * (float)su * (float)nibu * xi;
            up_corr      += dminu * (float)mu * xi;
        }
    }

    float gate_val = simd_sum(gate_partial) - simd_sum(gate_corr);
    float up_val   = simd_sum(up_partial)   - simd_sum(up_corr);

    if (simd_lane == 0u) {
        float silu = gate_val / (1.0f + exp(-gate_val));
        y_act[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = silu * up_val;
    }
}

// ── moe_batched_gemm_q4_indexed_v2t_gu_v2 ────────────────────────────────────
// v2t_gu + sumy correction trick + scale/activation preloading + paired nibble
// reads. Same buffer layout and grid/TG geometry as v2t_gu.
//
// Key improvements over v2t_gu:
//   1. Scale pre-load: sg[8]/mg[8] for gate and su[8]/mu[8] for up extracted
//      once per block (eliminates redundant byte ops in the inner nibble loop).
//   2. Activation pre-load: xl[8] loaded into registers from x_cache before the
//      nibble loop (avoids SRAM re-reads in the hot path).
//   3. Sumy trick: total correction accumulated as sum_k(dm[k]*simd_sum(xl[k]))
//      rather than per-element dm*xi inside the inner loop.  Removes 16 MADs per
//      thread per block (2 correction MADs × 8 sub-blocks for gate+up combined).
//      total_gate_corr is thread-uniform so no extra simd_sum needed at reduce.
//   4. Paired nibble reads: pi-loop (4 iters) instead of k-loop (8 iters).
//      One weight byte covers k=2*pi (low nibble) and k=2*pi+1 (high nibble),
//      halving weight byte reads per row per block for gate and up.
//
// Grid: (ceil(rows/8)*256, routes, 1)   TG: (256, 1, 1)   shmem: cols*4 bytes.
kernel void moe_batched_gemm_q4_indexed_v2t_gu_v2(
    device const uchar* w_all         [[buffer(0)]],
    device const uint*  route_ids     [[buffer(1)]],
    device const float* x             [[buffer(2)]],
    device       float* y_act         [[buffer(3)]],  // silu(gate) * up
    constant     ulong& gate_offset   [[buffer(4)]],
    constant     ulong& up_offset     [[buffer(5)]],
    constant     uint&  routes        [[buffer(6)]],
    constant     uint&  rows          [[buffer(7)]],
    constant     uint&  cols          [[buffer(8)]],
    threadgroup  float* x_cache       [[threadgroup(0)]],
    uint2               tid2          [[thread_position_in_threadgroup]],
    uint2               tgp           [[threadgroup_position_in_grid]],
    uint                simd_lane     [[thread_index_in_simdgroup]],
    uint                simd_id       [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid2.x;
    // Cooperative x preload into threadgroup SRAM — same as v2t_gu.
    for (uint i = tid; i < cols; i += 256u) x_cache[i] = x[(uint64_t)i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_row = tgp.x * 8u + simd_id;
    uint route    = tgp.y;
    if (route >= routes || base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 144ul;

    uint64_t gate_row_off = gate_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t up_row_off   = up_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    float gate_partial = 0.0f, up_partial = 0.0f;
    float total_gate_corr = 0.0f, total_up_corr = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo_g = gate_row_off + (uint64_t)b * 144ul;
        uint64_t bo_u = up_row_off   + (uint64_t)b * 144ul;

        float dg    = fp16_at(w_all, bo_g);
        float dming = fp16_at(w_all, bo_g + 2ul);
        float du    = fp16_at(w_all, bo_u);
        float dminu = fp16_at(w_all, bo_u + 2ul);

        // ── Step 1: Pre-load sub-block scale and min bytes (gate + up) ──────
        uchar sg[8], mg[8], su[8], mu[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sg[sub] = w_all[bo_g + 4u + sub]      & 0x3Fu;
            mg[sub] = w_all[bo_g + 4u + 4u + sub] & 0x3Fu;
            su[sub] = w_all[bo_u + 4u + sub]      & 0x3Fu;
            mu[sub] = w_all[bo_u + 4u + 4u + sub] & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sg[4u+j] = (w_all[bo_g + 4u + 8u + j] & 0x0Fu)
                     | ((w_all[bo_g + 4u + j]      >> 6u) << 4u);
            mg[4u+j] = (w_all[bo_g + 4u + 8u + j] >> 4u)
                     | ((w_all[bo_g + 4u + 4u + j] >> 6u) << 4u);
            su[4u+j] = (w_all[bo_u + 4u + 8u + j] & 0x0Fu)
                     | ((w_all[bo_u + 4u + j]      >> 6u) << 4u);
            mu[4u+j] = (w_all[bo_u + 4u + 8u + j] >> 4u)
                     | ((w_all[bo_u + 4u + 4u + j] >> 6u) << 4u);
        }

        // Pre-compute d*scale and dmin*scale per sub-block.
        float dsg[8], dmg[8], dsu[8], dmu[8];
        for (uint k = 0; k < 8u; ++k) {
            dsg[k] = dg    * (float)sg[k];
            dmg[k] = dming * (float)mg[k];
            dsu[k] = du    * (float)su[k];
            dmu[k] = dminu * (float)mu[k];
        }

        // ── Step 2: Pre-load activations from x_cache into registers ────────
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x_cache[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        // ── Step 3: Sumy trick — sub-block activation sums ──────────────────
        // simd_sum returns the same value to all 32 threads → sumy is
        // thread-uniform.  total_gate_corr / total_up_corr are therefore
        // thread-uniform and need no further simd_sum at the reduce step.
        float sumy[8];
        for (uint k = 0; k < 8u; ++k) sumy[k] = simd_sum(xl[k]);
        for (uint k = 0; k < 8u; ++k) {
            total_gate_corr += dmg[k] * sumy[k];
            total_up_corr   += dmu[k] * sumy[k];
        }

        // ── Step 4: Paired nibble dot product (no correction term) ──────────
        // One weight byte per pair: low nibble = sub-block 2*pi, high = 2*pi+1.
        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qg = w_all[bo_g + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uchar qu = w_all[bo_u + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            gate_partial += dsg[k0] * (float)(qg & 0x0Fu) * xl[k0]
                          + dsg[k1] * (float)(qg >> 4u)   * xl[k1];
            up_partial   += dsu[k0] * (float)(qu & 0x0Fu) * xl[k0]
                          + dsu[k1] * (float)(qu >> 4u)   * xl[k1];
        }
    }

    // total_gate_corr is thread-uniform — subtract directly (no simd_sum needed).
    float gate_val = simd_sum(gate_partial) - total_gate_corr;
    float up_val   = simd_sum(up_partial)   - total_up_corr;

    if (simd_lane == 0u) {
        float silu = gate_val / (1.0f + exp(-gate_val));
        y_act[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = silu * up_val;
    }
}

// ── v2.3.0 A4.2: function-constant-specialized moe_q4_v2t_gu_v2 ──────────────
// Identical math/body to `moe_batched_gemm_q4_indexed_v2t_gu_v2` above. The
// two model-constant shape params (rows = moe_intermediate, cols = hidden)
// are baked into the compiled pipeline via MTLFunctionConstantValues. For
// DeepSeek-V2-Lite: kFcMoeRows=1408, kFcMoeCols=2048.
//
// The compiler then knows `blocks_per_row = kFcMoeCols/256 = 8` at compile
// time, so the outer `for b in 0..blocks_per_row` loop is unrollable and
// `per_matrix_bytes = rows * blocks_per_row * 144` collapses to a constant.
// The dispatcher passes only `gate_offset`, `up_offset`, `routes` at
// runtime (the values that genuinely vary per layer/per-token).
constant uint kFcMoeRows [[function_constant(10)]];
constant uint kFcMoeCols [[function_constant(11)]];

kernel void moe_batched_gemm_q4_indexed_v2t_gu_v2_fc(
    device const uchar* w_all         [[buffer(0)]],
    device const uint*  route_ids     [[buffer(1)]],
    device const float* x             [[buffer(2)]],
    device       float* y_act         [[buffer(3)]],
    constant     ulong& gate_offset   [[buffer(4)]],
    constant     ulong& up_offset     [[buffer(5)]],
    constant     uint&  routes        [[buffer(6)]],
    threadgroup  float* x_cache       [[threadgroup(0)]],
    uint2               tid2          [[thread_position_in_threadgroup]],
    uint2               tgp           [[threadgroup_position_in_grid]],
    uint                simd_lane     [[thread_index_in_simdgroup]],
    uint                simd_id       [[simdgroup_index_in_threadgroup]])
{
    uint tid = tid2.x;
    for (uint i = tid; i < kFcMoeCols; i += 256u) x_cache[i] = x[(uint64_t)i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_row = tgp.x * 8u + simd_id;
    uint route    = tgp.y;
    if (route >= routes || base_row >= kFcMoeRows) return;

    uint expert = route_ids[route];
    const uint blocks_per_row = kFcMoeCols / 256u;
    const uint64_t per_matrix_bytes = (uint64_t)kFcMoeRows * (uint64_t)blocks_per_row * 144ul;

    uint64_t gate_row_off = gate_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t up_row_off   = up_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    float gate_partial = 0.0f, up_partial = 0.0f;
    float total_gate_corr = 0.0f, total_up_corr = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo_g = gate_row_off + (uint64_t)b * 144ul;
        uint64_t bo_u = up_row_off   + (uint64_t)b * 144ul;

        float dg    = fp16_at(w_all, bo_g);
        float dming = fp16_at(w_all, bo_g + 2ul);
        float du    = fp16_at(w_all, bo_u);
        float dminu = fp16_at(w_all, bo_u + 2ul);

        uchar sg[8], mg[8], su[8], mu[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sg[sub] = w_all[bo_g + 4u + sub]      & 0x3Fu;
            mg[sub] = w_all[bo_g + 4u + 4u + sub] & 0x3Fu;
            su[sub] = w_all[bo_u + 4u + sub]      & 0x3Fu;
            mu[sub] = w_all[bo_u + 4u + 4u + sub] & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sg[4u+j] = (w_all[bo_g + 4u + 8u + j] & 0x0Fu)
                     | ((w_all[bo_g + 4u + j]      >> 6u) << 4u);
            mg[4u+j] = (w_all[bo_g + 4u + 8u + j] >> 4u)
                     | ((w_all[bo_g + 4u + 4u + j] >> 6u) << 4u);
            su[4u+j] = (w_all[bo_u + 4u + 8u + j] & 0x0Fu)
                     | ((w_all[bo_u + 4u + j]      >> 6u) << 4u);
            mu[4u+j] = (w_all[bo_u + 4u + 8u + j] >> 4u)
                     | ((w_all[bo_u + 4u + 4u + j] >> 6u) << 4u);
        }

        float dsg[8], dmg[8], dsu[8], dmu[8];
        for (uint k = 0; k < 8u; ++k) {
            dsg[k] = dg    * (float)sg[k];
            dmg[k] = dming * (float)mg[k];
            dsu[k] = du    * (float)su[k];
            dmu[k] = dminu * (float)mu[k];
        }

        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x_cache[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        float sumy[8];
        for (uint k = 0; k < 8u; ++k) sumy[k] = simd_sum(xl[k]);
        for (uint k = 0; k < 8u; ++k) {
            total_gate_corr += dmg[k] * sumy[k];
            total_up_corr   += dmu[k] * sumy[k];
        }

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qg = w_all[bo_g + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uchar qu = w_all[bo_u + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            gate_partial += dsg[k0] * (float)(qg & 0x0Fu) * xl[k0]
                          + dsg[k1] * (float)(qg >> 4u)   * xl[k1];
            up_partial   += dsu[k0] * (float)(qu & 0x0Fu) * xl[k0]
                          + dsu[k1] * (float)(qu >> 4u)   * xl[k1];
        }
    }

    float gate_val = simd_sum(gate_partial) - total_gate_corr;
    float up_val   = simd_sum(up_partial)   - total_up_corr;

    if (simd_lane == 0u) {
        float silu = gate_val / (1.0f + exp(-gate_val));
        y_act[(uint64_t)route * (uint64_t)kFcMoeRows + (uint64_t)base_row] = silu * up_val;
    }
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

// v2t variant of moe_batched_gemm_q8_0_indexed.
// Grid: (ceil(rows/8)*256, routes, 1), TG (256,1,1), shmem = cols*4 bytes.
// 8 simdgroups per TG share one x_cache preload; each simdgroup owns one row.
// Q8_0 block = 34 bytes: 2B fp16 scale + 32B signed int8. Exactly 32 elements
// per block matches simdgroup width — no inner loop, one simd_sum per block.
// Eliminates ~1.4 GB/token of redundant x DRAM reads vs the scalar kernel.
kernel void moe_batched_gemm_q8_0_indexed_v2t(
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
    uint tid   = tid2.x;
    uint route = tgp.y;
    // x is route-major: x[route*cols .. route*cols+cols] is this route's activation.
    // Cooperative preload into threadgroup SRAM (stride-256 for cols=1408, 6 passes).
    for (uint i = tid; i < cols; i += 256u) {
        x_cache[i] = x[(uint64_t)route * cols + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_row = tgp.x * 8u + simd_id;
    if (route >= routes || base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 32u;                               // e.g. 1408/32 = 44
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 34ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 34ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 34ul;
        float d  = fp16_at(w_all, bo);
        int   qi = signed_u8(w_all[bo + 2ul + (uint64_t)simd_lane]);
        float xi = x_cache[b * 32u + simd_lane];
        partial += d * (float)qi * xi;
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = partial;
    }
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

// v2.1.0-T2.11 — v2t-pattern port for Q5_0 routed-down kernel.
//
// Mirrors moe_batched_gemm_q8_0_indexed_v2t exactly except for:
//   - block stride (22 bytes for Q5_0 vs 34 for Q8_0)
//   - inline 5-bit decode for the per-lane qi instead of signed_u8 read
//
// Each simdgroup (32 lanes) processes ONE row of ONE block at a time;
// 8 rows per threadgroup (simd_id 0..7 within the TG), 32 simd_lanes
// each handling one of the 32 values in the block. Threadgroup x_cache
// preloads the route's activation vector once and reuses it across the
// inner block-loop (avoiding cols × routes repeated global reads).
//
// Q5_0 block layout (22 bytes per block, 32 values):
//   [0..2)   fp16 scale d
//   [2..6)   qh — 4 bytes = 32 bits, one per value (5th/high bit)
//   [6..22)  qlo — 16 bytes = 32 nibbles (low 4 bits per value, packed
//            so byte[i] holds value i's nibble in low4 and value (i+16)'s
//            nibble in high4)
//
// Per-lane decode for lane `simd_lane` in block `b`:
//   packed_byte = w_all[bo + 6 + (simd_lane & 15)]
//   low4 = (simd_lane < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF)
//   high_bit = (qh32 >> simd_lane) & 1
//   q = (low4 | (high_bit << 4)) - 16
//   value = d * q
//
// Parity validated by tests/v2_1_q5_0_v2t_parity.rs.
kernel void moe_batched_gemm_q5_0_indexed_v2t(
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
    uint tid   = tid2.x;
    uint route = tgp.y;
    // Cooperative preload of this route's activation slice into TG SRAM.
    // Same stride-256 / 6-pass pattern as the Q8_0_v2t kernel for cols=1408.
    for (uint i = tid; i < cols; i += 256u) {
        x_cache[i] = x[(uint64_t)route * cols + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_row = tgp.x * 8u + simd_id;
    if (route >= routes || base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 32u;                               // 1408/32 = 44
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 22ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 22ul;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 22ul;
        float d = fp16_at(w_all, bo);
        // qh: 4 bytes = 32 bits, one bit per value's 5th/high bit.
        uint qh = ((uint)w_all[bo + 2ul])
                | ((uint)w_all[bo + 3ul] << 8)
                | ((uint)w_all[bo + 4ul] << 16)
                | ((uint)w_all[bo + 5ul] << 24);
        // Each simd_lane handles value index `simd_lane` (0..31).
        // Packed byte for value i is at offset 6 + (i & 15); low nibble
        // for i<16, high nibble for i>=16.
        uchar packed = w_all[bo + 6ul + (uint64_t)(simd_lane & 15u)];
        uint low  = (simd_lane < 16u)
                  ? ((uint)packed & 0x0Fu)
                  : (((uint)packed >> 4) & 0x0Fu);
        uint high = (qh >> simd_lane) & 0x01u;
        int qi    = (int)(low | (high << 4)) - 16;
        float xi  = x_cache[b * 32u + simd_lane];
        partial += d * (float)qi * xi;
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = partial;
    }
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

// v2.1.0-T2.12 — v2t-pattern port for Q6_K shared-down kernel.
//
// Mirrors the moe_batched_gemm_q5_0_indexed_v2t structure but adapted
// to Q6_K's 256-value-per-block layout (vs Q5_0's 32). Each simdgroup
// (32 lanes) processes one row of one block at a time; each lane
// handles 8 contiguous values in that block. 8 simdgroups per
// threadgroup → 8 rows per TG. The route's activation slice is
// pre-cached in threadgroup memory once per TG and reused across
// blocks_per_row iterations.
//
// Q6_K superblock layout (210 bytes / 256 values):
//   [0..128)    ql      — 128 bytes, low 4 bits per value
//   [128..192)  qh      — 64 bytes, high 2 bits per value
//   [192..208)  scales  — 16 signed int8 per-16-value sub-block scales
//   [208..210)  d       — fp16 superblock scale
//
// Lane → value-index mapping: lane L processes block-local tids
// L*8..L*8+7 (8 contiguous values). These all share the same
// (half_idx, group) so each lane reads exactly ONE scale byte per
// block. half_idx = L>>4 (0 or 1); group = (L>>2)&3 (0..3);
// l_base = (L&3)*8 (0, 8, 16, or 24).
//
// Per-value decode (matches q6_k_value()):
//   l        = l_base + k                    (k in 0..7)
//   ql_off   = (group & 1) ? 32 : 0          (which 32-byte ql half-row)
//   qlb      = ql[half_idx*64 + ql_off + l]
//   qlow     = (group < 2) ? qlb & 0xF       (low nibble for groups 0,1)
//                          : qlb >> 4        (high nibble for groups 2,3)
//   qhb      = qh[128 + half_idx*32 + l]
//   qhigh    = (qhb >> (group*2)) & 0x03
//   q        = (qlow | (qhigh << 4)) - 32    (signed 6-bit)
//   value    = d * scale * q
//
// Numerical parity vs basic: each value's math is identical, but
// summation order is simdsum-then-block-loop (v2t) vs all-tids-of-
// block-then-tree-reduce (basic). fp32 add is non-associative so
// ULP-level drift can shift greedy argmax; same caveat as Q5_0 v2t.
kernel void moe_batched_gemm_q6_k_indexed_v2t(
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
    uint tid   = tid2.x;
    uint route = tgp.y;
    // Cooperative preload of this route's activation slice (cols floats)
    // into TG SRAM; 256 threads / cols=2816 → ~11 reads per thread.
    for (uint i = tid; i < cols; i += 256u) {
        x_cache[i] = x[(uint64_t)route * cols + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_row = tgp.x * 8u + simd_id;
    if (route >= routes || base_row >= rows) return;

    uint expert = route_ids[route];
    uint blocks_per_row = cols / 256u;
    uint64_t per_matrix_bytes = (uint64_t)rows * (uint64_t)blocks_per_row * 210ul;
    uint64_t row_byte_off = (uint64_t)base_offset
                          + (uint64_t)expert * per_matrix_bytes
                          + (uint64_t)base_row * (uint64_t)blocks_per_row * 210ul;

    // Per-lane constants (independent of block index).
    uint half_idx       = simd_lane >> 4u;          // 0 or 1
    uint group          = (simd_lane >> 2u) & 3u;   // 0..3
    uint l_base         = (simd_lane & 3u) * 8u;    // 0, 8, 16, or 24
    uint scale_l_off    = l_base >> 4u;             // (l>>4) — same for all 8 of lane's values
    uint scale_byte_off = 192u + half_idx * 8u + scale_l_off + group * 2u;
    uint ql_group_off   = (group & 1u) * 32u;       // 0 if group∈{0,2}, 32 if group∈{1,3}
    bool group_high_nibble = (group >= 2u);
    uint qh_shift       = group * 2u;
    uint tid_base       = half_idx * 128u + group * 32u + l_base;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 210ul;
        float d = fp16_at(w_all, bo + 208ul);
        int scale = signed_u8(w_all[bo + (uint64_t)scale_byte_off]);
        float dscale = d * (float)scale;

        uint64_t ql_base = bo + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
        uint64_t qh_base = bo + 128ul + (uint64_t)half_idx * 32ul;

        float lane_acc = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            uint l = l_base + k;
            uchar qlb = w_all[ql_base + (uint64_t)l];
            uint qlow = group_high_nibble
                      ? (((uint)qlb >> 4) & 0x0Fu)
                      : ((uint)qlb & 0x0Fu);
            uchar qhb = w_all[qh_base + (uint64_t)l];
            uint qhigh = ((uint)qhb >> qh_shift) & 0x03u;
            int qi = (int)(qlow | (qhigh << 4)) - 32;
            float xi = x_cache[b * 256u + tid_base + k];
            lane_acc += (float)qi * xi;
        }
        partial += dscale * lane_acc;
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = partial;
    }
}

kernel void moe_batched_silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device       float* out  [[buffer(2)]],
    constant ArgbufN& args   [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    if (id >= args.n) return;
    float g = gate[id];
    out[id] = (g / (1.0f + exp(-g))) * up[id];
}

kernel void moe_route_accumulate(
    device const float* routed_out  [[buffer(0)]],   // (routes, hidden)
    device const float* weights     [[buffer(1)]],   // (routes)
    device const float* shared_out  [[buffer(2)]],   // (hidden) when has_shared=1
    device       float* out         [[buffer(3)]],   // (hidden)
    constant ArgbufRouteAcc& args   [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= args.hidden) return;
    float acc = args.has_shared != 0u ? shared_out[id] : 0.0f;
    for (uint r = 0; r < args.routes; ++r) {
        acc += weights[r] * routed_out[(uint64_t)r * args.hidden + id];
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

// G1.4 — fp32 GEMV for the MoE gate-logit projection (`ffn_gate_inp`).
// Tiny shape (n_routed_experts × hidden = 64 × 2048 for DeepSeek-V2-Lite)
// but proves MoE-shaped weight access. Same body as gemv_f32_attn; kept
// in its own file/kernel name per the manifest's gate split.
kernel void gemv_f32_moe(
    device const float* w     [[buffer(0)]],   // (rows, cols) row-major fp32
    device const float* x     [[buffer(1)]],   // (cols,)
    device       float* y     [[buffer(2)]],   // (rows,)
    constant ArgbufRowsCols& args  [[buffer(3)]],
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
