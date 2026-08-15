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
// One workgroup per token.
//
// Threadgroup memory layout (host allocates at least this many bytes):
//   work[n_experts]             — logits → softmax probs (masked in-place)
//   red_val[tg_size]            — float reduction scratch
//   red_idx[tg_size]            — uint reduction scratch (overlay as float*)
// Total floats: n_experts + 2*tg_size.
//
// Two paths, branched once on the uniform `args.tie_epsilon`:
//
//   tie_epsilon == 0 (default): parallel lexicographic max of the total
//     order (value, -index). Max / sum / top-k are associative over that
//     order, so a tree reduction is bit-identical to the serial scan for
//     every exact-tie pattern. Lowest index wins on exact ties.
//
//   tie_epsilon > 0: keep the historical single-thread scan. The epsilon-
//     window rule compares against a running best_val and is order-
//     dependent; no reduction reproduces it.
//
// Input/output are fp32: top-K selection compares softmax probabilities
// for *integer* expert-id tie-breaking, so any precision loss on input
// can flip ordering for two close experts. The upstream kernel
// (`gemv_f32_moe`) already produces fp32 logits, so f32 here is also
// the natural shape.

// Prefer (val_a, -idx_a) over (val_b, -idx_b): higher value wins; on an
// exact value tie the lower index wins. Matches the serial left-to-right
// strict-`>` scan (which retains the first/lowest index on equality).
static inline bool moe_topk_lex_prefer(float val_a, uint idx_a, float val_b, uint idx_b)
{
    return (val_a > val_b) || (val_a == val_b && idx_a < idx_b);
}

kernel void moe_topk_gate(
    device const float* logits    [[buffer(0)]],   // (n_tokens, n_experts) row-major fp32
    device       uint*  expert_ids[[buffer(1)]],   // (n_tokens, top_k)
    device       float* weights   [[buffer(2)]],   // (n_tokens, top_k) raw softmax probs
    constant ArgbufTopkGate& args [[buffer(3)]],
    threadgroup  float* shmem     [[threadgroup(0)]],   // n_experts + 2*tg_size floats
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[threadgroup_position_in_grid]],   // token index
    uint                tg_size   [[threads_per_threadgroup]])
{
    threadgroup float* work = shmem;
    threadgroup float* red_val = shmem + args.n_experts;
    threadgroup uint*  red_idx = (threadgroup uint*)(shmem + args.n_experts + tg_size);

    // Cooperative load — pure fp32 copy.
    for (uint i = tid; i < args.n_experts; i += tg_size) {
        work[i] = logits[(uint64_t)gid * args.n_experts + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Serial path: epsilon-window ties (order-dependent) ──────────────────
    if (args.tie_epsilon > 0.0f) {
        if (tid == 0) {
            float m = -INFINITY;
            for (uint i = 0; i < args.n_experts; ++i) if (work[i] > m) m = work[i];

            float sum = 0.0f;
            for (uint i = 0; i < args.n_experts; ++i) {
                work[i] = exp(work[i] - m);
                sum += work[i];
            }
            float inv = 1.0f / sum;
            for (uint i = 0; i < args.n_experts; ++i) work[i] *= inv;

            for (uint k = 0; k < args.top_k; ++k) {
                uint best_idx = 0;
                float best_val = -INFINITY;
                for (uint i = 0; i < args.n_experts; ++i) {
                    bool finite_pair = isfinite(best_val) && isfinite(work[i]);
                    bool tied = finite_pair
                        && abs(work[i] - best_val) <= args.tie_epsilon;
                    if ((work[i] > best_val && !tied) || (tied && i < best_idx)) {
                        best_val = work[i];
                        best_idx = i;
                    }
                }
                expert_ids[(uint64_t)gid * args.top_k + k] = best_idx;
                weights[(uint64_t)gid * args.top_k + k]    = best_val;
                work[best_idx] = -INFINITY;
            }
            // Optional top-k re-normalization (norm_topk_prob). Matches
            // qwen_complete_normalize_route_weights left-to-right sum/scale.
            if (args.normalize_topk != 0u) {
                device float* w = weights + (uint64_t)gid * args.top_k;
                float sum = 0.0f;
                for (uint i = 0u; i < args.top_k; ++i) sum += w[i];
                if (!isfinite(sum) || sum <= 0.0f) {
                    for (uint i = 0u; i < args.top_k; ++i) w[i] = NAN;
                } else {
                    const float inv_w = 1.0f / sum;
                    for (uint i = 0u; i < args.top_k; ++i) w[i] *= inv_w;
                }
            }
        }
        return;
    }

    // ── Parallel path: tie_epsilon == 0 ─────────────────────────────────────
    // Stable softmax max (associative; bit-identical for finite floats).
    {
        float local = -INFINITY;
        for (uint i = tid; i < args.n_experts; i += tg_size) {
            local = max(local, work[i]);
        }
        red_val[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) red_val[tid] = max(red_val[tid], red_val[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    float m = red_val[0];

    // Cooperative exp(x - m). The sum is a serial left-fold on tid 0 so the
    // inverse scale is bit-identical to the historical single-thread softmax
    // (FP add is not associative; a tree sum would drift weights and can flip
    // downstream tokens even when top-k *ids* match). Exp + scale still run
    // across the threadgroup; only the O(n) add chain is serial.
    for (uint i = tid; i < args.n_experts; i += tg_size) {
        work[i] = exp(work[i] - m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < args.n_experts; ++i) sum += work[i];
        red_val[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = red_val[0];
    for (uint i = tid; i < args.n_experts; i += tg_size) {
        work[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-K: k passes of parallel lexicographic (value, -index) max.
    // Mask each winner to -INFINITY between passes (total order still holds
    // over the remaining experts; exact ties keep the lowest index).
    for (uint k = 0; k < args.top_k; ++k) {
        float best_val = -INFINITY;
        uint  best_idx = 0xFFFFFFFFu;
        for (uint i = tid; i < args.n_experts; i += tg_size) {
            float v = work[i];
            if (moe_topk_lex_prefer(v, i, best_val, best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }
        red_val[tid] = best_val;
        red_idx[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float ov = red_val[tid + stride];
                uint  oi = red_idx[tid + stride];
                if (moe_topk_lex_prefer(ov, oi, red_val[tid], red_idx[tid])) {
                    red_val[tid] = ov;
                    red_idx[tid] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            uint win = red_idx[0];
            expert_ids[(uint64_t)gid * args.top_k + k] = win;
            weights[(uint64_t)gid * args.top_k + k]    = red_val[0];
            work[win] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Optional top-k re-normalization (norm_topk_prob). Matches
    // qwen_complete_normalize_route_weights left-to-right sum/scale.
    if (args.normalize_topk != 0u && tid == 0u) {
        device float* w = weights + (uint64_t)gid * args.top_k;
        float sum = 0.0f;
        for (uint i = 0u; i < args.top_k; ++i) sum += w[i];
        if (!isfinite(sum) || sum <= 0.0f) {
            for (uint i = 0u; i < args.top_k; ++i) w[i] = NAN;
        } else {
            const float inv_w = 1.0f / sum;
            for (uint i = 0u; i < args.top_k; ++i) w[i] *= inv_w;
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

// ── moe_batched_gemm_q4_indexed_v2t_gu_v3 ────────────────────────────────────
// v2t_gu_v2 with paired routes per threadgroup.  Two routed experts share the
// cooperative activation preload, so x is fetched from device memory once for
// both routes instead of once per route.  Each route still owns eight
// simdgroups (eight output rows), preserving the v2t_gu_v2 arithmetic and
// output layout.  Odd route counts use the same barrier-safe early return.
// Grid: (ceil(rows/8)*512, ceil(routes/2), 1)   TG: (512, 1, 1)
// shmem: cols*4 bytes.
kernel void moe_batched_gemm_q4_indexed_v2t_gu_v3(
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
    // One cooperative preload is shared by both routes in this threadgroup.
    for (uint i = tid; i < cols; i += 512u) x_cache[i] = x[(uint64_t)i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint route_in_pair = simd_id / 8u;
    uint row_simd       = simd_id & 7u;
    uint route           = tgp.y * 2u + route_in_pair;
    uint base_row        = tgp.x * 8u + row_simd;
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
        y_act[(uint64_t)route * (uint64_t)rows + (uint64_t)base_row] = silu * up_val;
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

// K5/K6 diagnostic: combine the routed/shared expert accumulation directly
// into the residual stream. The result is mathematically identical to
// `moe_route_accumulate` followed by the next layer's `add_inplace`, but avoids
// an intermediate hidden-width write/read. The runtime keeps this opt-in so
// the ordinary source-preserving graph remains byte-for-byte unchanged.
kernel void moe_route_accumulate_add(
    device const float* routed_out  [[buffer(0)]],   // (routes, hidden)
    device const float* weights     [[buffer(1)]],   // (routes)
    device const float* shared_out  [[buffer(2)]],   // (hidden) when has_shared=1
    device       float* residual    [[buffer(3)]],   // (hidden), updated in place
    constant ArgbufRouteAcc& args   [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= args.hidden) return;
    float acc = residual[id];
    if (args.has_shared != 0u) acc += shared_out[id];
    for (uint r = 0; r < args.routes; ++r) {
        acc += weights[r] * routed_out[(uint64_t)r * args.hidden + id];
    }
    residual[id] = acc;
}

// Mixtral K6 bounded path: top-2 expert outputs remain in their persistent
// device buffers and are combined directly into the residual stream.  The
// scalar route weights are deliberately supplied by the already-authoritative
// CPU router; this kernel changes only where the weighted sum and residual add
// execute, not route selection or accumulation order.
kernel void moe_route_accumulate_two_add(
    device const float* routed_out0 [[buffer(0)]],
    device const float* routed_out1 [[buffer(1)]],
    constant float& weight0         [[buffer(2)]],
    constant float& weight1         [[buffer(3)]],
    device       float* residual    [[buffer(4)]],
    constant uint& hidden           [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    residual[id] += weight0 * routed_out0[id] + weight1 * routed_out1[id];
}

// ── DeepSeek-V4 P5B bounded MoE device-boundary authority ────────────────
//
// These symbols are deliberately isolated from the generic MoE graph above.
// They implement only the source-storage boundaries needed by the bounded
// layer-0 P5B receipt: BF16 SwiGLU (with the V4 clamps and optional route
// factor), source-layout FP4 QAT matvec, and the routed+shared BF16 combine.
// No Engine, token loop, or HCLI path selects them.
//
// `moe.metal` is concatenated before `matmul.metal`, so this family owns
// private helper names rather than relying on later DeepSeek component helpers.

static inline float deepseek_v4_p5b_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

static inline ushort deepseek_v4_p5b_bf16_encode_rne(float value)
{
    const uint bits = as_type<uint>(value);
    const uint low_lsb = (bits >> 16u) & 1u;
    return (ushort)((bits + 0x7fffu + low_lsb) >> 16u);
}

static inline float deepseek_v4_p5b_e4m3fn_value(uchar bits)
{
    const uint raw = (uint)bits;
    const uint exponent = (raw >> 3u) & 0x0fu;
    const uint mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) return 0.0f;
    const float magnitude = exponent == 0u
        ? (float)mantissa * 0.001953125f
        : as_type<float>(((exponent + 120u) << 23u) | (mantissa << 20u));
    return (raw & 0x80u) != 0u ? -magnitude : magnitude;
}

static inline float deepseek_v4_p5b_e8m0fnu_value(uchar bits)
{
    if ((uint)bits == 0xffu) return 0.0f;
    return (uint)bits == 0u
        ? as_type<float>(0x00400000u)
        : as_type<float>(((uint)bits) << 23u);
}

static inline float deepseek_v4_p5b_e2m1fn_value(uchar packed, bool high_nibble)
{
    const uint nibble = high_nibble ? (((uint)packed >> 4u) & 0x0fu)
                                     : ((uint)packed & 0x0fu);
    float magnitude = 0.0f;
    switch (nibble & 0x07u) {
        case 0u: magnitude = 0.0f; break;
        case 1u: magnitude = 0.5f; break;
        case 2u: magnitude = 1.0f; break;
        case 3u: magnitude = 1.5f; break;
        case 4u: magnitude = 2.0f; break;
        case 5u: magnitude = 3.0f; break;
        case 6u: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
    }
    return (nibble & 0x08u) != 0u ? -magnitude : magnitude;
}

static inline float deepseek_v4_p5b_silu(float value)
{
    // Match the source-oracle's overflow-avoiding formulation rather than a
    // generic expression whose large-negative intermediate may differ.
    if (value >= 0.0f) return value / (1.0f + exp(-value));
    const float e = exp(value);
    return value * e / (1.0f + e);
}

// Source Expert.forward storage boundary:
// BF16 W1/W3 outputs -> clamp -> SiLU*up -> optional route factor -> BF16.
// `route_weight` is 1.0 for the shared expert and the selected source f32
// route weight for the routed expert.  It is intentionally applied before W2.
kernel void deepseek_v4_p5b_swiglu_route_bf16_authority(
    device const ushort* gate_bf16 [[buffer(0)]],
    device const ushort* up_bf16   [[buffer(1)]],
    device       ushort* output_bf16 [[buffer(2)]],
    constant float& route_weight [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float gate = min(deepseek_v4_p5b_bf16_value(gate_bf16[index]), 10.0f);
    const float up = clamp(deepseek_v4_p5b_bf16_value(up_bf16[index]), -10.0f, 10.0f);
    output_bf16[index] = deepseek_v4_p5b_bf16_encode_rne(
        deepseek_v4_p5b_silu(gate) * up * route_weight);
}

// Source-native FP4 `fp4_gemm` shape after device QAT.  The accumulation is
// explicitly 32-K block-local, then scaled by the corresponding 128-K
// activation E8M0 and 32-K weight E8M0 values, matching the CPU source
// authority rather than the older F32-x FP4 component kernel's association.
#pragma clang fp contract(off)
kernel void deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* weight_scales  [[buffer(1)]],
    device const uchar* quantized      [[buffer(2)]],
    device const uchar* act_scales     [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& packed_cols          [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kFp4Block = 32u;
    constexpr uint kActBlock = 128u;
    if (row >= rows || packed_cols == 0u || scale_cols == 0u
        || packed_cols * 2u != scale_cols * kFp4Block) return;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * kFp4Block;
        for (uint offset = 0u; offset < kFp4Block; ++offset) {
            const uint col = start + offset;
            const uchar packed = packed_weights[weight_base + (ulong)(col >> 1u)];
            const float activation = deepseek_v4_p5b_e4m3fn_value(quantized[col]);
            const float weight = deepseek_v4_p5b_e2m1fn_value(packed, (col & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = deepseek_v4_p5b_e8m0fnu_value(
            act_scales[block / (kActBlock / kFp4Block)]);
        const float weight_scale = deepseek_v4_p5b_e8m0fnu_value(
            weight_scales[scale_base + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[row] = row_accumulator;
}
#pragma clang fp contract(on)

// The source MoE loop adds routed and shared BF16 W2 results in F32 before it
// casts the combined result back to the current BF16 dtype.  P5B has exactly
// one routed expert, whose source route weight was already applied before W2.
kernel void deepseek_v4_p5b_route_shared_combine_bf16_authority(
    device const ushort* routed_bf16 [[buffer(0)]],
    device const ushort* shared_bf16 [[buffer(1)]],
    device       ushort* output_bf16 [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float value = deepseek_v4_p5b_bf16_value(routed_bf16[index])
        + deepseek_v4_p5b_bf16_value(shared_bf16[index]);
    output_bf16[index] = deepseek_v4_p5b_bf16_encode_rne(value);
}

// ── DeepSeek-V4 P6A bounded six-expert wave authority ────────────────────
//
// This extension deliberately remains a layer-0, fixed-predecessor probe.
// Unlike P5B, Gate scores, the complete hash tid2eid lookup, gathered score
// normalization, and all six routed expert weights are device-resident.  The
// caller must still establish that its predecessor BF16 vector is legitimate;
// these kernels are not registered in a decoder or an HCLI path.

// Source Gate: one serial F32 row reduction over BF16 storage.  Keeping each
// row scalar makes the reduction order explicit and comparable to the CPU
// source transcription used by the P6A receipt.
#pragma clang fp contract(off)
kernel void deepseek_v4_p6a_gate_bf16_matvec_authority(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows || cols == 0u) return;
    float accumulator = 0.0f;
    const ulong base = (ulong)row * (ulong)cols;
    for (uint col = 0u; col < cols; ++col) {
        accumulator = accumulator
            + deepseek_v4_p5b_bf16_value(input_bf16[col])
            * deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]);
    }
    logits_f32[row] = accumulator;
}
#pragma clang fp contract(on)

// ── Isolated P0 Gate reduction candidates ──────────────────────────────────
//
// These are named, bounded diagnostic candidates for the frozen P0
// BF16[4096] Gate input. The associated sweep runner stages only the admitted
// Gate matrix and that frozen input, then compares the resulting F32[256]
// logits with a separately captured Torch F.linear calibration and an
// independent FP64 authority. C1-C3 and C5-C7 remain sweep-only. C4 alone is
// selected by the bounded reusable layer-0 P6 executor after its isolated
// admission; it remains outside an Engine, causal loop, HCLI endpoint, and
// TPS path. Keeping the original P6A authority kernel intact preserves the
// frozen baseline/trace control for future comparison.
//
// C1: preserve the existing scalar column order while making the fused
// multiply-add explicit.  This isolates product/add rounding from reduction
// association.
#pragma clang fp contract(off)
kernel void deepseek_v4_p0_gate_reduction_c1_serial_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows || cols == 0u) return;
    const ulong base = (ulong)row * (ulong)cols;
    float accumulator = 0.0f;
    for (uint col = 0u; col < cols; ++col) {
        const float activation = deepseek_v4_p5b_bf16_value(input_bf16[col]);
        const float weight = deepseek_v4_p5b_bf16_value(
            gate_weight_bf16[base + (ulong)col]);
        accumulator = metal::precise::fma(activation, weight, accumulator);
    }
    logits_f32[row] = accumulator;
}

// C2: four fixed, interleaved F32 FMA accumulators followed by an explicitly
// ordered fold.  The layout mirrors the scalar lanes of a width-four CPU SIMD
// dot product without assuming that any particular host BLAS implementation
// uses this exact microkernel.
kernel void deepseek_v4_p0_gate_reduction_c2_strided4_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kLanes = 4u;
    if (row >= rows || cols == 0u || (cols % kLanes) != 0u) return;
    const ulong base = (ulong)row * (ulong)cols;
    float lane0 = 0.0f;
    float lane1 = 0.0f;
    float lane2 = 0.0f;
    float lane3 = 0.0f;
    for (uint col = 0u; col < cols; col += kLanes) {
        lane0 = metal::precise::fma(
            deepseek_v4_p5b_bf16_value(input_bf16[col]),
            deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]), lane0);
        lane1 = metal::precise::fma(
            deepseek_v4_p5b_bf16_value(input_bf16[col + 1u]),
            deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col + 1ul]), lane1);
        lane2 = metal::precise::fma(
            deepseek_v4_p5b_bf16_value(input_bf16[col + 2u]),
            deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col + 2ul]), lane2);
        lane3 = metal::precise::fma(
            deepseek_v4_p5b_bf16_value(input_bf16[col + 3u]),
            deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col + 3ul]), lane3);
    }
    float accumulator = metal::precise::fma(1.0f, lane0, 0.0f);
    accumulator = metal::precise::fma(1.0f, lane1, accumulator);
    accumulator = metal::precise::fma(1.0f, lane2, accumulator);
    accumulator = metal::precise::fma(1.0f, lane3, accumulator);
    logits_f32[row] = accumulator;
}

// C3: thirty-two contiguous K=128 partials, each accumulated by FMA, then a
// deterministic increasing-block fold.  This is a K-block association probe;
// it intentionally remains one device thread per Gate row so its final fold
// has no hardware-defined reduction tree.
kernel void deepseek_v4_p0_gate_reduction_c3_block128_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    constexpr uint kBlocks = 32u;
    if (row >= rows || cols != kBlock * kBlocks) return;
    const ulong base = (ulong)row * (ulong)cols;
    float partials[kBlocks];
    for (uint block = 0u; block < kBlocks; ++block) {
        float partial = 0.0f;
        const uint start = block * kBlock;
        for (uint offset = 0u; offset < kBlock; ++offset) {
            const uint col = start + offset;
            partial = metal::precise::fma(
                deepseek_v4_p5b_bf16_value(input_bf16[col]),
                deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]), partial);
        }
        partials[block] = partial;
    }
    float accumulator = 0.0f;
    for (uint block = 0u; block < kBlocks; ++block) {
        accumulator = metal::precise::fma(1.0f, partials[block], accumulator);
    }
    logits_f32[row] = accumulator;
}

// C5: thirty-two fixed, interleaved FMA accumulators followed by an ordered
// lane fold.  This is the deterministic scalar counterpart to the C4
// SIMDgroup layout: it keeps the same i, i+32, ... input ownership but makes
// the final association explicit rather than delegating it to simd_sum.
kernel void deepseek_v4_p0_gate_reduction_c5_strided32_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kLanes = 32u;
    if (row >= rows || cols == 0u || (cols % kLanes) != 0u) return;
    const ulong base = (ulong)row * (ulong)cols;
    float partials[kLanes];
    for (uint lane = 0u; lane < kLanes; ++lane) {
        float partial = 0.0f;
        for (uint col = lane; col < cols; col += kLanes) {
            partial = metal::precise::fma(
                deepseek_v4_p5b_bf16_value(input_bf16[col]),
                deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]), partial);
        }
        partials[lane] = partial;
    }
    float accumulator = 0.0f;
    for (uint lane = 0u; lane < kLanes; ++lane) {
        accumulator = metal::precise::fma(1.0f, partials[lane], accumulator);
    }
    logits_f32[row] = accumulator;
}

// C6: sixteen contiguous K=256 FMA partials and a deterministic ordered
// fold.  The host-only association screen selected this compact blocked shape
// as a strong source-target candidate; the real-Metal sweep remains the only
// admissible device evidence.
kernel void deepseek_v4_p0_gate_reduction_c6_block256x16_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kBlock = 256u;
    constexpr uint kBlocks = 16u;
    if (row >= rows || cols != kBlock * kBlocks) return;
    const ulong base = (ulong)row * (ulong)cols;
    float partials[kBlocks];
    for (uint block = 0u; block < kBlocks; ++block) {
        float partial = 0.0f;
        const uint start = block * kBlock;
        for (uint offset = 0u; offset < kBlock; ++offset) {
            const uint col = start + offset;
            partial = metal::precise::fma(
                deepseek_v4_p5b_bf16_value(input_bf16[col]),
                deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]), partial);
        }
        partials[block] = partial;
    }
    float accumulator = 0.0f;
    for (uint block = 0u; block < kBlocks; ++block) {
        accumulator = metal::precise::fma(1.0f, partials[block], accumulator);
    }
    logits_f32[row] = accumulator;
}

// C7: sixty-four contiguous K=64 FMA partials and a deterministic ordered
// fold.  It tests the more finely blocked candidate found by the host-only
// screen without changing any execution-path authority.
kernel void deepseek_v4_p0_gate_reduction_c7_block64x64_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kBlock = 64u;
    constexpr uint kBlocks = 64u;
    if (row >= rows || cols != kBlock * kBlocks) return;
    const ulong base = (ulong)row * (ulong)cols;
    float partials[kBlocks];
    for (uint block = 0u; block < kBlocks; ++block) {
        float partial = 0.0f;
        const uint start = block * kBlock;
        for (uint offset = 0u; offset < kBlock; ++offset) {
            const uint col = start + offset;
            partial = metal::precise::fma(
                deepseek_v4_p5b_bf16_value(input_bf16[col]),
                deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]), partial);
        }
        partials[block] = partial;
    }
    float accumulator = 0.0f;
    for (uint block = 0u; block < kBlocks; ++block) {
        accumulator = metal::precise::fma(1.0f, partials[block], accumulator);
    }
    logits_f32[row] = accumulator;
}

// C4: one 32-lane SIMDgroup per Gate row.  Lane i accumulates the fixed
// strided column sequence i, i+32, ... with precise FMA; simd_sum then applies
// Metal's hardware reduction tree.  This is the only candidate that permits a
// hardware-defined final association, so the host admits it only when the
// pipeline reports a 32-thread execution width and records it separately.
kernel void deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    constexpr uint kSimdWidth = 32u;
    if (row >= rows || cols == 0u || tg_size != kSimdWidth) return;
    const ulong base = (ulong)row * (ulong)cols;
    float partial = 0.0f;
    for (uint col = lane; col < cols; col += kSimdWidth) {
        partial = metal::precise::fma(
            deepseek_v4_p5b_bf16_value(input_bf16[col]),
            deepseek_v4_p5b_bf16_value(gate_weight_bf16[base + (ulong)col]), partial);
    }
    const float total = simd_sum(partial);
    if (lane == 0u) logits_f32[row] = total;
}
#pragma clang fp contract(on)

// Source hash route: score every Gate logit with thresholded sqrt-softplus,
// gather `tid2eid[token_id]` from the native I64 table represented as LE u32
// word pairs, then normalize only the six gathered *unbiased* scores before
// applying route_scale.  A single device thread preserves the source slot and
// summation order and exposes a validity bit instead of silently accepting an
// invalid I64, non-finite score, or zero normalization sum.

// The host source uses `ln_1p`, which retains a positive sub-ULP score for
// very negative Gate logits.  Metal's portable `log(1.0f + u)` rounds that
// sum to one for small positive `u`, silently turning a valid source route
// score into zero.  The source branch only supplies u in [0, 1], so retain the
// first terms of ln(1+u) below 1e-4 (the cubic term is below one f32 ULP at
// the cutoff) and use the normal logarithm outside the cancellation region.
static inline float deepseek_v4_p6a_log1p_source_stable(float u)
{
    if (u < 0.0001f) {
        return u - 0.5f * u * u;
    }
    return metal::precise::log(1.0f + u);
}

#pragma clang fp contract(off)
kernel void deepseek_v4_p6a_hash_route_sqrtsoftplus_authority(
    device const float* logits_f32          [[buffer(0)]],
    device const uint* tid2eid_i64_le_words [[buffer(1)]],
    device       uint* selected_ids         [[buffer(2)]],
    device       float* selected_weights    [[buffer(3)]],
    device       float* original_scores     [[buffer(4)]],
    device       uint* valid                [[buffer(5)]],
    constant uint& token_id                 [[buffer(6)]],
    constant uint& expert_count             [[buffer(7)]],
    constant uint& top_k                    [[buffer(8)]],
    constant float& route_scale             [[buffer(9)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0u) return;
    valid[0] = 0u;
    if (expert_count == 0u || top_k != 6u || !isfinite(route_scale)) {
        valid[0] = 2u;
        return;
    }

    for (uint expert = 0u; expert < expert_count; ++expert) {
        const float logit = logits_f32[expert];
        if (!isfinite(logit)) {
            valid[0] = 16u + expert;
            return;
        }
        const float softplus = logit > 20.0f
            ? logit
            : (logit >= 0.0f
                ? logit + deepseek_v4_p6a_log1p_source_stable(metal::precise::exp(-logit))
                : deepseek_v4_p6a_log1p_source_stable(metal::precise::exp(logit)));
        const float score = metal::precise::sqrt(softplus);
        if (!(isfinite(score) && score > 0.0f)) {
            valid[0] = 512u + expert;
            return;
        }
        original_scores[expert] = score;
    }

    const ulong row_base = (ulong)token_id * (ulong)top_k * 2ul;
    float sum = 0.0f;
    for (uint slot = 0u; slot < top_k; ++slot) {
        const ulong word = row_base + (ulong)slot * 2ul;
        const uint lo = tid2eid_i64_le_words[word];
        const uint hi = tid2eid_i64_le_words[word + 1ul];
        // Valid source I64 experts are nonnegative and fit in the configured
        // routed-expert range, so the high word must be zero.
        if (hi != 0u || lo >= expert_count) {
            valid[0] = 1024u + slot;
            return;
        }
        selected_ids[slot] = lo;
        const float score = original_scores[lo];
        selected_weights[slot] = score;
        sum = sum + score;
    }
    if (!(isfinite(sum) && sum > 0.0f)) {
        valid[0] = 1536u;
        return;
    }
    for (uint slot = 0u; slot < top_k; ++slot) {
        const float weight = (selected_weights[slot] / sum) * route_scale;
        if (!isfinite(weight)) {
            valid[0] = 1792u + slot;
            return;
        }
        selected_weights[slot] = weight;
    }
    valid[0] = 1u;
}
#pragma clang fp contract(on)

// Learned-bias Gate route (layers >= n_hash_layers):
//   original_scores = sqrt(softplus(logits))
//   selection_scores = original_scores + bias
//   indices = topk(selection_scores, k=6)  // stable: higher score wins; ties keep lower expert id
//   weights = original_scores[indices]; weights /= sum; weights *= route_scale
// Bias affects selection only, never the gathered route weights (source Gate.forward).
#pragma clang fp contract(off)
kernel void deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority(
    device const float* logits_f32          [[buffer(0)]],
    device const float* bias_f32            [[buffer(1)]],
    device       uint* selected_ids         [[buffer(2)]],
    device       float* selected_weights    [[buffer(3)]],
    device       float* original_scores     [[buffer(4)]],
    device       uint* valid                [[buffer(5)]],
    constant uint& expert_count             [[buffer(6)]],
    constant uint& top_k                    [[buffer(7)]],
    constant float& route_scale             [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0u) return;
    valid[0] = 0u;
    if (expert_count == 0u || expert_count > 256u || top_k != 6u || !isfinite(route_scale)) {
        valid[0] = 2u;
        return;
    }

    for (uint expert = 0u; expert < expert_count; ++expert) {
        const float logit = logits_f32[expert];
        if (!isfinite(logit)) {
            valid[0] = 16u + expert;
            return;
        }
        const float softplus = logit > 20.0f
            ? logit
            : (logit >= 0.0f
                ? logit + deepseek_v4_p6a_log1p_source_stable(metal::precise::exp(-logit))
                : deepseek_v4_p6a_log1p_source_stable(metal::precise::exp(logit)));
        const float score = metal::precise::sqrt(softplus);
        if (!(isfinite(score) && score > 0.0f)) {
            valid[0] = 512u + expert;
            return;
        }
        original_scores[expert] = score;
    }

    // Serial top-k on selection scores = original + bias. Deterministic:
    // prefer higher score; on exact ties prefer the lower expert id.
    for (uint slot = 0u; slot < top_k; ++slot) {
        uint best_id = 0xffffffffu;
        float best_score = -INFINITY;
        for (uint expert = 0u; expert < expert_count; ++expert) {
            bool already = false;
            for (uint prev = 0u; prev < slot; ++prev) {
                if (selected_ids[prev] == expert) {
                    already = true;
                    break;
                }
            }
            if (already) continue;
            const float bias = bias_f32[expert];
            if (!isfinite(bias)) {
                valid[0] = 768u + expert;
                return;
            }
            const float sel = original_scores[expert] + bias;
            if (!isfinite(sel)) {
                valid[0] = 896u + expert;
                return;
            }
            if (best_id == 0xffffffffu
                || sel > best_score
                || (sel == best_score && expert < best_id)) {
                best_score = sel;
                best_id = expert;
            }
        }
        if (best_id == 0xffffffffu || best_id >= expert_count) {
            valid[0] = 1024u + slot;
            return;
        }
        selected_ids[slot] = best_id;
    }

    float sum = 0.0f;
    for (uint slot = 0u; slot < top_k; ++slot) {
        const float score = original_scores[selected_ids[slot]];
        selected_weights[slot] = score;
        sum = sum + score;
    }
    if (!(isfinite(sum) && sum > 0.0f)) {
        valid[0] = 1536u;
        return;
    }
    for (uint slot = 0u; slot < top_k; ++slot) {
        const float weight = (selected_weights[slot] / sum) * route_scale;
        if (!isfinite(weight)) {
            valid[0] = 1792u + slot;
            return;
        }
        selected_weights[slot] = weight;
    }
    valid[0] = 1u;
}
#pragma clang fp contract(on)

// Device-route-weighted source SwiGLU.  `route_slot` indexes the device
// result above; the host does not supply the floating route factor.  Each
// expert wave uses a distinct output allocation, so these invocations are
// safe to put in an explicitly concurrent command encoder.
kernel void deepseek_v4_p6a_swiglu_route_weight_buffer_bf16_authority(
    device const ushort* gate_bf16      [[buffer(0)]],
    device const ushort* up_bf16        [[buffer(1)]],
    device       ushort* output_bf16    [[buffer(2)]],
    device const float* route_weights   [[buffer(3)]],
    constant uint& route_slot           [[buffer(4)]],
    constant uint& count                [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float route_weight = route_weights[route_slot];
    const float gate = min(deepseek_v4_p5b_bf16_value(gate_bf16[index]), 10.0f);
    const float up = clamp(deepseek_v4_p5b_bf16_value(up_bf16[index]), -10.0f, 10.0f);
    output_bf16[index] = deepseek_v4_p5b_bf16_encode_rne(
        deepseek_v4_p5b_silu(gate) * up * route_weight);
}

// Exact source-loop combine for six routed outputs stored in ascending
// numeric-expert order plus the always-on shared output.  The sequential
// statements intentionally retain the `y = 0; y += expert_i; y += shared`
// association from the source model instead of collapsing the sum.
#pragma clang fp contract(off)
kernel void deepseek_v4_p6a_route6_shared_combine_bf16_authority(
    device const ushort* routed_0_bf16 [[buffer(0)]],
    device const ushort* routed_1_bf16 [[buffer(1)]],
    device const ushort* routed_2_bf16 [[buffer(2)]],
    device const ushort* routed_3_bf16 [[buffer(3)]],
    device const ushort* routed_4_bf16 [[buffer(4)]],
    device const ushort* routed_5_bf16 [[buffer(5)]],
    device const ushort* shared_bf16    [[buffer(6)]],
    device       ushort* output_bf16    [[buffer(7)]],
    constant uint& count                 [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    float value = 0.0f;
    value = value + deepseek_v4_p5b_bf16_value(routed_0_bf16[index]);
    value = value + deepseek_v4_p5b_bf16_value(routed_1_bf16[index]);
    value = value + deepseek_v4_p5b_bf16_value(routed_2_bf16[index]);
    value = value + deepseek_v4_p5b_bf16_value(routed_3_bf16[index]);
    value = value + deepseek_v4_p5b_bf16_value(routed_4_bf16[index]);
    value = value + deepseek_v4_p5b_bf16_value(routed_5_bf16[index]);
    value = value + deepseek_v4_p5b_bf16_value(shared_bf16[index]);
    output_bf16[index] = deepseek_v4_p5b_bf16_encode_rne(value);
}
#pragma clang fp contract(on)

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
