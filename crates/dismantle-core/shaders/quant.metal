// quant.metal — quantization formats.
//
// Kernels:
//   dequant_q4_k_m        — standalone Q4_K_M → fp16 dequant. Used by
//                           the Phase 0 reference path and by tests.
//                           [Phase 0]
//   dequant_q5_k_m        — Q5_K_M → fp16.
//                           [Phase 1]
//   dequant_q8_0          — Q8_0 → fp16.
//                           [Phase 1]
//   gemm_q4_k_m_fused     — quant-aware GEMM: weights stay 4-bit in
//                           DRAM, dequantized inside threadgroup memory
//                           on the fly. The dense-GEMM equivalent of the
//                           moe_grouped_gemm_q4 kernel.
//                           [Phase 1, wedge 2]

#include <metal_stdlib>
using namespace metal;

static inline float q3_k_fp16_at(device const uchar* p, uint64_t off)
{
    ushort bits = (ushort)p[off] | ((ushort)p[off + 1] << 8);
    return (float)as_type<half>(bits);
}

static inline int q3_k_scale(device const uchar* w_q3, uint64_t bo, uint scale_idx)
{
    uint low;
    if (scale_idx < 8u) {
        low = (uint)w_q3[bo + 96ul + (uint64_t)scale_idx] & 0x0Fu;
    } else {
        low = ((uint)w_q3[bo + 96ul + (uint64_t)(scale_idx - 8u)] >> 4) & 0x0Fu;
    }
    uint high = ((uint)w_q3[bo + 104ul + (uint64_t)(scale_idx & 3u)]
              >> (2u * (scale_idx >> 2))) & 0x03u;
    return (int)(low | (high << 4)) - 32;
}

static inline float q3_k_value(device const uchar* w_q3, uint64_t bo, uint c)
{
    float d = q3_k_fp16_at(w_q3, bo + 108ul);
    uint half_idx = c >> 7;
    uint local = c & 127u;
    uint group16 = local >> 4;
    uint j = group16 >> 1;
    uint second = group16 & 1u;
    uint lane = local & 15u;

    uint q_idx = half_idx * 32u + second * 16u + lane;
    uint h_idx = second * 16u + lane;
    uint shift = j * 2u;
    uint high_mask = 1u << (half_idx * 4u + j);
    int q = (int)(((uint)w_q3[bo + 32ul + (uint64_t)q_idx] >> shift) & 0x03u)
          - (((uint)w_q3[bo + (uint64_t)h_idx] & high_mask) != 0u ? 0 : 4);
    int scale = q3_k_scale(w_q3, bo, half_idx * 8u + group16);
    return d * (float)scale * (float)q;
}

// One thread = one Q8_0 block (32 elems). Cheap; called per-tensor at
// most once when materializing reference fp16 weights.
kernel void dequant_q8_0(
    device const uchar* src    [[buffer(0)]],
    device       half*  dst    [[buffer(1)]],
    constant     uint&  nblock [[buffer(2)]],
    uint                bid    [[thread_position_in_grid]])
{
    if (bid >= nblock) return;
    uint off = bid * 34;
    half d = as_type<half>((ushort)(src[off] | (uint(src[off + 1]) << 8)));
    uint dst_off = bid * 32;
    for (uint i = 0; i < 32; ++i) {
        char q = (char)src[off + 2 + i];
        dst[dst_off + i] = half((float)d * (float)q);
    }
}

// H2.4 — fp32 GEMV with Q4_K_M weights, dequant fused inside the FMA loop.
// Dense-GEMM counterpart to `moe_grouped_gemm_q4` in moe.metal — same
// kernel body, lives here in quant.metal so the dense path doesn't pull
// the MoE module. See moe.metal's `moe_grouped_gemm_q4` for the
// per-thread Q4_K_M index derivation; the body below is byte-identical.
//
// One workgroup per output row; tg_size MUST be 256 (Q4_K_M super-block
// size). cols must be a multiple of 256.
kernel void gemm_q4_k_m_fused(
    device const uchar* w_q4   [[buffer(0)]],   // (rows, cols) Q4_K_M
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
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

        float w_val = d * (float)s_byte * (float)nib
                    - dmin * (float)m_byte;

        float xv = x[(uint64_t)b * 256ul + (uint64_t)tid];
        partial += w_val * xv;
    }

    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) y[gid] = shmem[0];
}

// ── gemm_q4_k_m_fused_simd ───────────────────────────────────────────────────
// simdgroup_matrix variant of gemm_q4_k_m_fused.  Uses Apple Silicon
// matrix-multiply hardware via simdgroup_matrix<float,8,8>.
//
// One simdgroup (32 threads) per threadgroup; each threadgroup computes
// 8 output rows.  The activation vector x is broadcast across 8 columns
// to form an 8×8 matrix tile; the weight 8×8 tile is dequantized from
// Q4_K bytes.  After accumulation, column 0 of the result gives y.
//
// Grid:  (ceil(rows/8)*32, 1, 1)   threadgroup: (32, 1, 1)
// Threadgroup memory layout (each slot is float, stride 8):
//   shmem[ 0.. 64): weight tile  W[8][8]
//   shmem[64..128): activation tile X[8][8]  (broadcast: X[k][n] = x[k] ∀n)
//   shmem[128..192): result tile  D[8][8]    (temp for extract + zero-init)
#include <metal_simdgroup_matrix>

kernel void gemm_q4_k_m_fused_simd(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint                tid    [[thread_position_in_threadgroup]],
    uint                gid    [[threadgroup_position_in_grid]])
{
    uint base_row = gid * 8u;
    if (base_row >= rows) return;

    uint blocks_per_row = cols / 256u;

    threadgroup float* shmem_w   = shmem;        // [64]
    threadgroup float* shmem_x   = shmem + 64;   // [64]
    threadgroup float* shmem_out = shmem + 128;  // [64]

    // Zero-initialise accumulator via shmem_out.
    shmem_out[tid]      = 0.0f;
    shmem_out[tid + 32] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, shmem_out, 8, ulong2(0, 0));

    for (uint b = 0; b < blocks_per_row; ++b) {
        for (uint kt = 0; kt < 32u; ++kt) {

            // ── Fill weight tile shmem_w[8×8] ─────────────────────────────
            // Each of the 32 threads fills 2 elements (indices tid and tid+32).
            for (int e = 0; e < 2; ++e) {
                uint elem = tid + (uint)e * 32u;
                uint m    = elem >> 3u;   // 0..7  — output row within tile
                uint k    = elem &  7u;   // 0..7  — K offset within tile
                uint row  = base_row + m;

                if (row >= rows) {
                    shmem_w[elem] = 0.0f;
                    continue;
                }

                uint kk = kt * 8u + k;   // element index in Q4_K block (0..255)
                uint64_t bo = ((uint64_t)row * (uint64_t)blocks_per_row
                               + (uint64_t)b) * 144ul;

                ushort d_bits    = (ushort)w_q4[bo]
                                 | ((ushort)w_q4[bo + 1] << 8);
                ushort dmin_bits = (ushort)w_q4[bo + 2]
                                 | ((ushort)w_q4[bo + 3] << 8);
                float  d    = (float)as_type<half>(d_bits);
                float  dmin = (float)as_type<half>(dmin_bits);

                uint sub = kk >> 5u;   // sub-group index (0..7)
                uchar s_byte, m_byte;
                if (sub < 4u) {
                    s_byte = w_q4[bo + 4u + sub]      & 0x3Fu;
                    m_byte = w_q4[bo + 4u + 4u + sub] & 0x3Fu;
                } else {
                    uint j = sub - 4u;
                    s_byte = (w_q4[bo + 4u + 8u + j] & 0x0Fu)
                           | ((w_q4[bo + 4u + j]      >> 6u) << 4u);
                    m_byte = (w_q4[bo + 4u + 8u + j] >> 4u)
                           | ((w_q4[bo + 4u + 4u + j] >> 6u) << 4u);
                }

                uint pair  = sub >> 1u;
                bool upper = (sub & 1u) != 0u;
                uint i     = kk & 31u;
                uchar q    = w_q4[bo + 16ul + (uint64_t)pair * 32ul
                                            + (uint64_t)i];
                uint nib   = upper ? ((uint)(q >> 4) & 0x0Fu)
                                   : ((uint)q        & 0x0Fu);

                shmem_w[elem] = d * (float)s_byte * (float)nib
                              - dmin * (float)m_byte;
            }

            // ── Fill activation tile shmem_x[8×8] (broadcast) ────────────
            // X[k][n] = x[b*256 + kt*8 + k] for all n  →  shmem_x[k*8+n] = x[k].
            for (int e = 0; e < 2; ++e) {
                uint elem = tid + (uint)e * 32u;
                uint k    = elem >> 3u;   // K index (0..7)
                shmem_x[elem] = x[(uint64_t)b * 256ul
                                 + (uint64_t)kt * 8ul
                                 + (uint64_t)k];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── simdgroup multiply-accumulate ──────────────────────────────
            simdgroup_matrix<float, 8, 8> w_mat, x_mat;
            simdgroup_load(w_mat, shmem_w, 8, ulong2(0, 0));
            simdgroup_load(x_mat, shmem_x, 8, ulong2(0, 0));
            simdgroup_multiply_accumulate(acc, w_mat, x_mat, acc);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Write results ──────────────────────────────────────────────────────
    simdgroup_store(acc, shmem_out, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Column 0 of acc holds the dot-product result (broadcast makes all columns identical).
    if (tid < 8u && base_row + tid < rows) {
        y[base_row + tid] = shmem_out[tid * 8u];
    }
}

// ── gemm_q4_k_m_fused_v2 ─────────────────────────────────────────────────────
// Correct simd_sum design: 1 simdgroup per output row, 8 rows per threadgroup.
// Zero inner-loop barriers — simd_sum is a warp-level reduction, no shmem needed.
//
// Grid:  (ceil(rows/8)*256, 1, 1)   threadgroup: (256, 1, 1)
// 8 simdgroups of 32 threads each = 256 threads per TG.
// simd_id in [0..8) selects the output row within the TG; simd_lane in [0..32)
// covers 8 elements per Q4_K block (stride 32: elem = k*32 + simd_lane).

kernel void gemm_q4_k_m_fused_v2(
    device const uchar* w_q4   [[buffer(0)]],   // (rows, cols) Q4_K_M
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant ArgbufRowsCols& args [[buffer(3)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[threadgroup_position_in_grid]],
    uint                simd_lane    [[thread_index_in_simdgroup]],
    uint                simd_id      [[simdgroup_index_in_threadgroup]])
{
    // ROWS_PER_TG=8 (one simdgroup per row), TG_SIZE=256 (8 simdgroups).
    uint base_row = gid * 8u + simd_id;
    if (base_row >= args.rows) return;     // tail simdgroups do nothing

    uint  blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;

        ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        // Each thread covers 8 elements in this block via stride 32:
        // elem = k*32 + simd_lane, k in [0..8), simd_lane in [0..32)
        for (uint k = 0; k < 8u; ++k) {
            uint elem = k * 32u + simd_lane;
            uint sub  = elem >> 5;     // 0..7
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

    // 32-thread simdgroup reduction. Zero barriers.
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[base_row] = partial;
    }
}

// ── gemm_q3_k_fused_v2 ──────────────────────────────────────────────────────
// Q3_K GEMV using the same 256-thread / 8-row-per-TG geometry as the Q4_K
// v2 kernel. One simdgroup owns one row; each lane covers eight elements of
// the 256-element Q3_K super-block.

kernel void gemm_q3_k_fused_v2(
    device const uchar* w_q3   [[buffer(0)]],   // (rows, cols) Q3_K
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant ArgbufRowsCols& args [[buffer(3)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[threadgroup_position_in_grid]],
    uint                simd_lane    [[thread_index_in_simdgroup]],
    uint                simd_id      [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= args.rows) return;

    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 110ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 110ul;
        for (uint k = 0; k < 8u; ++k) {
            uint elem = k * 32u + simd_lane;
            float w_val = q3_k_value(w_q3, bo, elem);
            float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];
            partial += w_val * xv;
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[base_row] = partial;
    }
}

// ── gemm_q3_k_fused_2r ───────────────────────────────────────────────────────
// 2-rows-per-simdgroup FUSED Q3_K GEMV (byte-cut speed lever, 2026-05-31).
// Identical INLINE 6-bit scale decode + math as gemm_q3_k_fused_v2 (NO predec —
// predec ADDS scale bytes and breaks the byte-cut), but each simdgroup computes
// TWO output rows of the SAME matrix with two independent accumulator chains,
// sharing the single `x` load. The two chains give the compiler 2 in-flight
// weight-load streams per thread, hiding DRAM latency — the structure that makes
// gemm_q4_k_v4_predec_2r run ~56% peak. 16 rows/TG (8 simdgroups x 2 rows).
//
// BIT-IDENTICAL per-row to gemm_q3_k_fused_v2: each accumulator replays the
// exact same per-element `d*scale*q * xv` FMA in the same order; only the row
// pairing and shared `x` differ. `d` is read once per row per block (was once
// per element via q3_k_value); the value is identical so the product is too.
//
// Grid: (ceil(rows/16)*256, 1, 1)   threadgroup: (256, 1, 1)
kernel void gemm_q3_k_fused_2r(
    device const uchar* w_q3   [[buffer(0)]],   // (rows, cols) Q3_K, 110 B/block
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant ArgbufRowsCols& args [[buffer(3)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[threadgroup_position_in_grid]],
    uint                simd_lane    [[thread_index_in_simdgroup]],
    uint                simd_id      [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 16u + simd_id;
    if (row0 >= args.rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < args.rows;
    // Alias row1 to row0 when past the end so loads stay in-bounds; p1 is never
    // written. Production shapes are rows%16==0 so has1 holds.
    uint r1 = has1 ? row1 : row0;

    uint blocks_per_row = args.cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 110ul;
    uint64_t rb1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 110ul;
    float p0 = 0.0f;
    float p1 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 110ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 110ul;
        // `d` read once per row per block (identical to q3_k_value's per-element
        // read — same f16 → float value).
        float d0 = q3_k_fp16_at(w_q3, bo0 + 108ul);
        float d1 = q3_k_fp16_at(w_q3, bo1 + 108ul);

        for (uint k = 0; k < 8u; ++k) {
            uint elem = k * 32u + simd_lane;
            // Shared element index decode (same for both rows).
            uint half_idx = elem >> 7;
            uint local    = elem & 127u;
            uint group16  = local >> 4;
            uint j        = group16 >> 1;
            uint second   = group16 & 1u;
            uint lane     = local & 15u;
            uint q_idx    = half_idx * 32u + second * 16u + lane;
            uint h_idx    = second * 16u + lane;
            uint shift    = j * 2u;
            uint high_mask = 1u << (half_idx * 4u + j);
            uint scale_idx = half_idx * 8u + group16;

            // Shared activation load.
            float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];

            // Row 0.
            int q0 = (int)(((uint)w_q3[bo0 + 32ul + (uint64_t)q_idx] >> shift) & 0x03u)
                   - (((uint)w_q3[bo0 + (uint64_t)h_idx] & high_mask) != 0u ? 0 : 4);
            int s0 = q3_k_scale(w_q3, bo0, scale_idx);
            p0 += (d0 * (float)s0 * (float)q0) * xv;

            // Row 1.
            int q1 = (int)(((uint)w_q3[bo1 + 32ul + (uint64_t)q_idx] >> shift) & 0x03u)
                   - (((uint)w_q3[bo1 + (uint64_t)h_idx] & high_mask) != 0u ? 0 : 4);
            int s1 = q3_k_scale(w_q3, bo1, scale_idx);
            p1 += (d1 * (float)s1 * (float)q1) * xv;
        }
    }

    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) {
        p1 = simd_sum(p1);
        if (simd_lane == 0u) y[row1] = p1;
    }
}

// ── gemm_q3_k_v4_predec ──────────────────────────────────────────────────────
// Q3_K decode GEMV with pre-decoded sub-block scales (byte-cut Stage 3). This
// is the fast Q3_K GEMV the oracle byte-cut win was blocked on: a Q3_K model
// otherwise runs the generic dequant path (~19 dec_tps vs ~32 on the Q4_K fast
// stack). Same 256-thread / 8-row-per-TG geometry and identical math as
// gemm_q3_k_fused_v2, but the 16 per-sub-block `d * scale[i]` f32 values are
// read from a parallel pre-decoded buffer (matches `predecode_q3_k_scale_table`
// in Rust) instead of unpacking the packed 6-bit scales + super-block `d` on
// every call. Q3_K is symmetric (no min term), so the table is 16 f32/block
// (vs Q4_K v4_predec's 8 ds/dm pairs = 16 f32/block).
//
// Pre-decoded scale layout (one f32 per 16-element sub-block, 16 sub-blocks):
//   scales[block_idx * 16 + sub] = (f32)d * (f32)scale[sub]
//
// Numerically equivalent to gemm_q3_k_fused_v2 within fp16 tolerance (atol 1e-3;
// measured ~1 ULP / ~1e-4). NOT bit-identical: predec loads a pre-rounded
// `d*scale` from the table, whereas the fused kernel computes `d*scale*q` inline
// and the Metal compiler may FMA-contract it without that intermediate f32 round.
// The pre-decode is the optimization; the 1-ULP delta is inherent to it, not a bug.
//
// Grid: (ceil(rows/8)*256, 1, 1)   threadgroup: (256, 1, 1)
kernel void gemm_q3_k_v4_predec(
    device const uchar* w_q3    [[buffer(0)]],   // (rows, cols) Q3_K, 110 B/block
    device const float* scales  [[buffer(1)]],   // (rows * blocks_per_row * 16) f32
    device const float* x       [[buffer(2)]],   // (cols,)
    device       float* y       [[buffer(3)]],   // (rows,)
    constant     uint&  rows    [[buffer(4)]],
    constant     uint&  cols    [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row   = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 110ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 110ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;

        for (uint k = 0; k < 8u; ++k) {
            uint elem     = k * 32u + simd_lane;
            uint half_idx = elem >> 7;        // 0..1: which 128-element half
            uint local    = elem & 127u;
            uint group16  = local >> 4u;      // 0..7: 16-element sub-block in half
            uint j        = group16 >> 1u;    // 0..3
            uint second   = group16 & 1u;
            uint lane     = local & 15u;

            uint q_idx     = half_idx * 32u + second * 16u + lane;
            uint h_idx     = second * 16u + lane;
            uint shift     = j * 2u;
            uint high_mask = 1u << (half_idx * 4u + j);
            int q = (int)(((uint)w_q3[bo + 32ul + (uint64_t)q_idx] >> shift) & 0x03u)
                  - (((uint)w_q3[bo + (uint64_t)h_idx] & high_mask) != 0u ? 0 : 4);

            float dl = scales[so + (uint64_t)(half_idx * 8u + group16)];
            float xv = x[(uint64_t)b * 256ul + (uint64_t)elem];
            partial += dl * (float)q * xv;
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── gemm_q4_k_m_simdmat ──────────────────────────────────────────────────────
// Wedge K — improved Q4_K_M GEMV. Three improvements over gemm_q4_k_m_fused_v2:
//
//   1. Scale pre-load: s_byte[8] and m_byte[8] extracted once per block,
//      before the nibble loop. Compiler can schedule freely.
//   2. Activation pre-load: xl[8] loaded into registers before the nibble
//      loop. Eliminates repeated device-memory reads in the hot path.
//   3. Paired nibble reads: elements k and k+1 share the same qs byte
//      (lower/upper nibbles). One byte read covers two k-iterations → 4
//      byte reads per thread per block instead of 8.
//
// Geometry: 4 simdgroups (128 threads) per TG, 1 row per simdgroup, 4 rows
// per TG. Doubles TG count vs v2 (8 rows/TG); improves GPU parallelism on
// small-row shapes (e.g. rows=1408 expert gate/up).
//
// Buffer layout identical to gemm_q4_k_m_fused_v2. No shmem.
// Grid: (ceil(rows/4)*128, 1, 1)   threadgroup: (128, 1, 1)

kernel void gemm_q4_k_m_simdmat(
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
    uint base_row = gid * 4u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 144ul;

        // Global scale factors (2 × f16 → f32)
        ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        // ── Step 1: Pre-load sub-block scale and min bytes ────────────────
        // 12 scale bytes at bo+4..bo+15 cover all 8 sub-blocks.
        uchar sb[8], mb[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sb[sub] = w_q4[bo + 4u + sub]      & 0x3Fu;
            mb[sub] = w_q4[bo + 4u + 4u + sub] & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sb[4u + j] = (w_q4[bo + 12u + j] & 0x0Fu)
                       | ((w_q4[bo + 4u + j]  >> 6u) << 4u);
            mb[4u + j] = (w_q4[bo + 12u + j]  >> 4u)
                       | ((w_q4[bo + 8u + j]   >> 6u) << 4u);
        }

        // Pre-compute d*s and dmin*m per sub-block
        float ds[8], dm[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds[sub] = d    * (float)sb[sub];
            dm[sub] = dmin * (float)mb[sub];
        }

        // ── Step 2: Pre-load activations into registers ───────────────────
        // elem = k*32 + simd_lane for k = 0..7
        float xl[8];
        for (uint k = 0; k < 8u; ++k) {
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];
        }

        // ── Step 3: Paired nibble reads — 4 byte reads instead of 8 ─────
        // For sub-block pair (2*pi, 2*pi+1):
        //   k_lo = 2*pi: sub=k_lo, pair=pi, upper=false → low  nibble of qs byte
        //   k_hi = 2*pi+1: sub=k_hi, pair=pi, upper=true → high nibble of qs byte
        // Both k_lo and k_hi access the SAME qs byte at bo+16 + pi*32 + simd_lane.
        for (uint pi = 0; pi < 4u; ++pi) {
            uchar qb = w_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uint k0 = pi * 2u;
            uint k1 = k0 + 1u;
            float nib0 = (float)(qb & 0x0Fu);
            float nib1 = (float)(qb >> 4u);
            partial += (ds[k0] * nib0 - dm[k0]) * xl[k0];
            partial += (ds[k1] * nib1 - dm[k1]) * xl[k1];
        }
    }

    // 32-thread simdgroup reduction. Zero barriers.
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[base_row] = partial;
    }
}

// ── gemm_q4_k_m_v3_8r ────────────────────────────────────────────────────────
// Phase B Approach 1 Iter 1: same improvements as simdmat (scale+activation
// preloading, paired nibble reads) but using v2's 8-rows/TG geometry (256 threads,
// 8 simdgroups). Fewer TGs → potentially less scheduling overhead on small shapes.
//
// Grid: (ceil(rows/8)*256, 1, 1)   threadgroup: (256, 1, 1)

kernel void gemm_q4_k_m_v3_8r(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
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

        uchar sb[8], mb[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sb[sub] = w_q4[bo + 4u + sub]      & 0x3Fu;
            mb[sub] = w_q4[bo + 8u + sub]      & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sb[4u + j] = (w_q4[bo + 12u + j] & 0x0Fu)
                       | ((w_q4[bo + 4u + j]  >> 6u) << 4u);
            mb[4u + j] = (w_q4[bo + 12u + j]  >> 4u)
                       | ((w_q4[bo + 8u + j]   >> 6u) << 4u);
        }

        float ds[8], dm[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds[sub] = d    * (float)sb[sub];
            dm[sub] = dmin * (float)mb[sub];
        }

        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uchar qb = w_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uint k0 = pi * 2u, k1 = k0 + 1u;
            partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xl[k0];
            partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xl[k1];
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── quantize_f32_to_int8_per_block ───────────────────────────────────────────
//
// GPU-side quant for the W4A8 path. Reads a length-`n` f32 activation and
// writes per-256-elem int8 + f32 scales using the same formula as the CPU
// reference (`quantize_to_int8_per_block`): scale = max|x|/127 per block,
// q[i] = round(x[i] / scale) clamped to [-127, 127].
//
// Grid:  (n, 1, 1)               (one thread per element)
// TG:    (256, 1, 1)             (one threadgroup per block)
// Shmem: 256 * sizeof(float)     (reduce buffer)
//
// Production wire-up replaces the CPU readback + quantize per layer with
// one GPU dispatch fused into the same TCB as the rmsnorm/gemv pair.
kernel void quantize_f32_to_int8_per_block(
    device const float*       x        [[buffer(0)]],
    device       signed char* x_int8   [[buffer(1)]],
    device       float*       x_scales [[buffer(2)]],
    threadgroup  float*       red      [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint block_off = tg_id * 256u;
    float xv = x[block_off + tid];
    red[tid] = fabs(xv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_abs = red[0];
    // Use metal::precise::divide for IEEE-correct round-to-nearest division,
    // matching the CPU reference. Metal's default `/` is relaxed (fast-math)
    // and produces 1-ULP deltas on some block max values.
    float scale = (max_abs > 0.0f)
                ? metal::precise::divide(max_abs, 127.0f)
                : 1.0f;
    if (tid == 0u) x_scales[tg_id] = scale;
    // CPU computes `1.0 / scale` once then multiplies; replicate that
    // (rather than `xv / scale`) so the rounding before `round(...)` is
    // identical across CPU and GPU.
    float inv = metal::precise::divide(1.0f, scale);
    float q = round(xv * inv);
    q = clamp(q, -127.0f, 127.0f);
    x_int8[block_off + tid] = (signed char)q;
}

// ── quantize_f32_to_int8_per_block_scaled ────────────────────────────────────
//
// AWQ Option B fused activation-divide + per-block int8 quant. Identical to
// `quantize_f32_to_int8_per_block` except each input element is divided by
// the corresponding entry of a per-channel smoothing vector `s` BEFORE the
// per-block min/max reduction:
//     x'[i] = x[i] / s[i]
//     scale = max|x'| / 127 per 256-elem block
//     q[i]  = round(x'[i] / scale) clamped to [-127, 127]
//
// `s` has length `n` (same as `x`); the i-th smoothing factor pairs with the
// i-th channel of `x`. Activation-side mate of the offline-baked weights
// produced by `tools/awq_bake/` (W'[r, c] = W[r, c] * s[c]) such that
//     (x / s) · (W * s).T == x · W.T
// holds mathematically while the int8/Q4_K quantizers see a more uniform
// magnitude profile.
//
// Grid/TG/shmem identical to the unscaled variant so the dispatch wrapper
// can re-use the same launch geometry.
kernel void quantize_f32_to_int8_per_block_scaled(
    device const float*       x        [[buffer(0)]],
    device const float*       s        [[buffer(1)]],
    device       signed char* x_int8   [[buffer(2)]],
    device       float*       x_scales [[buffer(3)]],
    threadgroup  float*       red      [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint block_off = tg_id * 256u;
    uint i = block_off + tid;
    float sv = s[i];
    // Smoothing factors come from a calibration corpus and are strictly > 0
    // for any channel touched by activation traffic; guard the zero case so
    // a pathological factor doesn't propagate a NaN through the reduction.
    float inv_s = (sv > 1e-12f) ? metal::precise::divide(1.0f, sv) : 0.0f;
    float xv = x[i] * inv_s;
    red[tid] = fabs(xv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] = max(red[tid], red[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_abs = red[0];
    float scale = (max_abs > 0.0f)
                ? metal::precise::divide(max_abs, 127.0f)
                : 1.0f;
    if (tid == 0u) x_scales[tg_id] = scale;
    float inv = metal::precise::divide(1.0f, scale);
    float q = round(xv * inv);
    q = clamp(q, -127.0f, 127.0f);
    x_int8[i] = (signed char)q;
}

// ── quantize_f32_to_int8_per_channel ─────────────────────────────────────────
//
// GPU-side quant using STATIC pre-computed per-channel scales. Pairs with
// `gemm_q4_k_a8_v3_8r_per_channel` for the per-channel W4A8 path. The scales
// come from a calibration pass (memory/w4a8_lmhead_calibration_2026_05_26.md)
// and are pinned at model load — they do NOT change per token.
//
// Per the CPU `quantize_to_int8_per_channel` reference:
//   q[i] = round(x[i] / scales[i]) clamped to [-127, 127]
//
// Grid:  (n, 1, 1)   one thread per element
// TG:    (256, 1, 1) flat, no shmem needed (no reduction)
//
// Unlike per-block quant, no scale-output buffer — scales are an INPUT
// (read-only, pre-computed). Output is just int8 bytes.
kernel void quantize_f32_to_int8_per_channel(
    device const float*       x        [[buffer(0)]],
    device const float*       scales   [[buffer(1)]],  // PER-CHANNEL, length n
    device       signed char* x_int8   [[buffer(2)]],
    constant     uint&        n        [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float s = scales[i];
    if (s > 1e-8f) {
        float inv = metal::precise::divide(1.0f, s);
        float q = round(x[i] * inv);
        q = clamp(q, -127.0f, 127.0f);
        x_int8[i] = (signed char)q;
    } else {
        // Zero-magnitude calibration channel (never active) → emit zero.
        x_int8[i] = 0;
    }
}

// ── W4A8 prototype: gemm_q4_k_a8_v3_8r ───────────────────────────────────────
//
// Same v3_8r geometry (8 rows/TG, 32 threads/row, 256 threads/TG) but with
// activations packed as per-block (256-element) int8 + f32 scale instead of
// raw f32. Bandwidth on the activation buffer drops 4× (8 bytes/block/thread
// vs 32 bytes/block/thread), trading a small per-block scale lookup.
//
// Activation layout per row:
//   x_int8 : (cols,)               int8   — quantized values
//   x_scales : (cols / 256,)       f32    — per-block scales: real = x_int8 * scale
//
// The activation is per-token (1D) for decode; for batched callers each
// batch element has its own (x_int8, x_scales) pair laid out contiguously.
//
// Math: per block, the dot product becomes:
//   sum_k weight[k] * (act_int8[k] * scale_block) = scale_block * sum_k(weight[k] * act_int8[k])
//
// We factor scale_block out and apply it once per block. Weight stays Q4_K,
// dequanted per element as in v3_8r. No simd_mma here — that's the prefill
// path (separate v3w_a8 kernel). This kernel exists to measure decode
// activation-BW savings in isolation.

kernel void gemm_q4_k_a8_v3_8r(
    device const uchar*       w_q4     [[buffer(0)]],
    device const signed char* x_int8   [[buffer(1)]],
    device const float*       x_scales [[buffer(2)]],
    device       float*       y        [[buffer(3)]],
    constant     uint&        rows     [[buffer(4)]],
    constant     uint&        cols     [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
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

        uchar sb[8], mb[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sb[sub] = w_q4[bo + 4u + sub] & 0x3Fu;
            mb[sub] = w_q4[bo + 8u + sub] & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sb[4u + j] = (w_q4[bo + 12u + j] & 0x0Fu)
                       | ((w_q4[bo + 4u + j]  >> 6u) << 4u);
            mb[4u + j] = (w_q4[bo + 12u + j]  >> 4u)
                       | ((w_q4[bo + 8u + j]   >> 6u) << 4u);
        }

        float ds[8], dm[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds[sub] = d    * (float)sb[sub];
            dm[sub] = dmin * (float)mb[sub];
        }

        // Per-block activation scale + int8 lane loads.
        float scale_b = x_scales[b];
        signed char xq[8];
        for (uint k = 0; k < 8u; ++k) {
            xq[k] = x_int8[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];
        }

        float block_acc = 0.0f;
        for (uint pi = 0; pi < 4u; ++pi) {
            uchar qb = w_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uint k0 = pi * 2u, k1 = k0 + 1u;
            block_acc += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * (float)xq[k0];
            block_acc += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * (float)xq[k1];
        }
        partial += block_acc * scale_b;
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── W4A8 per-channel — gemm_q4_k_a8_v3_8r_per_channel ───────────────────────
//
// Per-channel int8 activation × Q4_K weight GEMV. Same v3_8r geometry as
// the per-block version (8 rows/TG, 32 threads/row, 256 threads/TG)
// but reads ONE scale per ACTIVATION CHANNEL instead of one scale per
// 256-element block.
//
// Why per-channel: the per-block scheme suffers from super-outlier
// channels (e.g., on Qwen-3B, ch[1979] consistently has |x|=150 while
// neighbors are ~3-7). One outlier in a block dominates the block's
// scale = max/127, crushing the dynamic range for the other 255
// channels — see memory/w4a8_activation_distribution_2026_05_26.md.
//
// Per-channel scales (one f32 per hidden dim, total 8 KB at hidden=2048
// vs 32 bytes for per-block) give each channel its own scale, so the
// outlier's 1.18 scale doesn't punish the block-neighbors' 0.04 scales.
// Reconstruction RMSE drops 3.83× globally, 4.24× on outlier blocks
// (memory/w4a8_quality_redesign_2026_05_26.md UPDATE 2026-05-26).
//
// Buffers:
//   buffer(0): w_q4       — Q4_K weight bytes (rows * blocks * 144)
//   buffer(1): x_int8     — int8 activations (cols bytes)
//   buffer(2): x_scales   — f32 PER-CHANNEL scales (cols floats)
//   buffer(3): y          — f32 output (rows floats)
//   buffer(4): rows       — u32
//   buffer(5): cols       — u32
//
// The activation buffer layout matches per-block (x_int8 is cols
// length); only the scales buffer changes meaning: cols entries instead
// of cols/256.

kernel void gemm_q4_k_a8_v3_8r_per_channel(
    device const uchar*       w_q4     [[buffer(0)]],
    device const signed char* x_int8   [[buffer(1)]],
    device const float*       x_scales [[buffer(2)]],
    device       float*       y        [[buffer(3)]],
    constant     uint&        rows     [[buffer(4)]],
    constant     uint&        cols     [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
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

        uchar sb[8], mb[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sb[sub] = w_q4[bo + 4u + sub] & 0x3Fu;
            mb[sub] = w_q4[bo + 8u + sub] & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sb[4u + j] = (w_q4[bo + 12u + j] & 0x0Fu)
                       | ((w_q4[bo + 4u + j]  >> 6u) << 4u);
            mb[4u + j] = (w_q4[bo + 12u + j]  >> 4u)
                       | ((w_q4[bo + 8u + j]   >> 6u) << 4u);
        }

        float ds[8], dm[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds[sub] = d    * (float)sb[sub];
            dm[sub] = dmin * (float)mb[sub];
        }

        // Per-CHANNEL activation recovery: each element has its own scale.
        // Recover the f32 activation inline (x_rec[k] = int8 * channel_scale).
        // Reads 8 int8 bytes + 8 f32 scales from DRAM per thread per block;
        // L1 caches the scale array (8 KB at hidden=2048, fits easily).
        float x_rec[8];
        for (uint k = 0; k < 8u; ++k) {
            uint elem = (uint)b * 256u + k * 32u + simd_lane;
            x_rec[k] = (float)x_int8[elem] * x_scales[elem];
        }

        for (uint pi = 0; pi < 4u; ++pi) {
            uchar qb = w_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uint k0 = pi * 2u, k1 = k0 + 1u;
            partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * x_rec[k0];
            partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * x_rec[k1];
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── gemm_q4k_fast_v1 ─────────────────────────────────────────────────────────
// Same v3_8r geometry (8 rows/TG, 32 threads/row, 256 threads/TG) but reads
// from the Q4K_FAST sub-block-contiguous layout instead of GGUF Q4_K:
//
//   per super-block (160 bytes total, 256 elements):
//     for sub-block k in 0..8:
//       bytes [k*20 .. k*20+2]   sub_scale (fp16)  = d    * sb_idx[k]
//       bytes [k*20+2 .. k*20+4] sub_min   (fp16)  = dmin * mb_idx[k]
//       bytes [k*20+4 .. k*20+20] 16 bytes (32 nibbles, two 4-bit values/byte)
//
// Decode loop reads the fp16 sub_scale / sub_min once, then iterates the 16
// nibble bytes paired across pi in {0..4}, exactly mirroring v3_8r's pi
// loop. The per-thread element layout inside a sub-block matches v3_8r:
// thread `simd_lane` in [0..32) reads nibble byte `pi*32+simd_lane` of the
// 32-byte sub-block-pair... but Q4K_FAST groups 16 nibble bytes per sub-
// block (not per super-block half), so the indexing simplifies to:
//
//   for k in 0..8:                      // sub-block
//     for pi in 0..2:                   // 2 nibble bytes per (k, simd_lane/16 pair)
//
// Wait — v3_8r uses pi in [0..4) with each pi covering 32 nibble bytes that
// span TWO sub-blocks (k0 = pi*2, k1 = pi*2+1). Q4K_FAST stores all 16
// nibble bytes per sub-block contiguously, so the natural loop is per
// sub-block. Each sub-block has 32 4-bit values = 16 nibble bytes; thread
// `simd_lane` (0..32) reads `nibbles[simd_lane / 2]` low or high nibble
// based on `simd_lane & 1`. This is a re-indexing that produces the SAME
// per-element partial sums as v3_8r — verified by the parity test.

kernel void gemm_q4k_fast_v1(
    device const uchar* w_fast [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 160ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 160ul;

        // Match v3_8r's pi-paired iteration so each thread's per-element
        // accumulation order is bit-identical to the source kernel:
        //   pi in [0..4)
        //     k0 = pi*2,  k1 = pi*2+1
        //     qb = byte at bo + 16 + pi*32 + simd_lane   (v3_8r layout)
        //
        // In Q4K_FAST, that same byte lives at:
        //   bo + (k0)*20 + 4 + (simd_lane half within k0/k1 layout)
        //
        // v3_8r's pi-byte simd_lane in [0..32) splits into:
        //   lane in [0..16)  → low half: byte = sub_block k0 nibble lane
        //   lane in [16..32) → high half: byte = sub_block k1 nibble (lane-16)
        // Q4_K packs k0's 16 nibble bytes followed by k1's 16 nibble bytes
        // contiguously inside the (pi*32) span at bo+16+pi*32. Q4K_FAST
        // separates those into bo+k0*20+4 (16 bytes) and bo+k1*20+4 (16
        // bytes). To preserve v3_8r ordering, each lane reads the byte
        // corresponding to its half:

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u;
            uint k1 = k0 + 1u;
            ushort s0_bits = (ushort)w_fast[bo + k0 * 20ul + 0ul]
                           | ((ushort)w_fast[bo + k0 * 20ul + 1ul] << 8);
            ushort m0_bits = (ushort)w_fast[bo + k0 * 20ul + 2ul]
                           | ((ushort)w_fast[bo + k0 * 20ul + 3ul] << 8);
            ushort s1_bits = (ushort)w_fast[bo + k1 * 20ul + 0ul]
                           | ((ushort)w_fast[bo + k1 * 20ul + 1ul] << 8);
            ushort m1_bits = (ushort)w_fast[bo + k1 * 20ul + 2ul]
                           | ((ushort)w_fast[bo + k1 * 20ul + 3ul] << 8);
            float ds0 = (float)as_type<half>(s0_bits);
            float dm0 = (float)as_type<half>(m0_bits);
            float ds1 = (float)as_type<half>(s1_bits);
            float dm1 = (float)as_type<half>(m1_bits);

            float xl0 = x[(uint64_t)b * 256ul + (uint64_t)(k0 * 32u + simd_lane)];
            float xl1 = x[(uint64_t)b * 256ul + (uint64_t)(k1 * 32u + simd_lane)];

            // v3_8r reads ONE byte per pi (at bo+16+pi*32+simd_lane) and
            // uses its low nibble for k0, high nibble for k1. Q4K_FAST
            // splits k0/k1 into separate sub-block payloads, but the byte
            // positions within each sub-block use the SAME simd_lane
            // index — meaning lane reads byte simd_lane of k0's 16-byte
            // nibble run (covers TWO 4-bit values: low for k0 element
            // simd_lane*2, high for k0 element simd_lane*2+1).
            //
            // To stay bit-identical to v3_8r we need the SAME nibble
            // selection: v3_8r used `qb & 0x0F` for the k0 element at
            // index simd_lane and `qb >> 4` for the k1 element at the
            // SAME index simd_lane (both within their respective
            // sub-blocks). The byte that holds the k0 element at
            // sub-block-internal index `simd_lane` (with simd_lane in
            // [0..32)) is byte `simd_lane / 2` in the 16-byte payload;
            // low nibble if simd_lane is even, high nibble if odd.
            //
            // Wait — v3_8r's `qb & 0x0F` is the k0 element at sub-block
            // index `simd_lane`. The activation lookup uses
            // `xl[k0] = x[b*256 + k0*32 + simd_lane]` (simd_lane in
            // [0..32)), so per-thread there are exactly 32 elements per
            // sub-block (one per lane). Each nibble byte holds 2 elements
            // (low + high) of DIFFERENT sub-blocks (k0 low, k1 high).
            //
            // In Q4K_FAST each sub-block has its OWN 16-byte nibble
            // payload, holding 32 4-bit values. Thread `simd_lane` (one
            // element per sub-block) reads:
            //   byte_idx = simd_lane / 2u
            //   nib_sel  = simd_lane & 1u   (0=low nibble, 1=high nibble)
            uint byte_idx = simd_lane >> 1u;
            uint nib_sel  = simd_lane & 1u;
            uchar nb0 = w_fast[bo + k0 * 20ul + 4ul + (uint64_t)byte_idx];
            uchar nb1 = w_fast[bo + k1 * 20ul + 4ul + (uint64_t)byte_idx];
            uint q0 = (nib_sel == 0u) ? ((uint)nb0 & 0x0Fu) : (((uint)nb0 >> 4u) & 0x0Fu);
            uint q1 = (nib_sel == 0u) ? ((uint)nb1 & 0x0Fu) : (((uint)nb1 >> 4u) & 0x0Fu);

            partial += (ds0 * (float)q0 - dm0) * xl0;
            partial += (ds1 * (float)q1 - dm1) * xl1;
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── gemm_q4_k_m_v3_dual ──────────────────────────────────────────────────────
// Phase B Approach 1 Iter 2: 2 rows per simdgroup (N_R0=2), 4 simdgroups per TG
// (128 threads) — matches llama.cpp's N_R0_Q4_K=2, FC_mul_mv_nsg=4 geometry.
// Each simdgroup loads activations xl[8] ONCE and computes two output rows,
// halving the activation load bandwidth per row.
//
// Grid: (ceil(rows/8)*128, 1, 1)   threadgroup: (128, 1, 1)
// 4 simdgroups × 2 rows each = 8 rows per TG.

kernel void gemm_q4_k_m_v3_dual(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row0 = gid * 8u + simd_id * 2u;
    uint base_row1 = base_row0 + 1u;
    bool row1_valid = base_row1 < rows;
    if (base_row0 >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off0 = (uint64_t)base_row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_byte_off1 = (uint64_t)base_row1 * (uint64_t)blocks_per_row * 144ul;
    float partial0 = 0.0f, partial1 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = row_byte_off0 + (uint64_t)b * 144ul;

        ushort d_bits    = (ushort)w_q4[bo0]     | ((ushort)w_q4[bo0 + 1] << 8);
        ushort dmin_bits = (ushort)w_q4[bo0 + 2] | ((ushort)w_q4[bo0 + 3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);

        uchar sb[8], mb[8];
        for (uint sub = 0; sub < 4u; ++sub) {
            sb[sub] = w_q4[bo0 + 4u + sub]      & 0x3Fu;
            mb[sub] = w_q4[bo0 + 8u + sub]      & 0x3Fu;
        }
        for (uint j = 0; j < 4u; ++j) {
            sb[4u + j] = (w_q4[bo0 + 12u + j] & 0x0Fu)
                       | ((w_q4[bo0 + 4u + j]  >> 6u) << 4u);
            mb[4u + j] = (w_q4[bo0 + 12u + j]  >> 4u)
                       | ((w_q4[bo0 + 8u + j]   >> 6u) << 4u);
        }

        // Row 1 scales (different row, same block offset pattern)
        float ds0[8], dm0[8], ds1[8], dm1[8];
        if (row1_valid) {
            uint64_t bo1 = row_byte_off1 + (uint64_t)b * 144ul;
            ushort d1_bits    = (ushort)w_q4[bo1]     | ((ushort)w_q4[bo1 + 1] << 8);
            ushort dmin1_bits = (ushort)w_q4[bo1 + 2] | ((ushort)w_q4[bo1 + 3] << 8);
            float d1    = (float)as_type<half>(d1_bits);
            float dmin1 = (float)as_type<half>(dmin1_bits);
            uchar sb1[8], mb1[8];
            for (uint sub = 0; sub < 4u; ++sub) {
                sb1[sub] = w_q4[bo1 + 4u + sub]      & 0x3Fu;
                mb1[sub] = w_q4[bo1 + 8u + sub]      & 0x3Fu;
            }
            for (uint j = 0; j < 4u; ++j) {
                sb1[4u + j] = (w_q4[bo1 + 12u + j] & 0x0Fu)
                            | ((w_q4[bo1 + 4u + j]  >> 6u) << 4u);
                mb1[4u + j] = (w_q4[bo1 + 12u + j]  >> 4u)
                            | ((w_q4[bo1 + 8u + j]   >> 6u) << 4u);
            }
            for (uint sub = 0; sub < 8u; ++sub) {
                ds1[sub] = d1    * (float)sb1[sub];
                dm1[sub] = dmin1 * (float)mb1[sub];
            }
        }
        for (uint sub = 0; sub < 8u; ++sub) {
            ds0[sub] = d    * (float)sb[sub];
            dm0[sub] = dmin * (float)mb[sub];
        }

        // Shared activation load — one load amortized over 2 rows
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qb0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            partial0 += (ds0[k0] * (float)(qb0 & 0x0Fu) - dm0[k0]) * xl[k0];
            partial0 += (ds0[k1] * (float)(qb0 >> 4u)   - dm0[k1]) * xl[k1];
            if (row1_valid) {
                uint64_t bo1 = row_byte_off1 + (uint64_t)b * 144ul;
                uchar qb1 = w_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
                partial1 += (ds1[k0] * (float)(qb1 & 0x0Fu) - dm1[k0]) * xl[k0];
                partial1 += (ds1[k1] * (float)(qb1 >> 4u)   - dm1[k1]) * xl[k1];
            }
        }
    }

    partial0 = simd_sum(partial0);
    if (simd_lane == 0u) y[base_row0] = partial0;
    if (row1_valid) {
        partial1 = simd_sum(partial1);
        if (simd_lane == 0u) y[base_row1] = partial1;
    }
}

// ── gemm_q4_k_m_v3_llama ─────────────────────────────────────────────────────
// Phase B Approach 3: faithful port of llama.cpp mul_mv_q4_K_f32 key optimizations.
//
// Key differences from v3_dual:
//   N_R0=4 (4 rows/simdgroup vs 2) — 2× less activation bandwidth per row
//   Sumy trick: precompute sumy[k]=simd_sum(xl[k]) per sub-block, compute min
//   correction as sum_k(dm[k]*sumy[k]) outside inner loop — eliminates 32
//   dm*xl MADs per sub-block, replacing with 1 multiply per sub-block per row.
//
// Grid: (ceil(rows/8)*64, 1, 1)   threadgroup: (64, 1, 1)
// 2 simdgroups × 4 rows each = 8 rows per TG.

kernel void gemm_q4_k_m_v3_llama(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup handles 4 consecutive output rows
    uint base_row = gid * 8u + simd_id * 4u;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    float p[4]           = {0.0f, 0.0f, 0.0f, 0.0f};
    // total_corr accumulates the min correction across all blocks.
    // sumy[k]=simd_sum(xl[k]) is the SAME value for all 32 threads, so
    // total_corr must be subtracted AFTER simd_sum(p[r]) to avoid 32× error.
    float total_corr[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    bool  row_valid[4];
    for (uint r = 0; r < 4u; ++r)
        row_valid[r] = (base_row + r) < rows;

    for (uint b = 0; b < blocks_per_row; ++b) {
        // ── Load scales and mins for all 4 rows ──────────────────────────────
        float ds[4][8], dm[4][8];
        for (uint r = 0; r < 4u; ++r) {
            if (!row_valid[r]) continue;
            uint64_t bo = (uint64_t)(base_row + r) * (uint64_t)blocks_per_row * 144ul
                        + (uint64_t)b * 144ul;
            ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
            ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
            float  d    = (float)as_type<half>(d_bits);
            float  dmin = (float)as_type<half>(dmin_bits);
            uchar sb[8], mb[8];
            for (uint sub = 0; sub < 4u; ++sub) {
                sb[sub] = w_q4[bo + 4u + sub]  & 0x3Fu;
                mb[sub] = w_q4[bo + 8u + sub]  & 0x3Fu;
            }
            for (uint j = 0; j < 4u; ++j) {
                sb[4u+j] = (w_q4[bo + 12u + j] & 0x0Fu) | ((w_q4[bo + 4u + j] >> 6u) << 4u);
                mb[4u+j] = (w_q4[bo + 12u + j] >> 4u)   | ((w_q4[bo + 8u + j] >> 6u) << 4u);
            }
            for (uint sub = 0; sub < 8u; ++sub) {
                ds[r][sub] = d    * (float)sb[sub];
                dm[r][sub] = dmin * (float)mb[sub];
            }
        }

        // ── Shared activation load (once for 4 rows) ─────────────────────────
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        // ── Sumy: sub-block sums of activations (same value for all threads) ─
        // sumy[k] = sum over 32 lanes of xl[k] = sum of x in sub-block k.
        float sumy[8];
        for (uint k = 0; k < 8u; ++k)
            sumy[k] = simd_sum(xl[k]);

        // ── Accumulate min correction BEFORE simd_sum(p) ─────────────────────
        // total_corr[r] is the same across all threads (sumy is thread-uniform).
        // It will be subtracted after simd_sum to avoid the 32× replication.
        for (uint r = 0; r < 4u; ++r) {
            if (!row_valid[r]) continue;
            for (uint k = 0; k < 8u; ++k)
                total_corr[r] += dm[r][k] * sumy[k];
        }

        // ── Main dot product: scale × nibble × activation (no min term) ──────
        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            for (uint r = 0; r < 4u; ++r) {
                if (!row_valid[r]) continue;
                uint64_t bo = (uint64_t)(base_row + r) * (uint64_t)blocks_per_row * 144ul
                            + (uint64_t)b * 144ul;
                uchar qb = w_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
                p[r] += ds[r][k0] * (float)(qb & 0x0Fu) * xl[k0];
                p[r] += ds[r][k1] * (float)(qb >> 4u)   * xl[k1];
            }
        }
    }

    // ── Reduce and write: simd_sum(p[r]) then subtract total_corr ────────────
    // total_corr is the same for all 32 threads, so subtract ONCE after reduce.
    for (uint r = 0; r < 4u; ++r) {
        if (!row_valid[r]) continue;
        float val = simd_sum(p[r]) - total_corr[r];
        if (simd_lane == 0u) y[base_row + r] = val;
    }
}

// v0.5.10-A — Q4_K_M GEMV with f16 x and f16 y. Internal MAC in f32.
// Identical body to gemm_q4_k_m_fused; only x/y types differ.
kernel void gemm_q4_k_m_fused_f16(
    device const uchar* w_q4   [[buffer(0)]],   // (rows, cols) Q4_K_M
    device const half*  x      [[buffer(1)]],   // (cols,) f16
    device       half*  y      [[buffer(2)]],   // (rows,) f16
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

// v0.5.10-D — standalone Q6_K → f16 dequant.
// One threadgroup per 256-element block (210 bytes), one thread per element.
// Q6_K block layout: ql[128], qh[64], scales[16], d[2] = 210 bytes.
// Decoding logic mirrors q6_k_value() from moe.metal, inlined here to avoid
// cross-file includes.
// Grid: (nblock, 1, 1), TG: (256, 1, 1).
kernel void dequant_q6_k_f16(
    device const uchar* src    [[buffer(0)]],
    device       half*  dst    [[buffer(1)]],
    constant     uint&  nblock [[buffer(2)]],
    uint                tid    [[thread_position_in_threadgroup]],
    uint                gid    [[threadgroup_position_in_grid]])
{
    if (gid >= nblock) return;
    uint64_t bo = (uint64_t)gid * 210ul;

    // Decode Q6_K element for this thread (same logic as q6_k_value in moe.metal).
    ushort d_bits = (ushort)src[bo + 208u] | ((ushort)src[bo + 209u] << 8);
    float d = (float)as_type<half>(d_bits);
    uint half_idx = tid >> 7;
    uint local = tid & 127u;
    uint l = local & 31u;
    uint group = local >> 5;

    uint64_t ql_base = bo + (uint64_t)half_idx * 64ul;
    uint64_t qh_base = bo + 128ul + (uint64_t)half_idx * 32ul;
    uchar qhi = src[qh_base + (uint64_t)l];
    uint q;
    if (group == 0u) {
        q = ((uint)src[ql_base + (uint64_t)l] & 0x0Fu)
          | (((uint)(qhi >> 0) & 0x03u) << 4);
    } else if (group == 1u) {
        q = ((uint)src[ql_base + 32ul + (uint64_t)l] & 0x0Fu)
          | (((uint)(qhi >> 2) & 0x03u) << 4);
    } else if (group == 2u) {
        q = ((uint)(src[ql_base + (uint64_t)l] >> 4))
          | (((uint)(qhi >> 4) & 0x03u) << 4);
    } else {
        q = ((uint)(src[ql_base + 32ul + (uint64_t)l] >> 4))
          | (((uint)(qhi >> 6) & 0x03u) << 4);
    }
    int q_signed = (int)q - 32;

    // scales: 16 bytes at offset 192, one signed byte per 16-element sub-block.
    int scale = (int)(signed char)src[bo + 192ul + (uint64_t)half_idx * 8ul
                                    + (uint64_t)(l >> 4) + (uint64_t)group * 2ul];
    float val = d * (float)scale * (float)q_signed;
    dst[(uint64_t)gid * 256ul + (uint64_t)tid] = (half)val;
}

// P3 — Batched Q4_K_M GEMM: one weight matrix W applied to B activation
// vectors in parallel. Compared to running B back-to-back single-matrix
// GEMVs (each of which re-reads W from DRAM), this reads W *once* per
// row and broadcasts to B output dot products in registers.
//
// Bandwidth amortization: a single-token forward through Qwen-3B-Q4_K_M
// reads ~1.6 GB of weights. At B=4 prefill, the same weight read produces
// 4 token outputs — effective compute per byte ~4×. Translates to a
// near-linear prefill speedup until the kernel becomes compute-bound or
// the activation reads dominate.
//
// Geometry matches gemm_q4_k_m_fused_v2 (8 rows per TG, 32 threads per row,
// TG=256) so callers can swap dispatchers without re-thinking grid shape.
// `args.batch` carries B; supported values 1..=4 (uses float4 partial
// accumulator). B > 4 callers should issue multiple dispatches.
//
// Memory layout:
//   x_batch : (B, cols)    f32, row-major (each row is one activation vec)
//   y_batch : (B, rows)    f32, row-major
//
// Activation x is read B times per weight-block-decode iteration, which
// the GPU L1 cache absorbs cleanly for small B (the same x_batch[b, off]
// addresses are re-read across rows, so they live in cache).

struct ArgbufBatchedRowsCols {
    uint rows;
    uint cols;
    uint batch;
};

kernel void gemm_q4_k_m_batched_v2(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x_batch[[buffer(1)]],
    device       float* y_batch[[buffer(2)]],
    constant ArgbufBatchedRowsCols& args [[buffer(3)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= args.rows) return;

    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint B = min(args.batch, 4u);

    float4 partial = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint64_t bo = row_byte_off + (uint64_t)blk * 144ul;

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
            uint64_t x_off = (uint64_t)blk * 256ul + (uint64_t)elem;

            if (B >= 1u) partial.x += w_val * x_batch[0u * args.cols + x_off];
            if (B >= 2u) partial.y += w_val * x_batch[1u * args.cols + x_off];
            if (B >= 3u) partial.z += w_val * x_batch[2u * args.cols + x_off];
            if (B >= 4u) partial.w += w_val * x_batch[3u * args.cols + x_off];
        }
    }

    partial.x = simd_sum(partial.x);
    if (B >= 2u) partial.y = simd_sum(partial.y);
    if (B >= 3u) partial.z = simd_sum(partial.z);
    if (B >= 4u) partial.w = simd_sum(partial.w);

    if (simd_lane == 0u) {
        if (B >= 1u) y_batch[0u * args.rows + base_row] = partial.x;
        if (B >= 2u) y_batch[1u * args.rows + base_row] = partial.y;
        if (B >= 3u) y_batch[2u * args.rows + base_row] = partial.z;
        if (B >= 4u) y_batch[3u * args.rows + base_row] = partial.w;
    }
}

// P3 v3 — Batched Q4_K_M GEMM with cooperative shared-memory activation
// staging. v2 reads each thread's B=4 activations directly from DRAM at
// stride `cols` apart — on cols-large shapes (ffn_down 2048×11008) those
// addresses miss the L1 line and the kernel collapses to no better than
// B sequential GEMVs (microbench: batched 1551us ≈ sequential 1574us).
//
// v3 fix: all 256 threads in the TG cooperatively load the current
// cols-block-of-256 activation slice for all B lanes into threadgroup
// memory (4 KB / block at B=4). The 32 threads/row compute loop then
// reads activations from shmem (single-cycle L1) instead of DRAM.
//
// Each TG processes 8 rows × 32 threads/row; activation tile is shared
// across all rows in the TG so a single shmem load serves 8 row dot
// products × 4 batch lanes.
//
// Geometry identical to v2 (TG=256, 8 rows/TG). Same dispatch args.
// Supported B: 1..=4.

kernel void gemm_q4_k_m_batched_v3(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x_batch[[buffer(1)]],
    device       float* y_batch[[buffer(2)]],
    constant ArgbufBatchedRowsCols& args [[buffer(3)]],
    threadgroup float* x_tile  [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint B = min(args.batch, 4u);
    bool row_valid = base_row < args.rows;

    float4 partial = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        // ── Cooperative load of the activation tile for this block ────
        // x_tile layout: [B][256] f32 — batch lane outer, position inner
        // so that consecutive threads in a simdgroup hit contiguous
        // shmem slots (no bank conflicts).
        uint x_off_base = blk * 256u;
        // 256 threads load 256 elements per batch lane.
        for (uint b = 0; b < B; ++b) {
            x_tile[b * 256u + tid] = x_batch[b * args.cols + x_off_base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            uint64_t bo = row_byte_off + (uint64_t)blk * 144ul;

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
                // All B activations are in shmem at x_tile[b * 256 + elem].
                if (B >= 1u) partial.x += w_val * x_tile[0u * 256u + elem];
                if (B >= 2u) partial.y += w_val * x_tile[1u * 256u + elem];
                if (B >= 3u) partial.z += w_val * x_tile[2u * 256u + elem];
                if (B >= 4u) partial.w += w_val * x_tile[3u * 256u + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!row_valid) return;

    partial.x = simd_sum(partial.x);
    if (B >= 2u) partial.y = simd_sum(partial.y);
    if (B >= 3u) partial.z = simd_sum(partial.z);
    if (B >= 4u) partial.w = simd_sum(partial.w);

    if (simd_lane == 0u) {
        if (B >= 1u) y_batch[0u * args.rows + base_row] = partial.x;
        if (B >= 2u) y_batch[1u * args.rows + base_row] = partial.y;
        if (B >= 3u) y_batch[2u * args.rows + base_row] = partial.z;
        if (B >= 4u) y_batch[3u * args.rows + base_row] = partial.w;
    }
}

// P3 v3w — Batched Q4_K_M GEMM with shmem activation tile, widened to
// B in 1..=8. Same shmem-staged approach as v3; partial accumulator
// is two float4s. Shmem tile size is B*256 floats (8 KB at B=8, fits
// well under the 32 KB threadgroup memory limit).
//
// Predicates on B at each broadcast site so B<8 callers still pay
// only their share of the multiplies; the rest of the GEMM dataflow
// (weight decode, shmem load) is unchanged.

kernel void gemm_q4_k_m_batched_v3w(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x_batch[[buffer(1)]],
    device       float* y_batch[[buffer(2)]],
    constant ArgbufBatchedRowsCols& args [[buffer(3)]],
    threadgroup float* x_tile  [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint B = min(args.batch, 8u);
    bool row_valid = base_row < args.rows;

    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint x_off_base = blk * 256u;
        // Cooperative shmem load — each of 256 threads loads B floats.
        for (uint b = 0; b < B; ++b) {
            x_tile[b * 256u + tid] = x_batch[b * args.cols + x_off_base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            uint64_t bo = row_byte_off + (uint64_t)blk * 144ul;

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
                if (B >= 1u) partial_lo.x += w_val * x_tile[0u * 256u + elem];
                if (B >= 2u) partial_lo.y += w_val * x_tile[1u * 256u + elem];
                if (B >= 3u) partial_lo.z += w_val * x_tile[2u * 256u + elem];
                if (B >= 4u) partial_lo.w += w_val * x_tile[3u * 256u + elem];
                if (B >= 5u) partial_hi.x += w_val * x_tile[4u * 256u + elem];
                if (B >= 6u) partial_hi.y += w_val * x_tile[5u * 256u + elem];
                if (B >= 7u) partial_hi.z += w_val * x_tile[6u * 256u + elem];
                if (B >= 8u) partial_hi.w += w_val * x_tile[7u * 256u + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!row_valid) return;

    partial_lo.x = simd_sum(partial_lo.x);
    if (B >= 2u) partial_lo.y = simd_sum(partial_lo.y);
    if (B >= 3u) partial_lo.z = simd_sum(partial_lo.z);
    if (B >= 4u) partial_lo.w = simd_sum(partial_lo.w);
    if (B >= 5u) partial_hi.x = simd_sum(partial_hi.x);
    if (B >= 6u) partial_hi.y = simd_sum(partial_hi.y);
    if (B >= 7u) partial_hi.z = simd_sum(partial_hi.z);
    if (B >= 8u) partial_hi.w = simd_sum(partial_hi.w);

    if (simd_lane == 0u) {
        if (B >= 1u) y_batch[0u * args.rows + base_row] = partial_lo.x;
        if (B >= 2u) y_batch[1u * args.rows + base_row] = partial_lo.y;
        if (B >= 3u) y_batch[2u * args.rows + base_row] = partial_lo.z;
        if (B >= 4u) y_batch[3u * args.rows + base_row] = partial_lo.w;
        if (B >= 5u) y_batch[4u * args.rows + base_row] = partial_hi.x;
        if (B >= 6u) y_batch[5u * args.rows + base_row] = partial_hi.y;
        if (B >= 7u) y_batch[6u * args.rows + base_row] = partial_hi.z;
        if (B >= 8u) y_batch[7u * args.rows + base_row] = partial_hi.w;
    }
}

// P1 — Q4_K batched-prefill GEMM via hardware simdgroup-matrix (MMA).
//
// Same Q4_K dequant→threadgroup staging contract as gemm_q4_k_m_batched_v3w,
// but the scalar-FMA inner product + simd_sum reduction is replaced by Apple
// Silicon's simdgroup_matrix<float,8,8> multiply-accumulate. This is the
// in-tree port of silicon-builds/dismantle-q4k-mma `gemm_q4k_mma` (+15% at
// N=8 in the standalone microbench). One simdgroup (32 threads) per
// threadgroup computes one 8(rows)×8(N) output tile; K is stepped in 32-wide
// sub-blocks (4 depth-8 MMA steps per Q4_K sub-block, 32 steps per 256 block).
//
// Geometry (differs from v3w, which packs 8 simdgroups/256-thread TG):
//   Grid:        (ceil(rows/8)*32, 1, 1)
//   Threadgroup: (32, 1, 1)              — one simdgroup
//   8 rows/TG. N = batch (1..=8); columns N..8 of the tile are zero-padded.
// Shmem layout (float, 576 slots = 2.25 KB):
//   Ws[ 0  .. 256): weight tile W[8 rows][32 K]   (ld = 32)
//   Xs[256 .. 512): activation X[32 K][8 N]       (ld = 8)
//   Os[512 .. 576): result tile C[8 rows][8 N]    (ld = 8)
// Output layout matches v3w: y_batch[n*rows + (row0+m)] = C[m][n].
kernel void gemm_q4_k_m_batched_v3w_mma(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* x_batch[[buffer(1)]],
    device       float* y_batch[[buffer(2)]],
    constant ArgbufBatchedRowsCols& args [[buffer(3)]],
    threadgroup float* shmem   [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]])
{
    uint row0 = gid * 8u;
    if (row0 >= args.rows) return;

    uint blocks_per_row = args.cols / 256u;
    uint B = min(args.batch, 8u);

    threadgroup float* Ws = shmem;          // [256] = 8 rows x 32 K
    threadgroup float* Xs = shmem + 256u;   // [256] = 32 K x 8 N
    threadgroup float* Os = shmem + 512u;   // [64]  = 8 rows x 8 N

    // Zero-init accumulator via shmem (lanes write 64 slots: tid and tid+32).
    Os[tid]       = 0.0f;
    Os[tid + 32u] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, Os, 8, ulong2(0, 0));

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint64_t row_blk_off = (uint64_t)blk * 144ul;
        // BK=32 sub-block step: 8 steps cover the 256-wide Q4_K block.
        for (uint kt = 0; kt < 8u; ++kt) {
            // ── Dequant weight tile Ws[8 rows][32 K] ──────────────────────
            // 32 threads × 8 elems = 256 slots. Each thread fills column
            // `tid` (the K offset within the 32-wide step) for all 8 rows.
            uint kk_local = tid;              // 0..31 — K offset in this step
            uint kk = kt * 32u + kk_local;    // 0..255 — element in block
            uint sub   = kk >> 5u;            // 0..7
            uint pair  = sub >> 1u;
            bool upper = (sub & 1u) != 0u;
            uint i     = kk & 31u;
            for (uint m = 0u; m < 8u; ++m) {
                uint row = row0 + m;
                if (row >= args.rows) { Ws[m * 32u + kk_local] = 0.0f; continue; }
                uint64_t bo = ((uint64_t)row * (uint64_t)blocks_per_row) * 144ul
                            + row_blk_off;
                ushort d_bits    = (ushort)w_q4[bo]     | ((ushort)w_q4[bo + 1] << 8);
                ushort dmin_bits = (ushort)w_q4[bo + 2] | ((ushort)w_q4[bo + 3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
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
                uchar q  = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
                uint nib = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
                Ws[m * 32u + kk_local] = d * (float)s_byte * (float)nib
                                       - dmin * (float)m_byte;
            }
            // ── Stage activation tile Xs[32 K][8 N] ───────────────────────
            // Thread `tid` owns K-row `tid`; fill all 8 N columns (pad >=B).
            uint x_k = kt * 32u + tid;        // 0..255 — element in block
            for (uint n = 0u; n < 8u; ++n) {
                Xs[kk_local * 8u + n] = (n < B)
                    ? x_batch[(uint64_t)n * (uint64_t)args.cols
                              + (uint64_t)blk * 256ul + (uint64_t)x_k]
                    : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── 4 depth-8 MMA steps across the 32-wide K sub-block ────────
            for (uint d8 = 0u; d8 < 32u; d8 += 8u) {
                simdgroup_matrix<float, 8, 8> wm, xm;
                simdgroup_load(wm, Ws + d8, 32, ulong2(0, 0));      // W[:, d8:d8+8], ld=32
                simdgroup_load(xm, Xs + d8 * 8u, 8, ulong2(0, 0));  // X[d8:d8+8, :], ld=8
                simdgroup_multiply_accumulate(acc, wm, xm, acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Write results: C[m][n] → y_batch[n*rows + row0+m] ─────────────────
    simdgroup_store(acc, Os, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = 0u; e < 2u; ++e) {
        uint slot = tid + e * 32u;   // 0..63
        uint m = slot >> 3u;          // row 0..7
        uint n = slot & 7u;           // N   0..7
        uint row = row0 + m;
        if (n < B && row < args.rows) {
            y_batch[(uint64_t)n * (uint64_t)args.rows + (uint64_t)row] = Os[slot];
        }
    }
}

// P1 — predec twin of gemm_q4_k_m_batched_v3w_mma. Reads pre-decoded
// (ds, dm) sub-block scale pairs (16 f32/block) instead of decoding the
// Q4_K header per element, mirroring gemm_q4_k_m_batched_v3w_predec. Weight
// NIBBLES are still read from w_q4; only the per-sub-block scale decode is
// hoisted. Same MMA staging/geometry as gemm_q4_k_m_batched_v3w_mma.
kernel void gemm_q4_k_m_batched_v3w_mma_predec(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* x_batch[[buffer(2)]],
    device       float* y_batch[[buffer(3)]],
    constant ArgbufBatchedRowsCols& args [[buffer(4)]],
    threadgroup float* shmem   [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]])
{
    uint row0 = gid * 8u;
    if (row0 >= args.rows) return;

    uint blocks_per_row = args.cols / 256u;
    uint B = min(args.batch, 8u);

    threadgroup float* Ws = shmem;          // [256] = 8 rows x 32 K
    threadgroup float* Xs = shmem + 256u;   // [256] = 32 K x 8 N
    threadgroup float* Os = shmem + 512u;   // [64]  = 8 rows x 8 N

    Os[tid]       = 0.0f;
    Os[tid + 32u] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, Os, 8, ulong2(0, 0));

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint64_t row_blk_off = (uint64_t)blk * 144ul;
        uint64_t scale_blk_off = (uint64_t)blk * 16ul;
        for (uint kt = 0; kt < 8u; ++kt) {
            uint kk_local = tid;              // 0..31
            uint kk = kt * 32u + kk_local;    // 0..255
            uint sub   = kk >> 5u;            // 0..7
            uint pair  = sub >> 1u;
            bool upper = (sub & 1u) != 0u;
            uint i     = kk & 31u;
            for (uint m = 0u; m < 8u; ++m) {
                uint row = row0 + m;
                if (row >= args.rows) { Ws[m * 32u + kk_local] = 0.0f; continue; }
                uint64_t bo = ((uint64_t)row * (uint64_t)blocks_per_row) * 144ul
                            + row_blk_off;
                uint64_t so = ((uint64_t)row * (uint64_t)blocks_per_row) * 16ul
                            + scale_blk_off;
                float ds = scales[so + (uint64_t)(sub * 2u)];
                float dm = scales[so + (uint64_t)(sub * 2u + 1u)];
                uchar q  = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
                uint nib = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
                Ws[m * 32u + kk_local] = ds * (float)nib - dm;
            }
            uint x_k = kt * 32u + tid;
            for (uint n = 0u; n < 8u; ++n) {
                Xs[kk_local * 8u + n] = (n < B)
                    ? x_batch[(uint64_t)n * (uint64_t)args.cols
                              + (uint64_t)blk * 256ul + (uint64_t)x_k]
                    : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint d8 = 0u; d8 < 32u; d8 += 8u) {
                simdgroup_matrix<float, 8, 8> wm, xm;
                simdgroup_load(wm, Ws + d8, 32, ulong2(0, 0));
                simdgroup_load(xm, Xs + d8 * 8u, 8, ulong2(0, 0));
                simdgroup_multiply_accumulate(acc, wm, xm, acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    simdgroup_store(acc, Os, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = 0u; e < 2u; ++e) {
        uint slot = tid + e * 32u;
        uint m = slot >> 3u;
        uint n = slot & 7u;
        uint row = row0 + m;
        if (n < B && row < args.rows) {
            y_batch[(uint64_t)n * (uint64_t)args.rows + (uint64_t)row] = Os[slot];
        }
    }
}

// Batched Q4_K GEMM with PRE-DECODED sub-block scales (v3w + v4_predec merge).
// Identical to gemm_q4_k_m_batched_v3w except the per-element Q4_K header decode
// (d/dmin half-floats + 6-bit s/m unpack) is replaced by a lookup into the
// pre-decoded `scales` table (ds=d*s_byte, dm=dmin*m_byte per sub-block, 16
// floats/block, same layout as gemm_q4_k_v4_predec). Weight NIBBLES are still
// read from w_q4. This brings the single-path predec win (+~40%) to the batched
// decode/verify path. Build `scales` once via predecode_q4_k_scale_table.
kernel void gemm_q4_k_m_batched_v3w_predec(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* x_batch[[buffer(2)]],
    device       float* y_batch[[buffer(3)]],
    constant ArgbufBatchedRowsCols& args [[buffer(4)]],
    threadgroup float* x_tile  [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    uint B = min(args.batch, 8u);
    bool row_valid = base_row < args.rows;

    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint x_off_base = blk * 256u;
        for (uint b = 0; b < B; ++b) {
            x_tile[b * 256u + tid] = x_batch[b * args.cols + x_off_base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            uint64_t bo = row_byte_off  + (uint64_t)blk * 144ul;
            uint64_t so = row_scale_off + (uint64_t)blk * 16ul;
            float ds[8], dm[8];
            for (uint sub = 0; sub < 8u; ++sub) {
                ds[sub] = scales[so + (uint64_t)(sub * 2u)];
                dm[sub] = scales[so + (uint64_t)(sub * 2u + 1u)];
            }
            for (uint k = 0; k < 8u; ++k) {
                uint elem  = k * 32u + simd_lane;
                uint sub   = elem >> 5;
                uint pair  = sub >> 1;
                bool upper = (sub & 1u) != 0u;
                uint i     = elem & 31u;
                uchar q    = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
                uint nib   = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
                float w_val = ds[sub] * (float)nib - dm[sub];
                if (B >= 1u) partial_lo.x += w_val * x_tile[0u * 256u + elem];
                if (B >= 2u) partial_lo.y += w_val * x_tile[1u * 256u + elem];
                if (B >= 3u) partial_lo.z += w_val * x_tile[2u * 256u + elem];
                if (B >= 4u) partial_lo.w += w_val * x_tile[3u * 256u + elem];
                if (B >= 5u) partial_hi.x += w_val * x_tile[4u * 256u + elem];
                if (B >= 6u) partial_hi.y += w_val * x_tile[5u * 256u + elem];
                if (B >= 7u) partial_hi.z += w_val * x_tile[6u * 256u + elem];
                if (B >= 8u) partial_hi.w += w_val * x_tile[7u * 256u + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!row_valid) return;

    partial_lo.x = simd_sum(partial_lo.x);
    if (B >= 2u) partial_lo.y = simd_sum(partial_lo.y);
    if (B >= 3u) partial_lo.z = simd_sum(partial_lo.z);
    if (B >= 4u) partial_lo.w = simd_sum(partial_lo.w);
    if (B >= 5u) partial_hi.x = simd_sum(partial_hi.x);
    if (B >= 6u) partial_hi.y = simd_sum(partial_hi.y);
    if (B >= 7u) partial_hi.z = simd_sum(partial_hi.z);
    if (B >= 8u) partial_hi.w = simd_sum(partial_hi.w);

    if (simd_lane == 0u) {
        if (B >= 1u) y_batch[0u * args.rows + base_row] = partial_lo.x;
        if (B >= 2u) y_batch[1u * args.rows + base_row] = partial_lo.y;
        if (B >= 3u) y_batch[2u * args.rows + base_row] = partial_lo.z;
        if (B >= 4u) y_batch[3u * args.rows + base_row] = partial_lo.w;
        if (B >= 5u) y_batch[4u * args.rows + base_row] = partial_hi.x;
        if (B >= 6u) y_batch[5u * args.rows + base_row] = partial_hi.y;
        if (B >= 7u) y_batch[6u * args.rows + base_row] = partial_hi.z;
        if (B >= 8u) y_batch[7u * args.rows + base_row] = partial_hi.w;
    }
}

// Track 3.5 — SwiGLU-fused v3w_predec: replaces (silu_mul + ffn_down) with ONE dispatch.
// Identical to gemm_q4_k_m_batched_v3w_predec except buffer(2)/buffer(5) are the raw
// gate and up activation buffers; x_tile is filled with silu(gate)*up inline.
// Saves 1 dispatch/layer × n_layers = 28 dispatches on Qwen-3B.
// gate_batch and up_batch must both be (B, cols) contiguous f32 row-major.
kernel void gemm_q4_k_m_batched_v3w_predec_swiglu(
    device const uchar* w_q4      [[buffer(0)]],
    device const float* scales    [[buffer(1)]],
    device const float* gate_batch[[buffer(2)]],
    device       float* y_batch   [[buffer(3)]],
    constant ArgbufBatchedRowsCols& args [[buffer(4)]],
    device const float* up_batch  [[buffer(5)]],
    threadgroup float* x_tile  [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    uint B = min(args.batch, 8u);
    bool row_valid = base_row < args.rows;

    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint x_off_base = blk * 256u;
        for (uint b = 0; b < B; ++b) {
            uint idx = b * args.cols + x_off_base + tid;
            float g = gate_batch[idx];
            x_tile[b * 256u + tid] = (g / (1.0f + exp(-g))) * up_batch[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            uint64_t bo = row_byte_off  + (uint64_t)blk * 144ul;
            uint64_t so = row_scale_off + (uint64_t)blk * 16ul;
            float ds[8], dm[8];
            for (uint sub = 0; sub < 8u; ++sub) {
                ds[sub] = scales[so + (uint64_t)(sub * 2u)];
                dm[sub] = scales[so + (uint64_t)(sub * 2u + 1u)];
            }
            for (uint k = 0; k < 8u; ++k) {
                uint elem  = k * 32u + simd_lane;
                uint sub   = elem >> 5;
                uint pair  = sub >> 1;
                bool upper = (sub & 1u) != 0u;
                uint i     = elem & 31u;
                uchar q    = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
                uint nib   = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
                float w_val = ds[sub] * (float)nib - dm[sub];
                if (B >= 1u) partial_lo.x += w_val * x_tile[0u * 256u + elem];
                if (B >= 2u) partial_lo.y += w_val * x_tile[1u * 256u + elem];
                if (B >= 3u) partial_lo.z += w_val * x_tile[2u * 256u + elem];
                if (B >= 4u) partial_lo.w += w_val * x_tile[3u * 256u + elem];
                if (B >= 5u) partial_hi.x += w_val * x_tile[4u * 256u + elem];
                if (B >= 6u) partial_hi.y += w_val * x_tile[5u * 256u + elem];
                if (B >= 7u) partial_hi.z += w_val * x_tile[6u * 256u + elem];
                if (B >= 8u) partial_hi.w += w_val * x_tile[7u * 256u + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!row_valid) return;
    partial_lo.x = simd_sum(partial_lo.x);
    if (B >= 2u) partial_lo.y = simd_sum(partial_lo.y);
    if (B >= 3u) partial_lo.z = simd_sum(partial_lo.z);
    if (B >= 4u) partial_lo.w = simd_sum(partial_lo.w);
    if (B >= 5u) partial_hi.x = simd_sum(partial_hi.x);
    if (B >= 6u) partial_hi.y = simd_sum(partial_hi.y);
    if (B >= 7u) partial_hi.z = simd_sum(partial_hi.z);
    if (B >= 8u) partial_hi.w = simd_sum(partial_hi.w);
    if (simd_lane == 0u) {
        if (B >= 1u) y_batch[0u * args.rows + base_row] = partial_lo.x;
        if (B >= 2u) y_batch[1u * args.rows + base_row] = partial_lo.y;
        if (B >= 3u) y_batch[2u * args.rows + base_row] = partial_lo.z;
        if (B >= 4u) y_batch[3u * args.rows + base_row] = partial_lo.w;
        if (B >= 5u) y_batch[4u * args.rows + base_row] = partial_hi.x;
        if (B >= 6u) y_batch[5u * args.rows + base_row] = partial_hi.y;
        if (B >= 7u) y_batch[6u * args.rows + base_row] = partial_hi.z;
        if (B >= 8u) y_batch[7u * args.rows + base_row] = partial_hi.w;
    }
}

// Extends gemm_q4_k_m_batched_v3w_predec to B=1..16. Adds two more float4
// accumulators (partial_lo2, partial_hi2) for slots 8..15.
// Shmem: B*256*sizeof(float), up to 16 KiB at B=16 (within M3 Pro 32 KiB limit).
kernel void gemm_q4_k_m_batched_v3w_predec_b16(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* x_batch[[buffer(2)]],
    device       float* y_batch[[buffer(3)]],
    constant ArgbufBatchedRowsCols& args [[buffer(4)]],
    threadgroup float* x_tile  [[threadgroup(0)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    uint B = min(args.batch, 16u);
    bool row_valid = base_row < args.rows;

    float4 partial_lo = float4(0.0f);
    float4 partial_hi = float4(0.0f);
    float4 partial_lo2 = float4(0.0f);
    float4 partial_hi2 = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint x_off_base = blk * 256u;
        for (uint b = 0; b < B; ++b) {
            x_tile[b * 256u + tid] = x_batch[b * args.cols + x_off_base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            uint64_t bo = row_byte_off  + (uint64_t)blk * 144ul;
            uint64_t so = row_scale_off + (uint64_t)blk * 16ul;
            float ds[8], dm[8];
            for (uint sub = 0; sub < 8u; ++sub) {
                ds[sub] = scales[so + (uint64_t)(sub * 2u)];
                dm[sub] = scales[so + (uint64_t)(sub * 2u + 1u)];
            }
            for (uint k = 0; k < 8u; ++k) {
                uint elem  = k * 32u + simd_lane;
                uint sub   = elem >> 5;
                uint pair  = sub >> 1;
                bool upper = (sub & 1u) != 0u;
                uint i     = elem & 31u;
                uchar q    = w_q4[bo + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
                uint nib   = upper ? ((uint)(q >> 4) & 0x0Fu) : ((uint)q & 0x0Fu);
                float w_val = ds[sub] * (float)nib - dm[sub];
                if (B >= 1u) partial_lo.x += w_val * x_tile[0u * 256u + elem];
                if (B >= 2u) partial_lo.y += w_val * x_tile[1u * 256u + elem];
                if (B >= 3u) partial_lo.z += w_val * x_tile[2u * 256u + elem];
                if (B >= 4u) partial_lo.w += w_val * x_tile[3u * 256u + elem];
                if (B >= 5u) partial_hi.x += w_val * x_tile[4u * 256u + elem];
                if (B >= 6u) partial_hi.y += w_val * x_tile[5u * 256u + elem];
                if (B >= 7u) partial_hi.z += w_val * x_tile[6u * 256u + elem];
                if (B >= 8u) partial_hi.w += w_val * x_tile[7u * 256u + elem];
                if (B >= 9u)  partial_lo2.x += w_val * x_tile[8u  * 256u + elem];
                if (B >= 10u) partial_lo2.y += w_val * x_tile[9u  * 256u + elem];
                if (B >= 11u) partial_lo2.z += w_val * x_tile[10u * 256u + elem];
                if (B >= 12u) partial_lo2.w += w_val * x_tile[11u * 256u + elem];
                if (B >= 13u) partial_hi2.x += w_val * x_tile[12u * 256u + elem];
                if (B >= 14u) partial_hi2.y += w_val * x_tile[13u * 256u + elem];
                if (B >= 15u) partial_hi2.z += w_val * x_tile[14u * 256u + elem];
                if (B >= 16u) partial_hi2.w += w_val * x_tile[15u * 256u + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!row_valid) return;

    partial_lo.x = simd_sum(partial_lo.x);
    if (B >= 2u) partial_lo.y = simd_sum(partial_lo.y);
    if (B >= 3u) partial_lo.z = simd_sum(partial_lo.z);
    if (B >= 4u) partial_lo.w = simd_sum(partial_lo.w);
    if (B >= 5u) partial_hi.x = simd_sum(partial_hi.x);
    if (B >= 6u) partial_hi.y = simd_sum(partial_hi.y);
    if (B >= 7u) partial_hi.z = simd_sum(partial_hi.z);
    if (B >= 8u) partial_hi.w = simd_sum(partial_hi.w);
    if (B >= 9u)  partial_lo2.x = simd_sum(partial_lo2.x);
    if (B >= 10u) partial_lo2.y = simd_sum(partial_lo2.y);
    if (B >= 11u) partial_lo2.z = simd_sum(partial_lo2.z);
    if (B >= 12u) partial_lo2.w = simd_sum(partial_lo2.w);
    if (B >= 13u) partial_hi2.x = simd_sum(partial_hi2.x);
    if (B >= 14u) partial_hi2.y = simd_sum(partial_hi2.y);
    if (B >= 15u) partial_hi2.z = simd_sum(partial_hi2.z);
    if (B >= 16u) partial_hi2.w = simd_sum(partial_hi2.w);

    if (simd_lane == 0u) {
        if (B >= 1u) y_batch[0u * args.rows + base_row] = partial_lo.x;
        if (B >= 2u) y_batch[1u * args.rows + base_row] = partial_lo.y;
        if (B >= 3u) y_batch[2u * args.rows + base_row] = partial_lo.z;
        if (B >= 4u) y_batch[3u * args.rows + base_row] = partial_lo.w;
        if (B >= 5u) y_batch[4u * args.rows + base_row] = partial_hi.x;
        if (B >= 6u) y_batch[5u * args.rows + base_row] = partial_hi.y;
        if (B >= 7u) y_batch[6u * args.rows + base_row] = partial_hi.z;
        if (B >= 8u) y_batch[7u * args.rows + base_row] = partial_hi.w;
        if (B >= 9u)  y_batch[8u  * args.rows + base_row] = partial_lo2.x;
        if (B >= 10u) y_batch[9u  * args.rows + base_row] = partial_lo2.y;
        if (B >= 11u) y_batch[10u * args.rows + base_row] = partial_lo2.z;
        if (B >= 12u) y_batch[11u * args.rows + base_row] = partial_lo2.w;
        if (B >= 13u) y_batch[12u * args.rows + base_row] = partial_hi2.x;
        if (B >= 14u) y_batch[13u * args.rows + base_row] = partial_hi2.y;
        if (B >= 15u) y_batch[14u * args.rows + base_row] = partial_hi2.z;
        if (B >= 16u) y_batch[15u * args.rows + base_row] = partial_hi2.w;
    }
}

// ── gemm_q4_k_m_batched_v4r_predec ──────────────────────────────────────────
// Barrier-free drop-in replacement for gemm_q4_k_m_batched_v3w_predec.
//
// v3w_predec uses threadgroup shmem to stage B activation vectors before the
// weight loop, requiring two threadgroup_barrier() calls per block (one after
// load, one before the next block reuses the tile). For blocks_per_row=8
// (hidden=2048) this is 16 barriers per projection call — the primary
// performance bottleneck at B=2..8.
//
// v4r_predec reads x directly from device memory (no shmem staging, zero
// barriers) and processes 16 rows per TG via 2-row ILP (two rows per
// simdgroup instead of one). Same Q4_K decode + predec scale contract.
//
// Geometry: 256 threads/TG, 8 simdgroups × 32 threads, 2 rows per simdgroup.
// Grid: (ceil(rows/16) × 256, 1, 1) with TG=(256,1,1). No threadgroup memory.
// Buffer layout: identical to v3w_predec (slots 0-4).
//
// Bit-identical to v3w_predec when both accumulate in the same per-element
// FMA order (verified by gemm_q4k_batched_parity test).
kernel void gemm_q4_k_m_batched_v4r_predec(
    device const uchar* w_q4   [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* x_batch[[buffer(2)]],
    device       float* y_batch[[buffer(3)]],
    constant ArgbufBatchedRowsCols& args [[buffer(4)]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 2-row ILP: each simdgroup handles row0 and row1 = row0 + 8.
    uint row0 = gid * 16u + simd_id;
    if (row0 >= args.rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < args.rows;
    // When row1 is out of bounds alias to row0 so weight reads stay in-bounds;
    // p1 accumulators are discarded in that case (never written to y_batch).
    uint r1 = has1 ? row1 : row0;

    uint B = min(args.batch, 8u);
    uint blocks_per_row = args.cols / 256u;
    uint64_t rbo0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rso0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rbo1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rso1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 16ul;

    float4 p0_lo = float4(0.0f), p0_hi = float4(0.0f);
    float4 p1_lo = float4(0.0f), p1_hi = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint64_t bo0 = rbo0 + (uint64_t)blk * 144ul;
        uint64_t so0 = rso0 + (uint64_t)blk * 16ul;
        uint64_t bo1 = rbo1 + (uint64_t)blk * 144ul;
        uint64_t so1 = rso1 + (uint64_t)blk * 16ul;

        // Pre-decoded scales: 8 (d, m) pairs per block = 16 floats.
        float ds0[8], dm0[8], ds1[8], dm1[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds0[sub] = scales[so0 + (uint64_t)(sub * 2u)];
            dm0[sub] = scales[so0 + (uint64_t)(sub * 2u + 1u)];
            ds1[sub] = scales[so1 + (uint64_t)(sub * 2u)];
            dm1[sub] = scales[so1 + (uint64_t)(sub * 2u + 1u)];
        }

        // 8 weight elements per lane (256 elems / 32 lanes).
        // Decode pattern mirrors v3w_predec exactly: each (pair, i) byte
        // packs the lo nibble for sub=pair*2 and hi for sub=pair*2+1.
        // x is read directly from device memory — no shmem, no barriers.
        for (uint k = 0; k < 8u; ++k) {
            uint elem  = k * 32u + simd_lane;
            uint sub   = elem >> 5u;
            uint pair  = sub >> 1u;
            bool upper = (sub & 1u) != 0u;
            uint i     = elem & 31u;
            uchar q0   = w_q4[bo0 + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
            uchar q1   = w_q4[bo1 + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
            uint  nib0 = upper ? ((uint)(q0 >> 4) & 0x0Fu) : ((uint)q0 & 0x0Fu);
            uint  nib1 = upper ? ((uint)(q1 >> 4) & 0x0Fu) : ((uint)q1 & 0x0Fu);
            float wv0  = ds0[sub] * (float)nib0 - dm0[sub];
            float wv1  = ds1[sub] * (float)nib1 - dm1[sub];

            // x element: device read, no staging.
            uint64_t x_col = (uint64_t)blk * 256ul + (uint64_t)elem;
            if (B >= 1u) { float x = x_batch[0u * args.cols + x_col]; p0_lo.x += wv0*x; p1_lo.x += wv1*x; }
            if (B >= 2u) { float x = x_batch[1u * args.cols + x_col]; p0_lo.y += wv0*x; p1_lo.y += wv1*x; }
            if (B >= 3u) { float x = x_batch[2u * args.cols + x_col]; p0_lo.z += wv0*x; p1_lo.z += wv1*x; }
            if (B >= 4u) { float x = x_batch[3u * args.cols + x_col]; p0_lo.w += wv0*x; p1_lo.w += wv1*x; }
            if (B >= 5u) { float x = x_batch[4u * args.cols + x_col]; p0_hi.x += wv0*x; p1_hi.x += wv1*x; }
            if (B >= 6u) { float x = x_batch[5u * args.cols + x_col]; p0_hi.y += wv0*x; p1_hi.y += wv1*x; }
            if (B >= 7u) { float x = x_batch[6u * args.cols + x_col]; p0_hi.z += wv0*x; p1_hi.z += wv1*x; }
            if (B >= 8u) { float x = x_batch[7u * args.cols + x_col]; p0_hi.w += wv0*x; p1_hi.w += wv1*x; }
        }
    }

    // Reduce across the 32-thread simdgroup (per batch slot).
    p0_lo.x = simd_sum(p0_lo.x);
    if (B >= 2u) p0_lo.y = simd_sum(p0_lo.y);
    if (B >= 3u) p0_lo.z = simd_sum(p0_lo.z);
    if (B >= 4u) p0_lo.w = simd_sum(p0_lo.w);
    if (B >= 5u) p0_hi.x = simd_sum(p0_hi.x);
    if (B >= 6u) p0_hi.y = simd_sum(p0_hi.y);
    if (B >= 7u) p0_hi.z = simd_sum(p0_hi.z);
    if (B >= 8u) p0_hi.w = simd_sum(p0_hi.w);
    if (has1) {
        p1_lo.x = simd_sum(p1_lo.x);
        if (B >= 2u) p1_lo.y = simd_sum(p1_lo.y);
        if (B >= 3u) p1_lo.z = simd_sum(p1_lo.z);
        if (B >= 4u) p1_lo.w = simd_sum(p1_lo.w);
        if (B >= 5u) p1_hi.x = simd_sum(p1_hi.x);
        if (B >= 6u) p1_hi.y = simd_sum(p1_hi.y);
        if (B >= 7u) p1_hi.z = simd_sum(p1_hi.z);
        if (B >= 8u) p1_hi.w = simd_sum(p1_hi.w);
    }

    if (simd_lane != 0u) return;

    // Write row0.
    if (B >= 1u) y_batch[0u * args.rows + row0] = p0_lo.x;
    if (B >= 2u) y_batch[1u * args.rows + row0] = p0_lo.y;
    if (B >= 3u) y_batch[2u * args.rows + row0] = p0_lo.z;
    if (B >= 4u) y_batch[3u * args.rows + row0] = p0_lo.w;
    if (B >= 5u) y_batch[4u * args.rows + row0] = p0_hi.x;
    if (B >= 6u) y_batch[5u * args.rows + row0] = p0_hi.y;
    if (B >= 7u) y_batch[6u * args.rows + row0] = p0_hi.z;
    if (B >= 8u) y_batch[7u * args.rows + row0] = p0_hi.w;

    // Write row1 (only when in bounds).
    if (has1) {
        if (B >= 1u) y_batch[0u * args.rows + row1] = p1_lo.x;
        if (B >= 2u) y_batch[1u * args.rows + row1] = p1_lo.y;
        if (B >= 3u) y_batch[2u * args.rows + row1] = p1_lo.z;
        if (B >= 4u) y_batch[3u * args.rows + row1] = p1_lo.w;
        if (B >= 5u) y_batch[4u * args.rows + row1] = p1_hi.x;
        if (B >= 6u) y_batch[5u * args.rows + row1] = p1_hi.y;
        if (B >= 7u) y_batch[6u * args.rows + row1] = p1_hi.z;
        if (B >= 8u) y_batch[7u * args.rows + row1] = p1_hi.w;
    }
}

// Track 3.5 — SwiGLU-fused v4r_predec: replaces (silu_mul + ffn_down) with ONE dispatch.
// Like gemm_q4_k_m_batched_v4r_predec but reads gate+up from device memory and applies
// silu(gate)*up inline at each element. Saves 1 dispatch/layer.
kernel void gemm_q4_k_m_batched_v4r_predec_swiglu(
    device const uchar* w_q4      [[buffer(0)]],
    device const float* scales    [[buffer(1)]],
    device const float* gate_batch[[buffer(2)]],
    device       float* y_batch   [[buffer(3)]],
    constant ArgbufBatchedRowsCols& args [[buffer(4)]],
    device const float* up_batch  [[buffer(5)]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 16u + simd_id;
    if (row0 >= args.rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < args.rows;
    uint r1 = has1 ? row1 : row0;

    uint B = min(args.batch, 8u);
    uint blocks_per_row = args.cols / 256u;
    uint64_t rbo0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rso0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rbo1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rso1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 16ul;

    float4 p0_lo = float4(0.0f), p0_hi = float4(0.0f);
    float4 p1_lo = float4(0.0f), p1_hi = float4(0.0f);

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        uint64_t bo0 = rbo0 + (uint64_t)blk * 144ul;
        uint64_t so0 = rso0 + (uint64_t)blk * 16ul;
        uint64_t bo1 = rbo1 + (uint64_t)blk * 144ul;
        uint64_t so1 = rso1 + (uint64_t)blk * 16ul;

        float ds0[8], dm0[8], ds1[8], dm1[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds0[sub] = scales[so0 + (uint64_t)(sub * 2u)];
            dm0[sub] = scales[so0 + (uint64_t)(sub * 2u + 1u)];
            ds1[sub] = scales[so1 + (uint64_t)(sub * 2u)];
            dm1[sub] = scales[so1 + (uint64_t)(sub * 2u + 1u)];
        }
        for (uint k = 0; k < 8u; ++k) {
            uint elem  = k * 32u + simd_lane;
            uint sub   = elem >> 5u;
            uint pair  = sub >> 1u;
            bool upper = (sub & 1u) != 0u;
            uint i     = elem & 31u;
            uchar q0   = w_q4[bo0 + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
            uchar q1   = w_q4[bo1 + 16ul + (uint64_t)pair * 32ul + (uint64_t)i];
            uint  nib0 = upper ? ((uint)(q0 >> 4) & 0x0Fu) : ((uint)q0 & 0x0Fu);
            uint  nib1 = upper ? ((uint)(q1 >> 4) & 0x0Fu) : ((uint)q1 & 0x0Fu);
            float wv0  = ds0[sub] * (float)nib0 - dm0[sub];
            float wv1  = ds1[sub] * (float)nib1 - dm1[sub];

            uint64_t x_col = (uint64_t)blk * 256ul + (uint64_t)elem;
            // SwiGLU: compute silu(gate)*up inline instead of reading pre-computed act.
            if (B >= 1u) { float g=gate_batch[0u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[0u*args.cols+x_col]; p0_lo.x+=wv0*x; p1_lo.x+=wv1*x; }
            if (B >= 2u) { float g=gate_batch[1u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[1u*args.cols+x_col]; p0_lo.y+=wv0*x; p1_lo.y+=wv1*x; }
            if (B >= 3u) { float g=gate_batch[2u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[2u*args.cols+x_col]; p0_lo.z+=wv0*x; p1_lo.z+=wv1*x; }
            if (B >= 4u) { float g=gate_batch[3u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[3u*args.cols+x_col]; p0_lo.w+=wv0*x; p1_lo.w+=wv1*x; }
            if (B >= 5u) { float g=gate_batch[4u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[4u*args.cols+x_col]; p0_hi.x+=wv0*x; p1_hi.x+=wv1*x; }
            if (B >= 6u) { float g=gate_batch[5u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[5u*args.cols+x_col]; p0_hi.y+=wv0*x; p1_hi.y+=wv1*x; }
            if (B >= 7u) { float g=gate_batch[6u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[6u*args.cols+x_col]; p0_hi.z+=wv0*x; p1_hi.z+=wv1*x; }
            if (B >= 8u) { float g=gate_batch[7u*args.cols+x_col]; float x=(g/(1.0f+exp(-g)))*up_batch[7u*args.cols+x_col]; p0_hi.w+=wv0*x; p1_hi.w+=wv1*x; }
        }
    }
    p0_lo.x = simd_sum(p0_lo.x);
    if (B >= 2u) p0_lo.y = simd_sum(p0_lo.y);
    if (B >= 3u) p0_lo.z = simd_sum(p0_lo.z);
    if (B >= 4u) p0_lo.w = simd_sum(p0_lo.w);
    if (B >= 5u) p0_hi.x = simd_sum(p0_hi.x);
    if (B >= 6u) p0_hi.y = simd_sum(p0_hi.y);
    if (B >= 7u) p0_hi.z = simd_sum(p0_hi.z);
    if (B >= 8u) p0_hi.w = simd_sum(p0_hi.w);
    if (has1) {
        p1_lo.x = simd_sum(p1_lo.x);
        if (B >= 2u) p1_lo.y = simd_sum(p1_lo.y);
        if (B >= 3u) p1_lo.z = simd_sum(p1_lo.z);
        if (B >= 4u) p1_lo.w = simd_sum(p1_lo.w);
        if (B >= 5u) p1_hi.x = simd_sum(p1_hi.x);
        if (B >= 6u) p1_hi.y = simd_sum(p1_hi.y);
        if (B >= 7u) p1_hi.z = simd_sum(p1_hi.z);
        if (B >= 8u) p1_hi.w = simd_sum(p1_hi.w);
    }
    if (simd_lane != 0u) return;
    if (B >= 1u) y_batch[0u * args.rows + row0] = p0_lo.x;
    if (B >= 2u) y_batch[1u * args.rows + row0] = p0_lo.y;
    if (B >= 3u) y_batch[2u * args.rows + row0] = p0_lo.z;
    if (B >= 4u) y_batch[3u * args.rows + row0] = p0_lo.w;
    if (B >= 5u) y_batch[4u * args.rows + row0] = p0_hi.x;
    if (B >= 6u) y_batch[5u * args.rows + row0] = p0_hi.y;
    if (B >= 7u) y_batch[6u * args.rows + row0] = p0_hi.z;
    if (B >= 8u) y_batch[7u * args.rows + row0] = p0_hi.w;
    if (has1) {
        if (B >= 1u) y_batch[0u * args.rows + row1] = p1_lo.x;
        if (B >= 2u) y_batch[1u * args.rows + row1] = p1_lo.y;
        if (B >= 3u) y_batch[2u * args.rows + row1] = p1_lo.z;
        if (B >= 4u) y_batch[3u * args.rows + row1] = p1_lo.w;
        if (B >= 5u) y_batch[4u * args.rows + row1] = p1_hi.x;
        if (B >= 6u) y_batch[5u * args.rows + row1] = p1_hi.y;
        if (B >= 7u) y_batch[6u * args.rows + row1] = p1_hi.z;
        if (B >= 8u) y_batch[7u * args.rows + row1] = p1_hi.w;
    }
}

// P2 — Q6_K-weight × fp32-vec → fp32 GEMV (single-matrix). Adapted from
// moe_batched_gemm_q6_k_indexed_v2t with the route/batch layer stripped.
// Matches gemm_q4_k_m_fused_v2 dispatch shape: TG=256, 8 rows/TG, one
// simdgroup per row (32 threads). Drops the f16-fallback dequant penalty
// for Q6_K weights in Q4_K_M mix-quant GGUFs.
//
// Q6_K block layout (210 B / 256 elems):
//   ql[128] — 4-bit lows, packed two-per-byte
//   qh[64]  — 2-bit highs, packed four-per-byte
//   scales[16] — int8 per-16-elem scales
//   d[2]    — half-precision block scale
kernel void gemm_q6_k_fused_v2(
    device const uchar* w_q6   [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant ArgbufRowsCols& args [[buffer(3)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= args.rows) return;

    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 210ul;

    // Per-lane constants (same scheme as the MoE v2t Q6_K kernel).
    uint half_idx          = simd_lane >> 4u;          // 0 or 1
    uint group             = (simd_lane >> 2u) & 3u;   // 0..3
    uint l_base            = (simd_lane & 3u) * 8u;    // 0, 8, 16, 24
    uint scale_l_off       = l_base >> 4u;
    uint scale_byte_off    = 192u + half_idx * 8u + scale_l_off + group * 2u;
    uint ql_group_off      = (group & 1u) * 32u;
    bool group_high_nibble = (group >= 2u);
    uint qh_shift          = group * 2u;
    uint tid_base          = half_idx * 128u + group * 32u + l_base;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 210ul;
        ushort d_bits = (ushort)w_q6[bo + 208u] | ((ushort)w_q6[bo + 209u] << 8);
        float d = (float)as_type<half>(d_bits);
        int scale = (int)(signed char)w_q6[bo + (uint64_t)scale_byte_off];
        float dscale = d * (float)scale;

        uint64_t ql_base = bo + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
        uint64_t qh_base = bo + 128ul + (uint64_t)half_idx * 32ul;

        float lane_acc = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            uint l = l_base + k;
            uchar qlb = w_q6[ql_base + (uint64_t)l];
            uint qlow = group_high_nibble
                      ? ((uint)(qlb >> 4) & 0x0Fu)
                      : ((uint)qlb & 0x0Fu);
            uchar qhb = w_q6[qh_base + (uint64_t)l];
            uint qhigh = ((uint)qhb >> qh_shift) & 0x03u;
            int qi = (int)(qlow | (qhigh << 4)) - 32;
            float xi = x[b * 256u + tid_base + k];
            lane_acc += (float)qi * xi;
        }
        partial += dscale * lane_acc;
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[base_row] = partial;
    }
}

// ── gemm_q4_k_v4_predec ──────────────────────────────────────────────────────
// Q4_K decode GEMV with pre-decoded sub-block scales.
//
// Same v3_8r geometry (256 threads/TG, 8 simdgroups, 8 rows/TG) and identical
// math, but the 8 (ds, dm) f32 pairs per block are read from a parallel
// pre-decoded buffer instead of being decoded inline from the packed 6-bit
// indices each call. The block d/dmin fp16 header and the 4-bit quants are
// still read from the same Q4_K bytes; only the per-sub-block-scale decode
// (which is invariant across forward passes) is hoisted to load time.
//
// Pre-decoded scale layout (matches `predecode_q4_k_scale_table` in Rust):
//   `scales[block_idx * 16 + sub*2 + 0]` = ds[sub] = (f32)d    * (f32)sb[sub]
//   `scales[block_idx * 16 + sub*2 + 1]` = dm[sub] = (f32)dmin * (f32)mb[sub]
// where `sb[sub]`, `mb[sub]` are the 6-bit sub-block scale/min indices.
//
// Bit-identical to gemm_q4_k_m_v3_8r when the host pre-decoder uses the same
// fp16->f32 widening for d/dmin and the same uchar->float widening for the
// 6-bit indices, then multiplies in f32.
//
// Grid: (ceil(rows/8)*256, 1, 1)   threadgroup: (256, 1, 1)

kernel void gemm_q4_k_v4_predec(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device       float* y       [[buffer(3)]],
    constant     uint&  rows    [[buffer(4)]],
    constant     uint&  cols    [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    float partial = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;

        // ds[k] at scales[so + k*2 + 0], dm[k] at scales[so + k*2 + 1].
        float ds[8], dm[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            ds[sub] = scales[so + (uint64_t)(sub * 2u)];
            dm[sub] = scales[so + (uint64_t)(sub * 2u + 1u)];
        }

        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uchar qb = w_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            uint k0 = pi * 2u, k1 = k0 + 1u;
            partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xl[k0];
            partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xl[k1];
        }
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── gemm_q4_k_v4_predec_pair ─────────────────────────────────────────────────
// Fused gate+up decode GEMV (path-to-50, 2026-05-30). Same v4_predec math and
// geometry, but ONE dispatch computes TWO outputs (gate, up) that share the
// same input activation `x`. The FFN gate and up projections are independent
// GEMVs reading the identical post-norm activation; fusing them halves the FFN
// projection dispatch count (2/layer -> 1/layer = -36 dispatches/token) and
// loads `x` once per simdgroup. Weights/scales for gate and up are read from
// their own buffers/offsets, so per-row arithmetic is bit-identical to two
// separate gemm_q4_k_v4_predec calls.
//
// Grid: (ceil(rows/8)*256, 1, 1)   threadgroup: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_pair(
    device const uchar* wg_q4   [[buffer(0)]],
    device const float* g_scales[[buffer(1)]],
    device const uchar* wu_q4   [[buffer(2)]],
    device const float* u_scales[[buffer(3)]],
    device const float* x       [[buffer(4)]],
    device       float* yg      [[buffer(5)]],
    device       float* yu      [[buffer(6)]],
    constant     uint&  rows    [[buffer(7)]],
    constant     uint&  cols    [[buffer(8)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    float partial_g = 0.0f;
    float partial_u = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;

        float dsg[8], dmg[8], dsu[8], dmu[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            dsg[sub] = g_scales[so + (uint64_t)(sub * 2u)];
            dmg[sub] = g_scales[so + (uint64_t)(sub * 2u + 1u)];
            dsu[sub] = u_scales[so + (uint64_t)(sub * 2u)];
            dmu[sub] = u_scales[so + (uint64_t)(sub * 2u + 1u)];
        }

        // x is shared between gate and up -- load it once.
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qbg = wg_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            partial_g += (dsg[k0] * (float)(qbg & 0x0Fu) - dmg[k0]) * xl[k0];
            partial_g += (dsg[k1] * (float)(qbg >> 4u)   - dmg[k1]) * xl[k1];
            uchar qbu = wu_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            partial_u += (dsu[k0] * (float)(qbu & 0x0Fu) - dmu[k0]) * xl[k0];
            partial_u += (dsu[k1] * (float)(qbu >> 4u)   - dmu[k1]) * xl[k1];
        }
    }

    partial_g = simd_sum(partial_g);
    partial_u = simd_sum(partial_u);
    if (simd_lane == 0u) {
        yg[base_row] = partial_g;
        yu[base_row] = partial_u;
    }
}

// ── gemm_q4_k_v4_predec_pair_2r ──────────────────────────────────────────────
// 2-row-per-simdgroup upgrade of gemm_q4_k_v4_predec_pair (Track A7, 2026-06-06).
// Each simdgroup computes 2 rows of gate AND 2 rows of up from the shared x
// activation vector.  vs the 1r pair:
//   • x is amortised across 4 partial sums instead of 2 → 2× activation-load reuse
//   • 4 accumulators (pg0, pg1, pu0, pu1) vs 2 → higher instruction-level parallelism
//   • 16 rows/TG instead of 8 → ½ the launch overhead for the dominant 11008-row
//     gate/up projections
// Parity: bit-identical to the 1r pair (same FMA order per row).
// Grid: (ceil(rows/16)*256, 1, 1)   TG: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_pair_2r(
    device const uchar* wg_q4   [[buffer(0)]],
    device const float* g_scales[[buffer(1)]],
    device const uchar* wu_q4   [[buffer(2)]],
    device const float* u_scales[[buffer(3)]],
    device const float* x       [[buffer(4)]],
    device       float* yg      [[buffer(5)]],
    device       float* yu      [[buffer(6)]],
    constant     uint&  rows    [[buffer(7)]],
    constant     uint&  cols    [[buffer(8)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 2 rows per simdgroup: row0 and row1 = row0 + 8.
    // 16 rows per TG (8 simdgroups × 2 rows each).
    uint row0 = gid * 16u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < rows;
    uint r1 = has1 ? row1 : row0; // alias to row0 so inner reads stay in-bounds

    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 16ul;

    float pg0 = 0.0f, pg1 = 0.0f, pu0 = 0.0f, pu1 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul;
        uint64_t so0 = rs0 + (uint64_t)b * 16ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 144ul;
        uint64_t so1 = rs1 + (uint64_t)b * 16ul;

        // Preload all scale pairs for both rows and both matrices.
        // 8 arrays × 8 f32 = 64 regs — within M3 per-thread register budget.
        float dsg0[8], dmg0[8], dsu0[8], dmu0[8];
        float dsg1[8], dmg1[8], dsu1[8], dmu1[8];
        for (uint s = 0; s < 8u; ++s) {
            dsg0[s] = g_scales[so0 + (uint64_t)(s * 2u)];
            dmg0[s] = g_scales[so0 + (uint64_t)(s * 2u + 1u)];
            dsu0[s] = u_scales[so0 + (uint64_t)(s * 2u)];
            dmu0[s] = u_scales[so0 + (uint64_t)(s * 2u + 1u)];
            dsg1[s] = g_scales[so1 + (uint64_t)(s * 2u)];
            dmg1[s] = g_scales[so1 + (uint64_t)(s * 2u + 1u)];
            dsu1[s] = u_scales[so1 + (uint64_t)(s * 2u)];
            dmu1[s] = u_scales[so1 + (uint64_t)(s * 2u + 1u)];
        }

        // Activation x shared across all 4 partial sums — load once per block.
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            float x0 = xl[k0], x1 = xl[k1];

            uchar qg0 = wg_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            pg0 += (dsg0[k0] * (float)(qg0 & 0x0Fu) - dmg0[k0]) * x0;
            pg0 += (dsg0[k1] * (float)(qg0 >> 4u)   - dmg0[k1]) * x1;

            uchar qu0 = wu_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            pu0 += (dsu0[k0] * (float)(qu0 & 0x0Fu) - dmu0[k0]) * x0;
            pu0 += (dsu0[k1] * (float)(qu0 >> 4u)   - dmu0[k1]) * x1;

            uchar qg1 = wg_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            pg1 += (dsg1[k0] * (float)(qg1 & 0x0Fu) - dmg1[k0]) * x0;
            pg1 += (dsg1[k1] * (float)(qg1 >> 4u)   - dmg1[k1]) * x1;

            uchar qu1 = wu_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            pu1 += (dsu1[k0] * (float)(qu1 & 0x0Fu) - dmu1[k0]) * x0;
            pu1 += (dsu1[k1] * (float)(qu1 >> 4u)   - dmu1[k1]) * x1;
        }
    }

    pg0 = simd_sum(pg0); pu0 = simd_sum(pu0);
    if (simd_lane == 0u) { yg[row0] = pg0; yu[row0] = pu0; }
    if (has1) {
        pg1 = simd_sum(pg1); pu1 = simd_sum(pu1);
        if (simd_lane == 0u) { yg[row1] = pg1; yu[row1] = pu1; }
    }
}

// ── gemm_q4_k_v4_predec_pair_4r ──────────────────────────────────────────────
// 4-row-per-simdgroup variant of the gate+up pair (Track B2, opt-in via
// DISMANTLE_QWEN_PAIR_4R=1).  Uses inline scale access (no preload array) so
// register pressure stays at ~20 floats vs 64 for the preloaded-2r approach,
// giving the compiler room to hold all 8 accumulators simultaneously.
// vs pair_2r:
//   • 8 FMA chains (pg0-3, pu0-3) vs 4 → wider ILP window
//   • 32 rows/TG instead of 16 → ½ the dispatch launch overhead
//   • inline scale reads instead of preloaded → compiler hides loads in pipeline
// Parity: bit-identical to pair_2r (same per-accumulator FMA order).
// Grid: (ceil(rows/32)*256, 1, 1)  TG: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_pair_4r(
    device const uchar* wg_q4   [[buffer(0)]],
    device const float* g_scales[[buffer(1)]],
    device const uchar* wu_q4   [[buffer(2)]],
    device const float* u_scales[[buffer(3)]],
    device const float* x       [[buffer(4)]],
    device       float* yg      [[buffer(5)]],
    device       float* yu      [[buffer(6)]],
    constant     uint&  rows    [[buffer(7)]],
    constant     uint&  cols    [[buffer(8)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 4 rows per simdgroup: row0, row0+8, row0+16, row0+24.
    // 32 rows per TG (8 simdgroups × 4 rows each).
    uint row0 = gid * 32u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u, row2 = row0 + 16u, row3 = row0 + 24u;
    bool has1 = row1 < rows, has2 = row2 < rows, has3 = row3 < rows;
    uint r1 = has1 ? row1 : row0;
    uint r2 = has2 ? row2 : row0;
    uint r3 = has3 ? row3 : row0;

    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb2 = (uint64_t)r2   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs2 = (uint64_t)r2   * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb3 = (uint64_t)r3   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs3 = (uint64_t)r3   * (uint64_t)blocks_per_row * 16ul;

    float pg0 = 0.f, pg1 = 0.f, pg2 = 0.f, pg3 = 0.f;
    float pu0 = 0.f, pu1 = 0.f, pu2 = 0.f, pu3 = 0.f;

    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b*144ul, so0 = rs0 + (uint64_t)b*16ul;
        uint64_t bo1 = rb1 + (uint64_t)b*144ul, so1 = rs1 + (uint64_t)b*16ul;
        uint64_t bo2 = rb2 + (uint64_t)b*144ul, so2 = rs2 + (uint64_t)b*16ul;
        uint64_t bo3 = rb3 + (uint64_t)b*144ul, so3 = rs3 + (uint64_t)b*16ul;

        // Activation x — preloaded once per block, shared across all 8 accumulators.
        float xl[8];
        for (uint k = 0u; k < 8u; ++k)
            xl[k] = x[(uint64_t)b*256ul + (uint64_t)(k*32u + simd_lane)];

        for (uint pi = 0u; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            float x0 = xl[k0], x1 = xl[k1];

            // Row 0 — gate then up (inline scale reads, same FMA order as pair_2r)
            uchar qg0 = wg_q4[bo0 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pg0 += (g_scales[so0+(uint64_t)(k0*2u)]   * (float)(qg0&0x0Fu) - g_scales[so0+(uint64_t)(k0*2u+1u)]) * x0;
            pg0 += (g_scales[so0+(uint64_t)(k1*2u)]   * (float)(qg0>>4u)   - g_scales[so0+(uint64_t)(k1*2u+1u)]) * x1;
            uchar qu0 = wu_q4[bo0 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pu0 += (u_scales[so0+(uint64_t)(k0*2u)]   * (float)(qu0&0x0Fu) - u_scales[so0+(uint64_t)(k0*2u+1u)]) * x0;
            pu0 += (u_scales[so0+(uint64_t)(k1*2u)]   * (float)(qu0>>4u)   - u_scales[so0+(uint64_t)(k1*2u+1u)]) * x1;

            // Row 1
            uchar qg1 = wg_q4[bo1 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pg1 += (g_scales[so1+(uint64_t)(k0*2u)]   * (float)(qg1&0x0Fu) - g_scales[so1+(uint64_t)(k0*2u+1u)]) * x0;
            pg1 += (g_scales[so1+(uint64_t)(k1*2u)]   * (float)(qg1>>4u)   - g_scales[so1+(uint64_t)(k1*2u+1u)]) * x1;
            uchar qu1 = wu_q4[bo1 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pu1 += (u_scales[so1+(uint64_t)(k0*2u)]   * (float)(qu1&0x0Fu) - u_scales[so1+(uint64_t)(k0*2u+1u)]) * x0;
            pu1 += (u_scales[so1+(uint64_t)(k1*2u)]   * (float)(qu1>>4u)   - u_scales[so1+(uint64_t)(k1*2u+1u)]) * x1;

            // Row 2
            uchar qg2 = wg_q4[bo2 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pg2 += (g_scales[so2+(uint64_t)(k0*2u)]   * (float)(qg2&0x0Fu) - g_scales[so2+(uint64_t)(k0*2u+1u)]) * x0;
            pg2 += (g_scales[so2+(uint64_t)(k1*2u)]   * (float)(qg2>>4u)   - g_scales[so2+(uint64_t)(k1*2u+1u)]) * x1;
            uchar qu2 = wu_q4[bo2 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pu2 += (u_scales[so2+(uint64_t)(k0*2u)]   * (float)(qu2&0x0Fu) - u_scales[so2+(uint64_t)(k0*2u+1u)]) * x0;
            pu2 += (u_scales[so2+(uint64_t)(k1*2u)]   * (float)(qu2>>4u)   - u_scales[so2+(uint64_t)(k1*2u+1u)]) * x1;

            // Row 3
            uchar qg3 = wg_q4[bo3 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pg3 += (g_scales[so3+(uint64_t)(k0*2u)]   * (float)(qg3&0x0Fu) - g_scales[so3+(uint64_t)(k0*2u+1u)]) * x0;
            pg3 += (g_scales[so3+(uint64_t)(k1*2u)]   * (float)(qg3>>4u)   - g_scales[so3+(uint64_t)(k1*2u+1u)]) * x1;
            uchar qu3 = wu_q4[bo3 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            pu3 += (u_scales[so3+(uint64_t)(k0*2u)]   * (float)(qu3&0x0Fu) - u_scales[so3+(uint64_t)(k0*2u+1u)]) * x0;
            pu3 += (u_scales[so3+(uint64_t)(k1*2u)]   * (float)(qu3>>4u)   - u_scales[so3+(uint64_t)(k1*2u+1u)]) * x1;
        }
    }

    pg0 = simd_sum(pg0); pu0 = simd_sum(pu0);
    if (simd_lane == 0u) { yg[row0] = pg0; yu[row0] = pu0; }
    if (has1) {
        pg1 = simd_sum(pg1); pu1 = simd_sum(pu1);
        if (simd_lane == 0u) { yg[row1] = pg1; yu[row1] = pu1; }
    }
    if (has2) {
        pg2 = simd_sum(pg2); pu2 = simd_sum(pu2);
        if (simd_lane == 0u) { yg[row2] = pg2; yu[row2] = pu2; }
    }
    if (has3) {
        pg3 = simd_sum(pg3); pu3 = simd_sum(pu3);
        if (simd_lane == 0u) { yg[row3] = pg3; yu[row3] = pu3; }
    }
}

// ── gemm_q4_k_v4_predec_pair_f16s ────────────────────────────────────────────
// f16-scales variant of _pair (A6.5, 2026-05-31). Identical fused gate+up math
// + geometry + FMA order, but BOTH the gate (`g_scales`) and up (`u_scales`)
// pre-decoded sub-block scale tables are read as `half` (2 B) instead of f32
// (4 B), then widened to float in register. The _pair kernel is 46.6% of decode
// and bandwidth-bound (A4/A5/A6); the f16 scales cut the scale-table traffic
// 192→160 B/block (−17%) on BOTH weight reads in the fused dispatch — the half
// of decode A3's non-pair f16s could not touch. NOT bit-identical (f16 scale
// rounding ~5e-4 relative) — gate at rel-L2 < 1e-2. Opt-in via
// DISMANTLE_QWEN_PREDEC_F16SCALES=1. Tables built by
// kernels::predecode_q4_k_scale_table_f16 (16 halfs/block, same element layout).
// Grid/threadgroup identical to _pair: (ceil(rows/8)*256,1,1) / (256,1,1).
kernel void gemm_q4_k_v4_predec_pair_f16s(
    device const uchar* wg_q4   [[buffer(0)]],
    device const half*  g_scales[[buffer(1)]],
    device const uchar* wu_q4   [[buffer(2)]],
    device const half*  u_scales[[buffer(3)]],
    device const float* x       [[buffer(4)]],
    device       float* yg      [[buffer(5)]],
    device       float* yu      [[buffer(6)]],
    constant     uint&  rows    [[buffer(7)]],
    constant     uint&  cols    [[buffer(8)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    float partial_g = 0.0f;
    float partial_u = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;

        float dsg[8], dmg[8], dsu[8], dmu[8];
        for (uint sub = 0; sub < 8u; ++sub) {
            dsg[sub] = (float)g_scales[so + (uint64_t)(sub * 2u)];
            dmg[sub] = (float)g_scales[so + (uint64_t)(sub * 2u + 1u)];
            dsu[sub] = (float)u_scales[so + (uint64_t)(sub * 2u)];
            dmu[sub] = (float)u_scales[so + (uint64_t)(sub * 2u + 1u)];
        }

        // x is shared between gate and up -- load it once.
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qbg = wg_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            partial_g += (dsg[k0] * (float)(qbg & 0x0Fu) - dmg[k0]) * xl[k0];
            partial_g += (dsg[k1] * (float)(qbg >> 4u)   - dmg[k1]) * xl[k1];
            uchar qbu = wu_q4[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            partial_u += (dsu[k0] * (float)(qbu & 0x0Fu) - dmu[k0]) * xl[k0];
            partial_u += (dsu[k1] * (float)(qbu >> 4u)   - dmu[k1]) * xl[k1];
        }
    }

    partial_g = simd_sum(partial_g);
    partial_u = simd_sum(partial_u);
    if (simd_lane == 0u) {
        yg[base_row] = partial_g;
        yu[base_row] = partial_u;
    }
}

// ── gemm_q4_k_v4_predec_2r ───────────────────────────────────────────────────
// 2-rows-per-simdgroup predec GEMV (path-to-50, 2026-05-30). Identical math to
// gemm_q4_k_v4_predec, but each simdgroup computes TWO output rows of the SAME
// matrix with two independent accumulator chains, sharing the single `x` load.
// The two chains give the compiler 2 in-flight weight-load streams per thread,
// hiding DRAM latency the same way the gate+up pair kernel did — but for any
// single GEMV (q/o/ffn_down). 16 rows/TG (8 simdgroups x 2 rows).
//
// Grid: (ceil(rows/16)*256, 1, 1)   threadgroup: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_2r(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device       float* y       [[buffer(3)]],
    constant     uint&  rows    [[buffer(4)]],
    constant     uint&  cols    [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 16u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < rows;
    // When row1 is past the end, alias it to row0 so the inner loop reads valid
    // memory (no OOB); its result p1 is simply never written. Avoids a per-block
    // branch on the hot path. For all production shapes rows%16==0 so has1 holds.
    uint r1 = has1 ? row1 : row0;

    uint  blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f;
    float p1 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul, so0 = rs0 + (uint64_t)b * 16ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 144ul, so1 = rs1 + (uint64_t)b * 16ul;

        float ds0[8], dm0[8], ds1[8], dm1[8];
        for (uint s = 0; s < 8u; ++s) {
            ds0[s] = scales[so0 + (uint64_t)(s * 2u)];
            dm0[s] = scales[so0 + (uint64_t)(s * 2u + 1u)];
            ds1[s] = scales[so1 + (uint64_t)(s * 2u)];
            dm1[s] = scales[so1 + (uint64_t)(s * 2u + 1u)];
        }

        // x shared across both rows — load once.
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p0 += (ds0[k0] * (float)(q0 & 0x0Fu) - dm0[k0]) * xl[k0];
            p0 += (ds0[k1] * (float)(q0 >> 4u)   - dm0[k1]) * xl[k1];
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p1 += (ds1[k0] * (float)(q1 & 0x0Fu) - dm1[k0]) * xl[k0];
            p1 += (ds1[k1] * (float)(q1 >> 4u)   - dm1[k1]) * xl[k1];
        }
    }

    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) {
        p1 = simd_sum(p1);
        if (simd_lane == 0u) y[row1] = p1;
    }
}

// ── gemm_q4_k_v4_predec_2r_add ──────────────────────────────────────────────
// o_proj tail helper: same math and 2-row geometry as gemm_q4_k_v4_predec_2r,
// but the completed GEMV row is added directly into the residual stream instead
// of being materialized to a temporary y buffer. The following dispatch can run
// rmsnorm_f32 on the updated residual, preserving the required vector-wide norm
// synchronization while skipping the o_proj_out write/read round-trip.
//
// Bindings:
//   0: w_q4      Q4_K bytes
//   1: scales    predecoded f32 (ds, dm) pairs
//   2: x         GEMV input activation (attn_out)
//   3: residual  f32 IN/OUT; residual[row] += GEMV(row, x)
//   4: rows
//   5: cols
kernel void gemm_q4_k_v4_predec_2r_add(
    device const uchar* w_q4     [[buffer(0)]],
    device const float* scales   [[buffer(1)]],
    device const float* x        [[buffer(2)]],
    device       float* residual [[buffer(3)]],
    constant     uint&  rows     [[buffer(4)]],
    constant     uint&  cols     [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 16u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < rows;
    uint r1 = has1 ? row1 : row0;

    uint  blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f;
    float p1 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul, so0 = rs0 + (uint64_t)b * 16ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 144ul, so1 = rs1 + (uint64_t)b * 16ul;

        float ds0[8], dm0[8], ds1[8], dm1[8];
        for (uint s = 0; s < 8u; ++s) {
            ds0[s] = scales[so0 + (uint64_t)(s * 2u)];
            dm0[s] = scales[so0 + (uint64_t)(s * 2u + 1u)];
            ds1[s] = scales[so1 + (uint64_t)(s * 2u)];
            dm1[s] = scales[so1 + (uint64_t)(s * 2u + 1u)];
        }

        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p0 += (ds0[k0] * (float)(q0 & 0x0Fu) - dm0[k0]) * xl[k0];
            p0 += (ds0[k1] * (float)(q0 >> 4u)   - dm0[k1]) * xl[k1];
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p1 += (ds1[k0] * (float)(q1 & 0x0Fu) - dm1[k0]) * xl[k0];
            p1 += (ds1[k1] * (float)(q1 >> 4u)   - dm1[k1]) * xl[k1];
        }
    }

    p0 = simd_sum(p0);
    if (simd_lane == 0u) residual[row0] = residual[row0] + p0;
    if (has1) {
        p1 = simd_sum(p1);
        if (simd_lane == 0u) residual[row1] = residual[row1] + p1;
    }
}

// ── gemm_q4_k_v4_predec_4r_add ───────────────────────────────────────────────
// 4-row-per-simdgroup upgrade of gemm_q4_k_v4_predec_2r_add (Track B4,
// opt-in via DISMANTLE_QWEN_OPROJ_4R=1).  Same in-place residual-add geometry
// but 8 FMA chains (vs 4 for 2r) and inline scale access to keep register
// pressure low (~20 floats per thread).  vs 2r_add:
//   • 4 accumulators → wider ILP for the GPU pipeline
//   • 32 rows/TG instead of 16 → ½ the dispatch overhead
//   • inline scale reads — no preload array, compiler hides loads behind FMAs
// Bit-identical to 2r_add (same per-accumulator FMA order).
// Grid: (ceil(rows/32)*256,1,1)  TG: (256,1,1)
kernel void gemm_q4_k_v4_predec_4r_add(
    device const uchar* w_q4     [[buffer(0)]],
    device const float* scales   [[buffer(1)]],
    device const float* x        [[buffer(2)]],
    device       float* residual [[buffer(3)]],
    constant     uint&  rows     [[buffer(4)]],
    constant     uint&  cols     [[buffer(5)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 32u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u, row2 = row0 + 16u, row3 = row0 + 24u;
    bool has1 = row1 < rows, has2 = row2 < rows, has3 = row3 < rows;
    uint r1 = has1 ? row1 : row0;
    uint r2 = has2 ? row2 : row0;
    uint r3 = has3 ? row3 : row0;

    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1   * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb2 = (uint64_t)r2   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs2 = (uint64_t)r2   * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb3 = (uint64_t)r3   * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs3 = (uint64_t)r3   * (uint64_t)blocks_per_row * 16ul;

    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;

    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b*144ul, so0 = rs0 + (uint64_t)b*16ul;
        uint64_t bo1 = rb1 + (uint64_t)b*144ul, so1 = rs1 + (uint64_t)b*16ul;
        uint64_t bo2 = rb2 + (uint64_t)b*144ul, so2 = rs2 + (uint64_t)b*16ul;
        uint64_t bo3 = rb3 + (uint64_t)b*144ul, so3 = rs3 + (uint64_t)b*16ul;

        float xl[8];
        for (uint k = 0u; k < 8u; ++k)
            xl[k] = x[(uint64_t)b*256ul + (uint64_t)(k*32u + simd_lane)];

        for (uint pi = 0u; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            float x0 = xl[k0], x1 = xl[k1];

            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p0 += (scales[so0+(uint64_t)(k0*2u)]   * (float)(q0&0x0Fu) - scales[so0+(uint64_t)(k0*2u+1u)]) * x0;
            p0 += (scales[so0+(uint64_t)(k1*2u)]   * (float)(q0>>4u)   - scales[so0+(uint64_t)(k1*2u+1u)]) * x1;

            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p1 += (scales[so1+(uint64_t)(k0*2u)]   * (float)(q1&0x0Fu) - scales[so1+(uint64_t)(k0*2u+1u)]) * x0;
            p1 += (scales[so1+(uint64_t)(k1*2u)]   * (float)(q1>>4u)   - scales[so1+(uint64_t)(k1*2u+1u)]) * x1;

            uchar q2 = w_q4[bo2 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p2 += (scales[so2+(uint64_t)(k0*2u)]   * (float)(q2&0x0Fu) - scales[so2+(uint64_t)(k0*2u+1u)]) * x0;
            p2 += (scales[so2+(uint64_t)(k1*2u)]   * (float)(q2>>4u)   - scales[so2+(uint64_t)(k1*2u+1u)]) * x1;

            uchar q3 = w_q4[bo3 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p3 += (scales[so3+(uint64_t)(k0*2u)]   * (float)(q3&0x0Fu) - scales[so3+(uint64_t)(k0*2u+1u)]) * x0;
            p3 += (scales[so3+(uint64_t)(k1*2u)]   * (float)(q3>>4u)   - scales[so3+(uint64_t)(k1*2u+1u)]) * x1;
        }
    }

    p0 = simd_sum(p0);
    if (simd_lane == 0u) residual[row0] = residual[row0] + p0;
    if (has1) { p1 = simd_sum(p1); if (simd_lane == 0u) residual[row1] = residual[row1] + p1; }
    if (has2) { p2 = simd_sum(p2); if (simd_lane == 0u) residual[row2] = residual[row2] + p2; }
    if (has3) { p3 = simd_sum(p3); if (simd_lane == 0u) residual[row3] = residual[row3] + p3; }
}

// ── gemm_q4_k_v4_predec_2r_f16s ──────────────────────────────────────────────
// f16-scales variant of _2r (Stage 2, 2026-05-30). Identical math + 2-row ILP,
// but the pre-decoded sub-block scales are read as `half` (2 B) instead of f32
// (4 B), cutting predec bytes/block 192→160 (−17%) on the bandwidth-bound Q4_K
// GEMV (the profiling-confirmed 76%-of-time wall). Scales widen to float in
// register. NOT bit-identical (f16 scale rounding) — gate at atol 1e-3 fp16.
// Opt-in via DISMANTLE_QWEN_PREDEC_F16SCALES=1. Scale table built by
// kernels::predecode_q4_k_scale_table_f16 (16 halfs/block, same element layout).
// Grid/threadgroup identical to _2r: (ceil(rows/16)*256,1,1) / (256,1,1).
kernel void gemm_q4_k_v4_predec_2r_f16s(
    device const uchar* w_q4    [[buffer(0)]],
    device const half*  scales  [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device       float* y       [[buffer(3)]],
    constant     uint&  rows    [[buffer(4)]],
    constant     uint&  cols    [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 16u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < rows;
    uint r1 = has1 ? row1 : row0;

    uint  blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f;
    float p1 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul, so0 = rs0 + (uint64_t)b * 16ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 144ul, so1 = rs1 + (uint64_t)b * 16ul;

        float ds0[8], dm0[8], ds1[8], dm1[8];
        for (uint s = 0; s < 8u; ++s) {
            ds0[s] = (float)scales[so0 + (uint64_t)(s * 2u)];
            dm0[s] = (float)scales[so0 + (uint64_t)(s * 2u + 1u)];
            ds1[s] = (float)scales[so1 + (uint64_t)(s * 2u)];
            dm1[s] = (float)scales[so1 + (uint64_t)(s * 2u + 1u)];
        }

        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p0 += (ds0[k0] * (float)(q0 & 0x0Fu) - dm0[k0]) * xl[k0];
            p0 += (ds0[k1] * (float)(q0 >> 4u)   - dm0[k1]) * xl[k1];
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p1 += (ds1[k0] * (float)(q1 & 0x0Fu) - dm1[k0]) * xl[k0];
            p1 += (ds1[k1] * (float)(q1 >> 4u)   - dm1[k1]) * xl[k1];
        }
    }

    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) {
        p1 = simd_sum(p1);
        if (simd_lane == 0u) y[row1] = p1;
    }
}

// ── gemm_q4_k_v4_predec_4r ───────────────────────────────────────────────────
// 4-rows-per-simdgroup predec GEMV (Stage 2, 2026-05-30). Direct extension of
// _2r: each simdgroup computes FOUR output rows of the same matrix with four
// independent accumulator chains sharing one `x` load — 4 in-flight weight-load
// streams per thread to hide DRAM latency further on decode (M=1). Identical
// per-row math => bit-identical. 32 rows/TG (8 simdgroups x 4 rows). Opt-in via
// DISMANTLE_QWEN_PREDEC_4R=1; bench decides vs _2r (register pressure may bite).
//
// Grid: (ceil(rows/32)*256, 1, 1)   threadgroup: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_4r(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device       float* y       [[buffer(3)]],
    constant     uint&  rows    [[buffer(4)]],
    constant     uint&  cols    [[buffer(5)]],
    uint                gid       [[threadgroup_position_in_grid]],
    uint                simd_lane [[thread_index_in_simdgroup]],
    uint                simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 32u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u, row2 = row0 + 16u, row3 = row0 + 24u;
    bool has1 = row1 < rows, has2 = row2 < rows, has3 = row3 < rows;
    // Alias missing rows to row0 so inner reads stay in-bounds; results unwritten.
    uint r1 = has1 ? row1 : row0;
    uint r2 = has2 ? row2 : row0;
    uint r3 = has3 ? row3 : row0;

    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb2 = (uint64_t)r2 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs2 = (uint64_t)r2 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb3 = (uint64_t)r3 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs3 = (uint64_t)r3 * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul, so0 = rs0 + (uint64_t)b * 16ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 144ul, so1 = rs1 + (uint64_t)b * 16ul;
        uint64_t bo2 = rb2 + (uint64_t)b * 144ul, so2 = rs2 + (uint64_t)b * 16ul;
        uint64_t bo3 = rb3 + (uint64_t)b * 144ul, so3 = rs3 + (uint64_t)b * 16ul;

        // x shared across all four rows — load once.
        float xl[8];
        for (uint k = 0; k < 8u; ++k)
            xl[k] = x[(uint64_t)b * 256ul + (uint64_t)(k * 32u + simd_lane)];

        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            float x0 = xl[k0], x1 = xl[k1];
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p0 += (scales[so0 + (uint64_t)(k0 * 2u)] * (float)(q0 & 0x0Fu) - scales[so0 + (uint64_t)(k0 * 2u + 1u)]) * x0;
            p0 += (scales[so0 + (uint64_t)(k1 * 2u)] * (float)(q0 >> 4u)   - scales[so0 + (uint64_t)(k1 * 2u + 1u)]) * x1;
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p1 += (scales[so1 + (uint64_t)(k0 * 2u)] * (float)(q1 & 0x0Fu) - scales[so1 + (uint64_t)(k0 * 2u + 1u)]) * x0;
            p1 += (scales[so1 + (uint64_t)(k1 * 2u)] * (float)(q1 >> 4u)   - scales[so1 + (uint64_t)(k1 * 2u + 1u)]) * x1;
            uchar q2 = w_q4[bo2 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p2 += (scales[so2 + (uint64_t)(k0 * 2u)] * (float)(q2 & 0x0Fu) - scales[so2 + (uint64_t)(k0 * 2u + 1u)]) * x0;
            p2 += (scales[so2 + (uint64_t)(k1 * 2u)] * (float)(q2 >> 4u)   - scales[so2 + (uint64_t)(k1 * 2u + 1u)]) * x1;
            uchar q3 = w_q4[bo3 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p3 += (scales[so3 + (uint64_t)(k0 * 2u)] * (float)(q3 & 0x0Fu) - scales[so3 + (uint64_t)(k0 * 2u + 1u)]) * x0;
            p3 += (scales[so3 + (uint64_t)(k1 * 2u)] * (float)(q3 >> 4u)   - scales[so3 + (uint64_t)(k1 * 2u + 1u)]) * x1;
        }
    }

    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) { p1 = simd_sum(p1); if (simd_lane == 0u) y[row1] = p1; }
    if (has2) { p2 = simd_sum(p2); if (simd_lane == 0u) y[row2] = p2; }
    if (has3) { p3 = simd_sum(p3); if (simd_lane == 0u) y[row3] = p3; }
}

// ── gemm_q6_k_kv_pair ────────────────────────────────────────────────────────
// Track 3.8 — Fused K+V Q6_K GEMV pair. Computes both K and V projections in
// one dispatch, sharing the x (x_norm_buf) read. Saves 1 dispatch/layer × n_layers
// = 28 on Qwen-3B. Both K and V have the same shape (kv_dim × hidden).
//
// The caller binds the same mmap model buffer at two indices with different
// byte offsets so w_k[0] and w_v[0] each start at their respective weight rows.
//
// Buffer layout:
//   0: w_k   (uchar*, K weight bytes, starting at K weight offset)
//   1: w_v   (uchar*, V weight bytes, starting at V weight offset)
//   2: x     (float*, hidden-length input, x_norm_buf)
//   3: y_k   (float*, kv_dim-length K output)
//   4: y_v   (float*, kv_dim-length V output)
//   5: args  (ArgbufRowsCols: rows=kv_dim, cols=hidden)
//
// Grid: (ceil(rows/8)*256, 1, 1)  TG: (256, 1, 1)  — identical to gemm_q6_k_fused_v2.
kernel void gemm_q6_k_kv_pair(
    device const uchar* w_k  [[buffer(0)]],
    device const uchar* w_v  [[buffer(1)]],
    device const float* x    [[buffer(2)]],
    device       float* y_k  [[buffer(3)]],
    device       float* y_v  [[buffer(4)]],
    constant ArgbufRowsCols& args [[buffer(5)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= args.rows) return;

    uint blocks_per_row = args.cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 210ul;

    // Per-lane constants — identical to gemm_q6_k_fused_v2.
    uint half_idx          = simd_lane >> 4u;
    uint group             = (simd_lane >> 2u) & 3u;
    uint l_base            = (simd_lane & 3u) * 8u;
    uint scale_l_off       = l_base >> 4u;
    uint scale_byte_off    = 192u + half_idx * 8u + scale_l_off + group * 2u;
    uint ql_group_off      = (group & 1u) * 32u;
    bool group_high_nibble = (group >= 2u);
    uint qh_shift          = group * 2u;
    uint tid_base          = half_idx * 128u + group * 32u + l_base;

    float pk = 0.0f, pv = 0.0f;

    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bk = row_byte_off + (uint64_t)b * 210ul;
        uint64_t bv = bk; // same offset structure, different buffers

        // Scale/delta for K and V
        ushort dk_bits = (ushort)w_k[bk+208u] | ((ushort)w_k[bk+209u] << 8);
        float dscalek = (float)as_type<half>(dk_bits) * (float)(int)(signed char)w_k[bk + (uint64_t)scale_byte_off];

        ushort dv_bits = (ushort)w_v[bv+208u] | ((ushort)w_v[bv+209u] << 8);
        float dscalev = (float)as_type<half>(dv_bits) * (float)(int)(signed char)w_v[bv + (uint64_t)scale_byte_off];

        uint64_t kql = bk + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
        uint64_t kqh = bk + 128ul + (uint64_t)half_idx * 32ul;
        uint64_t vql = bv + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
        uint64_t vqh = bv + 128ul + (uint64_t)half_idx * 32ul;

        float acc_k = 0.0f, acc_v = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            uint l = l_base + k;
            float xi = x[b * 256u + tid_base + k]; // one x load for both K and V

            uchar kqlb = w_k[kql + (uint64_t)l];
            uint klow = group_high_nibble ? ((uint)(kqlb >> 4) & 0xFu) : ((uint)kqlb & 0xFu);
            uint khi  = ((uint)w_k[kqh + (uint64_t)l] >> qh_shift) & 0x3u;
            acc_k += (float)((int)(klow | (khi << 4)) - 32) * xi;

            uchar vqlb = w_v[vql + (uint64_t)l];
            uint vlow = group_high_nibble ? ((uint)(vqlb >> 4) & 0xFu) : ((uint)vqlb & 0xFu);
            uint vhi  = ((uint)w_v[vqh + (uint64_t)l] >> qh_shift) & 0x3u;
            acc_v += (float)((int)(vlow | (vhi << 4)) - 32) * xi;
        }
        pk += dscalek * acc_k;
        pv += dscalev * acc_v;
    }

    pk = simd_sum(pk);
    pv = simd_sum(pv);
    if (simd_lane == 0u) {
        y_k[base_row] = pk;
        y_v[base_row] = pv;
    }
}

// ── gemm_q4k_predec_q6k_pair ─────────────────────────────────────────────────
// Track 3.9 — Cross-dtype K+V pair for layers where k_proj=Q4_K (predec) and
// v_proj=Q6_K. ONE dispatch computes both projections, each with its own weight
// format. Saves 1 dispatch/layer for mixed-dtype attention layers.
//
// Grid: (2 * ceil(kv_dim/8) * 256, 1, 1), TG: (256, 1, 1).
// First n_tg threadgroups → K (Q4_K predec); next n_tg → V (Q6_K inline).
//
// Buffer layout:
//   0: w_k    (uchar*, Q4_K weight bytes for K projection)
//   1: k_sc   (float*, pre-decoded scale table, 16 floats/block of 256)
//   2: w_v    (uchar*, Q6_K weight bytes for V projection)
//   3: x      (float*, hidden-length input)
//   4: y_k    (float*, kv_dim-length K output)
//   5: y_v    (float*, kv_dim-length V output)
//   6: rows   (uint, kv_dim)
//   7: cols   (uint, hidden, must be multiple of 256)
kernel void gemm_q4k_predec_q6k_pair(
    device const uchar* w_k    [[buffer(0)]],
    device const float* k_sc   [[buffer(1)]],
    device const uchar* w_v    [[buffer(2)]],
    device const float* x      [[buffer(3)]],
    device       float* y_k    [[buffer(4)]],
    device       float* y_v    [[buffer(5)]],
    constant     uint&  rows   [[buffer(6)]],
    constant     uint&  cols   [[buffer(7)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n_tg = (rows + 7u) / 8u;
    bool is_k = (gid < n_tg);
    uint eff_gid  = is_k ? gid : (gid - n_tg);
    uint base_row = eff_gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint blocks_per_row = cols / 256u;

    if (is_k) {
        // K projection: Q4_K predec math (identical to gemm_q4_k_v4_predec)
        uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
        uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
        float pk = 0.0f;
        for (uint b = 0; b < blocks_per_row; ++b) {
            uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
            uint64_t so = row_scale_off + (uint64_t)b * 16ul;
            float ds[8], dm[8];
            for (uint sub = 0u; sub < 8u; ++sub) {
                ds[sub] = k_sc[so + (uint64_t)(sub * 2u)];
                dm[sub] = k_sc[so + (uint64_t)(sub * 2u + 1u)];
            }
            for (uint pi = 0u; pi < 4u; ++pi) {
                uint k0 = pi * 2u, k1 = k0 + 1u;
                uchar qb = w_k[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
                float xi0 = x[(uint64_t)b * 256ul + (uint64_t)(k0 * 32u + simd_lane)];
                float xi1 = x[(uint64_t)b * 256ul + (uint64_t)(k1 * 32u + simd_lane)];
                pk += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xi0;
                pk += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xi1;
            }
        }
        pk = simd_sum(pk);
        if (simd_lane == 0u) y_k[base_row] = pk;
    } else {
        // V projection: Q6_K inline math (identical to gemm_q6_k_fused_v2)
        uint64_t row_byte_off   = (uint64_t)base_row * (uint64_t)blocks_per_row * 210ul;
        uint half_idx           = simd_lane >> 4u;
        uint group              = (simd_lane >> 2u) & 3u;
        uint l_base             = (simd_lane & 3u) * 8u;
        uint scale_l_off        = l_base >> 4u;
        uint scale_byte_off     = 192u + half_idx * 8u + scale_l_off + group * 2u;
        uint ql_group_off       = (group & 1u) * 32u;
        bool group_high_nibble  = (group >= 2u);
        uint qh_shift           = group * 2u;
        uint tid_base           = half_idx * 128u + group * 32u + l_base;
        float pv = 0.0f;
        for (uint b = 0u; b < blocks_per_row; ++b) {
            uint64_t bv     = row_byte_off + (uint64_t)b * 210ul;
            ushort dv_bits  = (ushort)w_v[bv+208u] | ((ushort)w_v[bv+209u] << 8);
            float dscalev   = (float)as_type<half>(dv_bits) *
                              (float)(int)(signed char)w_v[bv + (uint64_t)scale_byte_off];
            uint64_t vql    = bv + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
            uint64_t vqh    = bv + 128ul + (uint64_t)half_idx * 32ul;
            float acc = 0.0f;
            for (uint k = 0u; k < 8u; ++k) {
                uint l    = l_base + k;
                float xi  = x[b * 256u + tid_base + k];
                uchar vqlb = w_v[vql + (uint64_t)l];
                uint vlow  = group_high_nibble ? ((uint)(vqlb >> 4) & 0xFu) : ((uint)vqlb & 0xFu);
                uint vhi   = ((uint)w_v[vqh + (uint64_t)l] >> qh_shift) & 0x3u;
                acc       += (float)((int)(vlow | (vhi << 4)) - 32) * xi;
            }
            pv += dscalev * acc;
        }
        pv = simd_sum(pv);
        if (simd_lane == 0u) y_v[base_row] = pv;
    }
}

// ── gemm_q4k_predec_qkv_triple ───────────────────────────────────────────────
// Track 3.10 — Fused Q+K+V Q4_K predec GEMV triple. All three projections share
// the same input activation and use Q4_K predec format (pre-decoded scale tables).
// Saves 1 dispatch/layer for layers where all three (q_proj, k_proj, v_proj) are
// Q4_K with predec tables — replacing the Q dispatch + KV-pair dispatch (2→1).
//
// Q rows ≠ KV rows (GQA): Q has q_dim rows, K and V have kv_dim rows each.
// Grid: ((ceil(q_dim/8) + 2*ceil(kv_dim/8)) * 256, 1, 1), TG: (256, 1, 1).
// First n_tg_q TGs → Q; next n_tg_kv → K; last n_tg_kv → V.
//
// Buffer layout:
//   0: wq      (uchar*, Q4_K Q weight bytes)
//   1: q_sc    (float*, Q predec scale table)
//   2: wk      (uchar*, Q4_K K weight bytes)
//   3: k_sc    (float*, K predec scale table)
//   4: wv      (uchar*, Q4_K V weight bytes)
//   5: v_sc    (float*, V predec scale table)
//   6: x       (float*, hidden-length input)
//   7: yq      (float*, q_dim-length Q output)
//   8: yk      (float*, kv_dim-length K output)
//   9: yv      (float*, kv_dim-length V output)
//  10: q_rows  (uint, q_dim)
//  11: kv_rows (uint, kv_dim)
//  12: cols    (uint, hidden, must be multiple of 256)
kernel void gemm_q4k_predec_qkv_triple(
    device const uchar* wq     [[buffer(0)]],
    device const float* q_sc   [[buffer(1)]],
    device const uchar* wk     [[buffer(2)]],
    device const float* k_sc   [[buffer(3)]],
    device const uchar* wv     [[buffer(4)]],
    device const float* v_sc   [[buffer(5)]],
    device const float* x      [[buffer(6)]],
    device       float* yq     [[buffer(7)]],
    device       float* yk     [[buffer(8)]],
    device       float* yv     [[buffer(9)]],
    constant     uint&  q_rows  [[buffer(10)]],
    constant     uint&  kv_rows [[buffer(11)]],
    constant     uint&  cols    [[buffer(12)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n_tg_q  = (q_rows  + 7u) / 8u;
    uint n_tg_kv = (kv_rows + 7u) / 8u;

    device const uchar* w;
    device const float* sc;
    device       float* y;
    uint rows, eff_gid;

    if (gid < n_tg_q) {
        w = wq; sc = q_sc; y = yq; rows = q_rows; eff_gid = gid;
    } else if (gid < n_tg_q + n_tg_kv) {
        w = wk; sc = k_sc; y = yk; rows = kv_rows; eff_gid = gid - n_tg_q;
    } else {
        w = wv; sc = v_sc; y = yv; rows = kv_rows; eff_gid = gid - (n_tg_q + n_tg_kv);
    }

    uint base_row = eff_gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
    float partial = 0.0f;

    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;
        float ds[8], dm[8];
        for (uint sub = 0u; sub < 8u; ++sub) {
            ds[sub] = sc[so + (uint64_t)(sub * 2u)];
            dm[sub] = sc[so + (uint64_t)(sub * 2u + 1u)];
        }
        for (uint pi = 0u; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qb = w[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            float xi0 = x[(uint64_t)b * 256ul + (uint64_t)(k0 * 32u + simd_lane)];
            float xi1 = x[(uint64_t)b * 256ul + (uint64_t)(k1 * 32u + simd_lane)];
            partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xi0;
            partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xi1;
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) y[base_row] = partial;
}

// ── gemm_q4k_q4k_q6k_triple ──────────────────────────────────────────────────
// Track 3.11 — Mixed Q+K(Q4K predec)+V(Q6K) triple for layers where
// q/k_proj=Q4_K predec but v_proj=Q6_K. Fuses all three into one dispatch.
// Same segmented-grid approach as gemm_q4k_predec_qkv_triple but V uses Q6_K
// inline math. Saves 1 dispatch/layer vs q(separate)+kv_cross_dtype_pair.
//
// Grid: ((ceil(q_rows/8) + ceil(kv_rows/8) + ceil(kv_rows/8)) * 256, 1, 1)
// TG: (256, 1, 1)
// TG segments: [0, n_tg_q) → Q (Q4K predec)
//              [n_tg_q, n_tg_q+n_tg_kv) → K (Q4K predec)
//              [n_tg_q+n_tg_kv, total) → V (Q6K inline)
//
// Buffer layout:
//   0: wq      (uchar*, Q4_K Q weights)   1: q_sc (float*, Q predec scales)
//   2: wk      (uchar*, Q4_K K weights)   3: k_sc (float*, K predec scales)
//   4: wv      (uchar*, Q6_K V weights)
//   5: x       (float*, hidden input)
//   6: yq / 7: yk / 8: yv  (float* outputs)
//   9: q_rows  10: kv_rows  11: cols
kernel void gemm_q4k_q4k_q6k_triple(
    device const uchar* wq     [[buffer(0)]],
    device const float* q_sc   [[buffer(1)]],
    device const uchar* wk     [[buffer(2)]],
    device const float* k_sc   [[buffer(3)]],
    device const uchar* wv     [[buffer(4)]],
    device const float* x      [[buffer(5)]],
    device       float* yq     [[buffer(6)]],
    device       float* yk     [[buffer(7)]],
    device       float* yv     [[buffer(8)]],
    constant     uint&  q_rows  [[buffer(9)]],
    constant     uint&  kv_rows [[buffer(10)]],
    constant     uint&  cols    [[buffer(11)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint n_tg_q  = (q_rows  + 7u) / 8u;
    uint n_tg_kv = (kv_rows + 7u) / 8u;
    uint blocks_per_row = cols / 256u;

    if (gid < n_tg_q + n_tg_kv) {
        // Q or K — Q4_K predec math
        device const uchar* w;
        device const float* sc;
        device       float* y;
        uint rows, eff_gid;
        if (gid < n_tg_q) {
            w = wq; sc = q_sc; y = yq; rows = q_rows; eff_gid = gid;
        } else {
            w = wk; sc = k_sc; y = yk; rows = kv_rows; eff_gid = gid - n_tg_q;
        }
        uint base_row = eff_gid * 8u + simd_id;
        if (base_row >= rows) return;
        uint64_t row_byte_off  = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;
        uint64_t row_scale_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 16ul;
        float partial = 0.0f;
        for (uint b = 0u; b < blocks_per_row; ++b) {
            uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
            uint64_t so = row_scale_off + (uint64_t)b * 16ul;
            float ds[8], dm[8];
            for (uint sub = 0u; sub < 8u; ++sub) {
                ds[sub] = sc[so + (uint64_t)(sub * 2u)];
                dm[sub] = sc[so + (uint64_t)(sub * 2u + 1u)];
            }
            for (uint pi = 0u; pi < 4u; ++pi) {
                uint k0 = pi * 2u, k1 = k0 + 1u;
                uchar qb = w[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
                float xi0 = x[(uint64_t)b * 256ul + (uint64_t)(k0 * 32u + simd_lane)];
                float xi1 = x[(uint64_t)b * 256ul + (uint64_t)(k1 * 32u + simd_lane)];
                partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xi0;
                partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xi1;
            }
        }
        partial = simd_sum(partial);
        if (simd_lane == 0u) y[base_row] = partial;
    } else {
        // V — Q6_K inline math
        uint eff_gid  = gid - (n_tg_q + n_tg_kv);
        uint base_row = eff_gid * 8u + simd_id;
        if (base_row >= kv_rows) return;
        uint64_t row_byte_off   = (uint64_t)base_row * (uint64_t)blocks_per_row * 210ul;
        uint half_idx           = simd_lane >> 4u;
        uint group              = (simd_lane >> 2u) & 3u;
        uint l_base             = (simd_lane & 3u) * 8u;
        uint scale_l_off        = l_base >> 4u;
        uint scale_byte_off     = 192u + half_idx * 8u + scale_l_off + group * 2u;
        uint ql_group_off       = (group & 1u) * 32u;
        bool group_high_nibble  = (group >= 2u);
        uint qh_shift           = group * 2u;
        uint tid_base           = half_idx * 128u + group * 32u + l_base;
        float pv = 0.0f;
        for (uint b = 0u; b < blocks_per_row; ++b) {
            uint64_t bv     = row_byte_off + (uint64_t)b * 210ul;
            ushort dv_bits  = (ushort)wv[bv+208u] | ((ushort)wv[bv+209u] << 8);
            float dscalev   = (float)as_type<half>(dv_bits) *
                              (float)(int)(signed char)wv[bv + (uint64_t)scale_byte_off];
            uint64_t vql    = bv + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
            uint64_t vqh    = bv + 128ul + (uint64_t)half_idx * 32ul;
            float acc = 0.0f;
            for (uint k = 0u; k < 8u; ++k) {
                uint l    = l_base + k;
                float xi  = x[b * 256u + tid_base + k];
                uchar vqlb = wv[vql + (uint64_t)l];
                uint vlow  = group_high_nibble ? ((uint)(vqlb >> 4) & 0xFu) : ((uint)vqlb & 0xFu);
                uint vhi   = ((uint)wv[vqh + (uint64_t)l] >> qh_shift) & 0x3u;
                acc       += (float)((int)(vlow | (vhi << 4)) - 32) * xi;
            }
            pv += dscalev * acc;
        }
        pv = simd_sum(pv);
        if (simd_lane == 0u) yv[base_row] = pv;
    }
}

struct ArgbufQkvRopeAppend {
    uint  q_rows;
    uint  kv_rows;
    uint  cols;
    uint  n_q_heads;
    uint  n_k_heads;
    uint  head_dim;
    uint  pos;
    uint  kv_off;
    uint  has_q_bias;
    uint  has_k_bias;
    uint  has_v_bias;
    float base;
};

static inline float q4k_predec_dot_row(
    device const uchar* w,
    device const float* sc,
    device const float* x,
    uint row,
    uint cols,
    uint simd_lane)
{
    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)row * (uint64_t)blocks_per_row * 16ul;
    float partial = 0.0f;

    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;
        float ds[8], dm[8];
        for (uint sub = 0u; sub < 8u; ++sub) {
            ds[sub] = sc[so + (uint64_t)(sub * 2u)];
            dm[sub] = sc[so + (uint64_t)(sub * 2u + 1u)];
        }
        for (uint pi = 0u; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qb = w[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            float xi0 = x[(uint64_t)b * 256ul + (uint64_t)(k0 * 32u + simd_lane)];
            float xi1 = x[(uint64_t)b * 256ul + (uint64_t)(k1 * 32u + simd_lane)];
            partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xi0;
            partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xi1;
        }
    }
    return simd_sum(partial);
}

// Track D3 (2026-06-06): f16-scales variant of q4k_predec_dot_row.
// Reads the predecoded sub-block scale table as half (2 B) instead of float
// (4 B). For Q projection (2048 rows × 2048 cols = 8 blocks/row): saves
// 2048 × 8 × 32 B = 512 KB per token. Widen to float in register.
static inline float q4k_predec_dot_row_f16s(
    device const uchar* w,
    device const half*  sc,
    device const float* x,
    uint row,
    uint cols,
    uint simd_lane)
{
    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off  = (uint64_t)row * (uint64_t)blocks_per_row * 144ul;
    uint64_t row_scale_off = (uint64_t)row * (uint64_t)blocks_per_row * 16ul;
    float partial = 0.0f;

    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off  + (uint64_t)b * 144ul;
        uint64_t so = row_scale_off + (uint64_t)b * 16ul;
        float ds[8], dm[8];
        for (uint sub = 0u; sub < 8u; ++sub) {
            ds[sub] = (float)sc[so + (uint64_t)(sub * 2u)];
            dm[sub] = (float)sc[so + (uint64_t)(sub * 2u + 1u)];
        }
        for (uint pi = 0u; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uchar qb = w[bo + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            float xi0 = x[(uint64_t)b * 256ul + (uint64_t)(k0 * 32u + simd_lane)];
            float xi1 = x[(uint64_t)b * 256ul + (uint64_t)(k1 * 32u + simd_lane)];
            partial += (ds[k0] * (float)(qb & 0x0Fu) - dm[k0]) * xi0;
            partial += (ds[k1] * (float)(qb >> 4u)   - dm[k1]) * xi1;
        }
    }
    return simd_sum(partial);
}

static inline float q6k_dot_row(
    device const uchar* w,
    device const float* x,
    uint row,
    uint cols,
    uint simd_lane)
{
    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off   = (uint64_t)row * (uint64_t)blocks_per_row * 210ul;
    uint half_idx           = simd_lane >> 4u;
    uint group              = (simd_lane >> 2u) & 3u;
    uint l_base             = (simd_lane & 3u) * 8u;
    uint scale_l_off        = l_base >> 4u;
    uint scale_byte_off     = 192u + half_idx * 8u + scale_l_off + group * 2u;
    uint ql_group_off       = (group & 1u) * 32u;
    bool group_high_nibble  = (group >= 2u);
    uint qh_shift           = group * 2u;
    uint tid_base           = half_idx * 128u + group * 32u + l_base;
    float partial = 0.0f;

    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint64_t bo     = row_byte_off + (uint64_t)b * 210ul;
        ushort d_bits   = (ushort)w[bo + 208u] | ((ushort)w[bo + 209u] << 8);
        float dscale    = (float)as_type<half>(d_bits) *
                          (float)(int)(signed char)w[bo + (uint64_t)scale_byte_off];
        uint64_t ql     = bo + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
        uint64_t qh     = bo + 128ul + (uint64_t)half_idx * 32ul;
        float acc = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            uint l    = l_base + k;
            float xi  = x[b * 256u + tid_base + k];
            uchar qlb = w[ql + (uint64_t)l];
            uint qlow = group_high_nibble ? ((uint)(qlb >> 4) & 0xFu) : ((uint)qlb & 0xFu);
            uint qhi  = ((uint)w[qh + (uint64_t)l] >> qh_shift) & 0x3u;
            acc      += (float)((int)(qlow | (qhi << 4)) - 32) * xi;
        }
        partial += dscale * acc;
    }
    return simd_sum(partial);
}

static inline void write_rope_pair(
    device float* dst,
    device const float* bias,
    uint has_bias,
    uint row0,
    float v0,
    float v1,
    constant ArgbufQkvRopeAppend& args)
{
    float x0 = v0 + (has_bias != 0u ? bias[row0] : 0.0f);
    float x1 = v1 + (has_bias != 0u ? bias[row0 + 1u] : 0.0f);
    uint pair = (row0 % args.head_dim) / 2u;
    float theta = (float)args.pos / pow(args.base, 2.0f * float(pair) / float(args.head_dim));
    float c = cos(theta), s = sin(theta);
    dst[row0]      = x0 * c - x1 * s;
    dst[row0 + 1u] = x0 * s + x1 * c;
}

kernel void gemm_q4k_predec_qkv_rope_append(
    device const uchar* wq      [[buffer(0)]],
    device const float* q_sc    [[buffer(1)]],
    device const uchar* wk      [[buffer(2)]],
    device const float* k_sc    [[buffer(3)]],
    device const uchar* wv      [[buffer(4)]],
    device const float* v_sc    [[buffer(5)]],
    device const float* x       [[buffer(6)]],
    device       float* q_out   [[buffer(7)]],
    device       float* k_cache [[buffer(8)]],
    device       float* v_cache [[buffer(9)]],
    device const float* q_bias  [[buffer(10)]],
    device const float* k_bias  [[buffer(11)]],
    device const float* v_bias  [[buffer(12)]],
    constant ArgbufQkvRopeAppend& args [[buffer(13)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint q_pairs = args.q_rows / 2u;
    uint k_pairs = args.kv_rows / 2u;
    uint n_tg_q  = (q_pairs + 7u) / 8u;
    uint n_tg_k  = (k_pairs + 7u) / 8u;
    uint n_tg_v  = (args.kv_rows + 7u) / 8u;

    if (gid < n_tg_q) {
        uint pair_idx = gid * 8u + simd_id;
        if (pair_idx >= q_pairs) return;
        uint row0 = pair_idx * 2u;
        float p0 = q4k_predec_dot_row(wq, q_sc, x, row0, args.cols, simd_lane);
        float p1 = q4k_predec_dot_row(wq, q_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0, p0, p1, args);
        }
    } else if (gid < n_tg_q + n_tg_k) {
        uint pair_idx = (gid - n_tg_q) * 8u + simd_id;
        if (pair_idx >= k_pairs) return;
        uint row0 = pair_idx * 2u;
        float p0 = q4k_predec_dot_row(wk, k_sc, x, row0, args.cols, simd_lane);
        float p1 = q4k_predec_dot_row(wk, k_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0, p0, p1, args);
        }
    } else {
        uint eff_gid = gid - (n_tg_q + n_tg_k);
        if (eff_gid >= n_tg_v) return;
        uint row = eff_gid * 8u + simd_id;
        if (row >= args.kv_rows) return;
        float pv = q4k_predec_dot_row(wv, v_sc, x, row, args.cols, simd_lane);
        if (simd_lane == 0u) {
            v_cache[args.kv_off + row] = pv + (args.has_v_bias != 0u ? v_bias[row] : 0.0f);
        }
    }
}

// Track C28: 4r variant of gemm_q4k_predec_qkv_rope_append.
// Q: 4 rows/simdgroup (2 RoPE pairs) — 64 TGs for Qwen-3B vs 128
// K: 4 rows/simdgroup (2 RoPE pairs) — 32 TGs vs 64
// V: 2 rows/simdgroup (no RoPE)      — 64 TGs vs 128
// Total: 160 TGs vs 320 TGs — same single dispatch, half the scheduling overhead.
// Requires q_rows % 4 == 0 and kv_rows % 4 == 0 (validated on Rust side).
kernel void gemm_q4k_predec_qkv_rope_append_4r(
    device const uchar* wq      [[buffer(0)]],
    device const float* q_sc    [[buffer(1)]],
    device const uchar* wk      [[buffer(2)]],
    device const float* k_sc    [[buffer(3)]],
    device const uchar* wv      [[buffer(4)]],
    device const float* v_sc    [[buffer(5)]],
    device const float* x       [[buffer(6)]],
    device       float* q_out   [[buffer(7)]],
    device       float* k_cache [[buffer(8)]],
    device       float* v_cache [[buffer(9)]],
    device const float* q_bias  [[buffer(10)]],
    device const float* k_bias  [[buffer(11)]],
    device const float* v_bias  [[buffer(12)]],
    constant ArgbufQkvRopeAppend& args [[buffer(13)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // Q: 4r/simdgroup (2 RoPE pairs). 8 simdgroups/TG → 32 rows/TG.
    // K: 4r/simdgroup (2 RoPE pairs). 8 simdgroups/TG → 32 rows/TG.
    // V: 2r/simdgroup (no RoPE).      8 simdgroups/TG → 16 rows/TG.
    uint q_quads = args.q_rows / 4u;   // number of 4-row quads in Q
    uint k_quads = args.kv_rows / 4u;  // number of 4-row quads in K
    uint v_pairs = args.kv_rows / 2u;  // number of 2-row pairs in V
    uint n_tg_q  = (q_quads + 7u) / 8u;
    uint n_tg_k  = (k_quads + 7u) / 8u;
    uint n_tg_v  = (v_pairs + 7u) / 8u;

    if (gid < n_tg_q) {
        // Q section: each simdgroup handles a quad (4 rows = 2 RoPE pairs).
        uint quad_idx = gid * 8u + simd_id;
        if (quad_idx >= q_quads) return;
        uint row0 = quad_idx * 4u;
        float p0 = q4k_predec_dot_row(wq, q_sc, x, row0,      args.cols, simd_lane);
        float p1 = q4k_predec_dot_row(wq, q_sc, x, row0 + 1u, args.cols, simd_lane);
        float p2 = q4k_predec_dot_row(wq, q_sc, x, row0 + 2u, args.cols, simd_lane);
        float p3 = q4k_predec_dot_row(wq, q_sc, x, row0 + 3u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0,      p0, p1, args);
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0 + 2u, p2, p3, args);
        }
    } else if (gid < n_tg_q + n_tg_k) {
        // K section: each simdgroup handles a quad (4 rows = 2 RoPE pairs).
        uint quad_idx = (gid - n_tg_q) * 8u + simd_id;
        if (quad_idx >= k_quads) return;
        uint row0 = quad_idx * 4u;
        float p0 = q4k_predec_dot_row(wk, k_sc, x, row0,      args.cols, simd_lane);
        float p1 = q4k_predec_dot_row(wk, k_sc, x, row0 + 1u, args.cols, simd_lane);
        float p2 = q4k_predec_dot_row(wk, k_sc, x, row0 + 2u, args.cols, simd_lane);
        float p3 = q4k_predec_dot_row(wk, k_sc, x, row0 + 3u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0,      p0, p1, args);
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0 + 2u, p2, p3, args);
        }
    } else {
        // V section: 2r/simdgroup (append 2 rows per simdgroup, no RoPE).
        uint eff_gid = gid - (n_tg_q + n_tg_k);
        if (eff_gid >= n_tg_v) return;
        uint pair_idx = eff_gid * 8u + simd_id;
        if (pair_idx >= v_pairs) return;
        uint row0 = pair_idx * 2u;
        float pv0 = q4k_predec_dot_row(wv, v_sc, x, row0,      args.cols, simd_lane);
        float pv1 = q4k_predec_dot_row(wv, v_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            v_cache[args.kv_off + row0]      = pv0 + (args.has_v_bias != 0u ? v_bias[row0]      : 0.0f);
            v_cache[args.kv_off + row0 + 1u] = pv1 + (args.has_v_bias != 0u ? v_bias[row0 + 1u] : 0.0f);
        }
    }
}

// Track D3 (2026-06-06): f16-scales variant of gemm_q4k_predec_qkv_rope_append.
// Reads q_sc, k_sc, v_sc as half* instead of float* — halves scale bandwidth.
// For Qwen-3B (Q=2048×2048, K/V=1024×2048): saves 512KB+256KB+256KB = 1MB/layer.
// Same dispatch geometry as the 2r base kernel (320 TGs for Qwen-3B).
kernel void gemm_q4k_predec_qkv_rope_append_f16s(
    device const uchar* wq      [[buffer(0)]],
    device const half*  q_sc    [[buffer(1)]],
    device const uchar* wk      [[buffer(2)]],
    device const half*  k_sc    [[buffer(3)]],
    device const uchar* wv      [[buffer(4)]],
    device const half*  v_sc    [[buffer(5)]],
    device const float* x       [[buffer(6)]],
    device       float* q_out   [[buffer(7)]],
    device       float* k_cache [[buffer(8)]],
    device       float* v_cache [[buffer(9)]],
    device const float* q_bias  [[buffer(10)]],
    device const float* k_bias  [[buffer(11)]],
    device const float* v_bias  [[buffer(12)]],
    constant ArgbufQkvRopeAppend& args [[buffer(13)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint q_pairs = args.q_rows / 2u;
    uint k_pairs = args.kv_rows / 2u;
    uint n_tg_q  = (q_pairs + 7u) / 8u;
    uint n_tg_k  = (k_pairs + 7u) / 8u;
    uint n_tg_v  = (args.kv_rows + 7u) / 8u;

    if (gid < n_tg_q) {
        uint pair_idx = gid * 8u + simd_id;
        if (pair_idx >= q_pairs) return;
        uint row0 = pair_idx * 2u;
        float p0 = q4k_predec_dot_row_f16s(wq, q_sc, x, row0, args.cols, simd_lane);
        float p1 = q4k_predec_dot_row_f16s(wq, q_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0, p0, p1, args);
        }
    } else if (gid < n_tg_q + n_tg_k) {
        uint pair_idx = (gid - n_tg_q) * 8u + simd_id;
        if (pair_idx >= k_pairs) return;
        uint row0 = pair_idx * 2u;
        float p0 = q4k_predec_dot_row_f16s(wk, k_sc, x, row0, args.cols, simd_lane);
        float p1 = q4k_predec_dot_row_f16s(wk, k_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0, p0, p1, args);
        }
    } else {
        uint eff_gid = gid - (n_tg_q + n_tg_k);
        if (eff_gid >= n_tg_v) return;
        uint row = eff_gid * 8u + simd_id;
        if (row >= args.kv_rows) return;
        float pv = q4k_predec_dot_row_f16s(wv, v_sc, x, row, args.cols, simd_lane);
        if (simd_lane == 0u) {
            v_cache[args.kv_off + row] = pv + (args.has_v_bias != 0u ? v_bias[row] : 0.0f);
        }
    }
}

// Track D3 (2026-06-06): f16-scales variant of gemm_q4k_predec_qkv_rope_append_4r.
// Same as _f16s above but 4r geometry: Q/K use 4 rows/simdgroup (2 RoPE pairs),
// V uses 2 rows/simdgroup. 160 TGs vs 320 for Qwen-3B — combines C28+D3 savings.
// Requires q_rows % 4 == 0 and kv_rows % 4 == 0 (validated on Rust side).
kernel void gemm_q4k_predec_qkv_rope_append_4r_f16s(
    device const uchar* wq      [[buffer(0)]],
    device const half*  q_sc    [[buffer(1)]],
    device const uchar* wk      [[buffer(2)]],
    device const half*  k_sc    [[buffer(3)]],
    device const uchar* wv      [[buffer(4)]],
    device const half*  v_sc    [[buffer(5)]],
    device const float* x       [[buffer(6)]],
    device       float* q_out   [[buffer(7)]],
    device       float* k_cache [[buffer(8)]],
    device       float* v_cache [[buffer(9)]],
    device const float* q_bias  [[buffer(10)]],
    device const float* k_bias  [[buffer(11)]],
    device const float* v_bias  [[buffer(12)]],
    constant ArgbufQkvRopeAppend& args [[buffer(13)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint q_quads = args.q_rows / 4u;
    uint k_quads = args.kv_rows / 4u;
    uint v_pairs = args.kv_rows / 2u;
    uint n_tg_q  = (q_quads + 7u) / 8u;
    uint n_tg_k  = (k_quads + 7u) / 8u;
    uint n_tg_v  = (v_pairs + 7u) / 8u;

    if (gid < n_tg_q) {
        uint quad_idx = gid * 8u + simd_id;
        if (quad_idx >= q_quads) return;
        uint row0 = quad_idx * 4u;
        float p0 = q4k_predec_dot_row_f16s(wq, q_sc, x, row0,      args.cols, simd_lane);
        float p1 = q4k_predec_dot_row_f16s(wq, q_sc, x, row0 + 1u, args.cols, simd_lane);
        float p2 = q4k_predec_dot_row_f16s(wq, q_sc, x, row0 + 2u, args.cols, simd_lane);
        float p3 = q4k_predec_dot_row_f16s(wq, q_sc, x, row0 + 3u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0,      p0, p1, args);
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0 + 2u, p2, p3, args);
        }
    } else if (gid < n_tg_q + n_tg_k) {
        uint quad_idx = (gid - n_tg_q) * 8u + simd_id;
        if (quad_idx >= k_quads) return;
        uint row0 = quad_idx * 4u;
        float p0 = q4k_predec_dot_row_f16s(wk, k_sc, x, row0,      args.cols, simd_lane);
        float p1 = q4k_predec_dot_row_f16s(wk, k_sc, x, row0 + 1u, args.cols, simd_lane);
        float p2 = q4k_predec_dot_row_f16s(wk, k_sc, x, row0 + 2u, args.cols, simd_lane);
        float p3 = q4k_predec_dot_row_f16s(wk, k_sc, x, row0 + 3u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0,      p0, p1, args);
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0 + 2u, p2, p3, args);
        }
    } else {
        uint eff_gid = gid - (n_tg_q + n_tg_k);
        if (eff_gid >= n_tg_v) return;
        uint pair_idx = eff_gid * 8u + simd_id;
        if (pair_idx >= v_pairs) return;
        uint row0 = pair_idx * 2u;
        float pv0 = q4k_predec_dot_row_f16s(wv, v_sc, x, row0,      args.cols, simd_lane);
        float pv1 = q4k_predec_dot_row_f16s(wv, v_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            v_cache[args.kv_off + row0]      = pv0 + (args.has_v_bias != 0u ? v_bias[row0]      : 0.0f);
            v_cache[args.kv_off + row0 + 1u] = pv1 + (args.has_v_bias != 0u ? v_bias[row0 + 1u] : 0.0f);
        }
    }
}

kernel void gemm_q4k_q4k_q6k_rope_append(
    device const uchar* wq      [[buffer(0)]],
    device const float* q_sc    [[buffer(1)]],
    device const uchar* wk      [[buffer(2)]],
    device const float* k_sc    [[buffer(3)]],
    device const uchar* wv      [[buffer(4)]],
    device const float* x       [[buffer(5)]],
    device       float* q_out   [[buffer(6)]],
    device       float* k_cache [[buffer(7)]],
    device       float* v_cache [[buffer(8)]],
    device const float* q_bias  [[buffer(9)]],
    device const float* k_bias  [[buffer(10)]],
    device const float* v_bias  [[buffer(11)]],
    constant ArgbufQkvRopeAppend& args [[buffer(12)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint q_pairs = args.q_rows / 2u;
    uint k_pairs = args.kv_rows / 2u;
    uint n_tg_q  = (q_pairs + 7u) / 8u;
    uint n_tg_k  = (k_pairs + 7u) / 8u;
    uint n_tg_v  = (args.kv_rows + 7u) / 8u;

    if (gid < n_tg_q) {
        uint pair_idx = gid * 8u + simd_id;
        if (pair_idx >= q_pairs) return;
        uint row0 = pair_idx * 2u;
        float p0 = q4k_predec_dot_row(wq, q_sc, x, row0, args.cols, simd_lane);
        float p1 = q4k_predec_dot_row(wq, q_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(q_out, q_bias, args.has_q_bias, row0, p0, p1, args);
        }
    } else if (gid < n_tg_q + n_tg_k) {
        uint pair_idx = (gid - n_tg_q) * 8u + simd_id;
        if (pair_idx >= k_pairs) return;
        uint row0 = pair_idx * 2u;
        float p0 = q4k_predec_dot_row(wk, k_sc, x, row0, args.cols, simd_lane);
        float p1 = q4k_predec_dot_row(wk, k_sc, x, row0 + 1u, args.cols, simd_lane);
        if (simd_lane == 0u) {
            write_rope_pair(k_cache + args.kv_off, k_bias, args.has_k_bias, row0, p0, p1, args);
        }
    } else {
        uint eff_gid = gid - (n_tg_q + n_tg_k);
        if (eff_gid >= n_tg_v) return;
        uint row = eff_gid * 8u + simd_id;
        if (row >= args.kv_rows) return;
        float pv = q6k_dot_row(wv, x, row, args.cols, simd_lane);
        if (simd_lane == 0u) {
            v_cache[args.kv_off + row] = pv + (args.has_v_bias != 0u ? v_bias[row] : 0.0f);
        }
    }
}

// ── gemm_q6_k_fused_v2_swiglu ─────────────────────────────────────────────────
// Track 3.5 — SwiGLU-fused Q6_K GEMV.  Identical to gemm_q6_k_fused_v2 but
// the activation x[i] is replaced by silu(gate[i]) * up[i] inline, eliminating
// the separate silu_mul dispatch.  Saves 1 dispatch/layer × n_layers.
//
// Buffer layout:
//   0: w_q6    (uchar*, Q6_K packed weight bytes, rows * blocks * 210 B)
//   1: gate    (float*, intermediate-length gate projection output)
//   2: up      (float*, intermediate-length up   projection output)
//   3: y       (float*, hidden-length output)
//   4: rows    (uint, number of output rows)
//   5: cols    (uint, number of input columns, must be multiple of 256)
//
// Grid / threadgroup: same as gemm_q6_k_fused_v2 — (ceil(rows/8)*256,1,1) / (256,1,1)
kernel void gemm_q6_k_fused_v2_swiglu(
    device const uchar* w_q6  [[buffer(0)]],
    device const float* gate  [[buffer(1)]],
    device const float* up    [[buffer(2)]],
    device       float* y     [[buffer(3)]],
    constant     uint&  rows  [[buffer(4)]],
    constant     uint&  cols  [[buffer(5)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 210ul;

    uint half_idx          = simd_lane >> 4u;
    uint group             = (simd_lane >> 2u) & 3u;
    uint l_base            = (simd_lane & 3u) * 8u;
    uint scale_l_off       = l_base >> 4u;
    uint scale_byte_off    = 192u + half_idx * 8u + scale_l_off + group * 2u;
    uint ql_group_off      = (group & 1u) * 32u;
    bool group_high_nibble = (group >= 2u);
    uint qh_shift          = group * 2u;
    uint tid_base          = half_idx * 128u + group * 32u + l_base;

    float partial = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo = row_byte_off + (uint64_t)b * 210ul;
        ushort d_bits = (ushort)w_q6[bo + 208u] | ((ushort)w_q6[bo + 209u] << 8);
        float d = (float)as_type<half>(d_bits);
        int scale = (int)(signed char)w_q6[bo + (uint64_t)scale_byte_off];
        float dscale = d * (float)scale;

        uint64_t ql_base = bo + (uint64_t)half_idx * 64ul + (uint64_t)ql_group_off;
        uint64_t qh_base = bo + 128ul + (uint64_t)half_idx * 32ul;

        float lane_acc = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            uint l = l_base + k;
            uchar qlb = w_q6[ql_base + (uint64_t)l];
            uint qlow = group_high_nibble
                      ? ((uint)(qlb >> 4) & 0x0Fu)
                      : ((uint)qlb & 0x0Fu);
            uchar qhb = w_q6[qh_base + (uint64_t)l];
            uint qhigh = ((uint)qhb >> qh_shift) & 0x03u;
            int qi = (int)(qlow | (qhigh << 4)) - 32;
            uint xi_idx = b * 256u + tid_base + k;
            float g = gate[xi_idx];
            float xi = (g / (1.0f + exp(-g))) * up[xi_idx];
            lane_acc += (float)qi * xi;
        }
        partial += dscale * lane_acc;
    }

    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        y[base_row] = partial;
    }
}

// ── gemm_q4_k_v4_predec_swiglu ───────────────────────────────────────────────
// Track 3.5 — SwiGLU-fused Q4_K predec GEMV, 1-row-per-simdgroup base variant.
// Same as gemm_q4_k_v4_predec but x[i] → silu(gate[i])*up[i].
// Buffer layout:
//   0: w_q4   1: scales  2: gate  3: up  4: y  5: rows  6: cols
// Grid: (ceil(rows/8)*256, 1, 1)  TG: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_swiglu(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device const float* gate    [[buffer(2)]],
    device const float* up      [[buffer(3)]],
    device       float* y       [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 8u + simd_id;
    if (row0 >= rows) return;
    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul;
        uint64_t so0 = rs0 + (uint64_t)b * 16ul;
        float ds[8], dm[8];
        for (uint s = 0; s < 8u; ++s) {
            ds[s] = scales[so0 + (uint64_t)(s * 2u)];
            dm[s] = scales[so0 + (uint64_t)(s * 2u + 1u)];
        }
        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uint idx0 = b * 256u + k0 * 32u + simd_lane;
            uint idx1 = b * 256u + k1 * 32u + simd_lane;
            float g0 = gate[idx0]; float x0 = (g0 / (1.0f + exp(-g0))) * up[idx0];
            float g1 = gate[idx1]; float x1 = (g1 / (1.0f + exp(-g1))) * up[idx1];
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p0 += (ds[k0] * (float)(q0 & 0x0Fu) - dm[k0]) * x0;
            p0 += (ds[k1] * (float)(q0 >> 4u)   - dm[k1]) * x1;
        }
    }
    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
}

// ── gemm_q4_k_v4_predec_2r_swiglu ────────────────────────────────────────────
// Track 3.5 — SwiGLU-fused Q4_K predec GEMV, 2-rows-per-simdgroup (default).
// Same as gemm_q4_k_v4_predec_2r but x[i] → silu(gate[i])*up[i].
// Buffer layout: 0:w_q4  1:scales  2:gate  3:up  4:y  5:rows  6:cols
// Grid: (ceil(rows/16)*256, 1, 1)  TG: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_2r_swiglu(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device const float* gate    [[buffer(2)]],
    device const float* up      [[buffer(3)]],
    device       float* y       [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 16u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u;
    bool has1 = row1 < rows;
    uint r1 = has1 ? row1 : row0;
    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1  * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f, p1 = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b * 144ul, so0 = rs0 + (uint64_t)b * 16ul;
        uint64_t bo1 = rb1 + (uint64_t)b * 144ul, so1 = rs1 + (uint64_t)b * 16ul;
        float ds0[8], dm0[8], ds1[8], dm1[8];
        for (uint s = 0; s < 8u; ++s) {
            ds0[s] = scales[so0 + (uint64_t)(s * 2u)];
            dm0[s] = scales[so0 + (uint64_t)(s * 2u + 1u)];
            ds1[s] = scales[so1 + (uint64_t)(s * 2u)];
            dm1[s] = scales[so1 + (uint64_t)(s * 2u + 1u)];
        }
        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uint idx0 = b * 256u + k0 * 32u + simd_lane;
            uint idx1 = b * 256u + k1 * 32u + simd_lane;
            float g0 = gate[idx0]; float x0 = (g0 / (1.0f + exp(-g0))) * up[idx0];
            float g1 = gate[idx1]; float x1 = (g1 / (1.0f + exp(-g1))) * up[idx1];
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p0 += (ds0[k0] * (float)(q0 & 0x0Fu) - dm0[k0]) * x0;
            p0 += (ds0[k1] * (float)(q0 >> 4u)   - dm0[k1]) * x1;
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi * 32ul + (uint64_t)simd_lane];
            p1 += (ds1[k0] * (float)(q1 & 0x0Fu) - dm1[k0]) * x0;
            p1 += (ds1[k1] * (float)(q1 >> 4u)   - dm1[k1]) * x1;
        }
    }
    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) { p1 = simd_sum(p1); if (simd_lane == 0u) y[row1] = p1; }
}

// ── gemm_q4_k_v4_predec_4r_swiglu ────────────────────────────────────────────
// Track 3.5 — SwiGLU-fused Q4_K predec GEMV, 4-rows-per-simdgroup (opt-in).
// Same as gemm_q4_k_v4_predec_4r but x[i] → silu(gate[i])*up[i].
// Buffer layout: 0:w_q4  1:scales  2:gate  3:up  4:y  5:rows  6:cols
// Grid: (ceil(rows/32)*256, 1, 1)  TG: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_4r_swiglu(
    device const uchar* w_q4    [[buffer(0)]],
    device const float* scales  [[buffer(1)]],
    device const float* gate    [[buffer(2)]],
    device const float* up      [[buffer(3)]],
    device       float* y       [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 32u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u, row2 = row0 + 16u, row3 = row0 + 24u;
    bool has1 = row1 < rows, has2 = row2 < rows, has3 = row3 < rows;
    uint r1 = has1 ? row1 : row0, r2 = has2 ? row2 : row0, r3 = has3 ? row3 : row0;
    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1  * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb2 = (uint64_t)r2  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs2 = (uint64_t)r2  * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb3 = (uint64_t)r3  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs3 = (uint64_t)r3  * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b*144ul, so0 = rs0 + (uint64_t)b*16ul;
        uint64_t bo1 = rb1 + (uint64_t)b*144ul, so1 = rs1 + (uint64_t)b*16ul;
        uint64_t bo2 = rb2 + (uint64_t)b*144ul, so2 = rs2 + (uint64_t)b*16ul;
        uint64_t bo3 = rb3 + (uint64_t)b*144ul, so3 = rs3 + (uint64_t)b*16ul;
        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uint idx0 = b * 256u + k0 * 32u + simd_lane;
            uint idx1 = b * 256u + k1 * 32u + simd_lane;
            float g0 = gate[idx0]; float x0 = (g0 / (1.0f + exp(-g0))) * up[idx0];
            float g1 = gate[idx1]; float x1 = (g1 / (1.0f + exp(-g1))) * up[idx1];
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p0 += (scales[so0+(uint64_t)(k0*2u)] * (float)(q0&0x0Fu) - scales[so0+(uint64_t)(k0*2u+1u)]) * x0;
            p0 += (scales[so0+(uint64_t)(k1*2u)] * (float)(q0>>4u)   - scales[so0+(uint64_t)(k1*2u+1u)]) * x1;
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p1 += (scales[so1+(uint64_t)(k0*2u)] * (float)(q1&0x0Fu) - scales[so1+(uint64_t)(k0*2u+1u)]) * x0;
            p1 += (scales[so1+(uint64_t)(k1*2u)] * (float)(q1>>4u)   - scales[so1+(uint64_t)(k1*2u+1u)]) * x1;
            uchar q2 = w_q4[bo2 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p2 += (scales[so2+(uint64_t)(k0*2u)] * (float)(q2&0x0Fu) - scales[so2+(uint64_t)(k0*2u+1u)]) * x0;
            p2 += (scales[so2+(uint64_t)(k1*2u)] * (float)(q2>>4u)   - scales[so2+(uint64_t)(k1*2u+1u)]) * x1;
            uchar q3 = w_q4[bo3 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p3 += (scales[so3+(uint64_t)(k0*2u)] * (float)(q3&0x0Fu) - scales[so3+(uint64_t)(k0*2u+1u)]) * x0;
            p3 += (scales[so3+(uint64_t)(k1*2u)] * (float)(q3>>4u)   - scales[so3+(uint64_t)(k1*2u+1u)]) * x1;
        }
    }
    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) { p1 = simd_sum(p1); if (simd_lane == 0u) y[row1] = p1; }
    if (has2) { p2 = simd_sum(p2); if (simd_lane == 0u) y[row2] = p2; }
    if (has3) { p3 = simd_sum(p3); if (simd_lane == 0u) y[row3] = p3; }
}

// ── gemm_q4_k_v4_predec_f16s_4r_swiglu ──────────────────────────────────────
// Track D1 (2026-06-06): f16-scales variant of gemm_q4_k_v4_predec_4r_swiglu.
// Identical geometry (4 rows/simdgroup, 32 rows/TG) and SwiGLU fusion, but the
// pre-decoded sub-block scale table is read as half (2 B) instead of float (4 B),
// then widened to float in register.  For Qwen-3B ffn_down (rows=2048,
// cols=11008): scale table is 2048 × 43 × 32 B = 2.8 MB instead of 5.6 MB —
// cuts scale-table bandwidth 50%.  NOT bit-identical (f16 scale rounding ≈5e-4
// relative); only active under DISMANTLE_QWEN_PREDEC_F16SCALES.
// Buffer layout: 0:w_q4  1:scales(half)  2:gate  3:up  4:y  5:rows  6:cols
// Grid: (ceil(rows/32)*256, 1, 1)  TG: (256, 1, 1)
kernel void gemm_q4_k_v4_predec_f16s_4r_swiglu(
    device const uchar* w_q4    [[buffer(0)]],
    device const half*  scales  [[buffer(1)]],
    device const float* gate    [[buffer(2)]],
    device const float* up      [[buffer(3)]],
    device       float* y       [[buffer(4)]],
    constant     uint&  rows    [[buffer(5)]],
    constant     uint&  cols    [[buffer(6)]],
    uint  gid       [[threadgroup_position_in_grid]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]])
{
    uint row0 = gid * 32u + simd_id;
    if (row0 >= rows) return;
    uint row1 = row0 + 8u, row2 = row0 + 16u, row3 = row0 + 24u;
    bool has1 = row1 < rows, has2 = row2 < rows, has3 = row3 < rows;
    uint r1 = has1 ? row1 : row0, r2 = has2 ? row2 : row0, r3 = has3 ? row3 : row0;
    uint blocks_per_row = cols / 256u;
    uint64_t rb0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs0 = (uint64_t)row0 * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb1 = (uint64_t)r1  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs1 = (uint64_t)r1  * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb2 = (uint64_t)r2  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs2 = (uint64_t)r2  * (uint64_t)blocks_per_row * 16ul;
    uint64_t rb3 = (uint64_t)r3  * (uint64_t)blocks_per_row * 144ul;
    uint64_t rs3 = (uint64_t)r3  * (uint64_t)blocks_per_row * 16ul;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    for (uint b = 0; b < blocks_per_row; ++b) {
        uint64_t bo0 = rb0 + (uint64_t)b*144ul, so0 = rs0 + (uint64_t)b*16ul;
        uint64_t bo1 = rb1 + (uint64_t)b*144ul, so1 = rs1 + (uint64_t)b*16ul;
        uint64_t bo2 = rb2 + (uint64_t)b*144ul, so2 = rs2 + (uint64_t)b*16ul;
        uint64_t bo3 = rb3 + (uint64_t)b*144ul, so3 = rs3 + (uint64_t)b*16ul;
        for (uint pi = 0; pi < 4u; ++pi) {
            uint k0 = pi * 2u, k1 = k0 + 1u;
            uint idx0 = b * 256u + k0 * 32u + simd_lane;
            uint idx1 = b * 256u + k1 * 32u + simd_lane;
            float g0 = gate[idx0]; float x0 = (g0 / (1.0f + exp(-g0))) * up[idx0];
            float g1 = gate[idx1]; float x1 = (g1 / (1.0f + exp(-g1))) * up[idx1];
            uchar q0 = w_q4[bo0 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p0 += ((float)scales[so0+(uint64_t)(k0*2u)] * (float)(q0&0x0Fu) - (float)scales[so0+(uint64_t)(k0*2u+1u)]) * x0;
            p0 += ((float)scales[so0+(uint64_t)(k1*2u)] * (float)(q0>>4u)   - (float)scales[so0+(uint64_t)(k1*2u+1u)]) * x1;
            uchar q1 = w_q4[bo1 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p1 += ((float)scales[so1+(uint64_t)(k0*2u)] * (float)(q1&0x0Fu) - (float)scales[so1+(uint64_t)(k0*2u+1u)]) * x0;
            p1 += ((float)scales[so1+(uint64_t)(k1*2u)] * (float)(q1>>4u)   - (float)scales[so1+(uint64_t)(k1*2u+1u)]) * x1;
            uchar q2 = w_q4[bo2 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p2 += ((float)scales[so2+(uint64_t)(k0*2u)] * (float)(q2&0x0Fu) - (float)scales[so2+(uint64_t)(k0*2u+1u)]) * x0;
            p2 += ((float)scales[so2+(uint64_t)(k1*2u)] * (float)(q2>>4u)   - (float)scales[so2+(uint64_t)(k1*2u+1u)]) * x1;
            uchar q3 = w_q4[bo3 + 16ul + (uint64_t)pi*32ul + (uint64_t)simd_lane];
            p3 += ((float)scales[so3+(uint64_t)(k0*2u)] * (float)(q3&0x0Fu) - (float)scales[so3+(uint64_t)(k0*2u+1u)]) * x0;
            p3 += ((float)scales[so3+(uint64_t)(k1*2u)] * (float)(q3>>4u)   - (float)scales[so3+(uint64_t)(k1*2u+1u)]) * x1;
        }
    }
    p0 = simd_sum(p0);
    if (simd_lane == 0u) y[row0] = p0;
    if (has1) { p1 = simd_sum(p1); if (simd_lane == 0u) y[row1] = p1; }
    if (has2) { p2 = simd_sum(p2); if (simd_lane == 0u) y[row2] = p2; }
    if (has3) { p3 = simd_sum(p3); if (simd_lane == 0u) y[row3] = p3; }
}
