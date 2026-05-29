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
