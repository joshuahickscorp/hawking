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
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[threadgroup_position_in_grid]],
    uint                simd_lane    [[thread_index_in_simdgroup]],
    uint                simd_id      [[simdgroup_index_in_threadgroup]])
{
    // ROWS_PER_TG=8 (one simdgroup per row), TG_SIZE=256 (8 simdgroups).
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;     // tail simdgroups do nothing

    uint  blocks_per_row = cols / 256u;
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
