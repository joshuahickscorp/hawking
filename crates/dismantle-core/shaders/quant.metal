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
