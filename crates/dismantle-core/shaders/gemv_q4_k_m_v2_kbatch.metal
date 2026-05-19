// Path B Stage 2.3 — K-batched Q4_K_M GEMV.
//
// Mirrors gemm_q4_k_m_fused_v2 (quant.metal:283) with per-simdgroup K
// register accumulators. Same dispatch geometry (ROWS_PER_TG=8,
// TG_SIZE=256, one simdgroup per output row), same per-block Q4_K_M
// decode logic, but each lane accumulates K dot-product partials
// against K input vectors against the SAME decoded weight values.
//
// Q4_K_M block: 256 elements, 144 bytes/block, super-block scales
// (d, dmin) at bytes 0..4 (fp16), 12-byte scale/min packed, 128-byte
// nibble-packed weight payload.
//
// Geometry:
//   grid  = (((rows + 7) / 8), 1, 1) × TG_SIZE
//   TG    = (256, 1, 1) — 8 simdgroups × 32 lanes
//   per simdgroup: 1 output row × K queries
//   per lane: 8 elements per block via stride-32 (elem = k*32 + simd_lane)
//
// Requires cols % 256 == 0 and k_batch ∈ [1, 8].
//
// Bit-equivalent to gemm_q4_k_m_fused_v2 at K=1 (per-block decode +
// partial accumulation logic is unchanged; the K>1 register slots
// receive zero contributions when k_batch < K_MAX_LIMIT).

kernel void gemm_q4_k_m_fused_v2_kbatch(
    device const uchar* w_q4         [[buffer(0)]],   // (rows × cols) Q4_K_M, 144 B/block
    device const float* x_kbatch     [[buffer(1)]],   // (k_batch × cols) f32 row-major
    device       float* y_kbatch     [[buffer(2)]],   // (k_batch × rows) f32 row-major
    constant     uint&  rows         [[buffer(3)]],
    constant     uint&  cols         [[buffer(4)]],
    constant     uint&  k_batch      [[buffer(5)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[threadgroup_position_in_grid]],
    uint                simd_lane    [[thread_index_in_simdgroup]],
    uint                simd_id      [[simdgroup_index_in_threadgroup]])
{
    uint base_row = gid * 8u + simd_id;
    if (base_row >= rows) return;

    uint  blocks_per_row = cols / 256u;
    uint64_t row_byte_off = (uint64_t)base_row * (uint64_t)blocks_per_row * 144ul;

    // Per-K accumulators (registers; up to 8 supported, unused slots stay 0).
    float partial[8];
    for (uint kk = 0; kk < 8u; ++kk) {
        partial[kk] = 0.0f;
    }

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

            // K-fold inner accumulation. w_val is decoded ONCE per (block, elem)
            // and dot-producted against k_batch separate query columns.
            uint64_t x_off_base = (uint64_t)b * 256ul + (uint64_t)elem;
            for (uint kk = 0; kk < k_batch; ++kk) {
                float xv = x_kbatch[(uint64_t)kk * (uint64_t)cols + x_off_base];
                partial[kk] += w_val * xv;
            }
        }
    }

    // 32-thread simdgroup reduction per K. Zero barriers.
    for (uint kk = 0; kk < k_batch; ++kk) {
        float p = simd_sum(partial[kk]);
        if (simd_lane == 0u) {
            y_kbatch[(uint64_t)kk * (uint64_t)rows + (uint64_t)base_row] = p;
        }
    }
}
