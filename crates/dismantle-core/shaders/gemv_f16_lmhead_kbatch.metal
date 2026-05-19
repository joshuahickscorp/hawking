// Path B Stage 2.2 — K-batched f16 lm_head GEMV.
//
// Computes  y_kbatch[k, r] = Σ_c W[r, c] * x_kbatch[k, c]
// for k in [0, k_batch), r in [0, rows), where W is fp16 row-major
// (rows × cols) and x_kbatch / y_kbatch are fp32 row-major (k_batch × *).
//
// Geometry mirrors gemv_f16_simdmat (matmul.metal:97): grid =
// (ceil(rows/8) * 32, 1, 1), TG = (32, 1, 1), 192 floats of TG memory.
// 8 output rows × K queries per TG; K is supplied at dispatch time and
// must be ≤ 8.
//
// The K-batching exploits a property of the K=1 simdmat kernel: it
// broadcasts a single x across all 8 columns of the X tile, so the 8×8
// simdgroup_multiply_accumulate computes one dot product replicated 8
// times. Repurposing those columns to hold K distinct query rows
// (X[inner][n_query] = x_kbatch[n_query, c_base + inner]) lets the
// SAME 8×8 matmul compute K dot products per output row in one pass.
// At K=1 the kernel is bit-equivalent to gemv_f16_simdmat (X col 0
// holds the dot product, cols 1..7 hold 0 contributions).
//
// Requires cols % 8 == 0 and k_batch ∈ [1, 8].

kernel void gemv_f16_lmhead_kbatch(
    device const half*  w           [[buffer(0)]],  // (rows × cols) f16, row-major
    device const float* x_kbatch    [[buffer(1)]],  // (k_batch × cols) f32, row-major
    device       float* y_kbatch    [[buffer(2)]],  // (k_batch × rows) f32, row-major
    constant     uint&  rows        [[buffer(3)]],
    constant     uint&  cols        [[buffer(4)]],
    constant     uint&  k_batch     [[buffer(5)]],
    threadgroup  float* shmem       [[threadgroup(0)]],  // 192 floats = W tile + X tile + result tile
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint base_row = gid * 8u;
    if (base_row >= rows) return;

    threadgroup float* shmem_w   = shmem;         // W tile: [0..64)
    threadgroup float* shmem_x   = shmem + 64;    // X tile: [64..128)
    threadgroup float* shmem_out = shmem + 128;   // D tile: [128..192)

    // Zero-init result tile; simdgroup_load reads it to initialise acc to 0.
    shmem_out[tid]      = 0.0f;
    shmem_out[tid + 32] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, shmem_out, 8, ulong2(0, 0));

    uint n_chunks = cols / 8u;  // cols % 8 == 0 required by caller.

    for (uint chunk = 0; chunk < n_chunks; ++chunk) {
        uint c_base = chunk * 8u;

        // Fill W and X tiles (2 elements per thread, 64 slots).
        for (int e = 0; e < 2; ++e) {
            uint elem = tid + (uint)e * 32u;
            uint m = elem >> 3u;  // 0..7 — row index within 8×8 tile
            uint k = elem &  7u;  // 0..7 — col index within 8×8 tile

            // W[m_tile][k_tile] = W[base_row+m, c_base+k]; promote f16 → f32.
            uint row = base_row + m;
            shmem_w[elem] = (row < rows) ? float(w[(ulong)row * cols + c_base + k]) : 0.0f;

            // X[m_tile][k_tile] = x_kbatch[k_tile, c_base + m_tile]
            // for k_tile < k_batch; pad zero for out-of-range queries.
            // (m_tile is the K-inner index of the dot product; k_tile is the n
            // column of the X tile and selects which K-th query.)
            shmem_x[elem] = (k < k_batch) ? x_kbatch[(ulong)k * cols + c_base + m] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> w_mat, x_mat;
        simdgroup_load(w_mat, shmem_w, 8, ulong2(0, 0));
        simdgroup_load(x_mat, shmem_x, 8, ulong2(0, 0));
        simdgroup_multiply_accumulate(acc, w_mat, x_mat, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // acc[m][n] now holds Σ_inner W[base_row+m, inner] * x_kbatch[n, inner]
    // for n < k_batch. Write back as (K, rows)-major.
    simdgroup_store(acc, shmem_out, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8u && base_row + tid < rows) {
        uint m = tid;
        for (uint k = 0; k < k_batch; ++k) {
            y_kbatch[(ulong)k * rows + (base_row + m)] = shmem_out[m * 8u + k];
        }
    }
}
