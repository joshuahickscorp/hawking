// matmul.metal — simdgroup_matrix GEMV kernels (Wedge H, Path 2).
//
// One SIMD group (32 threads) per threadgroup; each threadgroup computes
// 8 output rows. The activation vector x is broadcast across all 8 columns
// of the x tile (X[k][n] = x[k] ∀n). After accumulation all 8 columns of acc
// hold the same partial dot-product; column 0 is extracted for output.
//
// Grid:  (ceil(rows/8)*32, 1, 1)  — one threadgroup per 8 output rows
// TG:    (32, 1, 1)               — one SIMD group
// Threadgroup memory layout (stride 8 per row, all float):
//   shmem[ 0.. 64): weight tile  W[8][8]    (f32)
//   shmem[64..128): act    tile  X[8][8]    (broadcast: X[k][n] = x[k])
//   shmem[128..192): result tile D[8][8]    (for simdgroup_store + zero-init)
//
// Requires: cols % 8 == 0. Handles rows % 8 != 0 by padding weight rows to 0.
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// v1.0.0-H — simdgroup_matrix GEMV: w (rows×cols f32) × x (cols f32) → y (rows f32).
kernel void gemv_simdgroup_f32(
    device const float* w       [[buffer(0)]],   // (rows × cols) f32, row-major
    device const float* x       [[buffer(1)]],   // (cols,) f32
    device       float* y       [[buffer(2)]],   // (rows,) f32
    constant     uint&  rows    [[buffer(3)]],
    constant     uint&  cols    [[buffer(4)]],
    threadgroup  float* shmem   [[threadgroup(0)]],  // 192 floats = 3 × 64
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint base_row = gid * 8u;
    if (base_row >= rows) return;

    threadgroup float* shmem_w   = shmem;         // [64]
    threadgroup float* shmem_x   = shmem + 64;    // [64]
    threadgroup float* shmem_out = shmem + 128;   // [64]

    // Zero-init accumulator via shmem_out (simdgroup_load initialises acc from it).
    shmem_out[tid]      = 0.0f;
    shmem_out[tid + 32] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, shmem_out, 8, ulong2(0, 0));

    uint n_chunks = cols / 8u;  // cols % 8 == 0 required

    for (uint chunk = 0; chunk < n_chunks; ++chunk) {
        uint c_base = chunk * 8u;

        // Fill weight tile shmem_w[8][8] and activation tile shmem_x[8][8].
        // Each thread fills 2 slots (elem = tid and tid + 32), covering all 64.
        for (int e = 0; e < 2; ++e) {
            uint elem = tid + (uint)e * 32u;
            uint m = elem >> 3u;   // 0..7 — row within 8×8 tile
            uint k = elem &  7u;   // 0..7 — col within 8×8 tile

            // Weight tile: W[base_row+m][c_base+k], zero-padded if row out of bounds.
            uint row = base_row + m;
            shmem_w[elem] = (row < rows) ? w[(ulong)row * cols + c_base + k] : 0.0f;

            // Activation tile: broadcast x[c_base+m] to all 8 cols of row m.
            // Layout: shmem_x[m*8 + n] = x[c_base+m] ∀n → B[m][n] = x[c_base+m].
            // So (A×B)[i][j] = Σ_m A[i][m] * x[c_base+m] = partial GEMV dot. ✓
            shmem_x[elem] = x[c_base + m];  // m = elem >> 3 = row of 8×8 tile
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> w_mat, x_mat;
        simdgroup_load(w_mat, shmem_w, 8, ulong2(0, 0));
        simdgroup_load(x_mat, shmem_x, 8, ulong2(0, 0));
        simdgroup_multiply_accumulate(acc, w_mat, x_mat, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Extract results: all columns of acc are identical (broadcast), so column 0 suffices.
    simdgroup_store(acc, shmem_out, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8u && base_row + tid < rows) {
        y[base_row + tid] = shmem_out[tid * 8u];  // shmem_out[row][col=0]
    }
}
