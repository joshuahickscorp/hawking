// Packed binary sign + FP16 group-scale matvec component for Ascension Qwen
// candidates.
//
// Layout: row-major sign bitstream (LSB-first within each byte) and one
// half-precision absmax scale per contiguous group of `group_size` columns.
// Qwen30 admits group_size=128; cols are multiples of the group size.

#include <metal_stdlib>
using namespace metal;

// Serial one-thread-per-row oracle. Preserves the exact left-to-right f32
// accumulation order used by the original control. Kept for component parity
// tests; the live Qwen30 path dispatches the tiled simdgroup kernel below.
kernel void qwen_binary_sign_scale_matvec_serial(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;

    float sum = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint group_start = group * group_size;
        const uint group_end = min(group_start + group_size, cols);
        const float scale = float(scales[scale_base + group]);
        // Byte-wise sign unpack: one load covers eight consecutive columns.
        // Accumulation order remains col = group_start .. group_end-1.
        uint col = group_start;
        // Align to a sign-byte boundary when possible without reordering.
        while (col < group_end && ((row_base + col) & 7u) != 0u) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
        while (col + 8u <= group_end) {
            const uchar byte = signs[(row_base + col) >> 3u];
            sum += ((byte & 0x01u) ? scale : -scale) * input[col];
            sum += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
            sum += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
            sum += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
            sum += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
            sum += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
            sum += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
            sum += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
            col += 8u;
        }
        while (col < group_end) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
    }
    output[row] = sum;
}

// Live default packed-binary GEMV.
//
// Geometry: one simdgroup (32 lanes) owns one output row; eight rows share a
// 256-thread threadgroup. Each iteration of the column loop covers a
// contiguous 32-wide tile so sign bytes and activations are coalesced across
// the simdgroup. Lane `simd_lane` multiplies column `base + simd_lane`; a
// single `simd_sum` reduces the row. Association differs from the serial
// oracle (component parity uses a small absolute tolerance).
//
// Grid: (ceil(rows / 8) * 256, 1, 1), threadgroup: (256, 1, 1).
kernel void qwen_binary_sign_scale_matvec(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;

    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;

    // Tile columns in contiguous 32-wide blocks. For group_size multiple of 32
    // (Qwen30 admits 128), every tile lies inside a single scale group, so the
    // scale is identical for all lanes and is loaded once per tile.
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float scale = float(scales[scale_base + col / group_size]);
        const uint flat = row_base + col;
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        partial += (positive ? scale : -scale) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// Prior opt-in candidate: strided column ownership (lane handles col,
// col+32, ...). Retained so existing `--packed-matvec-kernel simdgroup-candidate`
// receipts keep a distinct entry point. Prefer the tiled default above.
kernel void qwen_binary_sign_scale_matvec_simdgroup_candidate(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;

    float sum = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint col = simd_lane; col < cols; col += 32u) {
        const uint flat = row_base + col;
        const float scale = float(scales[scale_base + col / group_size]);
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        sum += (positive ? scale : -scale) * input[col];
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) output[row] = sum;
}
