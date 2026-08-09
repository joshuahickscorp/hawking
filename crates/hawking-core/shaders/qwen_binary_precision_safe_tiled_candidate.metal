// Isolated, unregistered Qwen30 packed-binary matvec candidate.
//
// This file is intentionally NOT included by `metal::library_source()` and is
// therefore unable to alter the live Qwen30 runtime.  It exists only for a
// later GPU-cleanroom component test paired with
// `ascension_qwen30_packed_matvec_exactness`.
//
// Geometry / safety contract:
// - one 32-lane SIMDgroup owns one output row;
// - lane N owns the contiguous 128-value group N;
// - Qwen30 candidates must use group_size=128 and groups_per_row<=32;
// - each lane uses Neumaier compensated accumulation over its own group;
// - lane zero combines completed group sums in monotonically increasing group
//   order with the same compensated rule.
//
// This changes floating-point associativity and is NOT bit-identical to the
// scalar control.  It cannot be promoted from component parity alone.

#include <metal_stdlib>
using namespace metal;

kernel void qwen_binary_sign_scale_matvec_tiled_neumaier_candidate(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint threadgroup_id             [[threadgroup_position_in_grid]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    threadgroup float* group_sums   [[threadgroup(0)]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = threadgroup_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;

    // Refuse unsupported geometry by leaving the output untouched. The host
    // cleanroom contract must reject such a dispatch rather than treating it
    // as generic Qwen support.
    if (group_size != 128u || groups_per_row == 0u || groups_per_row > kSimdWidth
        || cols != groups_per_row * group_size) {
        return;
    }

    const uint local = simd_id * kSimdWidth + simd_lane;
    precise float sum = 0.0f;
    precise float compensation = 0.0f;
    if (simd_lane < groups_per_row) {
        const uint group = simd_lane;
        const uint row_base = row * cols;
        const uint group_start = group * group_size;
        const float scale = float(scales[row * groups_per_row + group]);
        for (uint within = 0u; within < group_size; ++within) {
            const uint col = group_start + within;
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            precise const float term = (positive ? scale : -scale) * input[col];
            precise const float next = sum + term;
            if (fabs(sum) >= fabs(term)) {
                compensation += (sum - next) + term;
            } else {
                compensation += (term - next) + sum;
            }
            sum = next;
        }
        group_sums[local] = sum + compensation;
    } else {
        group_sums[local] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0u) {
        precise float total = 0.0f;
        precise float total_compensation = 0.0f;
        const uint base = simd_id * kSimdWidth;
        for (uint group = 0u; group < groups_per_row; ++group) {
            precise const float term = group_sums[base + group];
            precise const float next = total + term;
            if (fabs(total) >= fabs(term)) {
                total_compensation += (total - next) + term;
            } else {
                total_compensation += (term - next) + total;
            }
            total = next;
        }
        output[row] = total + total_compensation;
    }
}
