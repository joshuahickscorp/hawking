// Packed binary sign + FP16 group-scale matvec component for Ascension Qwen
// candidates.
//
// This is a deliberately bounded operator: one thread evaluates one output
// row.  It consumes a row-major sign bitstream and one half-precision absmax
// scale per contiguous group.  It is neither a Qwen decoder nor a throughput
// claim; callers must establish their own source, packing, and model-level
// admission evidence.

#include <metal_stdlib>
using namespace metal;

kernel void qwen_binary_sign_scale_matvec(
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
        for (uint col = group_start; col < group_end; ++col) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
        }
    }
    output[row] = sum;
}

// Candidate packed-binary GEMV geometry for the admitted Qwen30 runtime.
//
// The scalar baseline above gives every output row one thread and deliberately
// remains the correctness control.  This candidate gives each row one
// simdgroup: all 32 lanes decode disjoint columns and `simd_sum` reduces the
// exact same sign/FP16-scale products. Eight rows share a 256-thread
// threadgroup. It changes only associativity of f32 accumulation, so it must
// be admitted by the Metal-vs-CPU component parity gate before a full-runtime
// candidate is allowed to select it.
//
// Grid: (ceil(rows / 8) * 256, 1, 1), threadgroup: (256, 1, 1).
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
