// Isolated direct-packed Qwen30 routed-expert gate/up pair candidate.
//
// The baseline runtime currently emits two independent sign+scale matvec
// dispatches for each selected expert's gate and up projection.  This
// candidate keeps the *same exact admitted 128-value sign/FP16-scale layout*
// but evaluates both projections in one row-owned thread.  It is a component
// candidate only: no Qwen layer/token runtime selects it until protected CPU
// parity and complete-token re-profiling have independently passed.

#include <metal_stdlib>
using namespace metal;

inline float qwen_direct_packed_value(
    device const uchar* signs,
    device const half* scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uint bit = element % group_size;
    const uchar packed = signs[group * (group_size / 8u) + bit / 8u];
    const bool positive = ((packed >> (bit & 7u)) & 1u) != 0u;
    const float scale = float(scales[group]);
    return positive ? scale : -scale;
}

// Baseline shape, retained inside this diagnostic-only library so both arms
// compile with identical options and consume the same source-bound buffers.
kernel void qwen_direct_packed_matvec_baseline(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float sum = 0.0f;
    const uint row_base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        sum = fma(qwen_direct_packed_value(signs, scales, row_base + col, group_size), input[col], sum);
    }
    output[row] = sum;
}

// Command-topology candidate: one dispatch produces the two independent
// routed-expert projection vectors.  No intermediate is read or written by
// the other projection, so the pair has no data dependency.
kernel void qwen_direct_packed_gate_up_pair_candidate(
    device const uchar* gate_signs  [[buffer(0)]],
    device const half* gate_scales  [[buffer(1)]],
    device const uchar* up_signs    [[buffer(2)]],
    device const half* up_scales    [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float* gate_output       [[buffer(5)]],
    device float* up_output         [[buffer(6)]],
    constant uint& rows             [[buffer(7)]],
    constant uint& cols             [[buffer(8)]],
    constant uint& group_size       [[buffer(9)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        const float x = input[col];
        gate_sum = fma(qwen_direct_packed_value(gate_signs, gate_scales, row_base + col, group_size), x, gate_sum);
        up_sum = fma(qwen_direct_packed_value(up_signs, up_scales, row_base + col, group_size), x, up_sum);
    }
    gate_output[row] = gate_sum;
    up_output[row] = up_sum;
}
