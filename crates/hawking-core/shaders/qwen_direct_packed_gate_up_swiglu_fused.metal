// Direct-packed Qwen30 routed-expert gate/up/SwiGLU fusion candidate.
//
// This is deliberately an isolated component kernel.  It consumes the exact
// HQ30G1B1 fixed-group payload body (LSB-first sign bits plus FP16 group
// scales) and writes the activated intermediate vector directly.  It does not
// decode either matrix into a materialized f32/f16 weight tensor.
//
// The production runtime is not wired to this file.  A runtime owner must
// first prove all-layer route parity and re-profile a complete native token.

#include <metal_stdlib>
using namespace metal;

inline float qwen_direct_packed_gate_up_value(
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

// Control topology for the component ledger: two direct-packed matvec
// dispatches followed by a separate SwiGLU dispatch.  It mirrors the current
// Qwen30 runtime's gate, up, and activation materialization boundary, but is
// used only within the component experiment.
kernel void qwen_direct_packed_gate_up_baseline_matvec(
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
        sum = fma(
            qwen_direct_packed_gate_up_value(signs, scales, row_base + col, group_size),
            input[col],
            sum);
    }
    output[row] = sum;
}

kernel void qwen_direct_packed_gate_up_baseline_swiglu(
    device const float* gate       [[buffer(0)]],
    device const float* up         [[buffer(1)]],
    device float* activation       [[buffer(2)]],
    constant uint& rows            [[buffer(3)]],
    uint row                        [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const float g = gate[row];
    activation[row] = (g / (1.0f + exp(-g))) * up[row];
}

// One row-owned thread performs the gate and up reductions while `input[col]`
// is resident in a scalar.  It then applies SwiGLU and writes only the final
// intermediate value.  The two direct packed projection bodies remain
// distinct and exact; only command topology and temporary activation traffic
// change.
kernel void qwen_direct_packed_gate_up_swiglu_fused_candidate(
    device const uchar* gate_signs  [[buffer(0)]],
    device const half* gate_scales  [[buffer(1)]],
    device const uchar* up_signs    [[buffer(2)]],
    device const half* up_scales    [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float* activation        [[buffer(5)]],
    constant uint& rows             [[buffer(6)]],
    constant uint& cols             [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        const float x = input[col];
        gate_sum = fma(
            qwen_direct_packed_gate_up_value(
                gate_signs, gate_scales, row_base + col, group_size),
            x,
            gate_sum);
        up_sum = fma(
            qwen_direct_packed_gate_up_value(
                up_signs, up_scales, row_base + col, group_size),
            x,
            up_sum);
    }
    activation[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
}
