// Direct-packed Qwen30 gate/up paired-topology candidate with scalar-order
// arithmetic preserved.
//
// This is intentionally distinct from the earlier explicit-FMA fusion
// experiment. The current Qwen30 scalar control source performs one
// non-fused `sum += weight * input` update for each column.  The CPU
// discriminator established that explicit FMA changes bit patterns for real
// admitted expert tensors, so this candidate uses the compiler-supported
// no-contract/no-reassociate controls around explicit product and accumulator
// variables to retain that source-level arithmetic while only changing the
// paired gate/up command topology and intermediate traffic.
//
// It remains a diagnostic kernel: only an explicitly selected, isolated
// runtime may use it, and device route-major parity plus exact template A/B
// completion parity are mandatory before any integration discussion.

#include <metal_stdlib>
using namespace metal;

// This MSL compiler does not accept a C++ `precise` type qualifier. These are
// the compiler-supported controls used by the existing exactness shaders: do
// not contract the explicit product/add recurrence into FMA or reassociate its
// increasing-column accumulation. Device parity remains the only proof that
// the compiled result matches the scalar control.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float qwen_direct_packed_paired_scalar_order_value(
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

// One row-owned thread retains the scalar control's increasing-column order
// separately for gate and up. The no-contract/no-reassociate region prevents
// contraction of the explicit product/add recurrence into the rejected FMA
// arithmetic. The candidate
// eliminates only materialized gate/up intermediates; it writes the same
// route-major SwiGLU activation boundary consumed by the existing down wave.
kernel void qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate(
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
        const float gate_weight = qwen_direct_packed_paired_scalar_order_value(
            gate_signs, gate_scales, row_base + col, group_size);
        const float up_weight = qwen_direct_packed_paired_scalar_order_value(
            up_signs, up_scales, row_base + col, group_size);
        const float gate_product = gate_weight * x;
        const float up_product = up_weight * x;
        gate_sum = gate_sum + gate_product;
        up_sum = up_sum + up_product;
    }
    activation[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
}

#pragma clang fp reassociate(on)
#pragma clang fp contract(on)
