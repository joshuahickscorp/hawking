// Direct-packed HQ30GR2 sparse-residual gate/up/SwiGLU diagnostic kernel.
//
// This shader is deliberately not registered as a generic Qwen runtime
// selection.  It is for one separately typed quality-candidate diagnostic:
// embedded HQ30G1B1 direct bases plus the two validated, sorted FP16 sparse
// residual lists.  It never materializes dense weights or reads BF16 source
// tensors.  A host executor must prove CPU/device parity before allowing this
// output into any all-layer diagnostic graph.

#include <metal_stdlib>
using namespace metal;

// Match the validated scalar control recurrence.  The quality candidate is a
// representation experiment, not permission to alter arithmetic order.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline float qwen30_quality_repack_direct_value(
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

inline float qwen30_quality_repack_add_sparse_row(
    float sum,
    uint row,
    uint cols,
    device const uint* residual_indices,
    device const half* residual_values,
    uint residual_count,
    device const float* input)
{
    // Manifest admission proves globally sorted, in-range indices.  Keeping
    // the scan in serialized index order preserves the CPU reference's
    // base-then-sorted-sparse correction order for this diagnostic path.
    for (uint residual = 0u; residual < residual_count; ++residual) {
        const uint flat = residual_indices[residual];
        if ((flat / cols) == row) {
            const float correction_product = float(residual_values[residual]) * input[flat % cols];
            sum = sum + correction_product;
        }
    }
    return sum;
}

kernel void qwen30_quality_repack_sparse_gate_up_swiglu(
    device const uchar* gate_signs             [[buffer(0)]],
    device const half* gate_scales             [[buffer(1)]],
    device const uint* gate_residual_indices   [[buffer(2)]],
    device const half* gate_residual_values    [[buffer(3)]],
    device const uchar* up_signs               [[buffer(4)]],
    device const half* up_scales               [[buffer(5)]],
    device const uint* up_residual_indices     [[buffer(6)]],
    device const half* up_residual_values      [[buffer(7)]],
    device const float* input                  [[buffer(8)]],
    device float* activation                   [[buffer(9)]],
    constant uint& gate_residual_count         [[buffer(10)]],
    constant uint& up_residual_count           [[buffer(11)]],
    constant uint& rows                        [[buffer(12)]],
    constant uint& cols                        [[buffer(13)]],
    constant uint& group_size                  [[buffer(14)]],
    uint row                                   [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        const float x = input[col];
        const float gate_weight = qwen30_quality_repack_direct_value(
            gate_signs, gate_scales, row_base + col, group_size);
        const float up_weight = qwen30_quality_repack_direct_value(
            up_signs, up_scales, row_base + col, group_size);
        const float gate_product = gate_weight * x;
        const float up_product = up_weight * x;
        gate_sum = gate_sum + gate_product;
        up_sum = up_sum + up_product;
    }
    gate_sum = qwen30_quality_repack_add_sparse_row(
        gate_sum,
        row,
        cols,
        gate_residual_indices,
        gate_residual_values,
        gate_residual_count,
        input);
    up_sum = qwen30_quality_repack_add_sparse_row(
        up_sum,
        row,
        cols,
        up_residual_indices,
        up_residual_values,
        up_residual_count,
        input);
    activation[row] = (gate_sum / (1.0f + exp(-gate_sum))) * up_sum;
}

#pragma clang fp reassociate(on)
#pragma clang fp contract(on)
