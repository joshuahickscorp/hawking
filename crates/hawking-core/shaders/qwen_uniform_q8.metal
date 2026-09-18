// Flash routed-body Q8/G32 candidate.
//
// This is intentionally a separate candidate from the proven Q4/G64 shader:
// one unsigned byte stores one offset-binary signed code (q = code - 128),
// followed by one FP16 scale per contiguous group of 32 source weights.  An
// optional sparse exact residual stream is consumed in the same fused kernels;
// its row pointers are zero-filled in the residual-free arm.  This preserves
// the route-LUT and device-only compact-bank ABI while giving the accumulated
// 48-layer body a higher-fidelity representation to falsify.
// The shared/control weights remain source BF16.

#include <metal_stdlib>
using namespace metal;

constant uint QWEN_UNIFORM_Q8_GROUP_SIZE = 32u;

static inline void qwen_uniform_q8_unpack8_dual(
    device const uchar* gate_codes,
    device const half* gate_scales,
    uint gate_group,
    device const uchar* up_codes,
    device const half* up_scales,
    uint up_group,
    uint local,
    device const float* input,
    uint col,
    thread float& acc_gate,
    thread float& acc_up)
{
    const float gate_scale = float(gate_scales[gate_group]);
    const float up_scale = float(up_scales[up_group]);
    const uint gate_base = gate_group * QWEN_UNIFORM_Q8_GROUP_SIZE + local;
    const uint up_base = up_group * QWEN_UNIFORM_Q8_GROUP_SIZE + local;
    for (uint i = 0u; i < 8u; ++i) {
        const float x = input[col + i];
        acc_gate += float(int(gate_codes[gate_base + i]) - 128) * gate_scale * x;
        acc_up += float(int(up_codes[up_base + i]) - 128) * up_scale * x;
    }
}

static inline float qwen_uniform_q8_sparse_residual_dot(
    device const uint* row_ptr,
    device const ushort* indices,
    device const float* values,
    uint row,
    device const float* input)
{
    float sum = 0.0f;
    const uint begin = row_ptr[row];
    const uint end = row_ptr[row + 1u];
    for (uint n = begin; n < end; ++n) {
        sum += values[n] * input[uint(indices[n])];
    }
    return sum;
}

// Route-major [top_k, intermediate] activated outputs plus one shared
// activated row. Grid: ceil(((top_k + 1) * intermediate) / 2) * 128.
kernel void qwen_uniform_q8_group32_compact_gate_up_shared_swiglu_geo_tpr64_tg128(
    device const uchar* gate_codes    [[buffer(0)]],
    device const half* gate_scales    [[buffer(1)]],
    device const uchar* up_codes      [[buffer(2)]],
    device const half* up_scales      [[buffer(3)]],
    device const uint* route_ids      [[buffer(4)]],
    device const uint* route_lut       [[buffer(5)]],
    device const float* input          [[buffer(6)]],
    device float* routed_out           [[buffer(7)]],
    device const ushort* shared_gate  [[buffer(8)]],
    device const ushort* shared_up    [[buffer(9)]],
    device float* shared_out           [[buffer(10)]],
    device const uint* gate_residual_row_ptr [[buffer(11)]],
    device const ushort* gate_residual_indices [[buffer(12)]],
    device const float* gate_residual_values [[buffer(13)]],
    device const uint* up_residual_row_ptr [[buffer(14)]],
    device const ushort* up_residual_indices [[buffer(15)]],
    device const float* up_residual_values [[buffer(16)]],
    constant uint& compact_experts     [[buffer(17)]],
    constant uint& top_k               [[buffer(18)]],
    constant uint& intermediate        [[buffer(19)]],
    constant uint& hidden              [[buffer(20)]],
    constant uint& source_experts      [[buffer(21)]],
    constant uint& groups_per_row      [[buffer(22)]],
    uint group_id                      [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_simdgroup]],
    uint simd_id                       [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint logical_row = group_id * 2u + team;
    const uint total_rows = (top_k + 1u) * intermediate;
    const bool live = logical_row < total_rows;
    const uint route = live ? logical_row / intermediate : 0u;
    const uint row = live ? logical_row - route * intermediate : 0u;
    const bool shared = live && route == top_k;

    uint slot = 0u;
    bool valid = live;
    if (!shared && valid) {
        const uint expert = route_ids[route];
        valid = expert < source_experts;
        if (valid) {
            slot = route_lut[expert];
            valid = slot < compact_experts;
        }
    } else if (shared) {
        valid = true;
    } else {
        valid = false;
    }

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    if (!shared && valid) {
        const uint expert_stride = intermediate * groups_per_row;
        const uint gate_row = slot * expert_stride + row * groups_per_row;
        const uint up_row = gate_row;
        for (uint col = lane_in_row * 8u; col < hidden; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q8_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q8_GROUP_SIZE;
            qwen_uniform_q8_unpack8_dual(
                gate_codes,
                gate_scales,
                gate_row + group,
                up_codes,
                up_scales,
                up_row + group,
                local,
                input,
                col,
                acc_gate,
                acc_up);
        }
        // Add the sparse exact correction once, on lane zero of the first
        // split.  The main Q8 dot remains split across both simdgroups.
        if (split == 0u && simd_lane == 0u) {
            const uint residual_row = slot * intermediate + row;
            acc_gate += qwen_uniform_q8_sparse_residual_dot(
                gate_residual_row_ptr,
                gate_residual_indices,
                gate_residual_values,
                residual_row,
                input);
            acc_up += qwen_uniform_q8_sparse_residual_dot(
                up_residual_row_ptr,
                up_residual_indices,
                up_residual_values,
                residual_row,
                input);
        }
    } else if (shared) {
        const uint row_base = row * hidden;
        for (uint col = lane_in_row * 8u; col < hidden; col += 512u) {
            const float x0 = input[col];
            const float x1 = input[col + 1u];
            const float x2 = input[col + 2u];
            const float x3 = input[col + 3u];
            const float x4 = input[col + 4u];
            const float x5 = input[col + 5u];
            const float x6 = input[col + 6u];
            const float x7 = input[col + 7u];
            acc_gate += as_type<float>(uint(shared_gate[row_base + col]) << 16u) * x0;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 1u]) << 16u) * x1;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 2u]) << 16u) * x2;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 3u]) << 16u) * x3;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 4u]) << 16u) * x4;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 5u]) << 16u) * x5;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 6u]) << 16u) * x6;
            acc_gate += as_type<float>(uint(shared_gate[row_base + col + 7u]) << 16u) * x7;
            acc_up += as_type<float>(uint(shared_up[row_base + col]) << 16u) * x0;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 1u]) << 16u) * x1;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 2u]) << 16u) * x2;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 3u]) << 16u) * x3;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 4u]) << 16u) * x4;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 5u]) << 16u) * x5;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 6u]) << 16u) * x6;
            acc_up += as_type<float>(uint(shared_up[row_base + col + 7u]) << 16u) * x7;
        }
    }
    acc_gate = simd_sum(acc_gate);
    acc_up = simd_sum(acc_up);
    if (simd_lane == 0u) {
        red[simd_id] = acc_gate;
        red[4u + simd_id] = acc_up;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && live) {
        const uint t = team * kSplit;
        const float gate = red[t] + red[t + 1u];
        const float up = red[4u + t] + red[4u + t + 1u];
        const float activated = (gate / (1.0f + exp(-gate))) * up;
        if (shared) {
            shared_out[row] = activated;
        } else {
            routed_out[route * intermediate + row] = valid ? activated : 0.0f;
        }
    }
}

// Q8/G32 routed down projection fused with shared BF16 down, route weighting,
// shared sigmoid gate, and the HyperConnection write. Grid: (hidden, 1, 1),
// TG 256.
kernel void qwen_uniform_q8_group32_compact_down_shared_direct_hc_geo_tpr64_tg128(
    device const uchar* codes            [[buffer(0)]],
    device const half* scales            [[buffer(1)]],
    device const uint* route_ids         [[buffer(2)]],
    device const uint* route_lut         [[buffer(3)]],
    device const float* activated        [[buffer(4)]],
    device const float* selected_weights [[buffer(5)]],
    device const ushort* shared_down     [[buffer(6)]],
    device const float* shared_activation [[buffer(7)]],
    device const float* shared_gate_logit [[buffer(8)]],
    device float* routed_sum_out         [[buffer(9)]],
    device float* shared_output_out      [[buffer(10)]],
    device float* shared_gated_out       [[buffer(11)]],
    device float* output                 [[buffer(12)]],
    device const float* residual         [[buffer(13)]],
    device const float* block_logits     [[buffer(14)]],
    device float* final_output           [[buffer(15)]],
    device const uint* residual_row_ptr  [[buffer(16)]],
    device const ushort* residual_indices [[buffer(17)]],
    device const float* residual_values  [[buffer(18)]],
    constant uint& compact_experts       [[buffer(19)]],
    constant uint& top_k                 [[buffer(20)]],
    constant uint& intermediate          [[buffer(21)]],
    constant uint& hidden                [[buffer(22)]],
    constant uint& source_experts        [[buffer(23)]],
    constant uint& streams               [[buffer(24)]],
    constant float& divisor              [[buffer(25)]],
    constant uint& groups_per_row        [[buffer(26)]],
    uint row                              [[thread_position_in_grid]])
{
    if (row >= hidden) return;
    float routed_sum = 0.0f;
    for (uint route = 0u; route < top_k; ++route) {
        const uint expert = route_ids[route];
        bool valid = expert < source_experts;
        uint slot = 0u;
        if (valid) {
            slot = route_lut[expert];
            valid = slot < compact_experts;
        }
        if (!valid) continue;
        const uint row_group_base = (slot * hidden + row) * groups_per_row;
        device const float* input = activated + route * intermediate;
        float expert_sum = 0.0f;
        for (uint col = 0u; col < intermediate; col += 8u) {
            const uint group = col / QWEN_UNIFORM_Q8_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q8_GROUP_SIZE;
            const uint group_index = row_group_base + group;
            const float scale = float(scales[group_index]);
            const uint code_base = group_index * QWEN_UNIFORM_Q8_GROUP_SIZE + local;
            for (uint i = 0u; i < 8u; ++i) {
                expert_sum += float(int(codes[code_base + i]) - 128) * scale * input[col + i];
            }
        }
        expert_sum += qwen_uniform_q8_sparse_residual_dot(
            residual_row_ptr,
            residual_indices,
            residual_values,
            slot * hidden + row,
            input);
        routed_sum += expert_sum * selected_weights[route];
    }

    const uint shared_row_base = row * intermediate;
    float shared_sum = 0.0f;
    for (uint col = 0u; col < intermediate; col += 8u) {
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col]) << 16u) * shared_activation[col];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 1u]) << 16u) * shared_activation[col + 1u];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 2u]) << 16u) * shared_activation[col + 2u];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 3u]) << 16u) * shared_activation[col + 3u];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 4u]) << 16u) * shared_activation[col + 4u];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 5u]) << 16u) * shared_activation[col + 5u];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 6u]) << 16u) * shared_activation[col + 6u];
        shared_sum += as_type<float>(uint(shared_down[shared_row_base + col + 7u]) << 16u) * shared_activation[col + 7u];
    }
    const float shared_gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
    const float shared_gated = shared_sum * shared_gate;
    const float moe = routed_sum + shared_gated;
    routed_sum_out[row] = routed_sum;
    shared_output_out[row] = shared_sum;
    shared_gated_out[row] = shared_gated;
    output[row] = moe;
    for (uint stream = 0u; stream < streams; ++stream) {
        const float hc_gate = 2.0f / (1.0f + exp(-block_logits[stream] / divisor));
        const uint offset = stream * hidden + row;
        final_output[offset] = residual[offset] + moe * hc_gate;
    }
}
