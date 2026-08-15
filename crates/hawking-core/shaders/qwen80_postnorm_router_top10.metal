// Isolated Qwen3-Coder-Next layer-0 post-attention routing component.
//
// This source is deliberately not registered with the shared Metal program
// until the Qwen80 runtime owner grants a quiet device lease.  It consumes
// only Qwen80 HQ30G1B1 direct-packed sign/FP16-scale payload segments:
//
//   post-attention residual [2048]
//     -> source residual RMSNorm (x * inv_rms * (1 + weight))
//     -> router gate [512, 2048]
//     -> stable source-compatible top-10 / selected-probability renormalize
//
// It is a component seam, never a complete Qwen80 layer, token, decoder, or
// performance path.  Buffer layout and scalar bindings are documented beside
// the host example `ascension_qwen80_direct_packed_postnorm_router_top10.rs`.

#include <metal_stdlib>
using namespace metal;

constant uint qwen80_postnorm_router_top10_hidden = 2048u;
constant uint qwen80_postnorm_router_top10_experts = 512u;
constant uint qwen80_postnorm_router_top10_top_k = 10u;
constant uint qwen80_postnorm_router_top10_group = 128u;

inline float qwen80_postnorm_router_top10_packed_value(
    const device uchar* signs,
    const device half* scales,
    uint flat_index,
    uint group_size)
{
    const uint group = flat_index / group_size;
    const uint within_group = flat_index % group_size;
    const uint sign_byte = group * (group_size / 8u) + (within_group / 8u);
    const uchar bit = (signs[sign_byte] >> (within_group % 8u)) & 1u;
    const float scale = float(scales[group]);
    return bit == 1u ? scale : -scale;
}

// Buffers:
//   0 residual input float[hidden]
//   1 direct-packed norm signs byte[hidden/8]
//   2 direct-packed norm scales half[hidden/group]
//   3 normalized output float[hidden]
//   4 hidden uint (must be 2048)
//   5 group_size uint (must be 128)
//   6 epsilon float (must be source 1e-6)
kernel void qwen80_postnorm_router_top10_rmsnorm(
    const device float* residual [[buffer(0)]],
    const device uchar* norm_signs [[buffer(1)]],
    const device half* norm_scales [[buffer(2)]],
    device float* normalized [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (hidden != qwen80_postnorm_router_top10_hidden ||
        group_size != qwen80_postnorm_router_top10_group ||
        !isfinite(epsilon) || epsilon <= 0.0f) {
        return;
    }

    threadgroup float partial[256];
    float sum_squares = 0.0f;
    for (uint index = tid; index < hidden; index += 256u) {
        const float value = residual[index];
        sum_squares += value * value;
    }
    partial[tid] = sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial[0] = rsqrt(partial[0] / float(hidden) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse_rms = partial[0];
    for (uint index = tid; index < hidden; index += 256u) {
        const float packed_weight = qwen80_postnorm_router_top10_packed_value(
            norm_signs, norm_scales, index, group_size);
        // Qwen3Next residual RMSNorm uses (1 + weight), not just weight.
        normalized[index] = residual[index] * inverse_rms * (1.0f + packed_weight);
    }
}

// Buffers:
//   0 router signs byte[experts * hidden / 8]
//   1 router scales half[experts * hidden / group]
//   2 normalized input float[hidden]
//   3 logits float[experts]
//   4 rows uint (must be 512)
//   5 columns uint (must be 2048)
//   6 group_size uint (must be 128)
kernel void qwen80_postnorm_router_top10_matvec(
    const device uchar* router_signs [[buffer(0)]],
    const device half* router_scales [[buffer(1)]],
    const device float* normalized [[buffer(2)]],
    device float* logits [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    // The host dispatches one 256-thread group per router row on grid Y:
    // (256, 512, 1).  A scalar builtin binds X and silently aliases every
    // group to row zero, so retain the coordinate dimensionality explicitly.
    const uint row = group_position.y;
    const uint lane = tid.x;
    if (row >= rows || rows != qwen80_postnorm_router_top10_experts ||
        columns != qwen80_postnorm_router_top10_hidden ||
        group_size != qwen80_postnorm_router_top10_group) {
        return;
    }

    threadgroup float partial[256];
    float subtotal = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        subtotal += qwen80_postnorm_router_top10_packed_value(
                        router_signs, router_scales, row_base + column, group_size) *
                    normalized[column];
    }
    partial[lane] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        logits[row] = partial[0];
    }
}

// The source router makes its top-k choice by repeatedly selecting the
// largest softmax probability.  When HAWKING_DS_ROUTE_TIE_EPS is nonzero, a
// finite candidate within the supplied epsilon chooses the lower expert ID.
// The host supplies exactly the active source policy epsilon in `tie_epsilon`.
//
// Buffers:
//   0 logits float[experts]
//   1 probabilities scratch float[experts]
//   2 selected expert IDs uint[top_k]
//   3 selected/renormalized weights float[top_k]
//   4 experts uint (must be 512)
//   5 top_k uint (must be 10)
//   6 tie_epsilon float (must be finite and >= 0)
kernel void qwen80_postnorm_router_top10_select(
    const device float* logits [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    device uint* selected_ids [[buffer(2)]],
    device float* selected_weights [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant float& tie_epsilon [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0u || experts != qwen80_postnorm_router_top10_experts ||
        top_k != qwen80_postnorm_router_top10_top_k || !isfinite(tie_epsilon) ||
        tie_epsilon < 0.0f) {
        return;
    }

    float maximum = -INFINITY;
    for (uint expert = 0u; expert < experts; ++expert) {
        maximum = max(maximum, logits[expert]);
    }
    if (!isfinite(maximum)) {
        return;
    }
    float sum = 0.0f;
    for (uint expert = 0u; expert < experts; ++expert) {
        const float value = exp(logits[expert] - maximum);
        probabilities[expert] = value;
        sum += value;
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return;
    }
    for (uint expert = 0u; expert < experts; ++expert) {
        probabilities[expert] /= sum;
    }

    float selected_sum = 0.0f;
    for (uint route_index = 0u; route_index < top_k; ++route_index) {
        uint best_index = 0u;
        float best_value = -INFINITY;
        for (uint expert = 0u; expert < experts; ++expert) {
            const float value = probabilities[expert];
            const bool finite_pair = isfinite(best_value) && isfinite(value);
            const bool tied = tie_epsilon > 0.0f && finite_pair &&
                              abs(value - best_value) <= tie_epsilon;
            if ((value > best_value && !tied) || (tied && expert < best_index)) {
                best_index = expert;
                best_value = value;
            }
        }
        if (!isfinite(best_value) || best_value < 0.0f) {
            return;
        }
        selected_ids[route_index] = best_index;
        selected_weights[route_index] = best_value;
        selected_sum += best_value;
        probabilities[best_index] = -INFINITY;
    }
    if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
        return;
    }
    for (uint route_index = 0u; route_index < top_k; ++route_index) {
        selected_weights[route_index] /= selected_sum;
    }
}
