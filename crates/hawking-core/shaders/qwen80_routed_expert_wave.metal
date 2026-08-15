// Isolated Qwen3-Coder-Next layer-0 one-routed-expert component.
//
// This file is intentionally unregistered until the Qwen80 runtime owner
// grants an explicit quiet Metal lease.  It covers only one already selected
// source route:
//
//   postnorm hidden [2048]
//     -> gate_proj [512,2048] and up_proj [512,2048]
//     -> SiLU(gate) * up [512]
//     -> down_proj [2048,512]
//     -> selected route weight * output [2048]
//
// Every projection reads HQ30G1B1 sign/FP16-scale payload segments directly;
// no decoded-weight materialization is permitted.  This is not a ten-route
// wave, shared expert, residual combine, complete layer, token, or TPS path.

#include <metal_stdlib>
using namespace metal;

constant uint qwen80_routed_expert_wave_hidden = 2048u;
constant uint qwen80_routed_expert_wave_intermediate = 512u;
constant uint qwen80_routed_expert_wave_group = 128u;

inline float qwen80_routed_expert_wave_packed_value(
    const device uchar* signs,
    const device half* scales,
    uint flat_index,
    uint group_size)
{
    const uint group = flat_index / group_size;
    const uint in_group = flat_index % group_size;
    const uint sign_byte = group * (group_size / 8u) + in_group / 8u;
    const uchar bit = (signs[sign_byte] >> (in_group % 8u)) & 1u;
    const float scale = float(scales[group]);
    return bit == 1u ? scale : -scale;
}

// One threadgroup per intermediate row. Buffers:
//  0 gate signs, 1 gate scales, 2 up signs, 3 up scales,
//  4 postnorm hidden float[2048], 5 gate float[512], 6 up float[512],
//  7 rows (=512), 8 columns (=2048), 9 group (=128).
kernel void qwen80_routed_expert_wave_gate_up(
    const device uchar* gate_signs [[buffer(0)]],
    const device half* gate_scales [[buffer(1)]],
    const device uchar* up_signs [[buffer(2)]],
    const device half* up_scales [[buffer(3)]],
    const device float* hidden [[buffer(4)]],
    device float* gate_output [[buffer(5)]],
    device float* up_output [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& columns [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    // The host varies the 512 expert rows on grid Y.  Keep both position
    // builtins vector-shaped, then explicitly reduce them to scalar lane/row
    // coordinates; scalar builtins silently alias this two-dimensional grid.
    const uint lane = tid.x;
    const uint row = group_position.y;
    if (row >= rows || rows != qwen80_routed_expert_wave_intermediate ||
        columns != qwen80_routed_expert_wave_hidden ||
        group_size != qwen80_routed_expert_wave_group) {
        return;
    }
    threadgroup float gate_partial[256];
    threadgroup float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        gate_sum += qwen80_routed_expert_wave_packed_value(
                        gate_signs, gate_scales, row_base + column, group_size) *
                    hidden[column];
        up_sum += qwen80_routed_expert_wave_packed_value(
                      up_signs, up_scales, row_base + column, group_size) *
                  hidden[column];
    }
    gate_partial[lane] = gate_sum;
    up_partial[lane] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            gate_partial[lane] += gate_partial[lane + stride];
            up_partial[lane] += up_partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        gate_output[row] = gate_partial[0];
        up_output[row] = up_partial[0];
    }
}

// Buffers: 0 gate float[512], 1 up float[512], 2 activated float[512],
// 3 elements (=512).  Source Qwen3Next MLP ordering is SiLU(gate) * up.
kernel void qwen80_routed_expert_wave_swiglu(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* activated [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= elements || elements != qwen80_routed_expert_wave_intermediate) {
        return;
    }
    const float gate_value = gate[index];
    activated[index] = (gate_value / (1.0f + exp(-gate_value))) * up[index];
}

// One threadgroup per hidden row. Buffers:
//  0 down signs, 1 down scales, 2 activated float[512],
//  3 unweighted down output float[2048], 4 weighted accumulator delta
//  float[2048], 5 rows (=2048), 6 columns (=512), 7 group (=128),
//  8 route_weight (already selected and
//  normalized by the source top-10 router).
kernel void qwen80_routed_expert_wave_down_weighted(
    const device uchar* down_signs [[buffer(0)]],
    const device half* down_scales [[buffer(1)]],
    const device float* activated [[buffer(2)]],
    device float* down_output [[buffer(3)]],
    device float* weighted_accumulator_delta [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    constant float& route_weight [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint row = group_position.y;
    if (row >= rows || rows != qwen80_routed_expert_wave_hidden ||
        columns != qwen80_routed_expert_wave_intermediate ||
        group_size != qwen80_routed_expert_wave_group ||
        !isfinite(route_weight) || route_weight < 0.0f) {
        return;
    }
    threadgroup float partial[256];
    float subtotal = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        subtotal += qwen80_routed_expert_wave_packed_value(
                        down_signs, down_scales, row_base + column, group_size) *
                    activated[column];
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
        const float unweighted = partial[0];
        down_output[row] = unweighted;
        weighted_accumulator_delta[row] = unweighted * route_weight;
    }
}
