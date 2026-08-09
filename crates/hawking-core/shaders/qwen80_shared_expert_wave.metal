// Isolated Qwen3-Coder-Next layer-0 shared-expert component.
//
// Unregistered by design until the Qwen80 runtime owner grants an explicit
// quiet Metal lease.  This file consumes only direct HQ30G1B1 payload segments:
//
//   postnorm hidden [2048]
//     -> shared gate/up [512,2048]
//     -> SiLU(gate) * up [512]
//     -> shared down [2048,512]
//     -> scalar shared_expert_gate [1,2048] -> sigmoid -> gated shared [2048]
//
// It stops before routed-expert accumulation, MoE combination, second
// residual, a layer, token, decoder, or performance measurement.  No decoded
// weight materialization is permitted.

#include <metal_stdlib>
using namespace metal;

constant uint qwen80_shared_expert_wave_hidden = 2048u;
constant uint qwen80_shared_expert_wave_intermediate = 512u;
constant uint qwen80_shared_expert_wave_group = 128u;

inline float qwen80_shared_expert_wave_packed_value(
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
// 0 shared gate signs, 1 shared gate scales, 2 shared up signs,
// 3 shared up scales, 4 postnorm hidden, 5 gate[512], 6 up[512],
// 7 rows (=512), 8 columns (=2048), 9 group (=128).
kernel void qwen80_shared_expert_wave_gate_up(
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
    const uint lane = tid.x;
    const uint row = group_position.y;
    if (row >= rows || rows != qwen80_shared_expert_wave_intermediate ||
        columns != qwen80_shared_expert_wave_hidden ||
        group_size != qwen80_shared_expert_wave_group) {
        return;
    }
    threadgroup float gate_partial[256];
    threadgroup float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        gate_sum += qwen80_shared_expert_wave_packed_value(
                        gate_signs, gate_scales, row_base + column, group_size) *
                    hidden[column];
        up_sum += qwen80_shared_expert_wave_packed_value(
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

// Buffers: 0 gate[512], 1 up[512], 2 activated[512], 3 elements (=512).
kernel void qwen80_shared_expert_wave_swiglu(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* activated [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= elements || elements != qwen80_shared_expert_wave_intermediate) {
        return;
    }
    const float gate_value = gate[index];
    activated[index] = (gate_value / (1.0f + exp(-gate_value))) * up[index];
}

// One threadgroup per hidden row. Buffers: 0 down signs, 1 down scales,
// 2 activated[512], 3 shared output[2048], 4 rows (=2048), 5 columns (=512),
// 6 group (=128).
kernel void qwen80_shared_expert_wave_down(
    const device uchar* down_signs [[buffer(0)]],
    const device half* down_scales [[buffer(1)]],
    const device float* activated [[buffer(2)]],
    device float* shared_output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint row = group_position.y;
    if (row >= rows || rows != qwen80_shared_expert_wave_hidden ||
        columns != qwen80_shared_expert_wave_intermediate ||
        group_size != qwen80_shared_expert_wave_group) {
        return;
    }
    threadgroup float partial[256];
    float subtotal = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        subtotal += qwen80_shared_expert_wave_packed_value(
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
        shared_output[row] = partial[0];
    }
}

// One reduction threadgroup for the scalar shared gate. Buffers: 0 gate signs,
// 1 gate scales, 2 postnorm hidden[2048], 3 gate logit[1], 4 columns (=2048),
// 5 group (=128).
kernel void qwen80_shared_expert_wave_scalar_gate(
    const device uchar* gate_signs [[buffer(0)]],
    const device half* gate_scales [[buffer(1)]],
    const device float* hidden [[buffer(2)]],
    device float* gate_logit [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint lane = tid.x;
    if (columns != qwen80_shared_expert_wave_hidden ||
        group_size != qwen80_shared_expert_wave_group) {
        return;
    }
    threadgroup float partial[256];
    float subtotal = 0.0f;
    for (uint column = lane; column < columns; column += 256u) {
        subtotal += qwen80_shared_expert_wave_packed_value(
                        gate_signs, gate_scales, column, group_size) *
                    hidden[column];
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
        gate_logit[0] = partial[0];
    }
}

// Buffers: 0 shared output[2048], 1 shared gate logit[1], 2 gated shared[2048],
// 3 hidden (=2048).  Sigmoid is applied once and broadcasts across the body.
kernel void qwen80_shared_expert_wave_apply_sigmoid_gate(
    const device float* shared_output [[buffer(0)]],
    const device float* gate_logit [[buffer(1)]],
    device float* gated_shared [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= hidden || hidden != qwen80_shared_expert_wave_hidden) {
        return;
    }
    const float logit = gate_logit[0];
    if (!isfinite(logit)) {
        return;
    }
    const float gate = 1.0f / (1.0f + exp(-logit));
    gated_shared[index] = shared_output[index] * gate;
}
