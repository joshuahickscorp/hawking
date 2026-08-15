// Staged generic Qwen3-Coder-Next all-ten routed-expert wave.
//
// This source is intentionally NOT in metal/mod.rs yet.  A future explicitly
// leased source-bound layer-0 capture may append it to the shared library only
// after its host bridge has one admitted catalog, a retained first-residual
// buffer, and durable outer reaping.  These kernels never decode weights into
// BF16/f32 tensors: every route directly reads its HQ30G1B1 sign/FP16-scale
// sections.  They cover only the ten routed bodies; the host must separately
// bind postnorm/router, shared expert, and fixed residual combine.

#include <metal_stdlib>
using namespace metal;

constant uint qwen80_all_ten_route_hidden = 2048u;
constant uint qwen80_all_ten_route_intermediate = 512u;
constant uint qwen80_all_ten_route_count = 10u;
constant uint qwen80_all_ten_route_group = 128u;

inline float qwen80_all_ten_route_packed_value(
    const device uchar* signs,
    const device half* scales,
    uint route,
    uint route_sign_bytes,
    uint route_scale_elements,
    uint flat_index,
    uint group_size)
{
    const uint group = flat_index / group_size;
    const uint in_group = flat_index % group_size;
    const uint sign_byte = route * route_sign_bytes +
                           group * (group_size / 8u) + in_group / 8u;
    const uint scale_index = route * route_scale_elements + group;
    const uchar bit = (signs[sign_byte] >> (in_group % 8u)) & 1u;
    const float scale = float(scales[scale_index]);
    return bit == 1u ? scale : -scale;
}

// Compare the dynamic router output with the immutable all-ten source plan
// before a capture can accept the downstream fixed route-body outputs.
// The graph keeps router outputs and expected route controls in distinct
// buffers: the router must not be allowed to overwrite the authority it is
// being checked against.  This is a capture refusal guard, not a substitute
// for the host's post-fence ledger validation.
//
// Buffers: 0 observed IDs u32[10], 1 observed weights f32[10],
// 2 expected IDs u32[10], 3 expected weights f32[10], 4 guard u32[1],
// 5 count (=10), 6 weight tolerance.
kernel void qwen80_all_ten_routed_wave_route_guard(
    const device uint* observed_ids [[buffer(0)]],
    const device float* observed_weights [[buffer(1)]],
    const device uint* expected_ids [[buffer(2)]],
    const device float* expected_weights [[buffer(3)]],
    device uint* guard [[buffer(4)]],
    constant uint& route_count [[buffer(5)]],
    constant float& weight_tolerance [[buffer(6)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane != 0u) {
        return;
    }
    bool accepted = route_count == qwen80_all_ten_route_count &&
                    isfinite(weight_tolerance) && weight_tolerance >= 0.0f;
    for (uint route = 0u; route < qwen80_all_ten_route_count; ++route) {
        const float observed_weight = observed_weights[route];
        const float expected_weight = expected_weights[route];
        accepted = accepted && observed_ids[route] == expected_ids[route] &&
                   isfinite(observed_weight) && isfinite(expected_weight) &&
                   fabs(observed_weight - expected_weight) <= weight_tolerance;
    }
    guard[0] = accepted ? 1u : 0u;
}

// One 256-thread group per [route, intermediate row].  Grid is
// (256, 512, 10); the route coordinate MUST come from group_position.z so no
// scalar-X ABI can silently alias all ten source waves onto route zero.
//
// Buffers:
//  0 gate signs  byte[10 * 512*2048/8]
//  1 gate scales half[10 * 512*2048/128]
//  2 up signs    byte[same]
//  3 up scales   half[same]
//  4 postnorm hidden float[2048]
//  5 gate output float[10 * 512]
//  6 up output   float[10 * 512]
//  7 route_count (=10), 8 rows (=512), 9 columns (=2048), 10 group (=128)
kernel void qwen80_all_ten_routed_wave_gate_up(
    const device uchar* gate_signs [[buffer(0)]],
    const device half* gate_scales [[buffer(1)]],
    const device uchar* up_signs [[buffer(2)]],
    const device half* up_scales [[buffer(3)]],
    const device float* hidden [[buffer(4)]],
    device float* gate_output [[buffer(5)]],
    device float* up_output [[buffer(6)]],
    constant uint& route_count [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& columns [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint row = group_position.y;
    const uint route = group_position.z;
    if (route >= route_count || route_count != qwen80_all_ten_route_count ||
        row >= rows || rows != qwen80_all_ten_route_intermediate ||
        columns != qwen80_all_ten_route_hidden ||
        group_size != qwen80_all_ten_route_group) {
        return;
    }
    const uint route_sign_bytes = (rows * columns) / 8u;
    const uint route_scale_elements = (rows * columns) / group_size;
    threadgroup float gate_partial[256];
    threadgroup float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        const uint flat = row_base + column;
        gate_sum += qwen80_all_ten_route_packed_value(
                        gate_signs, gate_scales, route, route_sign_bytes,
                        route_scale_elements, flat, group_size) * hidden[column];
        up_sum += qwen80_all_ten_route_packed_value(
                      up_signs, up_scales, route, route_sign_bytes,
                      route_scale_elements, flat, group_size) * hidden[column];
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
        const uint output = route * rows + row;
        gate_output[output] = gate_partial[0];
        up_output[output] = up_partial[0];
    }
}

// Grid is (512, 10, 1).  Buffers: 0 gate[10*512], 1 up[10*512],
// 2 activated[10*512], 3 route_count (=10), 4 elements (=512).
kernel void qwen80_all_ten_routed_wave_swiglu(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* activated [[buffer(2)]],
    constant uint& route_count [[buffer(3)]],
    constant uint& elements [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    const uint element = tid.x;
    const uint route = tid.y;
    if (route >= route_count || route_count != qwen80_all_ten_route_count ||
        element >= elements || elements != qwen80_all_ten_route_intermediate) {
        return;
    }
    const uint index = route * elements + element;
    const float gate_value = gate[index];
    activated[index] = (gate_value / (1.0f + exp(-gate_value))) * up[index];
}

// One 256-thread group per [route, hidden row].  Grid is (256, 2048, 10).
// Buffers:
//  0 down signs byte[10 * 2048*512/8]
//  1 down scales half[10 * 2048*512/128]
//  2 activated float[10 * 512]
//  3 source-normalized weights float[10]
//  4 weighted route outputs float[10 * 2048]
//  5 route_count (=10), 6 rows (=2048), 7 columns (=512), 8 group (=128)
kernel void qwen80_all_ten_routed_wave_down_weighted(
    const device uchar* down_signs [[buffer(0)]],
    const device half* down_scales [[buffer(1)]],
    const device float* activated [[buffer(2)]],
    const device float* route_weights [[buffer(3)]],
    device float* weighted_output [[buffer(4)]],
    constant uint& route_count [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& columns [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint row = group_position.y;
    const uint route = group_position.z;
    if (route >= route_count || route_count != qwen80_all_ten_route_count ||
        row >= rows || rows != qwen80_all_ten_route_hidden ||
        columns != qwen80_all_ten_route_intermediate ||
        group_size != qwen80_all_ten_route_group) {
        return;
    }
    const float weight = route_weights[route];
    if (!isfinite(weight) || weight < 0.0f) {
        return;
    }
    const uint route_sign_bytes = (rows * columns) / 8u;
    const uint route_scale_elements = (rows * columns) / group_size;
    threadgroup float partial[256];
    float subtotal = 0.0f;
    const uint row_base = row * columns;
    const uint activation_base = route * columns;
    for (uint column = lane; column < columns; column += 256u) {
        subtotal += qwen80_all_ten_route_packed_value(
                        down_signs, down_scales, route, route_sign_bytes,
                        route_scale_elements, row_base + column, group_size) *
                    activated[activation_base + column];
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
        weighted_output[route * rows + row] = partial[0] * weight;
    }
}
