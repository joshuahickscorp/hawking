// Isolated Qwen3-Coder-Next layer-0 MoE aggregation/second-residual seam.
//
// Unregistered by design until an explicit Qwen80 quiet lease is granted. It
// accepts only already-weighted route-index ordered deltas and a separately
// sigmoid-gated shared result. No router, expert matrix, shared gate, or first
// residual computation happens here.
//
// Fixed source-shaped operation order for every hidden element:
//   sum = route_delta[0] + ... + route_delta[9]   (f32 index order)
//   sum += gated_shared
//   sum += first_residual
//
// It is a component boundary, not a full Qwen80 layer/token/decoder/TPS path.

#include <metal_stdlib>
using namespace metal;

constant uint qwen80_moe_wave_aggregate_second_residual_hidden = 2048u;
constant uint qwen80_moe_wave_aggregate_second_residual_routes = 10u;

// Buffers: 0 route-index ordered weighted deltas float[10 * 2048],
// 1 routed sum float[2048], 2 routes (=10), 3 hidden (=2048).
kernel void qwen80_moe_wave_aggregate_second_residual_route_sum(
    const device float* route_weighted_deltas [[buffer(0)]],
    device float* routed_sum [[buffer(1)]],
    constant uint& routes [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= hidden || routes != qwen80_moe_wave_aggregate_second_residual_routes ||
        hidden != qwen80_moe_wave_aggregate_second_residual_hidden) {
        return;
    }
    float sum = 0.0f;
    // Do not reorder: route index is source top-10 selection order.
    for (uint route = 0u; route < routes; ++route) {
        sum += route_weighted_deltas[route * hidden + index];
    }
    routed_sum[index] = sum;
}

// Buffers: 0 routed sum float[2048], 1 sigmoid-gated shared float[2048],
// 2 first residual float[2048], 3 second residual float[2048], 4 hidden.
kernel void qwen80_moe_wave_aggregate_second_residual_add_shared_residual(
    const device float* routed_sum [[buffer(0)]],
    const device float* gated_shared [[buffer(1)]],
    const device float* first_residual [[buffer(2)]],
    device float* second_residual [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= hidden || hidden != qwen80_moe_wave_aggregate_second_residual_hidden) {
        return;
    }
    float value = routed_sum[index];
    value += gated_shared[index];
    value += first_residual[index];
    second_residual[index] = value;
}
