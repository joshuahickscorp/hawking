// sample.metal — wedge 3, on-GPU sampling.
//
// Logits stay on the GPU; only the sampled token id crosses the bus.
//
// Kernels:
//   sample_temperature    — in-place temperature scaling.
//                           [Phase 2.5]
//   sample_repetition     — repetition penalty over a context window.
//                           [Phase 2.5]
//   sample_topk_topp      — fused top-K + top-P + softmax + draw.
//                           Single launch.
//                           [Phase 2.5]
//   sample_constraint     — applies a constraint mask (JSON-schema /
//                           regex) by setting masked logits to -inf
//                           before sampling.
//                           [Phase 2.5]
//   sample_argmax_f32     — deterministic greedy argmax over fp32
//                           logits. Bootstrap kernel for token-only
//                           GPU readback.

#include <metal_stdlib>
using namespace metal;

kernel void sample_temperature(
    device       half*  logits [[buffer(0)]],
    constant     uint&  n      [[buffer(1)]],
    constant     float& temp   [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    if (temp <= 0.0f) return; // greedy: leave untouched
    logits[id] = half((float)logits[id] / temp);
}

kernel void sample_repetition(
    device       half*  logits  [[buffer(0)]],
    device const uint*  recent  [[buffer(1)]],
    constant     uint&  n_recent[[buffer(2)]],
    constant     float& penalty [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n_recent) return;
    uint t = recent[id];
    float v = (float)logits[t];
    logits[t] = half(v >= 0.0f ? v / penalty : v * penalty);
}

kernel void sample_topk_topp_stub(
    device const half*  logits [[buffer(0)]],
    device       uint*  token  [[buffer(1)]],
    constant     uint&  top_k  [[buffer(2)]],
    constant     float& top_p  [[buffer(3)]],
    constant     uint&  seed   [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]])
{
    (void)tid;
}

kernel void sample_argmax_f32(
    device const float* logits [[buffer(0)]],
    device       uint*  token  [[buffer(1)]],
    constant     uint&  n      [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id != 0 || n == 0) return;
    uint best = 0;
    float best_v = -INFINITY;
    for (uint i = 0; i < n; ++i) {
        float v = logits[i];
        if (v > best_v) {
            best = i;
            best_v = v;
        }
    }
    token[0] = best;
}

kernel void sample_constraint(
    device       half*  logits [[buffer(0)]],
    device const uchar* mask   [[buffer(1)]],
    constant     uint&  n      [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    if (mask[id] == 0) {
        logits[id] = half(-INFINITY);
    }
}
