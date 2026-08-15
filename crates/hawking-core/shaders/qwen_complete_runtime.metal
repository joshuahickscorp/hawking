// Native device-side glue for the admitted Qwen30 complete-binary artifact.
//
// These kernels intentionally consume the exact compact `HQ30G1B1` body
// layout after the Rust reader has checked its header, shape, and hash.  They
// never accept BF16 source tensors or a decoded host-weight substitute.  The
// host runtime owns catalog admission and command ordering; this file owns the
// small operations that are otherwise awkward to express with the generic
// Q4/FP16 model kernels.

#include <metal_stdlib>
using namespace metal;

// Decode one logical binary-sign/FP16-scale value.  Signs are LSB-first and
// scales are a flat group array, exactly matching qwen_binary.metal.
static inline float qwen_complete_binary_value(
    device const uchar* signs,
    device const half* scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uchar byte = signs[element >> 3u];
    const bool positive = ((byte >> (element & 7u)) & 1u) != 0u;
    const float scale = float(scales[group]);
    return positive ? scale : -scale;
}

// Decode a checked compact vector into a persistent f32 control buffer.  This
// is used for RMSNorm weights only; matrix bodies remain packed and are read
// by the direct fused-decode matvec path.
kernel void qwen_complete_binary_decode_vector(
    device const uchar* signs [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& elements    [[buffer(3)]],
    constant uint& group_size  [[buffer(4)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= elements) return;
    output[id] = qwen_complete_binary_value(signs, scales, id, group_size);
}

// Direct packed embedding lookup.  This preserves the admitted weight body
// until device execution: no host f32/f16 embedding table is manufactured.
kernel void qwen_complete_binary_embedding_lookup(
    device const uchar* signs [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& token       [[buffer(3)]],
    constant uint& hidden      [[buffer(4)]],
    constant uint& vocab       [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= hidden || token >= vocab) return;
    const uint element = token * hidden + id;
    output[id] = qwen_complete_binary_value(signs, scales, element, group_size);
}

// Device-resident autoregressive feedback: the previous step's argmax id stays
// in a device buffer (`sampled_token`) and is gathered here without a host
// round-trip.  Buffer layout matches `sample_argmax_f32` (one uint).
kernel void qwen_complete_binary_embedding_lookup_device_token(
    device const uchar* signs [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    device const uint* token_id [[buffer(3)]],
    constant uint& hidden      [[buffer(4)]],
    constant uint& vocab       [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    uint id                     [[thread_position_in_grid]])
{
    const uint token = token_id[0];
    if (id >= hidden || token >= vocab) return;
    const uint element = token * hidden + id;
    output[id] = qwen_complete_binary_value(signs, scales, element, group_size);
}

// One threadgroup handles one independently normalized row.  It supports the
// Qwen30 32 Q heads and 4 KV heads (each width 128) without copying their
// activations through the host.  `input` and `output` may alias.
kernel void qwen_complete_rmsnorm_rows_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& width       [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint2 tg_pos                [[threadgroup_position_in_grid]])
{
    // The host dispatches a 2D grid: (256, rows, 1) threads with (256,1,1)
    // threadgroups, so the threadgroups form (1, rows, 1) and the row index
    // lives in .y. Binding a scalar uint here takes only .x, which is always
    // 0 -- every threadgroup then recomputed row 0 and rows 1.. were never
    // written, leaving stale buffer contents in q_norm for heads 1..31.
    const uint row = tg_pos.y;
    if (row >= rows) return;
    float sum = 0.0f;
    const uint base = row * width;
    for (uint i = tid; i < width; i += 256u) {
        const float value = input[base + i];
        sum = fma(value, value, sum);
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(scratch[0] / float(width) + eps);
    for (uint i = tid; i < width; i += 256u) {
        output[base + i] = input[base + i] * inv * weight[i];
    }
}

// Qwen3-MoE normalizes selected softmax probability weights after top-k when
// `norm_topk_prob` is true.  This is a control-plane operation on only eight
// values, but doing it on device avoids silently moving router arithmetic to
// the CPU reference path.
kernel void qwen_complete_normalize_route_weights(
    device float* weights      [[buffer(0)]],
    constant uint& count       [[buffer(1)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id != 0u) return;
    float sum = 0.0f;
    for (uint i = 0u; i < count; ++i) sum += weights[i];
    if (!isfinite(sum) || sum <= 0.0f) {
        for (uint i = 0u; i < count; ++i) weights[i] = NAN;
        return;
    }
    const float inv = 1.0f / sum;
    for (uint i = 0u; i < count; ++i) weights[i] *= inv;
}

// Offset-aware SwiGLU for a route-major expert workspace.  Each route lives
// in its own `[intermediate]` slice of the three buffers.
kernel void qwen_complete_silu_mul_offset(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& elements    [[buffer(3)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= elements) return;
    const float g = gate[id];
    output[id] = (g / (1.0f + exp(-g))) * up[id];
}

// Add the normalized route-major expert result directly into the residual.
// No shared expert exists in Qwen3-Coder-30B-A3B-Instruct's source catalog.
kernel void qwen_complete_weighted_expert_add(
    device const float* routed [[buffer(0)]],
    device const float* weights[[buffer(1)]],
    device float* residual     [[buffer(2)]],
    constant uint& hidden      [[buffer(3)]],
    constant uint& routes      [[buffer(4)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    float value = residual[id];
    for (uint route = 0u; route < routes; ++route) {
        value = fma(weights[route], routed[route * hidden + id], value);
    }
    residual[id] = value;
}

// A final fail-closed guard for native logits.  A direct packed artifact that
// causes numerical overflow must never quietly turn into a valid-looking
// token through argmax's comparison semantics.  One scalar flag is written
// on-device and inspected by the host only after command completion.
kernel void qwen_complete_any_nonfinite_f32(
    device const float* input       [[buffer(0)]],
    device atomic_uint* invalid     [[buffer(1)]],
    constant uint& elements         [[buffer(2)]],
    uint id                         [[thread_position_in_grid]])
{
    if (id >= elements) return;
    if (!isfinite(input[id])) {
        atomic_store_explicit(&invalid[0], 1u, memory_order_relaxed);
    }
}

// Fuse per-head Q/K RMSNorm into the rope + KV-append wave.
// Replaces qwen_complete_rmsnorm_rows_f32 (Q) + same (K) +
// rope_qk_kv_append_vbias_f32. RMSNorm math is copied from
// qwen_complete_rmsnorm_rows_f32 (fma tree + rsqrt). RoPE is the
// Qwen3 split-half pairing used by rope_qk_kv_append_vbias_f32.
//
// Grid: (256, n_q_heads + n_k_heads + 1, 1), TG (256, 1, 1).
//   tg.y in [0, n_q):            RMSNorm + RoPE Q in-place
//   tg.y in [n_q, n_q+n_k):      RMSNorm K in-place, RoPE into k_cache
//   tg.y == n_q+n_k:             copy V into v_cache
kernel void qwen_complete_qk_rmsnorm_rope_kv_append_f32(
    device float* q_buf              [[buffer(0)]],
    device float* k_tok              [[buffer(1)]],
    device const float* v_tok        [[buffer(2)]],
    device const float* q_weight     [[buffer(3)]],
    device const float* k_weight     [[buffer(4)]],
    device float* k_cache            [[buffer(5)]],
    device float* v_cache            [[buffer(6)]],
    constant uint& n_q_heads         [[buffer(7)]],
    constant uint& n_k_heads         [[buffer(8)]],
    constant uint& head_dim          [[buffer(9)]],
    constant uint& pos               [[buffer(10)]],
    constant float& rope_base        [[buffer(11)]],
    constant uint& kv_dim            [[buffer(12)]],
    constant uint& kv_off            [[buffer(13)]],
    constant float& eps              [[buffer(14)]],
    threadgroup float* scratch       [[threadgroup(0)]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint2 tg_pos                      [[threadgroup_position_in_grid]])
{
    const uint n_qk = n_q_heads + n_k_heads;
    const uint row = tg_pos.y;
    if (head_dim == 0u) {
        return;
    }

    if (row < n_qk) {
        const bool is_q = row < n_q_heads;
        const uint head = is_q ? row : (row - n_q_heads);
        device float* vec = is_q ? q_buf : k_tok;
        device const float* weight = is_q ? q_weight : k_weight;
        const uint base = head * head_dim;

        float sum = 0.0f;
        for (uint i = tid; i < head_dim; i += 256u) {
            const float value = vec[base + i];
            sum = fma(value, value, sum);
        }
        scratch[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = rsqrt(scratch[0] / float(head_dim) + eps);
        for (uint i = tid; i < head_dim; i += 256u) {
            vec[base + i] = vec[base + i] * inv * weight[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint pairs_per_head = head_dim / 2u;
        if (tid < pairs_per_head) {
            const uint off0 = base + tid;
            const uint off1 = off0 + pairs_per_head;
            const float x0 = vec[off0];
            const float x1 = vec[off1];
            const float theta = (float)pos
                / pow(rope_base, 2.0f * float(tid) / float(head_dim));
            const float c = cos(theta);
            const float s = sin(theta);
            const float y0 = x0 * c - x1 * s;
            const float y1 = x0 * s + x1 * c;
            if (is_q) {
                q_buf[off0] = y0;
                q_buf[off1] = y1;
            } else {
                k_cache[kv_off + off0] = y0;
                k_cache[kv_off + off1] = y1;
            }
        }
        return;
    }

    if (row == n_qk) {
        for (uint i = tid; i < kv_dim; i += 256u) {
            v_cache[kv_off + i] = v_tok[i];
        }
    }
}
