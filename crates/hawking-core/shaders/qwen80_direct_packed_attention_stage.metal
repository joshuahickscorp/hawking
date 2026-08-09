// Bounded direct-packed Qwen3-Coder-Next full-attention component stage.
//
// This shader owns the non-matvec portions of one cached layer-3 GQA mixer
// probe. Q/K/V/O projections remain in the existing direct binary sign +
// FP16-scale matvec primitive; this file keeps Q/K RMSNorm, partial RoPE,
// KV append, and the source sigmoid attention gate on the device.
//
// It is deliberately a parity baseline. It does not compose the surrounding
// input/post-attention norms, residuals, MoE, later layers, lm_head, sampler,
// token loop, HCLI, or any throughput measurement.

#include <metal_stdlib>
using namespace metal;

inline float qwen80_attention_stage_binary_value(
    device const uchar* signs,
    device const ushort* scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uint local = element - group * group_size;
    const float scale = (float)as_type<half>(scales[group]);
    const uint byte_index = group * (group_size / 8u) + local / 8u;
    const bool positive = ((uint)signs[byte_index] & (1u << (local & 7u))) != 0u;
    return positive ? scale : -scale;
}

// One thread owns one query head. For heads below n_kv_heads it additionally
// writes the matching RoPE-transformed K and unrotated V into the exact
// [sequence][KV-head][head-dim] cache layout consumed by `mha_decode_f32`.
//
// q_proj has source layout [head][query(256), gate(256)], not two global
// query/gate halves. Qwen3-Next's q_norm/k_norm weights are residual scales,
// i.e. norm(x) * (1 + weight), not the conventional norm(x) * weight. RoPE
// rotates only the first 64 (= 256 * .25) dimensions using the
// non-interleaved rotate_half layout.
kernel void qwen80_attention_qk_norm_rope_cache(
    device const float* q_proj              [[buffer(0)]],
    device const float* k_proj              [[buffer(1)]],
    device const float* v_proj              [[buffer(2)]],
    device const uchar* q_norm_signs        [[buffer(3)]],
    device const ushort* q_norm_scales      [[buffer(4)]],
    device const uchar* k_norm_signs        [[buffer(5)]],
    device const ushort* k_norm_scales      [[buffer(6)]],
    device float* query                     [[buffer(7)]],
    device float* key_cache                 [[buffer(8)]],
    device float* value_cache               [[buffer(9)]],
    constant uint& sequence_slot            [[buffer(10)]],
    constant uint& n_heads                  [[buffer(11)]],
    constant uint& n_kv_heads               [[buffer(12)]],
    constant uint& head_dim                 [[buffer(13)]],
    constant uint& rotary_dim               [[buffer(14)]],
    constant uint& group_size               [[buffer(15)]],
    constant float& rope_theta              [[buffer(16)]],
    constant float& rms_epsilon             [[buffer(17)]],
    uint head                               [[thread_position_in_grid]])
{
    // The bounded Qwen80 contract is intentionally exact; accepting a nearby
    // shape here would turn this into a generic attention kernel and weaken
    // the source-bound parity probe.
    if (head >= n_heads || n_heads != 16u || n_kv_heads != 2u ||
        head_dim != 256u || rotary_dim != 64u || group_size != 128u ||
        rope_theta != 5000000.0f || rms_epsilon != 1.0e-6f) {
        return;
    }

    const uint q_base = head * head_dim;
    const uint q_projection_base = head * (2u * head_dim);
    float q_variance = 0.0f;
    for (uint dim = 0u; dim < head_dim; ++dim) {
        const float value = q_proj[q_projection_base + dim];
        q_variance += value * value;
    }
    const float q_inverse_rms = rsqrt(q_variance / float(head_dim) + rms_epsilon);
    for (uint dim = 0u; dim < head_dim; ++dim) {
        const float raw = q_proj[q_projection_base + dim];
        const float normed = raw * q_inverse_rms *
            (1.0f + qwen80_attention_stage_binary_value(
                q_norm_signs, q_norm_scales, dim, group_size));
        if (dim < rotary_dim) {
            const uint half_dim = rotary_dim / 2u;
            const uint frequency_index = dim < half_dim ? dim : dim - half_dim;
            const float inv_frequency = pow(rope_theta, -2.0f * float(frequency_index) / float(rotary_dim));
            const float angle = float(sequence_slot) * inv_frequency;
            const float cosine = cos(angle);
            const float sine = sin(angle);
            const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
            const float peer_raw = q_proj[q_projection_base + peer] * q_inverse_rms *
                (1.0f + qwen80_attention_stage_binary_value(
                    q_norm_signs, q_norm_scales, peer, group_size));
            query[q_base + dim] = dim < half_dim
                ? normed * cosine - peer_raw * sine
                : normed * cosine + peer_raw * sine;
        } else {
            query[q_base + dim] = normed;
        }
    }

    // The two KV heads are independent and are written by heads 0 and 1,
    // avoiding a write race while all 16 query heads are normalized.
    if (head < n_kv_heads) {
        const uint kv_base = head * head_dim;
        float k_variance = 0.0f;
        for (uint dim = 0u; dim < head_dim; ++dim) {
            const float value = k_proj[kv_base + dim];
            k_variance += value * value;
        }
        const float k_inverse_rms = rsqrt(k_variance / float(head_dim) + rms_epsilon);
        const uint cache_base = (sequence_slot * n_kv_heads + head) * head_dim;
        for (uint dim = 0u; dim < head_dim; ++dim) {
            const float raw = k_proj[kv_base + dim];
            const float normed = raw * k_inverse_rms *
                (1.0f + qwen80_attention_stage_binary_value(
                    k_norm_signs, k_norm_scales, dim, group_size));
            if (dim < rotary_dim) {
                const uint half_dim = rotary_dim / 2u;
                const uint frequency_index = dim < half_dim ? dim : dim - half_dim;
                const float inv_frequency = pow(rope_theta, -2.0f * float(frequency_index) / float(rotary_dim));
                const float angle = float(sequence_slot) * inv_frequency;
                const float cosine = cos(angle);
                const float sine = sin(angle);
                const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
                const float peer_raw = k_proj[kv_base + peer] * k_inverse_rms *
                    (1.0f + qwen80_attention_stage_binary_value(
                        k_norm_signs, k_norm_scales, peer, group_size));
                key_cache[cache_base + dim] = dim < half_dim
                    ? normed * cosine - peer_raw * sine
                    : normed * cosine + peer_raw * sine;
            } else {
                key_cache[cache_base + dim] = normed;
            }
            value_cache[cache_base + dim] = v_proj[kv_base + dim];
        }
    }
}

// Apply the second half of each Qwen3-Next q_proj head as the source attention
// gate. The q_proj layout is [head][query(256), gate(256)]; gate is per Q
// head / head dimension and operates after causal GQA.
kernel void qwen80_attention_apply_sigmoid_gate(
    device const float* attention_output    [[buffer(0)]],
    device const float* q_proj              [[buffer(1)]],
    device float* gated_output              [[buffer(2)]],
    constant uint& elements                  [[buffer(3)]],
    constant uint& head_dim                  [[buffer(4)]],
    uint index                               [[thread_position_in_grid]])
{
    if (index >= elements) return;
    if (head_dim != 256u || elements != 16u * head_dim) return;
    const uint head = index / head_dim;
    const uint dimension = index - head * head_dim;
    const uint gate_offset = head * (2u * head_dim) + head_dim + dimension;
    const float gate = q_proj[gate_offset];
    const float sigmoid = 1.0f / (1.0f + exp(-gate));
    gated_output[index] = attention_output[index] * sigmoid;
}
