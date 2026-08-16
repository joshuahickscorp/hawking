// Device-resident Qwen80 hybrid-decode activations (uniform-Q4 neighbor ops).
//
// These kernels keep the residual / mixer / SwiGLU / DeltaNet / GQA control
// stream on Metal between the already-native Q4 matvecs. Weights that are
// small vectors are uploaded once as f32 (the catalog already decodes them
// that way). Math mirrors the host oracles in qwen80_complete_runtime.rs:
//   residual RMSNorm: x * rsqrt(mean(x^2)+eps) * (1+w)
//   SwiGLU: silu(gate) * up
//   DeltaNet conv + L2 + BA + gated RMSNorm
//   GQA per-head (1+w) RMSNorm + rotate_half RoPE on the first 64 dims
//
// Serial reductions follow the host left-to-right f32 sum so greedy tokens
// stay on the measured "Hi" sequence.

#include <metal_stdlib>
using namespace metal;

kernel void qwen80_residual_rmsnorm_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& hidden      [[buffer(3)]],
    constant float& eps        [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint tid                    [[thread_index_in_threadgroup]])
{
    float sum = 0.0f;
    for (uint index = tid; index < hidden; index += 256u) {
        const float value = input[index];
        sum += value * value;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = 1.0f / sqrt(scratch[0] / float(hidden) + eps);
    for (uint index = tid; index < hidden; index += 256u) {
        output[index] = input[index] * inverse_rms * (1.0f + weight[index]);
    }
}

kernel void qwen80_silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* output     [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    if (id >= n) return;
    const float g = gate[id];
    output[id] = (g / (1.0f + exp(-g))) * up[id];
}

inline float qwen80_causal_conv_update_f32(
    device float* conv_state,
    device const float* conv_weight,
    uint channel,
    float current,
    uint conv_kernel)
{
    const uint state_len = conv_kernel - 1u;
    const uint state_base = channel * state_len;
    const uint weight_base = channel * conv_kernel;
    float sum = 0.0f;
    for (uint tap = 0u; tap < state_len; ++tap) {
        sum += conv_state[state_base + tap] * conv_weight[weight_base + tap];
    }
    for (uint tap = 0u; tap + 1u < state_len; ++tap) {
        conv_state[state_base + tap] = conv_state[state_base + tap + 1u];
    }
    conv_state[state_base + state_len - 1u] = current;
    sum += current * conv_weight[weight_base + state_len];
    return sum / (1.0f + exp(-sum));
}

kernel void qwen80_qkvz_rearrange_conv_l2_f32(
    device const float* projected_qkvz [[buffer(0)]],
    device const float* conv_weight    [[buffer(1)]],
    device float* conv_state           [[buffer(2)]],
    device float* repeated_query       [[buffer(3)]],
    device float* repeated_key         [[buffer(4)]],
    device float* convolved_value      [[buffer(5)]],
    device float* z                    [[buffer(6)]],
    constant uint& key_heads           [[buffer(7)]],
    constant uint& values_per_key_head [[buffer(8)]],
    constant uint& key_head_dim        [[buffer(9)]],
    constant uint& value_head_dim      [[buffer(10)]],
    constant uint& conv_kernel         [[buffer(11)]],
    constant float& eps                [[buffer(12)]],
    threadgroup float* scratch         [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint key_head = group.y;
    if (key_head >= key_heads) return;
    const uint value_rows_per_key_head = values_per_key_head * value_head_dim;
    const uint qkvz_rows_per_key_head = key_head_dim * 2u + value_rows_per_key_head * 2u;
    const uint qkvz_base = key_head * qkvz_rows_per_key_head;
    const uint key_elements = key_heads * key_head_dim;
    const uint value_base = key_head * value_rows_per_key_head;

    threadgroup float* query_local = scratch;
    threadgroup float* key_local = scratch + 128u;
    threadgroup float* query_sums = scratch + 256u;
    threadgroup float* key_sums = scratch + 512u;

    if (tid < key_head_dim) {
        const uint query_channel = key_head * key_head_dim + tid;
        const uint key_channel = key_elements + query_channel;
        query_local[tid] = qwen80_causal_conv_update_f32(
            conv_state, conv_weight, query_channel,
            projected_qkvz[qkvz_base + tid], conv_kernel);
        key_local[tid] = qwen80_causal_conv_update_f32(
            conv_state, conv_weight, key_channel,
            projected_qkvz[qkvz_base + key_head_dim + tid], conv_kernel);
    }
    if (tid < value_rows_per_key_head) {
        const uint value_channel = key_elements * 2u + value_base + tid;
        convolved_value[value_base + tid] = qwen80_causal_conv_update_f32(
            conv_state, conv_weight, value_channel,
            projected_qkvz[qkvz_base + key_head_dim * 2u + tid], conv_kernel);
        z[value_base + tid] = projected_qkvz[
            qkvz_base + key_head_dim * 2u + value_rows_per_key_head + tid];
    }
    query_sums[tid] = tid < key_head_dim ? query_local[tid] * query_local[tid] : 0.0f;
    key_sums[tid] = tid < key_head_dim ? key_local[tid] * key_local[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            query_sums[tid] += query_sums[tid + stride];
            key_sums[tid] += key_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < key_head_dim) {
        const float query_scale = rsqrt(query_sums[0] + eps) * rsqrt(float(key_head_dim));
        const float key_scale = rsqrt(key_sums[0] + eps);
        const uint value_head_base = key_head * values_per_key_head;
        for (uint repeat = 0u; repeat < values_per_key_head; ++repeat) {
            const uint destination = (value_head_base + repeat) * key_head_dim + tid;
            repeated_query[destination] = query_local[tid] * query_scale;
            repeated_key[destination] = key_local[tid] * key_scale;
        }
    }
}

kernel void qwen80_ba_to_decay_beta_f32(
    device const float* projected_ba [[buffer(0)]],
    device const float* a_log        [[buffer(1)]],
    device const float* dt_bias      [[buffer(2)]],
    device float* decay              [[buffer(3)]],
    device float* beta               [[buffer(4)]],
    constant uint& key_heads         [[buffer(5)]],
    constant uint& values_per_key_head [[buffer(6)]],
    uint value_head                   [[thread_position_in_grid]])
{
    const uint value_heads = key_heads * values_per_key_head;
    if (value_head >= value_heads) return;
    const uint key_head = value_head / values_per_key_head;
    const uint within = value_head % values_per_key_head;
    const uint ba_base = key_head * (2u * values_per_key_head);
    const float b = projected_ba[ba_base + within];
    const float a = projected_ba[ba_base + values_per_key_head + within];
    const float x = a + dt_bias[value_head];
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    const float g = -exp(a_log[value_head]) * softplus;
    decay[value_head] = exp(g);
    beta[value_head] = 1.0f / (1.0f + exp(-b));
}

kernel void qwen80_deltanet_gated_rmsnorm_f32(
    device const float* input     [[buffer(0)]],
    device const float* gate      [[buffer(1)]],
    device const float* weight    [[buffer(2)]],
    device float* output          [[buffer(3)]],
    constant uint& heads          [[buffer(4)]],
    constant uint& value_head_dim [[buffer(5)]],
    constant float& eps           [[buffer(6)]],
    uint head                      [[thread_position_in_grid]])
{
    if (head >= heads) return;
    const uint base = head * value_head_dim;
    float sum = 0.0f;
    for (uint index = 0u; index < value_head_dim; ++index) {
        const float value = input[base + index];
        sum += value * value;
    }
    const float inverse_rms = 1.0f / sqrt(sum / float(value_head_dim) + eps);
    for (uint index = 0u; index < value_head_dim; ++index) {
        const float z = gate[base + index];
        const float silu = z / (1.0f + exp(-z));
        output[base + index] = input[base + index] * inverse_rms * weight[index] * silu;
    }
}

// One threadgroup per value head. 128 threads walk the key axis so the
// 128x128 recurrent update is not serialized on a single GPU thread.
// Reductions sum key index 0..127 in order to stay close to the host oracle.
kernel void qwen80_gated_delta_decode_tg(
    device float* state            [[buffer(0)]],
    device const float* query      [[buffer(1)]],
    device const float* key        [[buffer(2)]],
    device const float* value      [[buffer(3)]],
    device const float* decay      [[buffer(4)]],
    device const float* beta       [[buffer(5)]],
    device float* output           [[buffer(6)]],
    constant uint& heads           [[buffer(7)]],
    constant uint& key_dim         [[buffer(8)]],
    constant uint& value_dim       [[buffer(9)]],
    threadgroup float* scratch     [[threadgroup(0)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint3 group                     [[threadgroup_position_in_grid]])
{
    const uint head = group.y;
    if (head >= heads || key_dim != 128u || value_dim != 128u) return;
    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const float d = decay[head];
    const float b = beta[head];
    const uint ki = tid;

    for (uint vi = 0u; vi < value_dim; ++vi) {
        const uint index = state_base + ki * value_dim + vi;
        const float decayed = state[index] * d;
        state[index] = decayed;
        scratch[tid] = ki < key_dim ? decayed * key[key_base + ki] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float kv_mem = 0.0f;
            for (uint i = 0u; i < key_dim; ++i) kv_mem += scratch[i];
            scratch[0] = kv_mem;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float delta = (value[value_base + vi] - scratch[0]) * b;
        state[index] += key[key_base + ki] * delta;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint vi = 0u; vi < value_dim; ++vi) {
        const uint index = state_base + ki * value_dim + vi;
        scratch[tid] = ki < key_dim ? state[index] * query[key_base + ki] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float sum = 0.0f;
            for (uint i = 0u; i < key_dim; ++i) sum += scratch[i];
            output[value_base + vi] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qwen80_gqa_qk_norm_rope_cache_f32(
    device const float* q_proj     [[buffer(0)]],
    device const float* k_proj     [[buffer(1)]],
    device const float* v_proj     [[buffer(2)]],
    device const float* q_norm     [[buffer(3)]],
    device const float* k_norm     [[buffer(4)]],
    device float* query            [[buffer(5)]],
    device float* key_cache        [[buffer(6)]],
    device float* value_cache      [[buffer(7)]],
    constant uint& sequence_slot   [[buffer(8)]],
    constant uint& n_heads         [[buffer(9)]],
    constant uint& n_kv_heads      [[buffer(10)]],
    constant uint& head_dim        [[buffer(11)]],
    constant uint& rotary_dim      [[buffer(12)]],
    constant float& rope_theta     [[buffer(13)]],
    constant float& rms_epsilon    [[buffer(14)]],
    uint head                       [[thread_position_in_grid]])
{
    if (head >= n_heads ||
        !gk_gqa_geometry_ok(n_heads, n_kv_heads, head_dim, rotary_dim, rope_theta, rms_epsilon)) {
        return;
    }

    const uint q_base = head * head_dim;
    const uint q_projection_base = head * (2u * head_dim);
    float q_sum = 0.0f;
    for (uint dim = 0u; dim < head_dim; ++dim) {
        const float value = q_proj[q_projection_base + dim];
        q_sum += value * value;
    }
    const float q_inverse_rms = 1.0f / sqrt(q_sum / float(head_dim) + rms_epsilon);
    const uint half_dim = rotary_dim / 2u;
    for (uint dim = 0u; dim < head_dim; ++dim) {
        const float raw = q_proj[q_projection_base + dim];
        const float normed = raw * q_inverse_rms * (1.0f + q_norm[dim]);
        if (dim < rotary_dim) {
            const uint frequency_index = dim < half_dim ? dim : dim - half_dim;
            const float inv_frequency =
                pow(rope_theta, -2.0f * float(frequency_index) / float(rotary_dim));
            const float angle = float(sequence_slot) * inv_frequency;
            const float cosine = cos(angle);
            const float sine = sin(angle);
            const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
            const float peer_raw = q_proj[q_projection_base + peer] * q_inverse_rms
                * (1.0f + q_norm[peer]);
            query[q_base + dim] = dim < half_dim
                ? normed * cosine - peer_raw * sine
                : normed * cosine + peer_raw * sine;
        } else {
            query[q_base + dim] = normed;
        }
    }

    if (head < n_kv_heads) {
        const uint kv_base = head * head_dim;
        float k_sum = 0.0f;
        for (uint dim = 0u; dim < head_dim; ++dim) {
            const float value = k_proj[kv_base + dim];
            k_sum += value * value;
        }
        const float k_inverse_rms = 1.0f / sqrt(k_sum / float(head_dim) + rms_epsilon);
        const uint cache_base = (sequence_slot * n_kv_heads + head) * head_dim;
        for (uint dim = 0u; dim < head_dim; ++dim) {
            const float raw = k_proj[kv_base + dim];
            const float normed = raw * k_inverse_rms * (1.0f + k_norm[dim]);
            if (dim < rotary_dim) {
                const uint frequency_index = dim < half_dim ? dim : dim - half_dim;
                const float inv_frequency =
                    pow(rope_theta, -2.0f * float(frequency_index) / float(rotary_dim));
                const float angle = float(sequence_slot) * inv_frequency;
                const float cosine = cos(angle);
                const float sine = sin(angle);
                const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
                const float peer_raw = k_proj[kv_base + peer] * k_inverse_rms
                    * (1.0f + k_norm[peer]);
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
