// Qwen3.8 forks of Q80 device-activation kernels. Q80 shaders stay locked
// to 16/2/θ=5e6 and values_per_key=2. These entry points admit the Q38
// geometry (24/4/θ=1e7, values_per_key=3) without rewriting the math.

#include <metal_stdlib>
using namespace metal;

inline float qwen38_causal_conv_update_f32(
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

// Same as qwen80_qkvz_rearrange_conv_l2_f32 but loops value/Z rows so
// values_per_key_head=3 (384 rows) is not silently truncated at TG=256.
kernel void qwen38_qkvz_rearrange_conv_l2_f32(
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
    if (key_heads != 16u || values_per_key_head != 3u ||
        key_head_dim != 128u || value_head_dim != 128u || conv_kernel != 4u) {
        return;
    }
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
        query_local[tid] = qwen38_causal_conv_update_f32(
            conv_state, conv_weight, query_channel,
            projected_qkvz[qkvz_base + tid], conv_kernel);
        key_local[tid] = qwen38_causal_conv_update_f32(
            conv_state, conv_weight, key_channel,
            projected_qkvz[qkvz_base + key_head_dim + tid], conv_kernel);
    }
    for (uint row = tid; row < value_rows_per_key_head; row += 256u) {
        const uint value_channel = key_elements * 2u + value_base + row;
        convolved_value[value_base + row] = qwen38_causal_conv_update_f32(
            conv_state, conv_weight, value_channel,
            projected_qkvz[qkvz_base + key_head_dim * 2u + row], conv_kernel);
        z[value_base + row] = projected_qkvz[
            qkvz_base + key_head_dim * 2u + value_rows_per_key_head + row];
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

// rotate_half partial RoPE, first 64 of 256, θ=1e7, GQA 24:4.
kernel void qwen38_gqa_qk_norm_rope_cache_f32(
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
    if (head >= n_heads || n_heads != 24u || n_kv_heads != 4u ||
        head_dim != 256u || rotary_dim != 64u ||
        rope_theta != 10000000.0f || rms_epsilon != 1.0e-6f) {
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

// Same per-value-dim association as `qwen80_gated_delta_decode_tg`, but one
// threadgroup per (head, value_dim) instead of looping 128 value columns
// inside 48 heads. The vi columns do not share state, so this is the same
// serial-reduction arithmetic launched with 128× occupancy.
kernel void qwen38_gated_delta_decode_vi(
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
    const uint vi = group.z;
    if (head >= heads || vi >= value_dim || key_dim != 128u || value_dim != 128u) {
        return;
    }
    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const float d = decay[head];
    const float b = beta[head];
    const uint ki = tid;
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

    scratch[tid] = ki < key_dim ? state[index] * query[key_base + ki] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0u; i < key_dim; ++i) sum += scratch[i];
        output[value_base + vi] = sum;
    }
}

kernel void qwen38_attention_apply_sigmoid_gate(
    device const float* attention_output    [[buffer(0)]],
    device const float* q_proj              [[buffer(1)]],
    device float* gated_output              [[buffer(2)]],
    constant uint& elements                  [[buffer(3)]],
    constant uint& head_dim                  [[buffer(4)]],
    uint index                               [[thread_position_in_grid]])
{
    if (index >= elements) return;
    if (head_dim != 256u || elements != 24u * head_dim) return;
    const uint head = index / head_dim;
    const uint dimension = index - head * head_dim;
    const uint gate_offset = head * (2u * head_dim) + head_dim + dimension;
    const float gate = q_proj[gate_offset];
    const float sigmoid = 1.0f / (1.0f + exp(-gate));
    gated_output[index] = attention_output[index] * sigmoid;
}

// Interleave split in_proj_qkv + in_proj_z activations into the fused
// per-key-head QKVZ layout the rearrange kernel already consumes.
// Activation-only. Does not touch packed weights.
kernel void qwen38_fuse_split_qkvz_f32(
    device const float* qkv            [[buffer(0)]],
    device const float* z              [[buffer(1)]],
    device float* fused                [[buffer(2)]],
    constant uint& key_heads           [[buffer(3)]],
    constant uint& values_per_key_head [[buffer(4)]],
    constant uint& key_head_dim        [[buffer(5)]],
    constant uint& value_head_dim      [[buffer(6)]],
    uint idx                            [[thread_position_in_grid]])
{
    if (key_heads != 16u || values_per_key_head != 3u ||
        key_head_dim != 128u || value_head_dim != 128u) {
        return;
    }
    const uint value_rows = values_per_key_head * value_head_dim;
    const uint qkvz_per_key = key_head_dim * 2u + value_rows * 2u;
    const uint fused_n = key_heads * qkvz_per_key;
    if (idx >= fused_n) return;
    const uint key_head = idx / qkvz_per_key;
    const uint local = idx - key_head * qkvz_per_key;
    const uint key_elements = key_heads * key_head_dim;
    if (local < key_head_dim) {
        fused[idx] = qkv[key_head * key_head_dim + local];
    } else if (local < key_head_dim * 2u) {
        fused[idx] = qkv[key_elements + key_head * key_head_dim + (local - key_head_dim)];
    } else if (local < key_head_dim * 2u + value_rows) {
        fused[idx] = qkv[key_elements * 2u + key_head * value_rows
            + (local - key_head_dim * 2u)];
    } else {
        fused[idx] = z[key_head * value_rows + (local - key_head_dim * 2u - value_rows)];
    }
}

// Pack split in_proj_b + in_proj_a activations into [key_head][b×3, a×3].
kernel void qwen38_fuse_split_ba_f32(
    device const float* b              [[buffer(0)]],
    device const float* a              [[buffer(1)]],
    device float* fused                [[buffer(2)]],
    constant uint& key_heads           [[buffer(3)]],
    constant uint& values_per_key_head [[buffer(4)]],
    uint idx                            [[thread_position_in_grid]])
{
    if (key_heads != 16u || values_per_key_head != 3u) return;
    const uint ba_per_key = values_per_key_head * 2u;
    const uint fused_n = key_heads * ba_per_key;
    if (idx >= fused_n) return;
    const uint key_head = idx / ba_per_key;
    const uint local = idx - key_head * ba_per_key;
    const uint src = key_head * values_per_key_head + (local % values_per_key_head);
    fused[idx] = local < values_per_key_head ? b[src] : a[src];
}

// One-row gather of an HGRAVU01 (unsigned LSB, group scale) embedding.
// Same extract as gk_uniform_value / Q80 uniform factor. Never a dense W.
kernel void qwen38_hgravu_embedding_lookup(
    device const uchar* codes     [[buffer(0)]],
    device const half* scales     [[buffer(1)]],
    device float* hidden          [[buffer(2)]],
    constant uint& token          [[buffer(3)]],
    constant uint& hidden_size    [[buffer(4)]],
    constant uint& vocab          [[buffer(5)]],
    constant uint& group_size     [[buffer(6)]],
    constant uint& bits           [[buffer(7)]],
    constant uint& bound          [[buffer(8)]],
    uint dim                       [[thread_position_in_grid]])
{
    if (dim >= hidden_size || token >= vocab || group_size == 0u || bits == 0u) {
        return;
    }
    const uint element = token * hidden_size + dim;
    hidden[dim] = gk_uniform_value(codes, scales, element, group_size, bits, bound);
}

// Diagnostic sequential f32 copy. Used to put a bandwidth floor under
// conv/recurrent/GQA state traffic without the fused activation ALU.
kernel void qwen38_f32_stream_probe(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint i                   [[thread_position_in_grid]])
{
    if (i < n) {
        dst[i] = src[i];
    }
}
