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

// One threadgroup per head, threads split head_dim. Same convention the DeltaNet decode
// kernel uses, and the same accident being repaired: the scalar form above is indexed
// `uint head [[thread_position_in_grid]]`, so the whole dispatch is 24 THREADS, each walking
// head_dim 256 serially -- and computing pow/cos/sin for every (head, dim) pair even though
// the angle depends only on dim and sequence_slot. Twenty-four heads recompute the same 64
// angles. This form gives each head a threadgroup and one dim per thread, so each transcendental
// is evaluated once per thread instead of 256 times per thread.
//
// The RMS sum is reduced as a tree, so this is NOT bit-identical to the scalar kernel and is
// kept behind a flag rather than replacing it.
kernel void qwen38_gqa_qk_norm_rope_cache_tg(
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
    threadgroup float* red         [[threadgroup(0)]],
    uint head                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]])
{
    if (head >= n_heads || n_heads != 24u || n_kv_heads != 4u ||
        head_dim != 256u || rotary_dim != 64u ||
        rope_theta != 10000000.0f || rms_epsilon != 1.0e-6f) {
        return;
    }
    const uint half_dim = rotary_dim / 2u;
    const uint q_base = head * head_dim;
    const uint q_projection_base = head * (2u * head_dim);

    float local = 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        const float v = q_proj[q_projection_base + d];
        local += v * v;
    }
    red[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) { red[tid] += red[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float q_inverse_rms = 1.0f / sqrt(red[0] / float(head_dim) + rms_epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < head_dim; dim += tg_size) {
        const float raw = q_proj[q_projection_base + dim];
        const float normed = raw * q_inverse_rms * (1.0f + q_norm[dim]);
        if (dim < rotary_dim) {
            const uint fi = dim < half_dim ? dim : dim - half_dim;
            const float inv_f = pow(rope_theta, -2.0f * float(fi) / float(rotary_dim));
            const float angle = float(sequence_slot) * inv_f;
            const float c = cos(angle);
            const float sn = sin(angle);
            const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
            const float peer_raw = q_proj[q_projection_base + peer] * q_inverse_rms
                * (1.0f + q_norm[peer]);
            query[q_base + dim] = dim < half_dim
                ? normed * c - peer_raw * sn
                : normed * c + peer_raw * sn;
        } else {
            query[q_base + dim] = normed;
        }
    }

    if (head < n_kv_heads) {
        const uint kv_base = head * head_dim;
        float klocal = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            const float v = k_proj[kv_base + d];
            klocal += v * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[tid] = klocal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) { red[tid] += red[tid + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float k_inverse_rms = 1.0f / sqrt(red[0] / float(head_dim) + rms_epsilon);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint cache_base = (sequence_slot * n_kv_heads + head) * head_dim;
        for (uint dim = tid; dim < head_dim; dim += tg_size) {
            const float raw = k_proj[kv_base + dim];
            const float normed = raw * k_inverse_rms * (1.0f + k_norm[dim]);
            if (dim < rotary_dim) {
                const uint fi = dim < half_dim ? dim : dim - half_dim;
                const float inv_f = pow(rope_theta, -2.0f * float(fi) / float(rotary_dim));
                const float angle = float(sequence_slot) * inv_f;
                const float c = cos(angle);
                const float sn = sin(angle);
                const uint peer = dim < half_dim ? dim + half_dim : dim - half_dim;
                const float peer_raw = k_proj[kv_base + peer] * k_inverse_rms
                    * (1.0f + k_norm[peer]);
                key_cache[cache_base + dim] = dim < half_dim
                    ? normed * c - peer_raw * sn
                    : normed * c + peer_raw * sn;
            } else {
                key_cache[cache_base + dim] = normed;
            }
            value_cache[cache_base + dim] = v_proj[kv_base + dim];
        }
    }
}

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

// simd-reduced sibling of `qwen38_gated_delta_decode_vi`. Same launch
// geometry, same state arithmetic. The recurrent element is loaded once
// into a register, reused for decay and both reductions, and stored once.
// The vi kernel spends both of its 128-element reductions on thread 0
// alone while the other 127 lanes wait at a barrier; this replaces each
// with a simdgroup tree plus a 4-partial combine.
//
// NOT bit-identical: a tree reduction does not associate the same way a
// left-to-right serial loop does. Greedy token identity against the oracle is
// the gate this has to clear, not intermediate bitwise equality.
kernel void qwen38_gated_delta_decode_vi_simd(
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
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
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

    float s = state[index];
    const float decayed = s * d;

    float part = simd_sum(decayed * key[key_base + ki]);
    if (simd_lane == 0u) {
        scratch[simd_id] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float kv_mem = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float delta = (value[value_base + vi] - kv_mem) * b;
    s = decayed + key[key_base + ki] * delta;

    float out = simd_sum(s * query[key_base + ki]);
    if (simd_lane == 0u) {
        scratch[simd_id] = out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        output[value_base + vi] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    }
    state[index] = s;
}

// ba_to_decay_beta in-register. values_per_key=3 and key_heads=16 are
// compile-time so `head / 3u` is not the bind-time integer divide that
// measured 1.37x on affine2. Writes the same rec_out as the two-dispatch
// (ba_to_decay + vi_simd) pair; decay/beta buffers are not stored.
kernel void qwen38_gated_delta_decode_vi_simd_ba(
    device float* state                 [[buffer(0)]],
    device const float* query           [[buffer(1)]],
    device const float* key             [[buffer(2)]],
    device const float* value           [[buffer(3)]],
    device const float* projected_ba    [[buffer(4)]],
    device const float* a_log           [[buffer(5)]],
    device const float* dt_bias         [[buffer(6)]],
    device float* output                [[buffer(7)]],
    constant uint& heads                [[buffer(8)]],
    constant uint& key_dim              [[buffer(9)]],
    constant uint& value_dim            [[buffer(10)]],
    threadgroup float* scratch          [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint head = group.y;
    const uint vi = group.z;
    if (head >= heads || vi >= value_dim || key_dim != 128u || value_dim != 128u
        || heads != 48u) {
        return;
    }
    const uint key_head = head / 3u;
    const uint within = head - key_head * 3u;
    const uint ba_base = key_head * 6u;
    const float bb = projected_ba[ba_base + within];
    const float a = projected_ba[ba_base + 3u + within];
    const float x = a + dt_bias[head];
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    const float g = -exp(a_log[head]) * softplus;
    const float d = exp(g);
    const float b = 1.0f / (1.0f + exp(-bb));

    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const uint ki = tid;
    const uint index = state_base + ki * value_dim + vi;

    float s = state[index];
    const float decayed = s * d;

    float part = simd_sum(decayed * key[key_base + ki]);
    if (simd_lane == 0u) {
        scratch[simd_id] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float kv_mem = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float delta = (value[value_base + vi] - kv_mem) * b;
    s = decayed + key[key_base + ki] * delta;

    float out = simd_sum(s * query[key_base + ki]);
    if (simd_lane == 0u) {
        scratch[simd_id] = out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        output[value_base + vi] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    }
    state[index] = s;
}

// BAD control: identity decay/beta. Same signature as the honest fused
// kernel so a no-op (honest vi_simd still consuming precomputed decay)
// cannot score. Greedy tokens must diverge.
kernel void qwen38_gated_delta_decode_vi_simd_ba_plain(
    device float* state                 [[buffer(0)]],
    device const float* query           [[buffer(1)]],
    device const float* key             [[buffer(2)]],
    device const float* value           [[buffer(3)]],
    device const float* projected_ba    [[buffer(4)]],
    device const float* a_log           [[buffer(5)]],
    device const float* dt_bias         [[buffer(6)]],
    device float* output                [[buffer(7)]],
    constant uint& heads                [[buffer(8)]],
    constant uint& key_dim              [[buffer(9)]],
    constant uint& value_dim            [[buffer(10)]],
    threadgroup float* scratch          [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint head = group.y;
    const uint vi = group.z;
    if (head >= heads || vi >= value_dim || key_dim != 128u || value_dim != 128u
        || heads != 48u) {
        return;
    }
    (void)projected_ba;
    (void)a_log;
    (void)dt_bias;
    const float d = 1.0f;
    const float b = 1.0f;
    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const uint ki = tid;
    const uint index = state_base + ki * value_dim + vi;

    float s = state[index];
    const float decayed = s * d;

    float part = simd_sum(decayed * key[key_base + ki]);
    if (simd_lane == 0u) {
        scratch[simd_id] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float kv_mem = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float delta = (value[value_base + vi] - kv_mem) * b;
    s = decayed + key[key_base + ki] * delta;

    float out = simd_sum(s * query[key_base + ki]);
    if (simd_lane == 0u) {
        scratch[simd_id] = out;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        output[value_base + vi] = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    }
    state[index] = s;
}

// ba → decay/beta. values_per_key=3 is a literal so `head / 3u` is a
// multiply-high, not the bind-time integer divide that measured 1.37x.
inline void qwen38_ba_decay_beta_f32(
    device const float* projected_ba,
    device const float* a_log,
    device const float* dt_bias,
    uint head,
    thread float& d,
    thread float& b)
{
    const uint key_head = head / 3u;
    const uint within = head - key_head * 3u;
    const uint ba_base = key_head * 6u;
    const float bb = projected_ba[ba_base + within];
    const float a = projected_ba[ba_base + 3u + within];
    const float x = a + dt_bias[head];
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    const float g = -exp(a_log[head]) * softplus;
    d = exp(g);
    b = 1.0f / (1.0f + exp(-bb));
}

// CHANGE 1 — widen loads. Same ki-thread / vi-TG geometry as
// `qwen38_gated_delta_decode_vi_simd_ba`, but each TG owns four consecutive
// vi columns as float4. State is still [head][ki][vi]; the 16-byte load is
// contiguous in vi, 4× fewer TGs, ba math paid once. Stride across ki is
// unchanged — that is change 2.
kernel void qwen38_gated_delta_decode_vi_simd_ba_f4(
    device float* state                 [[buffer(0)]],
    device const float* query           [[buffer(1)]],
    device const float* key             [[buffer(2)]],
    device const float* value           [[buffer(3)]],
    device const float* projected_ba    [[buffer(4)]],
    device const float* a_log           [[buffer(5)]],
    device const float* dt_bias         [[buffer(6)]],
    device float* output                [[buffer(7)]],
    constant uint& heads                [[buffer(8)]],
    constant uint& key_dim              [[buffer(9)]],
    constant uint& value_dim            [[buffer(10)]],
    threadgroup float* scratch          [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint head = group.y;
    const uint vi_base = group.z * 4u;
    if (head >= heads || vi_base >= value_dim || key_dim != 128u
        || value_dim != 128u || heads != 48u) {
        return;
    }
    float d;
    float b;
    qwen38_ba_decay_beta_f32(projected_ba, a_log, dt_bias, head, d, b);

    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const uint ki = tid;
    const uint index = state_base + ki * value_dim + vi_base;
    const float kk = key[key_base + ki];
    const float qq = query[key_base + ki];

    float4 s = *((device const float4*)(state + index));
    float4 decayed = s * d;
    float4 part = float4(
        simd_sum(decayed.x * kk),
        simd_sum(decayed.y * kk),
        simd_sum(decayed.z * kk),
        simd_sum(decayed.w * kk));
    if (simd_lane == 0u) {
        scratch[simd_id * 4u + 0u] = part.x;
        scratch[simd_id * 4u + 1u] = part.y;
        scratch[simd_id * 4u + 2u] = part.z;
        scratch[simd_id * 4u + 3u] = part.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float4 kv = float4(
        scratch[0] + scratch[4] + scratch[8] + scratch[12],
        scratch[1] + scratch[5] + scratch[9] + scratch[13],
        scratch[2] + scratch[6] + scratch[10] + scratch[14],
        scratch[3] + scratch[7] + scratch[11] + scratch[15]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 vv = *((device const float4*)(value + value_base + vi_base));
    const float4 delta = (vv - kv) * b;
    s = decayed + kk * delta;

    float4 outp = float4(
        simd_sum(s.x * qq),
        simd_sum(s.y * qq),
        simd_sum(s.z * qq),
        simd_sum(s.w * qq));
    if (simd_lane == 0u) {
        scratch[simd_id * 4u + 0u] = outp.x;
        scratch[simd_id * 4u + 1u] = outp.y;
        scratch[simd_id * 4u + 2u] = outp.z;
        scratch[simd_id * 4u + 3u] = outp.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        *((device float4*)(output + value_base + vi_base)) = float4(
            scratch[0] + scratch[4] + scratch[8] + scratch[12],
            scratch[1] + scratch[5] + scratch[9] + scratch[13],
            scratch[2] + scratch[6] + scratch[10] + scratch[14],
            scratch[3] + scratch[7] + scratch[11] + scratch[15]);
    }
    *((device float4*)(state + index)) = s;
}

// CHANGE 2 — stage a 128×32 state tile in threadgroup memory and load it
// coalesced. Layout is still [head][ki][vi]; a single vi-column is a
// 512-byte stride, so 32 consecutive vi of each ki-row are gathered by
// consecutive threads (linear = ki*32+vi_local). Compute then walks ki
// serially on 32 vi-threads. 1R+1W of the tile, no extra state pass.
kernel void qwen38_gated_delta_decode_vi_simd_ba_tg32(
    device float* state                 [[buffer(0)]],
    device const float* query           [[buffer(1)]],
    device const float* key             [[buffer(2)]],
    device const float* value           [[buffer(3)]],
    device const float* projected_ba    [[buffer(4)]],
    device const float* a_log           [[buffer(5)]],
    device const float* dt_bias         [[buffer(6)]],
    device float* output                [[buffer(7)]],
    constant uint& heads                [[buffer(8)]],
    constant uint& key_dim              [[buffer(9)]],
    constant uint& value_dim            [[buffer(10)]],
    threadgroup float* scratch          [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint head = group.y;
    const uint tile = group.z;
    if (head >= heads || tile >= 4u || key_dim != 128u || value_dim != 128u
        || heads != 48u) {
        return;
    }
    float d;
    float b;
    qwen38_ba_decay_beta_f32(projected_ba, a_log, dt_bias, head, d, b);

    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const uint vi_base = tile * 32u;
    threadgroup float* tile_s = scratch;
    threadgroup float* red = scratch + 4096u;

    for (uint iter = 0u; iter < 32u; ++iter) {
        const uint linear = iter * 128u + tid;
        const uint ki_l = linear >> 5u;
        const uint vi_l = linear & 31u;
        tile_s[linear] = state[state_base + ki_l * 128u + vi_base + vi_l];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Same ki-thread simd-sum association as vi_simd_ba / f4. The tile
    // only changed how the 128×32 slab moved; greedy identity is the gate.
    const uint ki = tid;
    const float kk = key[key_base + ki];
    const float qq = query[key_base + ki];
    for (uint v0 = 0u; v0 < 32u; v0 += 4u) {
        float4 s = float4(
            tile_s[ki * 32u + v0 + 0u],
            tile_s[ki * 32u + v0 + 1u],
            tile_s[ki * 32u + v0 + 2u],
            tile_s[ki * 32u + v0 + 3u]);
        float4 decayed = s * d;
        float4 part = float4(
            simd_sum(decayed.x * kk),
            simd_sum(decayed.y * kk),
            simd_sum(decayed.z * kk),
            simd_sum(decayed.w * kk));
        if (simd_lane == 0u) {
            red[simd_id * 4u + 0u] = part.x;
            red[simd_id * 4u + 1u] = part.y;
            red[simd_id * 4u + 2u] = part.z;
            red[simd_id * 4u + 3u] = part.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float4 kv = float4(
            red[0] + red[4] + red[8] + red[12],
            red[1] + red[5] + red[9] + red[13],
            red[2] + red[6] + red[10] + red[14],
            red[3] + red[7] + red[11] + red[15]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float4 vv = *((device const float4*)(value + value_base + vi_base + v0));
        const float4 delta = (vv - kv) * b;
        s = decayed + kk * delta;
        float4 outp = float4(
            simd_sum(s.x * qq),
            simd_sum(s.y * qq),
            simd_sum(s.z * qq),
            simd_sum(s.w * qq));
        if (simd_lane == 0u) {
            red[simd_id * 4u + 0u] = outp.x;
            red[simd_id * 4u + 1u] = outp.y;
            red[simd_id * 4u + 2u] = outp.z;
            red[simd_id * 4u + 3u] = outp.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            *((device float4*)(output + value_base + vi_base + v0)) = float4(
                red[0] + red[4] + red[8] + red[12],
                red[1] + red[5] + red[9] + red[13],
                red[2] + red[6] + red[10] + red[14],
                red[3] + red[7] + red[11] + red[15]);
        }
        tile_s[ki * 32u + v0 + 0u] = s.x;
        tile_s[ki * 32u + v0 + 1u] = s.y;
        tile_s[ki * 32u + v0 + 2u] = s.z;
        tile_s[ki * 32u + v0 + 3u] = s.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint iter = 0u; iter < 32u; ++iter) {
        const uint linear = iter * 128u + tid;
        const uint ki_l = linear >> 5u;
        const uint vi_l2 = linear & 31u;
        state[state_base + ki_l * 128u + vi_base + vi_l2] = tile_s[linear];
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
