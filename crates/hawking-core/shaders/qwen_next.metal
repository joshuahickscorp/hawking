// Qwen3-Next single-token Gated DeltaNet recurrence.
//
// This deliberately small kernel is the exact recurrent-state substage used
// during cached one-token decode.  The caller supplies Q/K after L2
// normalisation (and the Q 1/sqrt(d) scale), plus exp(g) and sigmoid(b).  One
// GPU thread owns one value head and serialises its 128×128 state update; this
// is a correctness/parity starting point, not the final throughput kernel.
//
// State layout: [head][key_dim][value_dim], all f32.  Q/K/V layout is
// [head][dimension].  The implementation mirrors
// torch_recurrent_gated_delta_rule in Qwen3Next reference Transformers code:
//   S <- S * exp(g)
//   delta <- (v - S^T k) * beta
//   S <- S + k outer delta
//   o <- S^T q

#include <metal_stdlib>
using namespace metal;

kernel void qwen_next_gated_delta_decode_single(
    device       float* state       [[buffer(0)]],
    device const float* query       [[buffer(1)]],
    device const float* key         [[buffer(2)]],
    device const float* value       [[buffer(3)]],
    device const float* decay       [[buffer(4)]],
    device const float* beta        [[buffer(5)]],
    device       float* output      [[buffer(6)]],
    constant uint& heads             [[buffer(7)]],
    constant uint& key_dim           [[buffer(8)]],
    constant uint& value_dim         [[buffer(9)]],
    uint head                         [[thread_position_in_grid]])
{
    if (head >= heads) return;
    const uint state_base = head * key_dim * value_dim;
    const uint key_base = head * key_dim;
    const uint value_base = head * value_dim;
    const float d = decay[head];
    const float b = beta[head];

    // Update every output/value channel independently. This avoids a
    // cross-thread reduction in the exact baseline and keeps the state fully
    // device resident. A tiled SIMD implementation may replace it only after
    // parity against this operator is retained.
    for (uint vi = 0; vi < value_dim; ++vi) {
        float kv_mem = 0.0f;
        for (uint ki = 0; ki < key_dim; ++ki) {
            const uint index = state_base + ki * value_dim + vi;
            const float decayed = state[index] * d;
            state[index] = decayed;
            kv_mem += decayed * key[key_base + ki];
        }
        const float delta = (value[value_base + vi] - kv_mem) * b;
        for (uint ki = 0; ki < key_dim; ++ki) {
            const uint index = state_base + ki * value_dim + vi;
            state[index] += key[key_base + ki] * delta;
        }
    }

    for (uint vi = 0; vi < value_dim; ++vi) {
        float sum = 0.0f;
        for (uint ki = 0; ki < key_dim; ++ki) {
            sum += state[state_base + ki * value_dim + vi] * query[key_base + ki];
        }
        output[value_base + vi] = sum;
    }
}

// Decode exactly one value from Ascension's admitted Qwen complete-binary
// sign/FP16-group-scale layout.  Small control vectors (A_log and dt_bias)
// retain the same fixed 128-bit tail group as matrices, so they must use this
// path too rather than crossing back to a host-decoded parameter vector.
inline float qwen_next_complete_binary_value(
    device const uchar* signs,
    device const ushort* scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uint local = element - group * group_size;
    const ushort scale_bits = scales[group];
    const float scale = (float)as_type<half>(scale_bits);
    const uint byte_index = group * (group_size / 8u) + local / 8u;
    const bool positive = ((uint)signs[byte_index] & (1u << (local & 7u))) != 0u;
    return positive ? scale : -scale;
}

// Qwen3-Next Gated DeltaNet control projection for one cached decode token.
//
// `in_proj_ba` emits [key_head][b(value-heads-per-key),
// a(value-heads-per-key)].  The reference implementation splits that exact
// layout, applies beta=sigmoid(b), then g=-exp(A_log)*softplus(a+dt_bias),
// and passes exp(g) to the recurrent operator.  This kernel keeps both the
// projected BA values and the admitted compact A_log/dt_bias vectors on
// Metal.  It intentionally does not claim the upstream QKVZ/convolution path
// or a complete DeltaNet layer.
kernel void qwen_next_ba_to_decay_beta(
    device const float* projected_ba  [[buffer(0)]],
    device const uchar* a_log_signs   [[buffer(1)]],
    device const ushort* a_log_scales [[buffer(2)]],
    device const uchar* dt_bias_signs [[buffer(3)]],
    device const ushort* dt_bias_scales [[buffer(4)]],
    device float* decay               [[buffer(5)]],
    device float* beta                [[buffer(6)]],
    constant uint& key_heads          [[buffer(7)]],
    constant uint& values_per_key_head [[buffer(8)]],
    constant uint& group_size         [[buffer(9)]],
    uint value_head                   [[thread_position_in_grid]])
{
    const uint value_heads = key_heads * values_per_key_head;
    if (value_head >= value_heads) return;

    const uint key_head = value_head / values_per_key_head;
    const uint value_within_key_head = value_head % values_per_key_head;
    const uint ba_base = key_head * (2u * values_per_key_head);
    const float b = projected_ba[ba_base + value_within_key_head];
    const float a = projected_ba[ba_base + values_per_key_head + value_within_key_head];
    const float a_log = qwen_next_complete_binary_value(
        a_log_signs, a_log_scales, value_head, group_size);
    const float dt_bias = qwen_next_complete_binary_value(
        dt_bias_signs, dt_bias_scales, value_head, group_size);

    // Stable softplus in the precise source order.  `g` is always non-positive
    // for finite source controls, hence the resulting recurrence decay is in
    // (0, 1].
    const float x = a + dt_bias;
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    const float g = -exp(a_log) * softplus;
    decay[value_head] = exp(g);
    beta[value_head] = 1.0f / (1.0f + exp(-b));
}

// Direct-packed source Qwen3-Next input RMSNorm. The source weight is stored
// as a delta from one, so this is `x * rsqrt(mean(x*x)+eps) * (1 + weight)`.
// It keeps the compact sign/FP16-scale norm vector on device and is intended
// only for the bounded first-DeltaNet-layer parity stage below.
kernel void qwen_next_direct_packed_input_rmsnorm(
    device const float* input        [[buffer(0)]],
    device const uchar* weight_signs [[buffer(1)]],
    device const ushort* weight_scales [[buffer(2)]],
    device float* output             [[buffer(3)]],
    constant uint& hidden            [[buffer(4)]],
    constant uint& group_size        [[buffer(5)]],
    constant float& eps              [[buffer(6)]],
    threadgroup float* scratch       [[threadgroup(0)]],
    uint tid                          [[thread_index_in_threadgroup]])
{
    float sum = 0.0f;
    for (uint index = tid; index < hidden; index += 256u) {
        const float value = input[index];
        sum = fma(value, value, sum);
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = rsqrt(scratch[0] / float(hidden) + eps);
    for (uint index = tid; index < hidden; index += 256u) {
        const float weight = qwen_next_complete_binary_value(
            weight_signs, weight_scales, index, group_size);
        output[index] = input[index] * inverse_rms * (1.0f + weight);
    }
}

inline float qwen_next_causal_conv_update(
    device float* conv_state,
    device const uchar* conv_signs,
    device const ushort* conv_scales,
    uint channel,
    float current,
    uint conv_kernel,
    uint group_size)
{
    const uint state_len = conv_kernel - 1u;
    const uint state_base = channel * state_len;
    const uint weight_base = channel * conv_kernel;
    float sum = 0.0f;
    for (uint tap = 0u; tap < state_len; ++tap) {
        const float weight = qwen_next_complete_binary_value(
            conv_signs, conv_scales, weight_base + tap, group_size);
        sum = fma(conv_state[state_base + tap], weight, sum);
    }
    for (uint tap = 0u; tap + 1u < state_len; ++tap) {
        conv_state[state_base + tap] = conv_state[state_base + tap + 1u];
    }
    conv_state[state_base + state_len - 1u] = current;
    const float newest_weight = qwen_next_complete_binary_value(
        conv_signs, conv_scales, weight_base + state_len, group_size);
    sum = fma(current, newest_weight, sum);
    return sum / (1.0f + exp(-sum));
}

// The source `fix_query_key_value_ordering` implementation is encoded here
// exactly: per key head `Q128,K128,V256,Z256`; only Q/K/V enter the causal
// depthwise SiLU convolution; Q and K are then repeated from 16 key heads to
// 32 value heads after independent L2 normalization. One threadgroup owns one
// key head so the exact Q/K reductions cannot silently cross head boundaries.
kernel void qwen_next_qkvz_rearrange_conv_l2(
    device const float* projected_qkvz [[buffer(0)]],
    device const uchar* conv_signs     [[buffer(1)]],
    device const ushort* conv_scales   [[buffer(2)]],
    device float* conv_state           [[buffer(3)]],
    device float* repeated_query       [[buffer(4)]],
    device float* repeated_key         [[buffer(5)]],
    device float* convolved_value      [[buffer(6)]],
    device float* z                    [[buffer(7)]],
    constant uint& key_heads           [[buffer(8)]],
    constant uint& values_per_key_head [[buffer(9)]],
    constant uint& key_head_dim        [[buffer(10)]],
    constant uint& value_head_dim      [[buffer(11)]],
    constant uint& conv_kernel         [[buffer(12)]],
    constant uint& group_size          [[buffer(13)]],
    constant float& eps                [[buffer(14)]],
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
        query_local[tid] = qwen_next_causal_conv_update(
            conv_state,
            conv_signs,
            conv_scales,
            query_channel,
            projected_qkvz[qkvz_base + tid],
            conv_kernel,
            group_size);
        key_local[tid] = qwen_next_causal_conv_update(
            conv_state,
            conv_signs,
            conv_scales,
            key_channel,
            projected_qkvz[qkvz_base + key_head_dim + tid],
            conv_kernel,
            group_size);
    }
    if (tid < value_rows_per_key_head) {
        const uint value_channel = key_elements * 2u + value_base + tid;
        convolved_value[value_base + tid] = qwen_next_causal_conv_update(
            conv_state,
            conv_signs,
            conv_scales,
            value_channel,
            projected_qkvz[qkvz_base + key_head_dim * 2u + tid],
            conv_kernel,
            group_size);
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

// Source `Qwen3NextRMSNormGated` for the DeltaNet output. The compact norm
// vector is shared across the 32 value heads and has no `+1` residual scale.
kernel void qwen_next_deltanet_gated_rmsnorm(
    device const float* input          [[buffer(0)]],
    device const float* z              [[buffer(1)]],
    device const uchar* weight_signs   [[buffer(2)]],
    device const ushort* weight_scales [[buffer(3)]],
    device float* output               [[buffer(4)]],
    constant uint& heads               [[buffer(5)]],
    constant uint& value_head_dim      [[buffer(6)]],
    constant uint& group_size          [[buffer(7)]],
    constant float& eps                [[buffer(8)]],
    threadgroup float* scratch         [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint head = group.y;
    if (head >= heads) return;
    const uint base = head * value_head_dim;
    float sum = 0.0f;
    for (uint index = tid; index < value_head_dim; index += 256u) {
        const float value = input[base + index];
        sum = fma(value, value, sum);
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = rsqrt(scratch[0] / float(value_head_dim) + eps);
    for (uint index = tid; index < value_head_dim; index += 256u) {
        const float gate = z[base + index];
        const float silu = gate / (1.0f + exp(-gate));
        const float weight = qwen_next_complete_binary_value(
            weight_signs, weight_scales, index, group_size);
        output[base + index] = input[base + index] * inverse_rms * weight * silu;
    }
}

// First source residual boundary after the DeltaNet output projection.
kernel void qwen_next_add_residual(
    device const float* input  [[buffer(0)]],
    device const float* mixer  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& elements    [[buffer(3)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= elements) return;
    output[id] = input[id] + mixer[id];
}

// Source `Qwen3NextSparseMoeBlock`: gate the shared expert's complete MLP
// output by sigmoid(shared_expert_gate(x)).  The scalar gate projection stays
// on device and is never promoted to a host-computed MoE control value.
kernel void qwen_next_shared_expert_sigmoid_gate(
    device const float* shared_output [[buffer(0)]],
    device const float* gate_logit    [[buffer(1)]],
    device float* gated_output        [[buffer(2)]],
    constant uint& elements           [[buffer(3)]],
    uint id                            [[thread_position_in_grid]])
{
    if (id >= elements) return;
    const float gate = 1.0f / (1.0f + exp(-gate_logit[0]));
    gated_output[id] = shared_output[id] * gate;
}
