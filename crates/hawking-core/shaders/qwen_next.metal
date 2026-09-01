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

static inline float qwen_next_source_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

#pragma clang fp contract(off)
// Source-order dot helper for device-resident BF16 rows. Vector loads reduce
// address-generation and transaction overhead while each product is still
// accumulated in the original left-to-right order. The tail keeps the helper
// valid for non-multiple-of-four diagnostic geometries.
static inline float qwen_next_source_bf16_dot_vec4(
    device const ushort* weights,
    device const float* values,
    uint count)
{
    float acc = 0.0f;
    uint column = 0u;
    for (; column + 4u <= count; column += 4u) {
        const ushort4 packed_w = *(device const ushort4*)(weights + column);
        const float4 packed_x = *(device const float4*)(values + column);
        acc = acc + qwen_next_source_bf16_value(packed_w.x) * packed_x.x;
        acc = acc + qwen_next_source_bf16_value(packed_w.y) * packed_x.y;
        acc = acc + qwen_next_source_bf16_value(packed_w.z) * packed_x.z;
        acc = acc + qwen_next_source_bf16_value(packed_w.w) * packed_x.w;
    }
    for (; column < count; ++column) {
        acc = acc + qwen_next_source_bf16_value(weights[column]) * values[column];
    }
    return acc;
}

// Same exact-order reduction for the fused HC/router kernel's staged
// threadgroup activation. Keeping this address-space-specific avoids a
// device-memory spill just to reuse the packed-load path.
static inline float qwen_next_source_bf16_dot_vec4_threadgroup(
    device const ushort* weights,
    threadgroup const float* values,
    uint count)
{
    float acc = 0.0f;
    uint column = 0u;
    for (; column + 4u <= count; column += 4u) {
        const ushort4 packed_w = *(device const ushort4*)(weights + column);
        const float4 packed_x = *(threadgroup const float4*)(values + column);
        acc = acc + qwen_next_source_bf16_value(packed_w.x) * packed_x.x;
        acc = acc + qwen_next_source_bf16_value(packed_w.y) * packed_x.y;
        acc = acc + qwen_next_source_bf16_value(packed_w.z) * packed_x.z;
        acc = acc + qwen_next_source_bf16_value(packed_w.w) * packed_x.w;
    }
    for (; column < count; ++column) {
        acc = acc + qwen_next_source_bf16_value(weights[column]) * values[column];
    }
    return acc;
}

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

// Qwen3.8-Flash-Next stores the linear-attention projections as three source
// tensors (`in_proj_qkv` and `in_proj_z`) rather than the fused per-key-head
// QKVZ view used by the older Qwen3-Next path above.  Preserve the source
// global Q/K/V channel order while performing the causal convolution and
// Q/K normalization entirely on device.
inline float qwen_next_causal_conv_update_source_bf16(
    device float* conv_state,
    device const ushort* conv_weights,
    uint channel,
    float current,
    uint conv_kernel)
{
    const uint state_len = conv_kernel - 1u;
    const uint state_base = channel * state_len;
    const uint weight_base = channel * conv_kernel;
    float sum = 0.0f;
    for (uint tap = 0u; tap < state_len; ++tap) {
        sum = fma(conv_state[state_base + tap],
                  qwen_next_source_bf16_value(conv_weights[weight_base + tap]),
                  sum);
    }
    for (uint tap = 0u; tap + 1u < state_len; ++tap) {
        conv_state[state_base + tap] = conv_state[state_base + tap + 1u];
    }
    conv_state[state_base + state_len - 1u] = current;
    sum = fma(current,
              qwen_next_source_bf16_value(conv_weights[weight_base + state_len]),
              sum);
    return sum / (1.0f + exp(-sum));
}

// Source-BF16 split projection counterpart of
// `qwen_next_qkvz_rearrange_conv_l2`.  The output layout is the same as the
// existing recurrent kernels: repeated Q/K and convolved V are laid out by
// value head, with Z copied from its separate source projection.
kernel void qwen_next_qkv_split_rearrange_conv_l2(
    device const float* projected_qkv [[buffer(0)]],
    device const float* projected_z    [[buffer(1)]],
    device const ushort* conv_weights [[buffer(2)]],
    device float* conv_state          [[buffer(3)]],
    device float* repeated_query      [[buffer(4)]],
    device float* repeated_key        [[buffer(5)]],
    device float* convolved_value     [[buffer(6)]],
    device float* z                    [[buffer(7)]],
    constant uint& key_heads           [[buffer(8)]],
    constant uint& values_per_key_head [[buffer(9)]],
    constant uint& key_head_dim        [[buffer(10)]],
    constant uint& value_head_dim      [[buffer(11)]],
    constant uint& conv_kernel         [[buffer(12)]],
    constant float& eps                [[buffer(13)]],
    threadgroup float* scratch         [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint3 group                         [[threadgroup_position_in_grid]])
{
    const uint key_head = group.y;
    if (key_head >= key_heads) return;
    const uint value_rows_per_key_head = values_per_key_head * value_head_dim;
    const uint key_elements = key_heads * key_head_dim;
    const uint conv_base = key_head * key_head_dim;
    const uint value_base = key_head * value_rows_per_key_head;

    threadgroup float* query_local = scratch;
    threadgroup float* key_local = scratch + 128u;
    threadgroup float* query_sums = scratch + 256u;
    threadgroup float* key_sums = scratch + 512u;

    if (tid < key_head_dim) {
        const uint query_channel = conv_base + tid;
        const uint key_channel = key_elements + query_channel;
        query_local[tid] = qwen_next_causal_conv_update_source_bf16(
            conv_state, conv_weights, query_channel, projected_qkv[query_channel], conv_kernel);
        key_local[tid] = qwen_next_causal_conv_update_source_bf16(
            conv_state, conv_weights, key_channel, projected_qkv[key_channel], conv_kernel);
    }
    // Qwen3.8 has three value heads per key head (3 × 128 = 384 rows),
    // larger than the 256-thread group.  Stride the value tail so every
    // source channel is updated; a single `tid < value_rows_per_key_head`
    // guard silently left the final 128 channels at zero.
    for (uint value_offset = tid; value_offset < value_rows_per_key_head; value_offset += 256u) {
        const uint value_channel = key_elements * 2u + value_base + value_offset;
        convolved_value[value_base + value_offset] = qwen_next_causal_conv_update_source_bf16(
            conv_state,
            conv_weights,
            value_channel,
            projected_qkv[value_channel],
            conv_kernel);
        z[value_base + value_offset] = projected_z[value_base + value_offset];
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

// Split BF16 `in_proj_b` / `in_proj_a` source projections into the same
// recurrent controls as the fused Qwen3-Next path.  A_log and dt_bias remain
// BF16 device values, so this is also the source-native control authority.
kernel void qwen_next_ba_split_to_decay_beta_source_bf16(
    device const float* projected_b    [[buffer(0)]],
    device const float* projected_a    [[buffer(1)]],
    device const ushort* a_log_bf16    [[buffer(2)]],
    device const ushort* dt_bias_bf16  [[buffer(3)]],
    device float* decay                [[buffer(4)]],
    device float* beta                 [[buffer(5)]],
    constant uint& key_heads           [[buffer(6)]],
    constant uint& values_per_key_head [[buffer(7)]],
    uint value_head                    [[thread_position_in_grid]])
{
    const uint value_heads = key_heads * values_per_key_head;
    if (value_head >= value_heads) return;
    const float b = projected_b[value_head];
    const float a = projected_a[value_head];
    const float a_log = qwen_next_source_bf16_value(a_log_bf16[value_head]);
    const float dt_bias = qwen_next_source_bf16_value(dt_bias_bf16[value_head]);
    const float x = a + dt_bias;
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    const float g = -exp(a_log) * softplus;
    decay[value_head] = exp(g);
    beta[value_head] = 1.0f / (1.0f + exp(-b));
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

// Qwen3.8-Flash-Next source norm variant.  Its checkpoint stores the
// 128-wide output norm as BF16 and the model-local reference uses a sigmoid
// output gate (rather than the older Qwen3-Next SiLU gate).
kernel void qwen_next_deltanet_source_bf16_gated_rmsnorm(
    device const float* input          [[buffer(0)]],
    device const float* z              [[buffer(1)]],
    device const ushort* weight_bf16   [[buffer(2)]],
    device float* output               [[buffer(3)]],
    constant uint& heads               [[buffer(4)]],
    constant uint& value_head_dim      [[buffer(5)]],
    constant float& eps                [[buffer(6)]],
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
        const float gate = 1.0f / (1.0f + exp(-z[base + index]));
        const float weight = qwen_next_source_bf16_value(weight_bf16[index]);
        output[base + index] = input[base + index] * inverse_rms * weight * gate;
    }
}

// Full-attention Flash candidate: source-BF16 Q/K/V projection, Q/K RMSNorm
// + RoPE, and the current-token KV-cache write in one head-local launch.
// Q/K projection rows and the raw Q gate remain observable in their existing
// diagnostic buffers; V is copied to both its diagnostic buffer and cache.
// Q/K normalization keeps the previous scalar left-to-right reduction in
// lane zero, while the projection rows are independent.  The explicit Rust
// switch is required because this fused ABI has not yet received a physical
// Metal parity/timing receipt on the current host.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_qkv_gqa_rope_cache(
    device const ushort* q_weight       [[buffer(0)]],
    device const ushort* k_weight       [[buffer(1)]],
    device const ushort* v_weight       [[buffer(2)]],
    device const float* input           [[buffer(3)]],
    device const float* q_norm          [[buffer(4)]],
    device const float* k_norm          [[buffer(5)]],
    device float* q_projection          [[buffer(6)]],
    device float* k_projection          [[buffer(7)]],
    device float* v_projection          [[buffer(8)]],
    device float* query                 [[buffer(9)]],
    device float* key_cache             [[buffer(10)]],
    device float* value_cache           [[buffer(11)]],
    constant uint& sequence_slot        [[buffer(12)]],
    constant uint& n_heads              [[buffer(13)]],
    constant uint& n_kv_heads           [[buffer(14)]],
    constant uint& head_dim             [[buffer(15)]],
    constant uint& rotary_dim           [[buffer(16)]],
    constant uint& input_dim            [[buffer(17)]],
    constant float& rope_theta          [[buffer(18)]],
    constant float& rms_epsilon         [[buffer(19)]],
    threadgroup float* scratch          [[threadgroup(0)]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint3 group                          [[threadgroup_position_in_grid]])
{
    const uint head = group.x;
    if (head >= n_heads || tid >= head_dim) return;

    const uint q_row_base = head * (2u * head_dim);
    const ulong q_row = (ulong)(q_row_base + tid) * (ulong)input_dim;
    const ulong gate_row = (ulong)(q_row_base + head_dim + tid) * (ulong)input_dim;
    const float q_acc = qwen_next_source_bf16_dot_vec4(q_weight + q_row, input, input_dim);
    const float gate_acc = qwen_next_source_bf16_dot_vec4(q_weight + gate_row, input, input_dim);
    scratch[tid] = q_acc;
    q_projection[q_row_base + tid] = q_acc;
    q_projection[q_row_base + head_dim + tid] = gate_acc;

    if (head < n_kv_heads) {
        const uint kv_row_base = head * head_dim;
        const ulong k_row = (ulong)(kv_row_base + tid) * (ulong)input_dim;
        const ulong v_row = (ulong)(kv_row_base + tid) * (ulong)input_dim;
        const float k_acc = qwen_next_source_bf16_dot_vec4(k_weight + k_row, input, input_dim);
        const float v_acc = qwen_next_source_bf16_dot_vec4(v_weight + v_row, input, input_dim);
        scratch[head_dim + tid] = k_acc;
        k_projection[kv_row_base + tid] = k_acc;
        v_projection[kv_row_base + tid] = v_acc;
        const ulong cache_base = ((ulong)sequence_slot * (ulong)n_kv_heads + (ulong)head)
            * (ulong)head_dim;
        value_cache[cache_base + tid] = v_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float q_sum = 0.0f;
        for (uint dim = 0u; dim < head_dim; ++dim) {
            const float value = scratch[dim];
            q_sum += value * value;
        }
        scratch[2u * head_dim] = 1.0f / sqrt(q_sum / float(head_dim) + rms_epsilon);
        if (head < n_kv_heads) {
            float k_sum = 0.0f;
            for (uint dim = 0u; dim < head_dim; ++dim) {
                const float value = scratch[head_dim + dim];
                k_sum += value * value;
            }
            scratch[2u * head_dim + 1u] = 1.0f / sqrt(k_sum / float(head_dim) + rms_epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_dim = rotary_dim / 2u;
    const float q_inverse_rms = scratch[2u * head_dim];
    const float q_raw = scratch[tid];
    const float q_normed = q_raw * q_inverse_rms * (1.0f + q_norm[tid]);
    if (tid < rotary_dim) {
        const uint frequency_index = tid < half_dim ? tid : tid - half_dim;
        const float inv_frequency = pow(
            rope_theta, -2.0f * float(frequency_index) / float(rotary_dim));
        const float angle = float(sequence_slot) * inv_frequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
        const uint peer = tid < half_dim ? tid + half_dim : tid - half_dim;
        const float peer_raw = scratch[peer] * q_inverse_rms * (1.0f + q_norm[peer]);
        query[head * head_dim + tid] = tid < half_dim
            ? q_normed * cosine - peer_raw * sine
            : q_normed * cosine + peer_raw * sine;
    } else {
        query[head * head_dim + tid] = q_normed;
    }

    if (head < n_kv_heads) {
        const float k_inverse_rms = scratch[2u * head_dim + 1u];
        const float k_raw = scratch[head_dim + tid];
        const float k_normed = k_raw * k_inverse_rms * (1.0f + k_norm[tid]);
        const ulong cache_base = ((ulong)sequence_slot * (ulong)n_kv_heads + (ulong)head)
            * (ulong)head_dim;
        if (tid < rotary_dim) {
            const uint frequency_index = tid < half_dim ? tid : tid - half_dim;
            const float inv_frequency = pow(
                rope_theta, -2.0f * float(frequency_index) / float(rotary_dim));
            const float angle = float(sequence_slot) * inv_frequency;
            const float cosine = cos(angle);
            const float sine = sin(angle);
            const uint peer = tid < half_dim ? tid + half_dim : tid - half_dim;
            const float peer_raw = scratch[head_dim + peer]
                * k_inverse_rms * (1.0f + k_norm[peer]);
            key_cache[cache_base + tid] = tid < half_dim
                ? k_normed * cosine - peer_raw * sine
                : k_normed * cosine + peer_raw * sine;
        } else {
            key_cache[cache_base + tid] = k_normed;
        }
    }
}

// Source-BF16 Flash router fusion candidate: router GEMV, shared-expert gate
// scalar, softmax/top-k selection, and optional top-k renormalization share one
// token-local threadgroup.  Router logits are still written for diagnostics,
// but top-k consumes the threadgroup copy directly instead of forcing a
// producer dispatch, device round-trip, and consumer dispatch.  The GEMV dots
// and the tie-enabled serial selector retain the source left-to-right order;
// the epsilon-zero branch mirrors the existing parallel top-k selector.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_router_topk_shared(
    device const ushort* router_weights       [[buffer(0)]],
    device const ushort* shared_scalar_weights [[buffer(1)]],
    device const float* input                 [[buffer(2)]],
    device float* router_logits               [[buffer(3)]],
    device float* shared_scalar_output        [[buffer(4)]],
    device uint* route_ids                    [[buffer(5)]],
    device float* route_weights               [[buffer(6)]],
    constant uint& n_experts                  [[buffer(7)]],
    constant uint& top_k                      [[buffer(8)]],
    constant float& tie_epsilon               [[buffer(9)]],
    constant uint& normalize_topk             [[buffer(10)]],
    constant uint& input_dim                  [[buffer(11)]],
    threadgroup float* scratch                [[threadgroup(0)]],
    uint tid                                  [[thread_index_in_threadgroup]],
    uint tg_size                              [[threads_per_threadgroup]])
{
    threadgroup float* work = scratch;
    threadgroup float* red_val = scratch + n_experts;
    threadgroup uint* red_idx = (threadgroup uint*)(scratch + n_experts + tg_size);

    for (uint expert = tid; expert < n_experts; expert += tg_size) {
        const ulong row_base = (ulong)expert * (ulong)input_dim;
        const float acc = qwen_next_source_bf16_dot_vec4(
            router_weights + row_base, input, input_dim);
        router_logits[expert] = acc;
        work[expert] = acc;
    }
    if (tid == 0u) {
        shared_scalar_output[0] = qwen_next_source_bf16_dot_vec4(
            shared_scalar_weights, input, input_dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This branch is the exact serial policy used when the runtime requests
    // epsilon-window tie resolution. It deliberately leaves the selector on
    // lane zero because the comparator is order-dependent by contract.
    if (tie_epsilon > 0.0f) {
        if (tid == 0u) {
            float m = -INFINITY;
            for (uint i = 0u; i < n_experts; ++i) {
                if (work[i] > m) m = work[i];
            }

            float sum = 0.0f;
            for (uint i = 0u; i < n_experts; ++i) {
                work[i] = exp(work[i] - m);
                sum += work[i];
            }
            const float inv = 1.0f / sum;
            for (uint i = 0u; i < n_experts; ++i) work[i] *= inv;

            for (uint k = 0u; k < top_k; ++k) {
                uint best_idx = 0u;
                float best_val = -INFINITY;
                for (uint i = 0u; i < n_experts; ++i) {
                    const bool finite_pair = isfinite(best_val) && isfinite(work[i]);
                    const bool tied = finite_pair
                        && abs(work[i] - best_val) <= tie_epsilon;
                    if ((work[i] > best_val && !tied) || (tied && i < best_idx)) {
                        best_val = work[i];
                        best_idx = i;
                    }
                }
                route_ids[k] = best_idx;
                route_weights[k] = best_val;
                work[best_idx] = -INFINITY;
            }
            if (normalize_topk != 0u) {
                float selected_sum = 0.0f;
                for (uint i = 0u; i < top_k; ++i) selected_sum += route_weights[i];
                if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
                    for (uint i = 0u; i < top_k; ++i) route_weights[i] = NAN;
                } else {
                    const float inv_selected = 1.0f / selected_sum;
                    for (uint i = 0u; i < top_k; ++i) route_weights[i] *= inv_selected;
                }
            }
        }
        return;
    }

    // Epsilon-zero path copied from moe_topk_gate. Max reduction is
    // associative; the exponential sum and selected-weight normalization are
    // intentionally left-folded on lane zero to preserve the authority path.
    float local = -INFINITY;
    for (uint i = tid; i < n_experts; i += tg_size) {
        local = max(local, work[i]);
    }
    red_val[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red_val[tid] = max(red_val[tid], red_val[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float m = red_val[0];
    for (uint i = tid; i < n_experts; i += tg_size) {
        work[i] = exp(work[i] - m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0u; i < n_experts; ++i) sum += work[i];
        red_val[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red_val[0];
    for (uint i = tid; i < n_experts; i += tg_size) {
        work[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0u; k < top_k; ++k) {
        float best_val = -INFINITY;
        uint best_idx = 0xFFFFFFFFu;
        for (uint i = tid; i < n_experts; i += tg_size) {
            const float value = work[i];
            if ((value > best_val) || (value == best_val && i < best_idx)) {
                best_val = value;
                best_idx = i;
            }
        }
        red_val[tid] = best_val;
        red_idx[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                const float other_value = red_val[tid + stride];
                const uint other_idx = red_idx[tid + stride];
                if ((other_value > red_val[tid])
                    || (other_value == red_val[tid] && other_idx < red_idx[tid])) {
                    red_val[tid] = other_value;
                    red_idx[tid] = other_idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            const uint winner = red_idx[0];
            route_ids[k] = winner;
            route_weights[k] = red_val[0];
            work[winner] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (normalize_topk != 0u && tid == 0u) {
        float selected_sum = 0.0f;
        for (uint i = 0u; i < top_k; ++i) selected_sum += route_weights[i];
        if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
            for (uint i = 0u; i < top_k; ++i) route_weights[i] = NAN;
        } else {
            const float inv_selected = 1.0f / selected_sum;
            for (uint i = 0u; i < top_k; ++i) route_weights[i] *= inv_selected;
        }
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

// Device-resident Qwen3-Next MoE route reduction.  Each selected routed
// expert writes one contiguous hidden-width row; the router's normalized
// selected weights are applied without a host gather.
kernel void qwen_next_moe_weighted_sum(
    device const float* routed_outputs [[buffer(0)]],
    device const float* selected_weights [[buffer(1)]],
    device float* output                [[buffer(2)]],
    constant uint& expert_count         [[buffer(3)]],
    constant uint& hidden               [[buffer(4)]],
    uint id                             [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    float sum = 0.0f;
    for (uint expert = 0u; expert < expert_count; ++expert) {
        sum += routed_outputs[expert * hidden + id] * selected_weights[expert];
    }
    output[id] = sum;
}

// Qwen3-Next's complete candidate MoE block adds the sigmoid-gated shared
// expert output to the selected routed-expert sum on device.
kernel void qwen_next_moe_add_shared(
    device const float* routed_output [[buffer(0)]],
    device const float* shared_output [[buffer(1)]],
    device float* output              [[buffer(2)]],
    constant uint& elements           [[buffer(3)]],
    uint id                           [[thread_position_in_grid]])
{
    if (id >= elements) return;
    output[id] = routed_output[id] + shared_output[id];
}

// Fused Qwen3-Next MoE epilogue.  Preserve the source accumulation order
// (routed sum first, then sigmoid-gated shared output) while eliminating the
// intermediate routed-sum, shared-gate, and add dispatches.
kernel void qwen_next_moe_weighted_sum_add_shared_sigmoid(
    device const float* routed_outputs [[buffer(0)]],
    device const float* selected_weights [[buffer(1)]],
    device const float* shared_output [[buffer(2)]],
    device const float* shared_gate_logit [[buffer(3)]],
    device float* routed_sum_out        [[buffer(4)]],
    device float* shared_gated_out      [[buffer(5)]],
    device float* output                [[buffer(6)]],
    constant uint& expert_count         [[buffer(7)]],
    constant uint& hidden               [[buffer(8)]],
    uint id                             [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    float routed = 0.0f;
    for (uint expert = 0u; expert < expert_count; ++expert) {
        routed += routed_outputs[expert * hidden + id] * selected_weights[expert];
    }
    const float gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
    const float shared = shared_output[id] * gate;
    routed_sum_out[id] = routed;
    shared_gated_out[id] = shared;
    output[id] = routed + shared;
}

// Dense-bank MoE epilogue fused with the following HyperConnection write.
// Keep the routed sum, shared-gated output, and hidden-width MoE output
// observable, but consume the output in the same launch so the stream-major
// residual is not reread by a second dispatch.
#pragma clang fp contract(off)
kernel void qwen_next_moe_weighted_sum_add_shared_sigmoid_hc(
    device const float* routed_outputs [[buffer(0)]],
    device const float* selected_weights [[buffer(1)]],
    device const float* shared_output [[buffer(2)]],
    device const float* shared_gate_logit [[buffer(3)]],
    device float* routed_sum_out        [[buffer(4)]],
    device float* shared_gated_out      [[buffer(5)]],
    device float* output                [[buffer(6)]],
    device const float* residual        [[buffer(7)]],
    device const float* block_logits    [[buffer(8)]],
    device float* final_output          [[buffer(9)]],
    constant uint& expert_count         [[buffer(10)]],
    constant uint& hidden               [[buffer(11)]],
    constant uint& streams              [[buffer(12)]],
    constant float& divisor             [[buffer(13)]],
    uint id                             [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    float routed = 0.0f;
    for (uint expert = 0u; expert < expert_count; ++expert) {
        routed += routed_outputs[expert * hidden + id] * selected_weights[expert];
    }
    const float gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
    const float shared = shared_output[id] * gate;
    const float moe = routed + shared;
    routed_sum_out[id] = routed;
    shared_gated_out[id] = shared;
    output[id] = moe;
    for (uint stream = 0u; stream < streams; ++stream) {
        const float hc_gate = 2.0f / (1.0f + exp(-block_logits[stream] / divisor));
        const ulong offset = (ulong)stream * (ulong)hidden + (ulong)id;
        final_output[offset] = residual[offset] + moe * hc_gate;
    }
}

// Native source-BF16 routed expert wave.  The complete expert tensor remains
// resident in its native BF16 layout; route IDs produced by `moe_topk_gate`
// select each expert after the router without a host gather.  Gate and up are
// fused with SwiGLU, while the down projection is a separate device pass.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_expert_gate_up_swiglu(
    device const ushort* gate_up_weights [[buffer(0)]],
    device const uint* route_ids        [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    constant uint& experts              [[buffer(4)]],
    constant uint& top_k                [[buffer(5)]],
    constant uint& intermediate         [[buffer(6)]],
    constant uint& hidden               [[buffer(7)]],
    uint3 gid                            [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint route = gid.y;
    if (route >= top_k || row >= intermediate) return;
    const uint expert = route_ids[route];
    if (expert >= experts) {
        output[route * intermediate + row] = 0.0f;
        return;
    }
    const ulong expert_stride = (ulong)(2u * intermediate) * (ulong)hidden;
    const ulong gate_base = (ulong)expert * expert_stride + (ulong)row * (ulong)hidden;
    const ulong up_base = gate_base + (ulong)intermediate * (ulong)hidden;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint column = 0u; column < hidden; ++column) {
        gate_acc = gate_acc + qwen_next_source_bf16_value(gate_up_weights[gate_base + column]) * input[column];
        up_acc = up_acc + qwen_next_source_bf16_value(gate_up_weights[up_base + column]) * input[column];
    }
    output[route * intermediate + row] =
        (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
}

#pragma clang fp contract(off)
kernel void qwen_next_bf16_expert_down(
    device const ushort* down_weights [[buffer(0)]],
    device const uint* route_ids      [[buffer(1)]],
    device const float* activated     [[buffer(2)]],
    device float* output              [[buffer(3)]],
    constant uint& experts            [[buffer(4)]],
    constant uint& top_k              [[buffer(5)]],
    constant uint& intermediate       [[buffer(6)]],
    constant uint& hidden             [[buffer(7)]],
    uint3 gid                          [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint route = gid.y;
    if (route >= top_k || row >= hidden) return;
    const uint expert = route_ids[route];
    if (expert >= experts) {
        output[route * hidden + row] = 0.0f;
        return;
    }
    const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
    const ulong row_base = (ulong)expert * expert_stride + (ulong)row * (ulong)intermediate;
    float acc = 0.0f;
    for (uint column = 0u; column < intermediate; ++column) {
        acc = acc + qwen_next_source_bf16_value(down_weights[row_base + column])
            * activated[route * intermediate + column];
    }
    output[route * hidden + row] = acc;
}

// Compact routed-bank variants.  `route_ids` retain the original 512-expert
// IDs for parity receipts; `route_lut` maps each original ID to a contiguous
// slot in the compact bank (or UINT_MAX for an unselected expert).
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_gate_up_swiglu(
    device const ushort* gate_up_weights [[buffer(0)]],
    device const uint* route_ids        [[buffer(1)]],
    device const uint* route_lut        [[buffer(2)]],
    device const float* input           [[buffer(3)]],
    device float* output                [[buffer(4)]],
    constant uint& compact_experts      [[buffer(5)]],
    constant uint& top_k                [[buffer(6)]],
    constant uint& intermediate         [[buffer(7)]],
    constant uint& hidden               [[buffer(8)]],
    constant uint& source_experts       [[buffer(9)]],
    uint3 gid                            [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint route = gid.y;
    if (route >= top_k || row >= intermediate) return;
    const uint expert = route_ids[route];
    if (expert >= source_experts) {
        output[route * intermediate + row] = 0.0f;
        return;
    }
    const uint slot = route_lut[expert];
    if (slot >= compact_experts) {
        output[route * intermediate + row] = 0.0f;
        return;
    }
    const ulong expert_stride = (ulong)(2u * intermediate) * (ulong)hidden;
    const ulong gate_base = (ulong)slot * expert_stride + (ulong)row * (ulong)hidden;
    const ulong up_base = gate_base + (ulong)intermediate * (ulong)hidden;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint column = 0u; column < hidden; ++column) {
        gate_acc = gate_acc + qwen_next_source_bf16_value(gate_up_weights[gate_base + column]) * input[column];
        up_acc = up_acc + qwen_next_source_bf16_value(gate_up_weights[up_base + column]) * input[column];
    }
    output[route * intermediate + row] =
        (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
}

// Compact routed gate/up plus the shared-expert gate/up in one launch.  The
// final grid row is reserved for the shared expert; routed rows retain the
// compact-bank lookup and the same source-BF16 left-to-right dot order as the
// standalone kernels.  This removes a dispatch and an extra activation
// boundary without changing the later direct MoE/HyperConnection epilogue.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_gate_up_shared_swiglu(
    device const ushort* gate_up_weights        [[buffer(0)]],
    device const uint* route_ids                [[buffer(1)]],
    device const uint* route_lut                 [[buffer(2)]],
    device const float* input                   [[buffer(3)]],
    device float* routed_output                 [[buffer(4)]],
    device const ushort* shared_gate_weights    [[buffer(5)]],
    device const ushort* shared_up_weights      [[buffer(6)]],
    device float* shared_output                 [[buffer(7)]],
    constant uint& compact_experts              [[buffer(8)]],
    constant uint& top_k                        [[buffer(9)]],
    constant uint& intermediate                 [[buffer(10)]],
    constant uint& hidden                       [[buffer(11)]],
    constant uint& source_experts               [[buffer(12)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint route = gid.y;
    if (row >= intermediate || route > top_k) return;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (route == top_k) {
        const ulong row_base = (ulong)row * (ulong)hidden;
        for (uint column = 0u; column < hidden; ++column) {
            const float x = input[column];
            gate_acc = gate_acc
                + qwen_next_source_bf16_value(shared_gate_weights[row_base + column]) * x;
            up_acc = up_acc
                + qwen_next_source_bf16_value(shared_up_weights[row_base + column]) * x;
        }
        shared_output[row] = (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
        return;
    }

    const uint expert = route_ids[route];
    if (expert >= source_experts) {
        routed_output[route * intermediate + row] = 0.0f;
        return;
    }
    const uint slot = route_lut[expert];
    if (slot >= compact_experts) {
        routed_output[route * intermediate + row] = 0.0f;
        return;
    }
    const ulong expert_stride = (ulong)(2u * intermediate) * (ulong)hidden;
    const ulong gate_base = (ulong)slot * expert_stride + (ulong)row * (ulong)hidden;
    const ulong up_base = gate_base + (ulong)intermediate * (ulong)hidden;
    for (uint column = 0u; column < hidden; ++column) {
        const float x = input[column];
        gate_acc = gate_acc
            + qwen_next_source_bf16_value(gate_up_weights[gate_base + column]) * x;
        up_acc = up_acc
            + qwen_next_source_bf16_value(gate_up_weights[up_base + column]) * x;
    }
    routed_output[route * intermediate + row] =
        (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
}

// Exact-order packed-load sibling for the compact routed/shared gate-up
// launch. The four source values are loaded together, but products are still
// added to each accumulator in source column order. This changes only the
// memory transaction shape, not BF16 widening or reduction association.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_gate_up_shared_swiglu_vec4(
    device const ushort* gate_up_weights        [[buffer(0)]],
    device const uint* route_ids                [[buffer(1)]],
    device const uint* route_lut                 [[buffer(2)]],
    device const float* input                   [[buffer(3)]],
    device float* routed_output                 [[buffer(4)]],
    device const ushort* shared_gate_weights    [[buffer(5)]],
    device const ushort* shared_up_weights      [[buffer(6)]],
    device float* shared_output                 [[buffer(7)]],
    constant uint& compact_experts              [[buffer(8)]],
    constant uint& top_k                        [[buffer(9)]],
    constant uint& intermediate                 [[buffer(10)]],
    constant uint& hidden                       [[buffer(11)]],
    constant uint& source_experts               [[buffer(12)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint route = gid.y;
    if (row >= intermediate || route > top_k) return;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (route == top_k) {
        const ulong row_base = (ulong)row * (ulong)hidden;
        uint column = 0u;
        for (; column + 4u <= hidden; column += 4u) {
            const ushort4 gate_bits = *(device const ushort4*)(shared_gate_weights + row_base + column);
            const ushort4 up_bits = *(device const ushort4*)(shared_up_weights + row_base + column);
            const float4 x = *(device const float4*)(input + column);
            gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.x) * x.x;
            up_acc = up_acc + qwen_next_source_bf16_value(up_bits.x) * x.x;
            gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.y) * x.y;
            up_acc = up_acc + qwen_next_source_bf16_value(up_bits.y) * x.y;
            gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.z) * x.z;
            up_acc = up_acc + qwen_next_source_bf16_value(up_bits.z) * x.z;
            gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.w) * x.w;
            up_acc = up_acc + qwen_next_source_bf16_value(up_bits.w) * x.w;
        }
        for (; column < hidden; ++column) {
            gate_acc = gate_acc
                + qwen_next_source_bf16_value(shared_gate_weights[row_base + column]) * input[column];
            up_acc = up_acc
                + qwen_next_source_bf16_value(shared_up_weights[row_base + column]) * input[column];
        }
        shared_output[row] = (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
        return;
    }

    const uint expert = route_ids[route];
    if (expert >= source_experts) {
        routed_output[route * intermediate + row] = 0.0f;
        return;
    }
    const uint slot = route_lut[expert];
    if (slot >= compact_experts) {
        routed_output[route * intermediate + row] = 0.0f;
        return;
    }
    const ulong expert_stride = (ulong)(2u * intermediate) * (ulong)hidden;
    const ulong gate_base = (ulong)slot * expert_stride + (ulong)row * (ulong)hidden;
    const ulong up_base = gate_base + (ulong)intermediate * (ulong)hidden;
    uint column = 0u;
    for (; column + 4u <= hidden; column += 4u) {
        const ushort4 gate_bits = *(device const ushort4*)(gate_up_weights + gate_base + column);
        const ushort4 up_bits = *(device const ushort4*)(gate_up_weights + up_base + column);
        const float4 x = *(device const float4*)(input + column);
        gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.x) * x.x;
        up_acc = up_acc + qwen_next_source_bf16_value(up_bits.x) * x.x;
        gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.y) * x.y;
        up_acc = up_acc + qwen_next_source_bf16_value(up_bits.y) * x.y;
        gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.z) * x.z;
        up_acc = up_acc + qwen_next_source_bf16_value(up_bits.z) * x.z;
        gate_acc = gate_acc + qwen_next_source_bf16_value(gate_bits.w) * x.w;
        up_acc = up_acc + qwen_next_source_bf16_value(up_bits.w) * x.w;
    }
    for (; column < hidden; ++column) {
        gate_acc = gate_acc
            + qwen_next_source_bf16_value(gate_up_weights[gate_base + column]) * input[column];
        up_acc = up_acc
            + qwen_next_source_bf16_value(gate_up_weights[up_base + column]) * input[column];
    }
    routed_output[route * intermediate + row] =
        (gate_acc / (1.0f + exp(-gate_acc))) * up_acc;
}

#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_down(
    device const ushort* down_weights [[buffer(0)]],
    device const uint* route_ids      [[buffer(1)]],
    device const uint* route_lut      [[buffer(2)]],
    device const float* activated     [[buffer(3)]],
    device float* output              [[buffer(4)]],
    constant uint& compact_experts    [[buffer(5)]],
    constant uint& top_k              [[buffer(6)]],
    constant uint& intermediate       [[buffer(7)]],
    constant uint& hidden             [[buffer(8)]],
    constant uint& source_experts     [[buffer(9)]],
    uint3 gid                          [[thread_position_in_grid]])
{
    const uint row = gid.x;
    const uint route = gid.y;
    if (route >= top_k || row >= hidden) return;
    const uint expert = route_ids[route];
    if (expert >= source_experts) {
        output[route * hidden + row] = 0.0f;
        return;
    }
    const uint slot = route_lut[expert];
    if (slot >= compact_experts) {
        output[route * hidden + row] = 0.0f;
        return;
    }
    const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
    const ulong row_base = (ulong)slot * expert_stride + (ulong)row * (ulong)intermediate;
    float acc = 0.0f;
    for (uint column = 0u; column < intermediate; ++column) {
        acc = acc + qwen_next_source_bf16_value(down_weights[row_base + column])
            * activated[route * intermediate + column];
    }
    output[route * hidden + row] = acc;
}

// Fused compact routed down projection and weighted accumulation.  This is
// the direct-compute counterpart to qwen_next_bf16_compact_expert_down plus
// qwen_next_moe_weighted_sum: each hidden element walks the selected routes,
// applies the route weight, and writes one final routed-sum value.  It avoids
// materialising TOP_K full hidden-width expert outputs.  Route IDs remain in
// source-expert space and route_lut maps them to compact-bank slots.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_down_weighted_sum(
    device const ushort* down_weights [[buffer(0)]],
    device const uint* route_ids      [[buffer(1)]],
    device const uint* route_lut      [[buffer(2)]],
    device const float* activated     [[buffer(3)]],
    device const float* selected_weights [[buffer(4)]],
    device float* output              [[buffer(5)]],
    constant uint& compact_experts    [[buffer(6)]],
    constant uint& top_k               [[buffer(7)]],
    constant uint& intermediate        [[buffer(8)]],
    constant uint& hidden              [[buffer(9)]],
    constant uint& source_experts      [[buffer(10)]],
    uint row                            [[thread_position_in_grid]])
{
    if (row >= hidden) return;
    float routed_sum = 0.0f;
    for (uint route = 0u; route < top_k; ++route) {
        const uint expert = route_ids[route];
        if (expert >= source_experts) continue;
        const uint slot = route_lut[expert];
        if (slot >= compact_experts) continue;
        const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
        const ulong row_base = (ulong)slot * expert_stride + (ulong)row * (ulong)intermediate;
        float acc = 0.0f;
        for (uint column = 0u; column < intermediate; ++column) {
            acc = acc + qwen_next_source_bf16_value(down_weights[row_base + column])
                * activated[route * intermediate + column];
        }
        routed_sum = routed_sum + acc * selected_weights[route];
    }
    output[row] = routed_sum;
}

// Direct compact MoE epilogue.  This is the fast Flash route: selected
// source-BF16 expert down projections, the shared source-BF16 down
// projection, the scalar sigmoid gate, and the source-order routed/shared
// addition are evaluated by one hidden-row owner.  The diagnostic vectors are
// still written, so the fused path remains stage-observable without bringing
// back a TOP_K x hidden routed-output materialization or separate shared-down,
// sigmoid, and add launches.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_down_shared_direct(
    device const ushort* down_weights       [[buffer(0)]],
    device const uint* route_ids            [[buffer(1)]],
    device const uint* route_lut            [[buffer(2)]],
    device const float* activated           [[buffer(3)]],
    device const float* selected_weights    [[buffer(4)]],
    device const ushort* shared_down_weights [[buffer(5)]],
    device const float* shared_activation   [[buffer(6)]],
    device const float* shared_gate_logit   [[buffer(7)]],
    device float* routed_sum_out             [[buffer(8)]],
    device float* shared_output_out          [[buffer(9)]],
    device float* shared_gated_out           [[buffer(10)]],
    device float* output                    [[buffer(11)]],
    constant uint& compact_experts           [[buffer(12)]],
    constant uint& top_k                    [[buffer(13)]],
    constant uint& intermediate              [[buffer(14)]],
    constant uint& hidden                    [[buffer(15)]],
    constant uint& source_experts            [[buffer(16)]],
    uint row                                [[thread_position_in_grid]])
{
    if (row >= hidden) return;
    const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
    float routed_sum = 0.0f;
    for (uint route = 0u; route < top_k; ++route) {
        const uint expert = route_ids[route];
        if (expert >= source_experts) continue;
        const uint slot = route_lut[expert];
        if (slot >= compact_experts) continue;
        const ulong row_base = expert_stride * (ulong)slot
            + (ulong)row * (ulong)intermediate;
        float expert_sum = 0.0f;
        for (uint column = 0u; column < intermediate; ++column) {
            expert_sum = expert_sum
                + qwen_next_source_bf16_value(down_weights[row_base + column])
                    * activated[route * intermediate + column];
        }
        routed_sum = routed_sum + expert_sum * selected_weights[route];
    }

    const ulong shared_row_base = (ulong)row * (ulong)intermediate;
    float shared_sum = 0.0f;
    for (uint column = 0u; column < intermediate; ++column) {
        shared_sum = shared_sum
            + qwen_next_source_bf16_value(shared_down_weights[shared_row_base + column])
                * shared_activation[column];
    }
    const float shared_gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
    const float shared_gated = shared_sum * shared_gate;
    routed_sum_out[row] = routed_sum;
    shared_output_out[row] = shared_sum;
    shared_gated_out[row] = shared_gated;
    output[row] = routed_sum + shared_gated;
}

// Compact-bank counterpart to the dense MoE/HyperConnection fusion above.
// The source-BF16 expert reductions and all diagnostic outputs retain the
// direct kernel's exact order; the final stream-major state is written by the
// same hidden-row owner before the command buffer advances.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_down_shared_direct_hc(
    device const ushort* down_weights       [[buffer(0)]],
    device const uint* route_ids            [[buffer(1)]],
    device const uint* route_lut             [[buffer(2)]],
    device const float* activated           [[buffer(3)]],
    device const float* selected_weights    [[buffer(4)]],
    device const ushort* shared_down_weights [[buffer(5)]],
    device const float* shared_activation   [[buffer(6)]],
    device const float* shared_gate_logit   [[buffer(7)]],
    device float* routed_sum_out             [[buffer(8)]],
    device float* shared_output_out          [[buffer(9)]],
    device float* shared_gated_out            [[buffer(10)]],
    device float* output                     [[buffer(11)]],
    device const float* residual              [[buffer(12)]],
    device const float* block_logits          [[buffer(13)]],
    device float* final_output                [[buffer(14)]],
    constant uint& compact_experts            [[buffer(15)]],
    constant uint& top_k                      [[buffer(16)]],
    constant uint& intermediate               [[buffer(17)]],
    constant uint& hidden                     [[buffer(18)]],
    constant uint& source_experts             [[buffer(19)]],
    constant uint& streams                    [[buffer(20)]],
    constant float& divisor                   [[buffer(21)]],
    uint row                                  [[thread_position_in_grid]])
{
    if (row >= hidden) return;
    const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
    float routed_sum = 0.0f;
    for (uint route = 0u; route < top_k; ++route) {
        const uint expert = route_ids[route];
        if (expert >= source_experts) continue;
        const uint slot = route_lut[expert];
        if (slot >= compact_experts) continue;
        const ulong row_base = expert_stride * (ulong)slot
            + (ulong)row * (ulong)intermediate;
        float expert_sum = 0.0f;
        for (uint column = 0u; column < intermediate; ++column) {
            expert_sum = expert_sum
                + qwen_next_source_bf16_value(down_weights[row_base + column])
                    * activated[route * intermediate + column];
        }
        routed_sum = routed_sum + expert_sum * selected_weights[route];
    }

    const ulong shared_row_base = (ulong)row * (ulong)intermediate;
    float shared_sum = 0.0f;
    for (uint column = 0u; column < intermediate; ++column) {
        shared_sum = shared_sum
            + qwen_next_source_bf16_value(shared_down_weights[shared_row_base + column])
                * shared_activation[column];
    }
    const float shared_gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
    const float shared_gated = shared_sum * shared_gate;
    const float moe = routed_sum + shared_gated;
    routed_sum_out[row] = routed_sum;
    shared_output_out[row] = shared_sum;
    shared_gated_out[row] = shared_gated;
    output[row] = moe;
    for (uint stream = 0u; stream < streams; ++stream) {
        const float hc_gate = 2.0f / (1.0f + exp(-block_logits[stream] / divisor));
        const ulong offset = (ulong)stream * (ulong)hidden + (ulong)row;
        final_output[offset] = residual[offset] + moe * hc_gate;
    }
}

// Exact-order packed-load sibling for the direct compact MoE/HC epilogue.
// Route accumulation, shared accumulation, and stream-major HC writes retain
// the scalar authority's order; only four adjacent BF16/FP32 values are
// fetched as a unit. It is selected by an explicit A/B control and is not
// implied by HAWKING_FLASH_MOE_GEO, whose SIMD reduction has a different
// numerical association.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_down_shared_direct_hc_vec4(
    device const ushort* down_weights       [[buffer(0)]],
    device const uint* route_ids            [[buffer(1)]],
    device const uint* route_lut             [[buffer(2)]],
    device const float* activated            [[buffer(3)]],
    device const float* selected_weights     [[buffer(4)]],
    device const ushort* shared_down_weights [[buffer(5)]],
    device const float* shared_activation    [[buffer(6)]],
    device const float* shared_gate_logit    [[buffer(7)]],
    device float* routed_sum_out             [[buffer(8)]],
    device float* shared_output_out           [[buffer(9)]],
    device float* shared_gated_out            [[buffer(10)]],
    device float* output                     [[buffer(11)]],
    device const float* residual              [[buffer(12)]],
    device const float* block_logits          [[buffer(13)]],
    device float* final_output                [[buffer(14)]],
    constant uint& compact_experts            [[buffer(15)]],
    constant uint& top_k                      [[buffer(16)]],
    constant uint& intermediate               [[buffer(17)]],
    constant uint& hidden                     [[buffer(18)]],
    constant uint& source_experts             [[buffer(19)]],
    constant uint& streams                    [[buffer(20)]],
    constant float& divisor                   [[buffer(21)]],
    uint row                                  [[thread_position_in_grid]])
{
    if (row >= hidden) return;
    const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
    float routed_sum = 0.0f;
    for (uint route = 0u; route < top_k; ++route) {
        const uint expert = route_ids[route];
        if (expert >= source_experts) continue;
        const uint slot = route_lut[expert];
        if (slot >= compact_experts) continue;
        const ulong row_base = expert_stride * (ulong)slot
            + (ulong)row * (ulong)intermediate;
        const ulong activation_base = (ulong)route * (ulong)intermediate;
        float expert_sum = 0.0f;
        uint column = 0u;
        for (; column + 4u <= intermediate; column += 4u) {
            const ushort4 weights = *(device const ushort4*)(down_weights + row_base + column);
            const float4 x = *(device const float4*)(activated + activation_base + column);
            expert_sum = expert_sum + qwen_next_source_bf16_value(weights.x) * x.x;
            expert_sum = expert_sum + qwen_next_source_bf16_value(weights.y) * x.y;
            expert_sum = expert_sum + qwen_next_source_bf16_value(weights.z) * x.z;
            expert_sum = expert_sum + qwen_next_source_bf16_value(weights.w) * x.w;
        }
        for (; column < intermediate; ++column) {
            expert_sum = expert_sum
                + qwen_next_source_bf16_value(down_weights[row_base + column])
                    * activated[activation_base + column];
        }
        routed_sum = routed_sum + expert_sum * selected_weights[route];
    }

    const ulong shared_row_base = (ulong)row * (ulong)intermediate;
    float shared_sum = 0.0f;
    uint column = 0u;
    for (; column + 4u <= intermediate; column += 4u) {
        const ushort4 weights = *(device const ushort4*)(shared_down_weights + shared_row_base + column);
        const float4 x = *(device const float4*)(shared_activation + column);
        shared_sum = shared_sum + qwen_next_source_bf16_value(weights.x) * x.x;
        shared_sum = shared_sum + qwen_next_source_bf16_value(weights.y) * x.y;
        shared_sum = shared_sum + qwen_next_source_bf16_value(weights.z) * x.z;
        shared_sum = shared_sum + qwen_next_source_bf16_value(weights.w) * x.w;
    }
    for (; column < intermediate; ++column) {
        shared_sum = shared_sum
            + qwen_next_source_bf16_value(shared_down_weights[shared_row_base + column])
                * shared_activation[column];
    }
    const float shared_gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
    const float shared_gated = shared_sum * shared_gate;
    const float moe = routed_sum + shared_gated;
    routed_sum_out[row] = routed_sum;
    shared_output_out[row] = shared_sum;
    shared_gated_out[row] = shared_gated;
    output[row] = moe;
    for (uint stream = 0u; stream < streams; ++stream) {
        const float hc_gate = 2.0f / (1.0f + exp(-block_logits[stream] / divisor));
        const ulong offset = (ulong)stream * (ulong)hidden + (ulong)row;
        final_output[offset] = residual[offset] + moe * hc_gate;
    }
}

// SIMD candidate for the compact MoE/HC epilogue. Four SIMD groups own four
// hidden rows per threadgroup; each lane walks a strided slice of the source
// BF16 columns and simd_sum reduces the row.  The reduction association is
// intentionally non-authoritative, so Rust selects this only through the
// explicit HAWKING_FLASH_MOE_GEO switch after physical parity is measured.
#pragma clang fp contract(off)
kernel void qwen_next_bf16_compact_expert_down_shared_direct_hc_geo_tg128(
    device const ushort* down_weights       [[buffer(0)]],
    device const uint* route_ids            [[buffer(1)]],
    device const uint* route_lut             [[buffer(2)]],
    device const float* activated           [[buffer(3)]],
    device const float* selected_weights    [[buffer(4)]],
    device const ushort* shared_down_weights [[buffer(5)]],
    device const float* shared_activation   [[buffer(6)]],
    device const float* shared_gate_logit   [[buffer(7)]],
    device float* routed_sum_out             [[buffer(8)]],
    device float* shared_output_out          [[buffer(9)]],
    device float* shared_gated_out           [[buffer(10)]],
    device float* output                     [[buffer(11)]],
    device const float* residual              [[buffer(12)]],
    device const float* block_logits          [[buffer(13)]],
    device float* final_output                [[buffer(14)]],
    constant uint& compact_experts            [[buffer(15)]],
    constant uint& top_k                      [[buffer(16)]],
    constant uint& intermediate               [[buffer(17)]],
    constant uint& hidden                     [[buffer(18)]],
    constant uint& source_experts             [[buffer(19)]],
    constant uint& streams                    [[buffer(20)]],
    constant float& divisor                   [[buffer(21)]],
    uint group_id                             [[threadgroup_position_in_grid]],
    uint lane_id                              [[thread_index_in_simdgroup]],
    uint simdgroup_id                         [[simdgroup_index_in_threadgroup]])
{
    const uint row = group_id * 4u + simdgroup_id;
    if (row >= hidden) return;
    const ulong expert_stride = (ulong)hidden * (ulong)intermediate;
    float routed_sum = 0.0f;
    for (uint route = 0u; route < top_k; ++route) {
        const uint expert = route_ids[route];
        float expert_sum = 0.0f;
        if (expert < source_experts) {
            const uint slot = route_lut[expert];
            if (slot < compact_experts) {
                const ulong row_base = expert_stride * (ulong)slot
                    + (ulong)row * (ulong)intermediate;
                for (uint column = lane_id; column < intermediate; column += 32u) {
                    expert_sum = expert_sum
                        + qwen_next_source_bf16_value(down_weights[row_base + column])
                            * activated[route * intermediate + column];
                }
            }
        }
        routed_sum = routed_sum + simd_sum(expert_sum) * selected_weights[route];
    }

    const ulong shared_row_base = (ulong)row * (ulong)intermediate;
    float shared_sum = 0.0f;
    for (uint column = lane_id; column < intermediate; column += 32u) {
        shared_sum = shared_sum
            + qwen_next_source_bf16_value(shared_down_weights[shared_row_base + column])
                * shared_activation[column];
    }
    shared_sum = simd_sum(shared_sum);
    if (lane_id == 0u) {
        const float shared_gate = 1.0f / (1.0f + exp(-shared_gate_logit[0]));
        const float shared_gated = shared_sum * shared_gate;
        const float moe = routed_sum + shared_gated;
        routed_sum_out[row] = routed_sum;
        shared_output_out[row] = shared_sum;
        shared_gated_out[row] = shared_gated;
        output[row] = moe;
        for (uint stream = 0u; stream < streams; ++stream) {
            const float hc_gate = 2.0f / (1.0f + exp(-block_logits[stream] / divisor));
            const ulong offset = (ulong)stream * (ulong)hidden + (ulong)row;
            final_output[offset] = residual[offset] + moe * hc_gate;
        }
    }
}

// Bounded Flash-Next hyperconnection candidate boundary.  The source census
// exposes four 2560-wide streams (10240 values total); this helper injects a
// device-resident 2560-wide shared-expert result into one stream while
// preserving the other candidate state values.  It is intentionally a
// candidate graph primitive, not a claim that the complete model's runtime
// uses this exact stream slot or ordering.
kernel void qwen_next_expand_shared_to_hyper_state(
    device const float* base_state   [[buffer(0)]],
    device const float* shared_output [[buffer(1)]],
    device float* output             [[buffer(2)]],
    constant uint& hidden             [[buffer(3)]],
    constant uint& streams            [[buffer(4)]],
    constant uint& injected_stream    [[buffer(5)]],
    uint id                           [[thread_position_in_grid]])
{
    const uint elements = hidden * streams;
    if (id >= elements) return;
    output[id] = base_state[id];
    const uint injected_start = injected_stream * hidden;
    if (id >= injected_start && id < injected_start + hidden) {
        output[id] = shared_output[id - injected_start];
    }
}

// Candidate low-rank hyperconnection residual mix.  The four source
// block-inject logits gate the low-rank correction per 2560-wide stream; all
// inputs and outputs remain device-resident for this bounded graph.
kernel void qwen_next_hyperconnection_residual_mix_candidate(
    device const float* state         [[buffer(0)]],
    device const float* correction    [[buffer(1)]],
    device const float* block_logits  [[buffer(2)]],
    device float* output              [[buffer(3)]],
    constant uint& hidden              [[buffer(4)]],
    constant uint& streams             [[buffer(5)]],
    uint id                            [[thread_position_in_grid]])
{
    const uint elements = hidden * streams;
    if (id >= elements) return;
    const uint stream = id / hidden;
    const float gate = 1.0f / (1.0f + exp(-block_logits[stream]));
    output[id] = state[id] + correction[id] * gate;
}

// Exact Qwen3.8-Flash-Next HyperConnection read/write primitives.  The
// checkpoint stores hc_norm as BF16 and the reference implementation applies
// it as grouped RMSNorm with the additive Gemma-style scale (1 + weight).
static inline float qwen_next_hc_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

kernel void qwen_next_hyperconnection_grouped_rmsnorm(
    device const float* input       [[buffer(0)]],
    device const ushort* weight_bf16 [[buffer(1)]],
    device float* output             [[buffer(2)]],
    constant uint& hidden             [[buffer(3)]],
    constant uint& streams            [[buffer(4)]],
    constant float& eps               [[buffer(5)]],
    threadgroup float* scratch        [[threadgroup(0)]],
    uint stream                       [[threadgroup_position_in_grid]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]])
{
    // The former implementation launched one thread per element and made
    // every one of the hidden-wide threads recompute the same stream norm.
    // That is O(streams * hidden^2) work.  One TG owns one stream instead:
    // each lane contributes to one reduction, then all lanes write the
    // normalized stream.  The source BF16 widening and (1+w) scale remain
    // unchanged; only the redundant work is removed.
    if (stream >= streams) return;
    float sum = 0.0f;
    const uint stream_start = stream * hidden;
    for (uint index = tid; index < hidden; index += tg_size) {
        const float value = input[stream_start + index];
        sum += value * value;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = rsqrt(scratch[0] / float(hidden) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < hidden; index += tg_size) {
        const uint id = stream_start + index;
        const float scale = 1.0f + qwen_next_hc_bf16_value(weight_bf16[id]);
        output[id] = input[id] * inverse_rms * scale;
    }
}

kernel void qwen_next_hyperconnection_silu_scale(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& elements    [[buffer(2)]],
    constant float& divisor    [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    if (id >= elements) return;
    const float value = input[id] / divisor;
    output[id] = value / (1.0f + exp(-value));
}

kernel void qwen_next_hyperconnection_read_mix(
    device const float* normalized [[buffer(0)]],
    device const float* gate_logits [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant uint& hidden           [[buffer(3)]],
    constant uint& streams          [[buffer(4)]],
    uint id                         [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    float sum = 0.0f;
    for (uint stream = 0u; stream < streams; ++stream) {
        const uint offset = stream * hidden + id;
        const float gate = 1.0f / (1.0f + exp(-gate_logits[offset]));
        sum += gate * normalized[offset];
    }
    output[id] = sum / float(streams);
}

// One-token HyperConnection input organ.  The barriers are deliberate: this
// is a single-threadgroup staged superkernel, not a count-only fusion.  Each
// stage preserves the scalar source-BF16 accumulation order and also writes
// the historical intermediate buffers for parity inspection.
#pragma clang fp contract(off)
kernel void qwen_next_hyperconnection_input_fused(
    device const float* input          [[buffer(0)]],
    device const ushort* norm_weight  [[buffer(1)]],
    device const ushort* down_weight  [[buffer(2)]],
    device const ushort* up_weight    [[buffer(3)]],
    device float* normalized           [[buffer(4)]],
    device float* low_rank             [[buffer(5)]],
    device float* low_rank_activation  [[buffer(6)]],
    device float* gate_logits          [[buffer(7)]],
    device float* output               [[buffer(8)]],
    constant uint& hidden              [[buffer(9)]],
    constant uint& streams             [[buffer(10)]],
    constant uint& low_rank_width      [[buffer(11)]],
    constant float& eps                [[buffer(12)]],
    constant float& divisor            [[buffer(13)]],
    threadgroup float* scratch         [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]])
{
    const uint elements = hidden * streams;
    // Keep one serial reduction per stream so this fused candidate preserves
    // the source left-to-right norm order.  The old implementation repeated
    // that hidden-wide reduction for every output element.  `scratch` holds
    // one sum per stream; the following barriers are real device-memory
    // dependencies between the staged writes.
    if (tid < streams) {
        const uint start = tid * hidden;
        float sum = 0.0f;
        for (uint i = 0u; i < hidden; ++i) {
            const float value = input[start + i];
            sum += value * value;
        }
        scratch[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < elements; id += tg_size) {
        const uint stream = id / hidden;
        const float inv = rsqrt(scratch[stream] / float(hidden) + eps);
        normalized[id] = input[id] * inv * (1.0f + qwen_next_hc_bf16_value(norm_weight[id]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint row = tid; row < low_rank_width; row += tg_size) {
        const ulong base = (ulong)row * (ulong)elements;
        low_rank[row] = qwen_next_source_bf16_dot_vec4(
            down_weight + base, normalized, elements);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < low_rank_width; id += tg_size) {
        const float value = low_rank[id] / divisor;
        low_rank_activation[id] = value / (1.0f + exp(-value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint row = tid; row < elements; row += tg_size) {
        const ulong base = (ulong)row * (ulong)low_rank_width;
        gate_logits[row] = qwen_next_source_bf16_dot_vec4(
            up_weight + base, low_rank_activation, low_rank_width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < hidden; id += tg_size) {
        float sum = 0.0f;
        for (uint stream = 0u; stream < streams; ++stream) {
            const uint offset = stream * hidden + id;
            sum += (1.0f / (1.0f + exp(-gate_logits[offset]))) * normalized[offset];
        }
        output[id] = sum / float(streams);
    }
}

// Input-organ sibling that also evaluates the source BF16 block-injection
// projection while the normalized state is already resident in this
// threadgroup. The block rows are small (`streams` rows), so the final stage
// retains the exact one-thread-per-row source-order GEMV but removes a whole
// dispatch and the intervening normalized-buffer read from the hot graph.
#pragma clang fp contract(off)
kernel void qwen_next_hyperconnection_input_fused_with_block(
    device const float* input          [[buffer(0)]],
    device const ushort* norm_weight  [[buffer(1)]],
    device const ushort* down_weight  [[buffer(2)]],
    device const ushort* up_weight    [[buffer(3)]],
    device float* normalized           [[buffer(4)]],
    device float* low_rank             [[buffer(5)]],
    device float* low_rank_activation  [[buffer(6)]],
    device float* gate_logits          [[buffer(7)]],
    device float* output               [[buffer(8)]],
    device const ushort* block_weight  [[buffer(9)]],
    device float* block_logits         [[buffer(10)]],
    constant uint& hidden              [[buffer(11)]],
    constant uint& streams             [[buffer(12)]],
    constant uint& low_rank_width      [[buffer(13)]],
    constant float& eps                [[buffer(14)]],
    constant float& divisor            [[buffer(15)]],
    threadgroup float* scratch         [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]])
{
    const uint elements = hidden * streams;
    if (tid < streams) {
        const uint start = tid * hidden;
        float sum = 0.0f;
        for (uint i = 0u; i < hidden; ++i) {
            const float value = input[start + i];
            sum += value * value;
        }
        scratch[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < elements; id += tg_size) {
        const uint stream = id / hidden;
        const float inv = rsqrt(scratch[stream] / float(hidden) + eps);
        normalized[id] = input[id] * inv * (1.0f + qwen_next_hc_bf16_value(norm_weight[id]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint row = tid; row < low_rank_width; row += tg_size) {
        const ulong base = (ulong)row * (ulong)elements;
        low_rank[row] = qwen_next_source_bf16_dot_vec4(
            down_weight + base, normalized, elements);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < low_rank_width; id += tg_size) {
        const float value = low_rank[id] / divisor;
        low_rank_activation[id] = value / (1.0f + exp(-value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint row = tid; row < elements; row += tg_size) {
        const ulong base = (ulong)row * (ulong)low_rank_width;
        gate_logits[row] = qwen_next_source_bf16_dot_vec4(
            up_weight + base, low_rank_activation, low_rank_width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < hidden; id += tg_size) {
        float sum = 0.0f;
        for (uint stream = 0u; stream < streams; ++stream) {
            const uint offset = stream * hidden + id;
            sum += (1.0f / (1.0f + exp(-gate_logits[offset]))) * normalized[offset];
        }
        output[id] = sum / float(streams);
    }
    for (uint row = tid; row < streams; row += tg_size) {
        const ulong base = (ulong)row * (ulong)elements;
        block_logits[row] = qwen_next_source_bf16_dot_vec4(
            block_weight + base, normalized, elements);
    }
}

// Stronger Flash MLP-input sibling: keep the newly produced MLP input in
// threadgroup memory long enough to evaluate the source-BF16 router and shared
// scalar, then select/renormalize the routes before leaving the launch. This
// removes both the router producer/consumer boundary and the global
// mlp_input write/read edge. The retained device outputs are still written so
// parity receipts and the following expert kernels keep their established ABI.
#pragma clang fp contract(off)
kernel void qwen_next_hyperconnection_input_fused_with_block_router_topk(
    device const float* input                 [[buffer(0)]],
    device const ushort* norm_weight         [[buffer(1)]],
    device const ushort* down_weight         [[buffer(2)]],
    device const ushort* up_weight           [[buffer(3)]],
    device float* normalized                  [[buffer(4)]],
    device float* low_rank                    [[buffer(5)]],
    device float* low_rank_activation         [[buffer(6)]],
    device float* gate_logits                 [[buffer(7)]],
    device float* output                      [[buffer(8)]],
    device const ushort* block_weight         [[buffer(9)]],
    device float* block_logits                [[buffer(10)]],
    device const ushort* router_weights       [[buffer(11)]],
    device const ushort* shared_scalar_weights [[buffer(12)]],
    device float* router_logits               [[buffer(13)]],
    device float* shared_scalar_output        [[buffer(14)]],
    device uint* route_ids                    [[buffer(15)]],
    device float* route_weights               [[buffer(16)]],
    constant uint& hidden                     [[buffer(17)]],
    constant uint& streams                    [[buffer(18)]],
    constant uint& low_rank_width             [[buffer(19)]],
    constant float& eps                       [[buffer(20)]],
    constant float& divisor                   [[buffer(21)]],
    constant uint& n_experts                  [[buffer(22)]],
    constant uint& top_k                      [[buffer(23)]],
    constant float& tie_epsilon               [[buffer(24)]],
    constant uint& normalize_topk             [[buffer(25)]],
    threadgroup float* scratch                [[threadgroup(0)]],
    uint tid                                  [[thread_index_in_threadgroup]],
    uint tg_size                              [[threads_per_threadgroup]])
{
    const uint elements = hidden * streams;
    const uint stage_offset = streams;
    const uint work_offset = ((stage_offset + hidden + 3u) / 4u) * 4u;
    const uint red_val_offset = work_offset + n_experts;
    const uint red_idx_offset = ((red_val_offset + tg_size + 3u) / 4u) * 4u;
    threadgroup float* output_stage = scratch + stage_offset;
    threadgroup float* work = scratch + work_offset;
    threadgroup float* red_val = scratch + red_val_offset;
    threadgroup uint* red_idx = (threadgroup uint*)(scratch + red_idx_offset);

    if (tid < streams) {
        const uint start = tid * hidden;
        float sum = 0.0f;
        for (uint i = 0u; i < hidden; ++i) {
            const float value = input[start + i];
            sum += value * value;
        }
        scratch[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < elements; id += tg_size) {
        const uint stream = id / hidden;
        const float inv = rsqrt(scratch[stream] / float(hidden) + eps);
        normalized[id] = input[id] * inv * (1.0f + qwen_next_hc_bf16_value(norm_weight[id]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint row = tid; row < low_rank_width; row += tg_size) {
        const ulong base = (ulong)row * (ulong)elements;
        low_rank[row] = qwen_next_source_bf16_dot_vec4(
            down_weight + base, normalized, elements);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < low_rank_width; id += tg_size) {
        const float value = low_rank[id] / divisor;
        low_rank_activation[id] = value / (1.0f + exp(-value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint row = tid; row < elements; row += tg_size) {
        const ulong base = (ulong)row * (ulong)low_rank_width;
        gate_logits[row] = qwen_next_source_bf16_dot_vec4(
            up_weight + base, low_rank_activation, low_rank_width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint id = tid; id < hidden; id += tg_size) {
        float sum = 0.0f;
        for (uint stream = 0u; stream < streams; ++stream) {
            const uint offset = stream * hidden + id;
            sum += (1.0f / (1.0f + exp(-gate_logits[offset]))) * normalized[offset];
        }
        const float value = sum / float(streams);
        output[id] = value;
        output_stage[id] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = tid; row < streams; row += tg_size) {
        const ulong base = (ulong)row * (ulong)elements;
        block_logits[row] = qwen_next_source_bf16_dot_vec4(
            block_weight + base, normalized, elements);
    }

    // Router and shared scalar consume the staged MLP input. Their source
    // left-to-right dot order is identical to the standalone source-BF16
    // router kernel; only the producer buffer changes from device memory to
    // threadgroup memory.
    for (uint expert = tid; expert < n_experts; expert += tg_size) {
        const ulong row_base = (ulong)expert * (ulong)hidden;
        const float acc = qwen_next_source_bf16_dot_vec4_threadgroup(
            router_weights + row_base, output_stage, hidden);
        router_logits[expert] = acc;
        work[expert] = acc;
    }
    if (tid == 0u) {
        shared_scalar_output[0] = qwen_next_source_bf16_dot_vec4_threadgroup(
            shared_scalar_weights, output_stage, hidden);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Keep the exact order-dependent selector for tie windows. All threads
    // enter the branch after the router barrier; only lane zero owns the
    // serial policy and then every lane exits the kernel.
    if (tie_epsilon > 0.0f) {
        if (tid == 0u) {
            float m = -INFINITY;
            for (uint i = 0u; i < n_experts; ++i) {
                if (work[i] > m) m = work[i];
            }
            float sum = 0.0f;
            for (uint i = 0u; i < n_experts; ++i) {
                work[i] = exp(work[i] - m);
                sum += work[i];
            }
            const float inv = 1.0f / sum;
            for (uint i = 0u; i < n_experts; ++i) work[i] *= inv;
            for (uint k = 0u; k < top_k; ++k) {
                uint best_idx = 0u;
                float best_val = -INFINITY;
                for (uint i = 0u; i < n_experts; ++i) {
                    const bool finite_pair = isfinite(best_val) && isfinite(work[i]);
                    const bool tied = finite_pair
                        && abs(work[i] - best_val) <= tie_epsilon;
                    if ((work[i] > best_val && !tied) || (tied && i < best_idx)) {
                        best_val = work[i];
                        best_idx = i;
                    }
                }
                route_ids[k] = best_idx;
                route_weights[k] = best_val;
                work[best_idx] = -INFINITY;
            }
            if (normalize_topk != 0u) {
                float selected_sum = 0.0f;
                for (uint i = 0u; i < top_k; ++i) selected_sum += route_weights[i];
                if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
                    for (uint i = 0u; i < top_k; ++i) route_weights[i] = NAN;
                } else {
                    const float inv_selected = 1.0f / selected_sum;
                    for (uint i = 0u; i < top_k; ++i) route_weights[i] *= inv_selected;
                }
            }
        }
        return;
    }

    // Epsilon-zero path mirrors qwen_next_bf16_router_topk_shared. The max
    // reductions are associative; softmax sums and selected-weight sums stay
    // left-folded on lane zero for the existing authority contract.
    float local = -INFINITY;
    for (uint i = tid; i < n_experts; i += tg_size) {
        local = max(local, work[i]);
    }
    red_val[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red_val[tid] = max(red_val[tid], red_val[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float m = red_val[0];
    for (uint i = tid; i < n_experts; i += tg_size) {
        work[i] = exp(work[i] - m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint i = 0u; i < n_experts; ++i) sum += work[i];
        red_val[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = red_val[0];
    for (uint i = tid; i < n_experts; i += tg_size) {
        work[i] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0u; k < top_k; ++k) {
        float best_val = -INFINITY;
        uint best_idx = 0xFFFFFFFFu;
        for (uint i = tid; i < n_experts; i += tg_size) {
            const float value = work[i];
            if ((value > best_val) || (value == best_val && i < best_idx)) {
                best_val = value;
                best_idx = i;
            }
        }
        red_val[tid] = best_val;
        red_idx[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                const float other_value = red_val[tid + stride];
                const uint other_idx = red_idx[tid + stride];
                if ((other_value > red_val[tid])
                    || (other_value == red_val[tid] && other_idx < red_idx[tid])) {
                    red_val[tid] = other_value;
                    red_idx[tid] = other_idx;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            const uint winner = red_idx[0];
            route_ids[k] = winner;
            route_weights[k] = red_val[0];
            work[winner] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (normalize_topk != 0u && tid == 0u) {
        float selected_sum = 0.0f;
        for (uint i = 0u; i < top_k; ++i) selected_sum += route_weights[i];
        if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
            for (uint i = 0u; i < top_k; ++i) route_weights[i] = NAN;
        } else {
            const float inv_selected = 1.0f / selected_sum;
            for (uint i = 0u; i < top_k; ++i) route_weights[i] *= inv_selected;
        }
    }
}

kernel void qwen_next_hyperconnection_combine(
    device const float* residual    [[buffer(0)]],
    device const float* block_output [[buffer(1)]],
    device const float* block_logits  [[buffer(2)]],
    device float* output              [[buffer(3)]],
    constant uint& hidden              [[buffer(4)]],
    constant uint& streams             [[buffer(5)]],
    constant float& divisor            [[buffer(6)]],
    uint id                            [[thread_position_in_grid]])
{
    const uint elements = hidden * streams;
    if (id >= elements) return;
    const uint stream = id / hidden;
    const float gate = 2.0f / (1.0f + exp(-block_logits[stream] / divisor));
    output[id] = residual[id] + block_output[id % hidden] * gate;
}
