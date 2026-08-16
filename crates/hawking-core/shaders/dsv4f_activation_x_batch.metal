// Batched (M=N) counterparts of the streamed DSV4F authority linears.
//
// Each kernel keeps the same per-row reduction as the M=1 authority matvec
// so a GEMM of N tokens is N independent GEMVs in one dispatch against one
// resident weight buffer. File-local helpers use the dsv4f_ax_ prefix so
// concatenation with matmul.metal / moe.metal cannot collide.

#include <metal_stdlib>
using namespace metal;

static inline float dsv4f_ax_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

static inline ushort dsv4f_ax_bf16_encode_rne(float value)
{
    const uint bits = as_type<uint>(value);
    const uint low_lsb = (bits >> 16u) & 1u;
    return (ushort)((bits + 0x7fffu + low_lsb) >> 16u);
}

static inline float dsv4f_ax_e4m3fn_value(uchar bits)
{
    const uint raw = (uint)bits;
    const uint exponent = (raw >> 3u) & 0x0fu;
    const uint mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) return 0.0f;
    const float magnitude = exponent == 0u
        ? (float)mantissa * 0.001953125f
        : as_type<float>(((exponent + 120u) << 23u) | (mantissa << 20u));
    return (raw & 0x80u) != 0u ? -magnitude : magnitude;
}

static inline float dsv4f_ax_e8m0fnu_value(uchar bits)
{
    if ((uint)bits == 0xffu) return 0.0f;
    return (uint)bits == 0u
        ? as_type<float>(0x00400000u)
        : as_type<float>(((uint)bits) << 23u);
}

static inline uchar dsv4f_ax_act_quant_ue8m0_scale(float amax)
{
    const float clamped_amax = max(amax, 0.0001f);
    const float scaled = clamped_amax * (1.0f / 448.0f);
    const uint raw = as_type<uint>(scaled);
    const int exponent_field = (int)((raw >> 23u) & 0xffu);
    const uint mantissa = raw & 0x007fffffu;
    const int exponent = exponent_field - 127 + (mantissa != 0u ? 1 : 0);
    return (uchar)(exponent + 127);
}

static inline uchar dsv4f_ax_e4m3fn_encode_rne(float value)
{
    if (value == 0.0f) {
        return (as_type<uint>(value) & 0x80000000u) != 0u ? (uchar)0x80u : (uchar)0x00u;
    }
    uchar best_bits = (uchar)0u;
    float best_distance = INFINITY;
    bool found = false;
    for (uint raw = 0u; raw <= 255u; ++raw) {
        const uchar bits = (uchar)raw;
        const uint exponent = (raw >> 3u) & 0x0fu;
        const uint mantissa = raw & 0x07u;
        if (exponent == 0x0fu && mantissa == 0x07u) {
            continue;
        }
        const float candidate = dsv4f_ax_e4m3fn_value(bits);
        const float distance = fabs(candidate - value);
        if (!found || distance < best_distance
            || (distance == best_distance && ((raw & 1u) == 0u)
                && (((uint)best_bits & 1u) != 0u))) {
            best_bits = bits;
            best_distance = distance;
            found = true;
        }
    }
    return best_bits;
}

static inline float dsv4f_ax_e2m1fn_value(uchar packed, bool high_nibble)
{
    const uint nibble = high_nibble ? (((uint)packed >> 4u) & 0x0fu)
                                     : ((uint)packed & 0x0fu);
    float magnitude = 0.0f;
    switch (nibble & 0x07u) {
        case 0u: magnitude = 0.0f; break;
        case 1u: magnitude = 0.5f; break;
        case 2u: magnitude = 1.0f; break;
        case 3u: magnitude = 1.5f; break;
        case 4u: magnitude = 2.0f; break;
        case 5u: magnitude = 3.0f; break;
        case 6u: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
    }
    return (nibble & 0x08u) != 0u ? -magnitude : magnitude;
}

// Act-quant over [batch, cols]. One thread owns (token, 128-wide block).
kernel void dsv4f_ax_act_quant_bf16_ue8m0_batched(
    device const ushort* input_bf16 [[buffer(0)]],
    device       uchar* quantized   [[buffer(1)]],
    device       uchar* act_scales  [[buffer(2)]],
    constant uint& cols              [[buffer(3)]],
    constant uint& batch             [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    if (cols == 0u || (cols % kBlock) != 0u || batch == 0u) return;
    const uint nblocks = cols / kBlock;
    const uint token = tid / nblocks;
    const uint block = tid % nblocks;
    if (token >= batch) return;
    const ulong row_base = (ulong)token * (ulong)cols;
    const uint start = block * kBlock;
    float amax = 0.0f;
    for (uint offset = 0u; offset < kBlock; ++offset) {
        const float value = dsv4f_ax_bf16_value(input_bf16[row_base + (ulong)(start + offset)]);
        amax = max(amax, fabs(value));
    }
    const uchar scale_bits = dsv4f_ax_act_quant_ue8m0_scale(amax);
    const float scale = dsv4f_ax_e8m0fnu_value(scale_bits);
    act_scales[(ulong)token * (ulong)nblocks + (ulong)block] = scale_bits;
    for (uint offset = 0u; offset < kBlock; ++offset) {
        const float value = dsv4f_ax_bf16_value(input_bf16[row_base + (ulong)(start + offset)]);
        const float scaled = clamp(value / scale, -448.0f, 448.0f);
        quantized[row_base + (ulong)(start + offset)] = dsv4f_ax_e4m3fn_encode_rne(scaled);
    }
}

// FP8 GEMM: Y[batch, rows] = Q[batch, cols] @ W[rows, cols]^T with the
// authority 128-wide block reduction (same order as the M=1 matvec).
// Thread owns one (token, output row). Consecutive tokens of one row share
// the weight row in cache.
#pragma clang fp contract(off)
kernel void dsv4f_ax_fp8_e4m3fn_e8m0_gemm(
    device const uchar* weights       [[buffer(0)]],
    device const uchar* weight_scales [[buffer(1)]],
    device const uchar* quantized     [[buffer(2)]],
    device const uchar* act_scales    [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& cols                 [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    constant uint& batch                [[buffer(8)]],
    uint tid                            [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    if (rows == 0u || cols == 0u || batch == 0u || (cols % kBlock) != 0u) return;
    const uint token = tid % batch;
    const uint row = tid / batch;
    if (row >= rows) return;
    const uint scale_row = row / kBlock;
    const ulong weight_base = (ulong)row * (ulong)cols;
    const ulong act_base = (ulong)token * (ulong)cols;
    const ulong act_scale_base = (ulong)token * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * kBlock;
        for (uint offset = 0u; offset < kBlock; ++offset) {
            const uint col = start + offset;
            const float activation = dsv4f_ax_e4m3fn_value(quantized[act_base + (ulong)col]);
            const float weight = dsv4f_ax_e4m3fn_value(weights[weight_base + (ulong)col]);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = dsv4f_ax_e8m0fnu_value(
            act_scales[act_scale_base + (ulong)block]);
        const float weight_scale = dsv4f_ax_e8m0fnu_value(
            weight_scales[(ulong)scale_row * (ulong)scale_cols + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[(ulong)token * (ulong)rows + (ulong)row] = row_accumulator;
}
#pragma clang fp contract(on)

// FP4 GEMM: same 32-K block reduction as the P5B authority matvec.
#pragma clang fp contract(off)
kernel void dsv4f_ax_fp4_e2m1fn_x2_e8m0_gemm(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* weight_scales  [[buffer(1)]],
    device const uchar* quantized      [[buffer(2)]],
    device const uchar* act_scales     [[buffer(3)]],
    device       float* output          [[buffer(4)]],
    constant uint& rows                  [[buffer(5)]],
    constant uint& packed_cols           [[buffer(6)]],
    constant uint& scale_cols            [[buffer(7)]],
    constant uint& batch                 [[buffer(8)]],
    uint tid                             [[thread_position_in_grid]])
{
    constexpr uint kFp4Block = 32u;
    constexpr uint kActBlock = 128u;
    if (rows == 0u || packed_cols == 0u || scale_cols == 0u || batch == 0u
        || packed_cols * 2u != scale_cols * kFp4Block) return;
    const uint token = tid % batch;
    const uint row = tid / batch;
    if (row >= rows) return;
    const uint logical_k = packed_cols * 2u;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    const ulong act_base = (ulong)token * (ulong)logical_k;
    const ulong act_scale_base = (ulong)token * (ulong)(logical_k / kActBlock);
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * kFp4Block;
        for (uint offset = 0u; offset < kFp4Block; ++offset) {
            const uint col = start + offset;
            const uchar packed = packed_weights[weight_base + (ulong)(col >> 1u)];
            const float activation = dsv4f_ax_e4m3fn_value(quantized[act_base + (ulong)col]);
            const float weight = dsv4f_ax_e2m1fn_value(packed, (col & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = dsv4f_ax_e8m0fnu_value(
            act_scales[act_scale_base + (ulong)(block / (kActBlock / kFp4Block))]);
        const float weight_scale = dsv4f_ax_e8m0fnu_value(
            weight_scales[scale_base + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[(ulong)token * (ulong)rows + (ulong)row] = row_accumulator;
}
#pragma clang fp contract(on)

// Gate GEMM: Y[batch, 256] = X[batch, 4096] @ W[256, 4096]^T. Same serial
// BF16 reduction as the P6A authority matvec, so top-6 membership is stable.
#pragma clang fp contract(off)
kernel void dsv4f_ax_gate_bf16_gemm(
    device const ushort* gate_weight_bf16 [[buffer(0)]],
    device const ushort* input_bf16       [[buffer(1)]],
    device       float* logits_f32         [[buffer(2)]],
    constant uint& rows                    [[buffer(3)]],
    constant uint& cols                    [[buffer(4)]],
    constant uint& batch                   [[buffer(5)]],
    uint tid                               [[thread_position_in_grid]])
{
    if (rows == 0u || cols == 0u || batch == 0u) return;
    const uint token = tid % batch;
    const uint row = tid / batch;
    if (row >= rows) return;
    float accumulator = 0.0f;
    const ulong wbase = (ulong)row * (ulong)cols;
    const ulong xbase = (ulong)token * (ulong)cols;
    for (uint col = 0u; col < cols; ++col) {
        accumulator = accumulator
            + dsv4f_ax_bf16_value(input_bf16[xbase + (ulong)col])
            * dsv4f_ax_bf16_value(gate_weight_bf16[wbase + (ulong)col]);
    }
    logits_f32[(ulong)token * (ulong)rows + (ulong)row] = accumulator;
}
#pragma clang fp contract(on)

// Batched WO-A: per-token [8, 4096] attention against the converted BF16
// weights. Same convert-then-dot as the P4A authority einsum.
kernel void dsv4f_ax_wo_a_convert_bf16_einsum_batched(
    device const uchar* raw_weights [[buffer(0)]],
    device const uchar* weight_scales [[buffer(1)]],
    device const ushort* attention_bf16 [[buffer(2)]],
    device       ushort* output_bf16 [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& scale_cols [[buffer(6)]],
    constant uint& ranks_per_group [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    if (rows == 0u || cols == 0u || batch == 0u || ranks_per_group == 0u
        || (cols % kBlock) != 0u || scale_cols != cols / kBlock) return;
    const uint token = tid % batch;
    const uint row = tid / batch;
    if (row >= rows) return;
    const uint group = row / ranks_per_group;
    // attention is [batch, groups, cols]; groups = rows / ranks_per_group.
    const ulong attn_stride = (ulong)(rows / ranks_per_group) * (ulong)cols;
    const ulong token_input = (ulong)token * attn_stride + (ulong)group * (ulong)cols;
    const ulong weight_base = (ulong)row * (ulong)cols;
    const uint scale_row = row / kBlock;
    float accumulator = 0.0f;
    for (uint column = 0u; column < cols; ++column) {
        const float raw_weight = dsv4f_ax_e4m3fn_value(raw_weights[weight_base + (ulong)column]);
        const float scale = dsv4f_ax_e8m0fnu_value(
            weight_scales[(ulong)scale_row * (ulong)scale_cols + (ulong)(column / kBlock)]);
        const float converted_bf16 = dsv4f_ax_bf16_value(
            dsv4f_ax_bf16_encode_rne(raw_weight * scale));
        accumulator = accumulator
            + dsv4f_ax_bf16_value(attention_bf16[token_input + (ulong)column]) * converted_bf16;
    }
    output_bf16[(ulong)token * (ulong)rows + (ulong)row] = dsv4f_ax_bf16_encode_rne(accumulator);
}

// Inverse of dsv4f_pack_worklist: given N tokens × top-6 selected ids,
// emit a compact CSR of (expert -> token rows). One thread. Host can also
// build this; the kernel exists so the capture path does not invent a
// second worklist grammar.
struct Dsv4fAxExpertCsrHeader {
    uint expert_count;
    uint token_count;
    uint top_k;
    uint nonempty;
};

kernel void dsv4f_ax_pack_expert_csr(
    device const uint* selected_ids      [[buffer(0)]], // [batch * top_k]
    device const float* selected_weights [[buffer(1)]], // [batch * top_k]
    device       uint* expert_counts     [[buffer(2)]], // [expert_count]
    device       uint* expert_offsets    [[buffer(3)]], // [expert_count + 1]
    device       uint* packed_tokens     [[buffer(4)]], // [batch * top_k]
    device       float* packed_weights   [[buffer(5)]], // [batch * top_k]
    device       uint* valid             [[buffer(6)]],
    constant uint& batch                  [[buffer(7)]],
    constant uint& top_k                  [[buffer(8)]],
    constant uint& expert_count           [[buffer(9)]],
    uint index                            [[thread_position_in_grid]])
{
    if (index != 0u) return;
    valid[0] = 0u;
    if (batch == 0u || top_k != 6u || expert_count == 0u) {
        valid[0] = 2u;
        return;
    }
    for (uint e = 0u; e < expert_count; ++e) {
        expert_counts[e] = 0u;
    }
    const uint nassign = batch * top_k;
    for (uint i = 0u; i < nassign; ++i) {
        const uint id = selected_ids[i];
        if (id >= expert_count || !isfinite(selected_weights[i])) {
            valid[0] = 16u;
            return;
        }
        expert_counts[id] += 1u;
    }
    uint running = 0u;
    uint nonempty = 0u;
    expert_offsets[0] = 0u;
    for (uint e = 0u; e < expert_count; ++e) {
        if (expert_counts[e] != 0u) nonempty += 1u;
        running += expert_counts[e];
        expert_offsets[e + 1u] = running;
    }
    for (uint e = 0u; e < expert_count; ++e) {
        expert_counts[e] = 0u;
    }
    for (uint token = 0u; token < batch; ++token) {
        for (uint slot = 0u; slot < top_k; ++slot) {
            const uint i = token * top_k + slot;
            const uint id = selected_ids[i];
            const uint dest = expert_offsets[id] + expert_counts[id];
            packed_tokens[dest] = token;
            packed_weights[dest] = selected_weights[i];
            expert_counts[id] += 1u;
        }
    }
    valid[0] = 1u;
    (void)nonempty;
}
