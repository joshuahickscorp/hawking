// G023 decode family: ONE source for Qwen (Q80 + Qwen3.8) and DSV4F.
//
// Function constants specialize codec / tile / I/O. Unused branches compile
// out. Do not put a codec switch inside the FMA — each kCodec keeps the
// inner loop that the reuse matrix classified as its association.
//
// STRUCTURAL leftovers stay in their original files: rice CSR, hgravs
// two-stage, MHC / sparse attn, act-quant, FP8, uniform-q4 (de-authorised).

#include <metal_stdlib>
using namespace metal;

// ── function constants ────────────────────────────────────────────────────
// 0 = SERIAL (1 thread / row), 1 = SIMD (1 SG / row), 2 = ROWBLOCK4
constant uint kGkTile [[function_constant(0)]];
constant uint gk_tile = is_function_constant_defined(kGkTile) ? kGkTile : 0u;

// 0 = BINARY_GROUP, 1 = HGRAVS01, 2 = FP4_E2M1
constant uint kGkCodec [[function_constant(1)]];
constant uint gk_codec = is_function_constant_defined(kGkCodec) ? kGkCodec : 0u;

// 0 = F32, 1 = BF16
constant uint kGkIo [[function_constant(2)]];
constant uint gk_io = is_function_constant_defined(kGkIo) ? kGkIo : 0u;

// 0 = no clamp, 1 = DSV4F (gate<=10, up in [-10,10])
constant uint kGkClamp [[function_constant(3)]];
constant uint gk_clamp = is_function_constant_defined(kGkClamp) ? kGkClamp : 0u;

// 0 = no route weight, 1 = per-slot worklist
constant uint kGkRoute [[function_constant(4)]];
constant uint gk_route = is_function_constant_defined(kGkRoute) ? kGkRoute : 0u;

// Worklist width. DSV4F = 6, Q80 = 10. Default 6 so DSV4F wrappers
// compile without a specialization dictionary.
constant uint kGkWorklistK [[function_constant(5)]];
constant uint gk_k = is_function_constant_defined(kGkWorklistK) ? kGkWorklistK : 6u;

constant constexpr uint GK_CODEC_BINARY = 0u;
constant constexpr uint GK_CODEC_HGRAVS = 1u;
constant constexpr uint GK_CODEC_FP4 = 2u;
constant constexpr uint GK_TILE_SERIAL = 0u;
constant constexpr uint GK_TILE_SIMD = 1u;
constant constexpr uint GK_TILE_R4 = 2u;
constant constexpr uint GK_FP4_BLOCK = 32u;
constant constexpr uint GK_ACT_BLOCK = 128u;
constant constexpr uint GK_K_MAX = 10u;

// ── IDENTICAL codecs ──────────────────────────────────────────────────────

static inline float gk_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

static inline ushort gk_bf16_encode_rne(float value)
{
    const uint bits = as_type<uint>(value);
    const uint low_lsb = (bits >> 16u) & 1u;
    return (ushort)((bits + 0x7fffu + low_lsb) >> 16u);
}

static inline float gk_e4m3fn_value(uchar bits)
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

static inline float gk_e8m0fnu_value(uchar bits)
{
    if ((uint)bits == 0xffu) return 0.0f;
    return (uint)bits == 0u
        ? as_type<float>(0x00400000u)
        : as_type<float>(((uint)bits) << 23u);
}

static inline float gk_e2m1fn_value(uchar packed, bool high_nibble)
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

static inline float gk_silu_f32(float value)
{
    return value / (1.0f + exp(-value));
}

static inline float gk_silu_dsv4f(float value)
{
    if (value >= 0.0f) return value / (1.0f + exp(-value));
    const float e = exp(value);
    return value * e / (1.0f + e);
}

// Q80 / Q38 binary_group: 1-bit LSB signs + fp16 scale / group.
// Copied from q80_mixed_decode.metal so shipping association is unchanged.

static inline float gk_binary_group_serial_row(
    device const uchar* signs,
    device const half* scales,
    device const float* input,
    uint row,
    uint cols,
    uint group_size,
    uint groups_per_row)
{
    float sum = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint group_start = group * group_size;
        const uint group_end = min(group_start + group_size, cols);
        const float scale = float(scales[scale_base + group]);
        uint col = group_start;
        while (col < group_end && ((row_base + col) & 7u) != 0u) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
        while (col + 8u <= group_end) {
            const uchar byte = signs[(row_base + col) >> 3u];
            sum += ((byte & 0x01u) ? scale : -scale) * input[col];
            sum += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
            sum += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
            sum += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
            sum += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
            sum += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
            sum += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
            sum += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
            col += 8u;
        }
        while (col < group_end) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
    }
    return sum;
}

static inline float gk_binary_lane_term(
    device const uchar* signs,
    device const half* scales,
    device const float* input,
    uint row_base,
    uint scale_base,
    uint col,
    uint group_size)
{
    const float scale = float(scales[scale_base + col / group_size]);
    const uint flat = row_base + col;
    const uchar byte = signs[flat >> 3u];
    const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
    return (positive ? scale : -scale) * input[col];
}

// Byte/shift of bit 0 of `element` packed LSB-first at `bits` bits.
// Equivalent to (element * bits) >> 3 and (element * bits) & 7, but the
// factors stay in uint32 for bits in 1..=8 and any uint element (every
// tensor that fits this kernel signature). Cost vs the wrapping mul:
// +1 32-bit mul, +1 add, +1 and, +1 shr. No 64-bit regs.
static inline uint gk_packed_lsb_byte(uint element, uint bits)
{
    const uint r = element & 7u;
    return (element >> 3u) * bits + ((r * bits) >> 3u);
}

static inline uint gk_packed_lsb_shift(uint element, uint bits)
{
    return ((element & 7u) * bits) & 7u;
}

static inline uint gk_uniform_extract_wide(
    device const uchar* codes,
    uint element,
    uint bits)
{
    const uint byte0 = gk_packed_lsb_byte(element, bits);
    const uint shift = gk_packed_lsb_shift(element, bits);
    uint packed = uint(codes[byte0]);
    if (shift + bits > 8u) {
        packed |= uint(codes[byte0 + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

static inline float gk_uniform_value_wide(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    const uint group = element / group_size;
    const uint code = gk_uniform_extract_wide(codes, element, bits);
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
}

static inline uint gk_uniform_extract(
    device const uchar* codes,
    uint element,
    uint bits)
{
    uint byte_i = gk_packed_lsb_byte(element, bits);
    uint sh = gk_packed_lsb_shift(element, bits);
    uint value = 0u;
    for (uint b = 0u; b < bits; ++b) {
        const uchar byte = codes[byte_i];
        value |= ((uint(byte) >> sh) & 1u) << b;
        sh += 1u;
        if (sh == 8u) {
            sh = 0u;
            byte_i += 1u;
        }
    }
    return value;
}

static inline float gk_uniform_value(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    const uint group = element / group_size;
    const uint code = gk_uniform_extract(codes, element, bits);
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
}

// DSV4F FP4 isolated row. Copied from dsv4f_fp4_matvec_split.

static inline float gk_fp4_serial_row(
    device const uchar* packed_weights,
    device const uchar* weight_scales,
    device const uchar* quantized,
    device const uchar* act_scales,
    uint row,
    uint packed_cols,
    uint scale_cols)
{
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * GK_FP4_BLOCK;
        for (uint offset = 0u; offset < GK_FP4_BLOCK; ++offset) {
            const uint col = start + offset;
            const uchar packed = packed_weights[weight_base + (ulong)(col >> 1u)];
            const float activation = gk_e4m3fn_value(quantized[col]);
            const float weight = gk_e2m1fn_value(packed, (col & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = gk_e8m0fnu_value(
            act_scales[block / (GK_ACT_BLOCK / GK_FP4_BLOCK)]);
        const float weight_scale = gk_e8m0fnu_value(
            weight_scales[scale_base + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    return row_accumulator;
}

// Q80/Q38 GQA geometry: same algebra, two admitted (heads, kv, theta) pairs.

static inline bool gk_gqa_geometry_ok(
    uint n_heads,
    uint n_kv_heads,
    uint head_dim,
    uint rotary_dim,
    float rope_theta,
    float rms_epsilon)
{
    if (head_dim != 256u || rotary_dim != 64u || rms_epsilon != 1.0e-6f
        || n_kv_heads == 0u || (n_heads % n_kv_heads) != 0u) {
        return false;
    }
    const bool q80 = n_heads == 16u && n_kv_heads == 2u
        && rope_theta == 5000000.0f;
    const bool q38 = n_heads == 24u && n_kv_heads == 4u
        && rope_theta == 10000000.0f;
    return q80 || q38;
}

// ── worklist ABI (matches dsv4f_native_token_graph.metal) ─────────────────

struct GkWorklistEntry {
    uint expert_id;
    uint slab_slot;
    float route_weight;
    uint ready;
};

static_assert(sizeof(GkWorklistEntry) == 16, "GkWorklistEntry ABI drift");

struct GkExpertRef {
    device const uchar* packed_weights;
    device const uchar* weight_scales;
};

static_assert(sizeof(GkExpertRef) == 16, "GkExpertRef ABI drift");

static inline float gk_fp4_worklist_serial_row(
    device const GkWorklistEntry* worklist,
    device const GkExpertRef* refs,
    device const uchar* quantized,
    device const uchar* act_scales,
    uint slot,
    uint row,
    uint packed_cols,
    uint scale_cols,
    uint act_is_per_slot)
{
    const GkWorklistEntry entry = worklist[slot];
    if (entry.ready != 1u || entry.slab_slot >= gk_k) return 0.0f;
    const GkExpertRef ref = refs[entry.slab_slot];
    if (ref.packed_weights == nullptr || ref.weight_scales == nullptr) return 0.0f;

    const uint logical_k = packed_cols * 2u;
    const ulong act_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)logical_k
        : 0ul;
    const ulong act_scale_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)(logical_k / GK_ACT_BLOCK)
        : 0ul;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * GK_FP4_BLOCK;
        for (uint offset = 0u; offset < GK_FP4_BLOCK; ++offset) {
            const uint col = start + offset;
            const uchar packed = ref.packed_weights[weight_base + (ulong)(col >> 1u)];
            const float activation = gk_e4m3fn_value(quantized[act_base + (ulong)col]);
            const float weight = gk_e2m1fn_value(packed, (col & 1u) != 0u);
            block_accumulator = block_accumulator + activation * weight;
        }
        const float activation_scale = gk_e8m0fnu_value(
            act_scales[act_scale_base + (ulong)(block / (GK_ACT_BLOCK / GK_FP4_BLOCK))]);
        const float weight_scale = gk_e8m0fnu_value(
            ref.weight_scales[scale_base + (ulong)block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    return row_accumulator;
}

// ── family entry points ───────────────────────────────────────────────────
// Named kernels keep the existing pipeline() cache. Function constants
// default so get_function(name, None) is the shipping specialization.

// Q80 / Q38 gate_proj and the binary half of up_proj.
kernel void gk_matvec_binary(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint row                         [[thread_position_in_grid]])
{
    if (gk_tile != GK_TILE_SERIAL) return;
    if (row >= rows) return;
    output[row] = gk_binary_group_serial_row(
        signs, scales, input, row, cols, group_size, groups_per_row);
}

kernel void gk_matvec_binary_simd(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;
    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        partial += gk_binary_lane_term(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

// Q80 / Q38 hgravs01 factor (down_proj L or R, or uniform-n).
kernel void gk_matvec_hgravs(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float sum = 0.0f;
    const uint row_base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        sum += gk_uniform_value(codes, scales, row_base + col, group_size, bits, bound)
            * input[col];
    }
    output[row] = sum;
}

kernel void gk_matvec_hgravs_simd(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;
    float partial = 0.0f;
    const uint row_base = row * cols;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        partial += gk_uniform_value_wide(
            codes, scales, row_base + col, group_size, bits, bound) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

// Isolated DSV4F FP4 (same association as one worklist slot).
#pragma clang fp contract(off)
kernel void gk_matvec_fp4(
    device const uchar* packed_weights [[buffer(0)]],
    device const uchar* weight_scales  [[buffer(1)]],
    device const uchar* quantized      [[buffer(2)]],
    device const uchar* act_scales     [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& packed_cols          [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    uint row                            [[thread_position_in_grid]])
{
    if (rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * GK_FP4_BLOCK) {
        return;
    }
    if (row >= rows) return;
    output[row] = gk_fp4_serial_row(
        packed_weights, weight_scales, quantized, act_scales,
        row, packed_cols, scale_cols);
}
#pragma clang fp contract(on)

#pragma clang fp contract(off)
kernel void gk_worklist_fp4(
    device const GkWorklistEntry* worklist [[buffer(0)]],
    device const GkExpertRef* refs         [[buffer(1)]],
    device const uchar* quantized          [[buffer(2)]],
    device const uchar* act_scales         [[buffer(3)]],
    device       float* output             [[buffer(4)]],
    constant uint& rows                     [[buffer(5)]],
    constant uint& packed_cols              [[buffer(6)]],
    constant uint& scale_cols               [[buffer(7)]],
    constant uint& top_k                    [[buffer(8)]],
    constant uint& act_is_per_slot          [[buffer(9)]],
    uint tid                                [[thread_position_in_grid]])
{
    if (top_k != gk_k || rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * GK_FP4_BLOCK) {
        return;
    }
    const uint slot = tid / rows;
    const uint row = tid % rows;
    if (slot >= gk_k) return;
    output[(ulong)slot * (ulong)rows + (ulong)row] = gk_fp4_worklist_serial_row(
        worklist, refs, quantized, act_scales,
        slot, row, packed_cols, scale_cols, act_is_per_slot);
}
#pragma clang fp contract(on)

#pragma clang fp contract(off)
kernel void gk_worklist_fp4_simd(
    device const GkWorklistEntry* worklist [[buffer(0)]],
    device const GkExpertRef* refs         [[buffer(1)]],
    device const uchar* quantized          [[buffer(2)]],
    device const uchar* act_scales         [[buffer(3)]],
    device       float* output             [[buffer(4)]],
    constant uint& rows                     [[buffer(5)]],
    constant uint& packed_cols              [[buffer(6)]],
    constant uint& scale_cols               [[buffer(7)]],
    constant uint& top_k                    [[buffer(8)]],
    constant uint& act_is_per_slot          [[buffer(9)]],
    uint group_id                           [[threadgroup_position_in_grid]],
    uint simd_lane                          [[thread_index_in_simdgroup]],
    uint simd_id                            [[simdgroup_index_in_threadgroup]])
{
    if (top_k != gk_k || rows == 0u || packed_cols == 0u
        || packed_cols * 2u != scale_cols * GK_FP4_BLOCK) {
        return;
    }
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint groups_per_slot = (rows + kSimdgroupsPerThreadgroup - 1u)
        / kSimdgroupsPerThreadgroup;
    if (groups_per_slot == 0u) return;
    const uint slot = group_id / groups_per_slot;
    const uint row = (group_id % groups_per_slot) * kSimdgroupsPerThreadgroup + simd_id;
    if (slot >= gk_k || row >= rows) return;
    const GkWorklistEntry entry = worklist[slot];
    if (entry.ready != 1u || entry.slab_slot >= gk_k) return;
    const GkExpertRef ref = refs[entry.slab_slot];
    if (ref.packed_weights == nullptr || ref.weight_scales == nullptr) return;

    const uint logical_k = packed_cols * 2u;
    const ulong act_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)logical_k
        : 0ul;
    const ulong act_scale_base = act_is_per_slot != 0u
        ? (ulong)entry.slab_slot * (ulong)(logical_k / GK_ACT_BLOCK)
        : 0ul;
    const ulong weight_base = (ulong)row * (ulong)packed_cols;
    const ulong scale_base = (ulong)row * (ulong)scale_cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        const uint col = block * GK_FP4_BLOCK + simd_lane;
        const uchar packed = ref.packed_weights[weight_base + (ulong)(col >> 1u)];
        const float activation = gk_e4m3fn_value(quantized[act_base + (ulong)col]);
        const float weight = gk_e2m1fn_value(packed, (col & 1u) != 0u);
        float block_accumulator = activation * weight;
        block_accumulator = simd_sum(block_accumulator);
        if (simd_lane == 0u) {
            const float activation_scale = gk_e8m0fnu_value(
                act_scales[act_scale_base + (ulong)(block / (GK_ACT_BLOCK / GK_FP4_BLOCK))]);
            const float weight_scale = gk_e8m0fnu_value(
                ref.weight_scales[scale_base + (ulong)block]);
            row_accumulator = row_accumulator
                + block_accumulator * (activation_scale * weight_scale);
        }
    }
    if (simd_lane == 0u) {
        output[(ulong)slot * (ulong)rows + (ulong)row] = row_accumulator;
    }
}
#pragma clang fp contract(on)

// Q80 / Q38 SwiGLU (no clamp, no route).
kernel void gk_swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* output     [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    if (id >= n) return;
    output[id] = gk_silu_f32(gate[id]) * up[id];
}

// DSV4F worklist SwiGLU (clamp + route_weight + bf16).
kernel void gk_swiglu_bf16_worklist(
    device const GkWorklistEntry* worklist [[buffer(0)]],
    device const ushort* gate_bf16         [[buffer(1)]],
    device const ushort* up_bf16           [[buffer(2)]],
    device       ushort* output_bf16       [[buffer(3)]],
    constant uint& width                    [[buffer(4)]],
    constant uint& top_k                    [[buffer(5)]],
    uint tid                                [[thread_position_in_grid]])
{
    if (top_k != gk_k || width == 0u) return;
    const uint slot = tid / width;
    const uint index = tid % width;
    if (slot >= gk_k) return;
    const GkWorklistEntry entry = worklist[slot];
    if (entry.ready != 1u) return;
    const ulong off = (ulong)slot * (ulong)width + (ulong)index;
    const float gate = min(gk_bf16_value(gate_bf16[off]), 10.0f);
    const float up = clamp(gk_bf16_value(up_bf16[off]), -10.0f, 10.0f);
    output_bf16[off] = gk_bf16_encode_rne(
        gk_silu_dsv4f(gate) * up * entry.route_weight);
}

// Source-order combine: y = sum_k routed[k] + shared.
#pragma clang fp contract(off)
kernel void gk_combine_bf16(
    device const ushort* routed_bf16 [[buffer(0)]],
    device const ushort* shared_bf16 [[buffer(1)]],
    device       ushort* output_bf16 [[buffer(2)]],
    constant uint& hidden             [[buffer(3)]],
    constant uint& top_k              [[buffer(4)]],
    uint index                        [[thread_position_in_grid]])
{
    if (index >= hidden || top_k != gk_k) return;
    float value = 0.0f;
    for (uint slot = 0u; slot < gk_k; ++slot) {
        value = value + gk_bf16_value(
            routed_bf16[(ulong)slot * (ulong)hidden + (ulong)index]);
    }
    value = value + gk_bf16_value(shared_bf16[index]);
    output_bf16[index] = gk_bf16_encode_rne(value);
}
#pragma clang fp contract(on)

// Sort selected (id, weight) into execution order. K is a function constant
// so the stack arrays have a compile-time size (6 or 10).
#pragma clang fp contract(off)
kernel void gk_pack_worklist(
    device const uint* selected_ids      [[buffer(0)]],
    device const float* selected_weights [[buffer(1)]],
    device       GkWorklistEntry* worklist [[buffer(2)]],
    device       uint* valid             [[buffer(3)]],
    constant uint& top_k                 [[buffer(4)]],
    constant uint& expert_count          [[buffer(5)]],
    uint index                           [[thread_position_in_grid]])
{
    if (index != 0u) return;
    valid[0] = 0u;
    if (top_k != gk_k || expert_count == 0u || gk_k == 0u || gk_k > GK_K_MAX) {
        valid[0] = 2u;
        return;
    }
    uint ids[GK_K_MAX];
    float weights[GK_K_MAX];
    uint slots[GK_K_MAX];
    for (uint slot = 0u; slot < gk_k; ++slot) {
        const uint id = selected_ids[slot];
        if (id >= expert_count || !isfinite(selected_weights[slot])) {
            valid[0] = 16u + slot;
            return;
        }
        ids[slot] = id;
        weights[slot] = selected_weights[slot];
        slots[slot] = slot;
    }
    for (uint a = 0u; a < gk_k; ++a) {
        uint best = a;
        for (uint b = a + 1u; b < gk_k; ++b) {
            if (ids[b] < ids[best] || (ids[b] == ids[best] && slots[b] < slots[best])) {
                best = b;
            }
        }
        if (best != a) {
            const uint tid = ids[a];
            ids[a] = ids[best];
            ids[best] = tid;
            const float tw = weights[a];
            weights[a] = weights[best];
            weights[best] = tw;
            const uint ts = slots[a];
            slots[a] = slots[best];
            slots[best] = ts;
        }
    }
    for (uint i = 1u; i < gk_k; ++i) {
        if (ids[i] == ids[i - 1u]) {
            valid[0] = 32u + i;
            return;
        }
    }
    for (uint slot = 0u; slot < gk_k; ++slot) {
        worklist[slot].expert_id = ids[slot];
        worklist[slot].slab_slot = slot;
        worklist[slot].route_weight = weights[slot];
        worklist[slot].ready = 1u;
    }
    valid[0] = 1u;
}
#pragma clang fp contract(on)
