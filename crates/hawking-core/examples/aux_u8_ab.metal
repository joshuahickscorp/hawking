// Native u8-aux consumer vs incumbent f16-aux affine-Q2, same geo_tpr64 geometry.
//
// Production inner loop is copied from alu_roofline_organs.metal /
// q80_mixed_decode.metal. This file is NOT bound on the decode path.
//
// Incumbent: device half* scale/bias, half-to-float in-register, then
//            w = float(q)*scale + bias; acc += w * x.
// Native:    device uchar* scale/bias, log-scale exp + linear bias
//            decoded in-register from the u8 code. 2-bit codes kept.
//            There is no device half aux buffer and no store of a
//            decoded scale/bias array. Expanding u8 back to f16 aux
//            and then calling the incumbent kernel is the forbidden
//            shape and is not implemented here.

#include <metal_stdlib>
using namespace metal;

struct AuxU8Endpoints {
    float scale_lmin;
    float scale_span;  // (lmax - lmin) / 255
    float bias_min;
    float bias_span;
};

static inline float affine_q2_unpack8(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        const float w = float(q) * scale + bias;
        sum += w * x[col + i];
    }
    return sum;
}

static inline float affine_q2_geo_acc_g64(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const float scale = float(scales[rgb]);
        const float bias = float(biases[rgb]);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

kernel void aux_u8_incumbent_affine_q2_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && group_size == 64u && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// In-register u8 aux decode. scale_u8 and bias_u8 are never widened to a
// device half array. Each inner iteration loads one uchar pair and turns
// them into two floats in registers, then the production dequant-FMA.
static inline float affine_q2_geo_acc_g64_u8(
    device const uchar* codes,
    device const uchar* scales_u8,
    device const uchar* biases_u8,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row,
    AuxU8Endpoints ep)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uchar s = scales_u8[rgb];
        const uchar b = biases_u8[rgb];
        // Loads of s,b must survive even when a span is 0 (s*0 still uses s).
        const float scale = exp(ep.scale_lmin + float(s) * ep.scale_span);
        const float bias = ep.bias_min + float(b) * ep.bias_span;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

kernel void aux_u8_native_affine_q2_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const uchar* scales_u8   [[buffer(1)]],
    device const uchar* biases_u8   [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant AuxU8Endpoints& ep     [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && group_size == 64u && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64_u8(
            codes, scales_u8, biases_u8, input, row, cols, lane_in_row, ep);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}
