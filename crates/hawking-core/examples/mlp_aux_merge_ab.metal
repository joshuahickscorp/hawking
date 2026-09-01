// Matched A/B: production MLP matvec vs merged 4-byte scale+bias aux.
//
// ARM A is the production kernel body copied from
// crates/hawking-core/shaders/q80_mixed_decode.metal
//   qwen_affine_q2_group32_matvec_geo_tpr64_tg128
//   + affine_q2_unpack8 / affine_q2_geo_acc_g32 / affine_q2_geo_acc_g64.
// Two planes: scales at buffer(1), biases at buffer(2).
//
// ARM B is that same loop with the two 2-byte broadcast aux streams merged
// into one device const half2* (one 4-byte load per group). Host interleaves
// the existing planes in memory; nothing is rewritten under models/.
//
// Probe-only. Not bound on the decode path. Production shaders are not modified.
//
// Counted payload per thread-iteration is 38 B on both arms:
//   A: 2 (codes) + 2 (scale) + 2 (bias) + 32 (x)
//   B: 2 (codes) + 4 (half2) + 32 (x)

#include <metal_stdlib>
using namespace metal;

static inline bool affine_q2_group_ok(uint group_size, uint cols) {
    return (group_size == 32u || group_size == 64u) && (cols % group_size) == 0u;
}

// Production 8-weight tile. Association: (q*scale+bias)*x per element, sequential +=.
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

static inline float affine_q2_geo_acc_g32(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 5u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 5u;
        const uint local = col & 31u;
        const uint rgb = row * groups_per_row + group;
        const float scale = float(scales[rgb]);
        const float bias = float(biases[rgb]);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 8u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
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

// ARM B: same loop, one half2 load of {scale, bias} instead of two half loads.
static inline float affine_q2_geo_acc_g32_merged(
    device const uchar* codes,
    device const half2* scale_bias,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 5u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 5u;
        const uint local = col & 31u;
        const uint rgb = row * groups_per_row + group;
        const half2 sb = scale_bias[rgb];
        const float scale = float(sb.x);
        const float bias = float(sb.y);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 8u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

static inline float affine_q2_geo_acc_g64_merged(
    device const uchar* codes,
    device const half2* scale_bias,
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
        const half2 sb = scale_bias[rgb];
        const float scale = float(sb.x);
        const float bias = float(sb.y);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

// ARM A — incumbent. Production kernel body verbatim, two aux planes.
kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128(
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
    if (row < rows && affine_q2_group_ok(group_size, cols)) {
        if (group_size == 32u) {
            acc = affine_q2_geo_acc_g32(codes, scales, biases, input, row, cols, lane_in_row);
        } else {
            acc = affine_q2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
        }
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

// ARM B — merged aux. Same geometry, same unpack, one 4-byte group record.
kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128_aux_merge(
    device const uchar* codes       [[buffer(0)]],
    device const half2* scale_bias  [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
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
    if (row < rows && affine_q2_group_ok(group_size, cols)) {
        if (group_size == 32u) {
            acc = affine_q2_geo_acc_g32_merged(codes, scale_bias, input, row, cols, lane_in_row);
        } else {
            acc = affine_q2_geo_acc_g64_merged(codes, scale_bias, input, row, cols, lane_in_row);
        }
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
