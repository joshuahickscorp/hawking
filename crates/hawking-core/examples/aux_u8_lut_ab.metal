// LUT-decode u8-aux consumer vs incumbent f16-aux and vs the exp-variant.
// Same geo_tpr64 geometry as aux_u8_ab.metal. Production shaders untouched.
//
// A u8 has 256 values. The log-scale exp and the linear bias are functions
// of one byte: two 256-entry float tables, 2 KB, not a transcendental.
//
// Placements measured (the experiment):
//   constant    — constant float[256] via set_bytes / constant address space
//   threadgroup — cooperative copy into threadgroup memory, then indexed
//   device      — device const float[256], hardware cache
//
// The inner loop still binds uchar scale/bias and does NOT expand the aux
// back to an f16 array. Expanding u8 → f16 aux and feeding the ordinary
// kernel is the forbidden shape and is not implemented here.
//
// Incumbent and exp-variant kernels are copied from aux_u8_ab.metal so the
// three-way is the same lowering aside from how scale/bias become floats.

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

// Exp-variant: in-register u8 aux decode. Identical to aux_u8_ab.metal.
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

// Fill the two 256-entry tables with the same Metal exp / linear map the
// exp-variant uses in-register. LUT[i] is then an exact reindexing of that
// function, not a host-libm approximation of it.
kernel void aux_u8_fill_lut256(
    device float* scale_lut         [[buffer(0)]],
    device float* bias_lut          [[buffer(1)]],
    constant AuxU8Endpoints& ep     [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]])
{
    if (tid >= 256u) {
        return;
    }
    const float s = float(tid);
    scale_lut[tid] = exp(ep.scale_lmin + s * ep.scale_span);
    bias_lut[tid] = ep.bias_min + s * ep.bias_span;
}

// ---------------------------------------------------------------------------
// LUT inner loops. Same codes / u8 aux / unpack as the exp-variant. The
// only change is scale/bias come from a 256-entry table indexed by the u8.
// Address space of the table is the independent variable.
// ---------------------------------------------------------------------------

static inline float affine_q2_geo_acc_g64_u8_lut_constant(
    device const uchar* codes,
    device const uchar* scales_u8,
    device const uchar* biases_u8,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row,
    constant float* scale_lut,
    constant float* bias_lut)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uchar s = scales_u8[rgb];
        const uchar b = biases_u8[rgb];
        const float scale = scale_lut[s];
        const float bias = bias_lut[b];
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

static inline float affine_q2_geo_acc_g64_u8_lut_device(
    device const uchar* codes,
    device const uchar* scales_u8,
    device const uchar* biases_u8,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row,
    device const float* scale_lut,
    device const float* bias_lut)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uchar s = scales_u8[rgb];
        const uchar b = biases_u8[rgb];
        const float scale = scale_lut[s];
        const float bias = bias_lut[b];
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

static inline float affine_q2_geo_acc_g64_u8_lut_threadgroup(
    device const uchar* codes,
    device const uchar* scales_u8,
    device const uchar* biases_u8,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row,
    threadgroup const float* scale_lut,
    threadgroup const float* bias_lut)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uchar s = scales_u8[rgb];
        const uchar b = biases_u8[rgb];
        const float scale = scale_lut[s];
        const float bias = bias_lut[b];
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    return acc;
}

kernel void aux_u8_lut_constant_affine_q2_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const uchar* scales_u8   [[buffer(1)]],
    device const uchar* biases_u8   [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant float* scale_lut       [[buffer(8)]],
    constant float* bias_lut        [[buffer(9)]],
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
        acc = affine_q2_geo_acc_g64_u8_lut_constant(
            codes, scales_u8, biases_u8, input, row, cols, lane_in_row,
            scale_lut, bias_lut);
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

kernel void aux_u8_lut_device_affine_q2_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const uchar* scales_u8   [[buffer(1)]],
    device const uchar* biases_u8   [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    device const float* scale_lut   [[buffer(8)]],
    device const float* bias_lut    [[buffer(9)]],
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
        acc = affine_q2_geo_acc_g64_u8_lut_device(
            codes, scales_u8, biases_u8, input, row, cols, lane_in_row,
            scale_lut, bias_lut);
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

kernel void aux_u8_lut_threadgroup_affine_q2_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const uchar* scales_u8   [[buffer(1)]],
    device const uchar* biases_u8   [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    device const float* scale_lut   [[buffer(8)]],
    device const float* bias_lut    [[buffer(9)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    threadgroup float tg_scale[256];
    threadgroup float tg_bias[256];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    const uint tid = simd_id * 32u + simd_lane; // 0..127
    tg_scale[tid] = scale_lut[tid];
    tg_bias[tid] = bias_lut[tid];
    tg_scale[tid + 128u] = scale_lut[tid + 128u];
    tg_bias[tid + 128u] = bias_lut[tid + 128u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    if (row < rows && group_size == 64u && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64_u8_lut_threadgroup(
            codes, scales_u8, biases_u8, input, row, cols, lane_in_row,
            tg_scale, tg_bias);
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
