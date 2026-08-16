// Private LUT decode for the codec-switching-activity lane.
// Not part of the production shader library. Decode is
//   q = lut[code]; value = q * scale
// Identity LUT is lut[c] = int(c) - bound, bit-identical to the
// shipped q80_hgravs01_two_stage_matvec path.

#include <metal_stdlib>
using namespace metal;

static inline uint act_extract_wide(device const uchar* codes, uint element, uint bits)
{
    const uint bit0 = element * bits;
    const uint byte0 = bit0 >> 3u;
    const uint shift = bit0 & 7u;
    uint packed = uint(codes[byte0]);
    if (shift + bits > 8u) {
        packed |= uint(codes[byte0 + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

static inline float act_value_bound(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    const uint group = element / group_size;
    const uint code = act_extract_wide(codes, element, bits);
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
}

static inline float act_value_lut(
    device const uchar* codes,
    device const half* scales,
    device const int* lut,
    uint element,
    uint group_size,
    uint bits)
{
    const uint group = element / group_size;
    const uint code = act_extract_wide(codes, element, bits);
    return float(lut[code]) * float(scales[group]);
}

// Same geometry as q80_hgravs01_two_stage_matvec (rank cap 160, x cap 512).
kernel void q80_hgravs01_two_stage_matvec_bound(
    device const uchar* right_codes [[buffer(0)]],
    device const half* right_scales [[buffer(1)]],
    device const uchar* left_codes  [[buffer(2)]],
    device const half* left_scales  [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float* output            [[buffer(5)]],
    constant uint& right_rows       [[buffer(6)]],
    constant uint& right_cols       [[buffer(7)]],
    constant uint& left_rows        [[buffer(8)]],
    constant uint& left_cols        [[buffer(9)]],
    constant uint& group_size       [[buffer(10)]],
    constant uint& bits             [[buffer(11)]],
    constant uint& bound             [[buffer(12)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRankCap = 160u;
    constexpr uint kXCap = 512u;
    threadgroup float mid[kRankCap];
    threadgroup float x_tg[kXCap];

    if (right_rows > kRankCap || right_rows != left_cols || right_cols > kXCap) {
        return;
    }

    for (uint i = lid; i < right_cols; i += 256u) {
        x_tg[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint rbase = 0u; rbase < right_rows; rbase += kSimdgroupsPerThreadgroup) {
        const uint r = rbase + simd_id;
        float partial = 0.0f;
        if (r < right_rows) {
            const uint row_base = r * right_cols;
            for (uint base = 0u; base < right_cols; base += kSimdWidth) {
                const uint col = base + simd_lane;
                if (col >= right_cols) {
                    continue;
                }
                partial += act_value_bound(
                    right_codes, right_scales, row_base + col, group_size, bits, bound)
                    * x_tg[col];
            }
            partial = simd_sum(partial);
            if (simd_lane == 0u) {
                mid[r] = partial;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lrow = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (lrow >= left_rows) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = lrow * left_cols;
    for (uint base = 0u; base < left_cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= left_cols) {
            continue;
        }
        partial += act_value_bound(
            left_codes, left_scales, row_base + col, group_size, bits, bound)
            * mid[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[lrow] = partial;
    }
}

kernel void q80_hgravs01_two_stage_matvec_lut(
    device const uchar* right_codes [[buffer(0)]],
    device const half* right_scales [[buffer(1)]],
    device const uchar* left_codes  [[buffer(2)]],
    device const half* left_scales  [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float* output            [[buffer(5)]],
    constant uint& right_rows       [[buffer(6)]],
    constant uint& right_cols       [[buffer(7)]],
    constant uint& left_rows        [[buffer(8)]],
    constant uint& left_cols        [[buffer(9)]],
    constant uint& group_size       [[buffer(10)]],
    constant uint& bits             [[buffer(11)]],
    device const int* right_lut     [[buffer(12)]],
    device const int* left_lut      [[buffer(13)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRankCap = 160u;
    constexpr uint kXCap = 512u;
    threadgroup float mid[kRankCap];
    threadgroup float x_tg[kXCap];

    if (right_rows > kRankCap || right_rows != left_cols || right_cols > kXCap) {
        return;
    }

    for (uint i = lid; i < right_cols; i += 256u) {
        x_tg[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint rbase = 0u; rbase < right_rows; rbase += kSimdgroupsPerThreadgroup) {
        const uint r = rbase + simd_id;
        float partial = 0.0f;
        if (r < right_rows) {
            const uint row_base = r * right_cols;
            for (uint base = 0u; base < right_cols; base += kSimdWidth) {
                const uint col = base + simd_lane;
                if (col >= right_cols) {
                    continue;
                }
                partial += act_value_lut(
                    right_codes, right_scales, right_lut, row_base + col, group_size, bits)
                    * x_tg[col];
            }
            partial = simd_sum(partial);
            if (simd_lane == 0u) {
                mid[r] = partial;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lrow = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (lrow >= left_rows) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = lrow * left_cols;
    for (uint base = 0u; base < left_cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= left_cols) {
            continue;
        }
        partial += act_value_lut(
            left_codes, left_scales, left_lut, row_base + col, group_size, bits)
            * mid[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[lrow] = partial;
    }
}
