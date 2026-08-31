// Matched-pair ALU vs memory-system probe for the production MLP affine2
// geo_tpr64 kernel and DeltaNet's dominant Q4 geo_tpr64 kernel.
//
// Production bodies are copied from q80_mixed_decode.metal /
// qwen_uniform_q4.metal. They are not bound on the decode path.
// ARM A (stripped): same addressing and loads, decode+dequant+FMA replaced
// by a XOR/add sink so the compiler cannot DCE the traffic.
// ARM B (halfk): same arithmetic, first half of K only.
// zero: launch+reduction floor (no weight/x loads).

#include <metal_stdlib>
using namespace metal;

// ── production affine2 q2 (HGRAVF01), group 32 or 64 ──────────────────────

static inline bool affine_q2_group_ok(uint group_size, uint cols) {
    return (group_size == 32u || group_size == 64u) && (cols % group_size) == 0u;
}

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

// Same addressing and loads as production g64. Decode+dequant+FMA replaced
// by a XOR/add of the raw words and the eight x floats. work_cols==0 means
// the full K; otherwise the loop stops at work_cols (stripped-half DCE check).
static inline float affine_q2_geo_stripped_g64(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    uint row,
    uint cols,
    uint work_cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 6u;
    const uint limit = work_cols == 0u ? cols : work_cols;
    uint csink = 0u;
    uint xsink = 0u;
    float ssink = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= limit; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        csink ^= packed16;
        ssink += float(scales[rgb]) + float(biases[rgb]);
        const device float* xp = input + col;
        xsink ^= as_type<uint>(xp[0]) ^ as_type<uint>(xp[1])
              ^ as_type<uint>(xp[2]) ^ as_type<uint>(xp[3])
              ^ as_type<uint>(xp[4]) ^ as_type<uint>(xp[5])
              ^ as_type<uint>(xp[6]) ^ as_type<uint>(xp[7]);
    }
    return ssink + float(csink) + as_type<float>(xsink);
}

static inline float affine_q2_geo_acc_g64_halfk(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    uint row,
    uint cols,
    uint work_cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 6u;
    const uint limit = work_cols == 0u ? (cols >> 1u) : work_cols;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= limit; col += 512u) {
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

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_stripped(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant uint& work_cols        [[buffer(8)]],
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
        acc = affine_q2_geo_stripped_g64(
            codes, scales, biases, input, row, cols, work_cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_halfk(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant uint& work_cols        [[buffer(8)]],
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
        acc = affine_q2_geo_acc_g64_halfk(
            codes, scales, biases, input, row, cols, work_cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_zero(
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
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint row = group_id * 2u + team;
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = float(group_id);
    }
    (void)codes;
    (void)scales;
    (void)biases;
    (void)input;
    (void)cols;
    (void)group_size;
}

// ── production Q4 geo_tpr64 (DeltaNet in_proj_qkvz) ───────────────────────

constant uint QWEN_UNIFORM_Q4_GROUP_SIZE = 64u;
constant uint QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP = 32u;

static inline float qwen_uniform_q4_unpack8(
    uint packed,
    float scale,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        sum += float(int(byte & 0x0fu) - 8) * scale * x[col + 2u * i];
        sum += float(int(byte >> 4u) - 8) * scale * x[col + 2u * i + 1u];
    }
    return sum;
}

kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
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
    if (row < rows) {
        const uint rgb0 = row * groups_per_row;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint rgb = rgb0 + group;
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_q4_geo_tpr64_tg128_stripped(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    constant uint& work_cols        [[buffer(7)]],
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
    if (row < rows) {
        const uint rgb0 = row * groups_per_row;
        const uint limit = work_cols == 0u ? cols : work_cols;
        uint csink = 0u;
        uint xsink = 0u;
        float ssink = 0.0f;
        for (uint col = lane_in_row * 8u; col < limit; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint rgb = rgb0 + group;
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            csink ^= packed;
            ssink += float(scales[rgb]);
            const device float* xp = input + col;
            xsink ^= as_type<uint>(xp[0]) ^ as_type<uint>(xp[1])
                  ^ as_type<uint>(xp[2]) ^ as_type<uint>(xp[3])
                  ^ as_type<uint>(xp[4]) ^ as_type<uint>(xp[5])
                  ^ as_type<uint>(xp[6]) ^ as_type<uint>(xp[7]);
        }
        acc = ssink + as_type<float>(csink) + as_type<float>(xsink);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_q4_geo_tpr64_tg128_halfk(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    constant uint& work_cols        [[buffer(7)]],
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
    if (row < rows) {
        const uint rgb0 = row * groups_per_row;
        const uint limit = work_cols == 0u ? (cols >> 1u) : work_cols;
        for (uint col = lane_in_row * 8u; col < limit; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint rgb = rgb0 + group;
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_q4_geo_tpr64_tg128_zero(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint row = group_id * 2u + team;
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = float(group_id);
    }
    (void)codes;
    (void)scales;
    (void)input;
    (void)cols;
    (void)groups_per_row;
}
