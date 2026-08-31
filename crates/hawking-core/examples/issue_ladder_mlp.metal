// Monotone total-ops/byte ladder + C1/C2 discriminators for one MLP layer.
//
// Same geometry as decode_cheapen_mlp.metal / alu_roofline_organs.metal:
// 128 threads/threadgroup, 2 rows/threadgroup, 8-wide tiles, stride 512,
// HGRAVF01 affine2 q2 group64. These kernels are not bound on the decode
// path. Production shaders are not modified.
//
// Static inner-loop tax is counted per 8-weight tile / 6 B
// (2 B codes + 2 B scale + 2 B bias) and is the authority in
// tools/future/mlp_issue_rate_ladder.py. This file implements that count:
//
//   tile overhead (every rung that keeps the access pattern):
//     integer 6  (group, local, rgb, code ptr)
//     conversion 2  (half->float scale, bias)
//     memory 3  (scale, bias, packed ushort)
//     control 1  (col loop)
//     + 8 x-float loads
//   per production weight slot: integer 2 (shift,mask) + conv 1 + fma 2 + mem 1
//   per xor-sink weight slot:   integer 1 (xor of x bits) + mem 1
//   ARM A (k=0): xor packed + 8 xor x + 2 fadd of scale/bias (keeps loads live)
//
// Ladder (monotone total ops / 6 B): production 60, k6 52, k4 44, k2 36, arm_a 31.

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

static inline void ladder_reduce_store(
    float acc,
    threadgroup float* red,
    uint simd_lane,
    uint simd_id,
    uint row,
    uint rows,
    device float* output)
{
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
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

// One production slot: 2 FMA + 2 integer + 1 conv + 1 x-load.
static inline float slot_prod(uint packed16, uint i, float scale, float bias, float xi) {
    const uint q = (packed16 >> (2u * i)) & 3u;
    const float w = float(q) * scale + bias;
    return w * xi;
}

static inline uint slot_xor(float xi) {
    return as_type<uint>(xi);
}

// Keep k production slots, xor-sink the rest of x so the 8 loads stay live.
// Unrolled so the compiler cannot emit both paths for a runtime k.
static inline float unpack_keep6(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    uint xsink = 0u;
    sum += slot_prod(packed16, 0u, scale, bias, x[col + 0u]);
    sum += slot_prod(packed16, 1u, scale, bias, x[col + 1u]);
    sum += slot_prod(packed16, 2u, scale, bias, x[col + 2u]);
    sum += slot_prod(packed16, 3u, scale, bias, x[col + 3u]);
    sum += slot_prod(packed16, 4u, scale, bias, x[col + 4u]);
    sum += slot_prod(packed16, 5u, scale, bias, x[col + 5u]);
    xsink ^= slot_xor(x[col + 6u]);
    xsink ^= slot_xor(x[col + 7u]);
    return sum + as_type<float>(xsink);
}

static inline float unpack_keep4(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    uint xsink = 0u;
    sum += slot_prod(packed16, 0u, scale, bias, x[col + 0u]);
    sum += slot_prod(packed16, 1u, scale, bias, x[col + 1u]);
    sum += slot_prod(packed16, 2u, scale, bias, x[col + 2u]);
    sum += slot_prod(packed16, 3u, scale, bias, x[col + 3u]);
    xsink ^= slot_xor(x[col + 4u]);
    xsink ^= slot_xor(x[col + 5u]);
    xsink ^= slot_xor(x[col + 6u]);
    xsink ^= slot_xor(x[col + 7u]);
    return sum + as_type<float>(xsink);
}

static inline float unpack_keep2(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    uint xsink = 0u;
    sum += slot_prod(packed16, 0u, scale, bias, x[col + 0u]);
    sum += slot_prod(packed16, 1u, scale, bias, x[col + 1u]);
    xsink ^= slot_xor(x[col + 2u]);
    xsink ^= slot_xor(x[col + 3u]);
    xsink ^= slot_xor(x[col + 4u]);
    xsink ^= slot_xor(x[col + 5u]);
    xsink ^= slot_xor(x[col + 6u]);
    xsink ^= slot_xor(x[col + 7u]);
    return sum + as_type<float>(xsink);
}

// ARM A: same addressing and loads, decode+dequant+FMA replaced by XOR/add sink.
static inline float unpack_arm_a(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    uint csink = packed16;
    float ssink = scale + bias;
    uint xsink = 0u;
    xsink ^= as_type<uint>(x[col + 0u]);
    xsink ^= as_type<uint>(x[col + 1u]);
    xsink ^= as_type<uint>(x[col + 2u]);
    xsink ^= as_type<uint>(x[col + 3u]);
    xsink ^= as_type<uint>(x[col + 4u]);
    xsink ^= as_type<uint>(x[col + 5u]);
    xsink ^= as_type<uint>(x[col + 6u]);
    xsink ^= as_type<uint>(x[col + 7u]);
    return ssink + float(csink) + as_type<float>(xsink);
}

// C1: same 16 FMA, independent accumulator chains. Combine adds are extra
// and reported; FMA-MAC count is identical to production.
static inline float unpack_ilp2(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float a0 = 0.0f;
    float a1 = 0.0f;
    a0 += slot_prod(packed16, 0u, scale, bias, x[col + 0u]);
    a1 += slot_prod(packed16, 1u, scale, bias, x[col + 1u]);
    a0 += slot_prod(packed16, 2u, scale, bias, x[col + 2u]);
    a1 += slot_prod(packed16, 3u, scale, bias, x[col + 3u]);
    a0 += slot_prod(packed16, 4u, scale, bias, x[col + 4u]);
    a1 += slot_prod(packed16, 5u, scale, bias, x[col + 5u]);
    a0 += slot_prod(packed16, 6u, scale, bias, x[col + 6u]);
    a1 += slot_prod(packed16, 7u, scale, bias, x[col + 7u]);
    return a0 + a1;
}

static inline float unpack_ilp4(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    a0 += slot_prod(packed16, 0u, scale, bias, x[col + 0u]);
    a1 += slot_prod(packed16, 1u, scale, bias, x[col + 1u]);
    a2 += slot_prod(packed16, 2u, scale, bias, x[col + 2u]);
    a3 += slot_prod(packed16, 3u, scale, bias, x[col + 3u]);
    a0 += slot_prod(packed16, 4u, scale, bias, x[col + 4u]);
    a1 += slot_prod(packed16, 5u, scale, bias, x[col + 5u]);
    a2 += slot_prod(packed16, 6u, scale, bias, x[col + 6u]);
    a3 += slot_prod(packed16, 7u, scale, bias, x[col + 7u]);
    return (a0 + a1) + (a2 + a3);
}

static inline float unpack_ilp8(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float a0 = slot_prod(packed16, 0u, scale, bias, x[col + 0u]);
    float a1 = slot_prod(packed16, 1u, scale, bias, x[col + 1u]);
    float a2 = slot_prod(packed16, 2u, scale, bias, x[col + 2u]);
    float a3 = slot_prod(packed16, 3u, scale, bias, x[col + 3u]);
    float a4 = slot_prod(packed16, 4u, scale, bias, x[col + 4u]);
    float a5 = slot_prod(packed16, 5u, scale, bias, x[col + 5u]);
    float a6 = slot_prod(packed16, 6u, scale, bias, x[col + 6u]);
    float a7 = slot_prod(packed16, 7u, scale, bias, x[col + 7u]);
    const float b0 = a0 + a1;
    const float b1 = a2 + a3;
    const float b2 = a4 + a5;
    const float b3 = a6 + a7;
    return (b0 + b1) + (b2 + b3);
}

#define LADDER_G64_ACC(name, unpack) \
static inline float name( \
    device const uchar* codes, \
    device const half* scales, \
    device const half* biases, \
    device const float* input, \
    uint row, \
    uint cols, \
    uint lane_in_row) \
{ \
    const uint groups_per_row = cols >> 6u; \
    float acc = 0.0f; \
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) { \
        const uint group = col >> 6u; \
        const uint local = col & 63u; \
        const uint rgb = row * groups_per_row + group; \
        const float scale = float(scales[rgb]); \
        const float bias = float(biases[rgb]); \
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u)))); \
        acc += unpack(packed16, scale, bias, input, col); \
    } \
    return acc; \
}

LADDER_G64_ACC(acc_keep6, unpack_keep6)
LADDER_G64_ACC(acc_keep4, unpack_keep4)
LADDER_G64_ACC(acc_keep2, unpack_keep2)
LADDER_G64_ACC(acc_arm_a, unpack_arm_a)
LADDER_G64_ACC(acc_ilp2,  unpack_ilp2)
LADDER_G64_ACC(acc_ilp4,  unpack_ilp4)
LADDER_G64_ACC(acc_ilp8,  unpack_ilp8)

#undef LADDER_G64_ACC

#define LADDER_KERNEL(kname, accfn) \
kernel void kname( \
    device const uchar* codes       [[buffer(0)]], \
    device const half*  scales      [[buffer(1)]], \
    device const half*  biases      [[buffer(2)]], \
    device const float* input       [[buffer(3)]], \
    device float*       output      [[buffer(4)]], \
    constant uint& rows             [[buffer(5)]], \
    constant uint& cols             [[buffer(6)]], \
    constant uint& group_size       [[buffer(7)]], \
    uint group_id                    [[threadgroup_position_in_grid]], \
    uint simd_lane                   [[thread_index_in_simdgroup]], \
    uint simd_id                     [[simdgroup_index_in_threadgroup]]) \
{ \
    threadgroup float red[4]; \
    constexpr uint kSplit = 2u; \
    const uint team = simd_id / kSplit; \
    const uint split = simd_id % kSplit; \
    const uint lane_in_row = split * 32u + simd_lane; \
    const uint row = group_id * 2u + team; \
    float acc = 0.0f; \
    if (row < rows && group_size == 64u && (cols % 64u) == 0u) { \
        acc = accfn(codes, scales, biases, input, row, cols, lane_in_row); \
    } \
    ladder_reduce_store(acc, red, simd_lane, simd_id, row, rows, output); \
}

LADDER_KERNEL(issue_ladder_k6,    acc_keep6)
LADDER_KERNEL(issue_ladder_k4,    acc_keep4)
LADDER_KERNEL(issue_ladder_k2,    acc_keep2)
LADDER_KERNEL(issue_ladder_arm_a, acc_arm_a)
LADDER_KERNEL(issue_ladder_ilp2,  acc_ilp2)
LADDER_KERNEL(issue_ladder_ilp4,  acc_ilp4)
LADDER_KERNEL(issue_ladder_ilp8,  acc_ilp8)

#undef LADDER_KERNEL

// Opaque keep: noinline so the compiler cannot DCE the live set across the
// tile loop. XOR-with-zero is bit-identical on acc.
__attribute__((noinline))
static float keep_live(float acc, float r) {
    return as_type<float>(as_type<uint>(acc) ^ (as_type<uint>(r) & 0u));
}

#define WS_KERNEL(kname, nlive) \
kernel void kname( \
    device const uchar* codes       [[buffer(0)]], \
    device const half*  scales      [[buffer(1)]], \
    device const half*  biases      [[buffer(2)]], \
    device const float* input       [[buffer(3)]], \
    device float*       output      [[buffer(4)]], \
    constant uint& rows             [[buffer(5)]], \
    constant uint& cols             [[buffer(6)]], \
    constant uint& group_size       [[buffer(7)]], \
    uint group_id                    [[threadgroup_position_in_grid]], \
    uint simd_lane                   [[thread_index_in_simdgroup]], \
    uint simd_id                     [[simdgroup_index_in_threadgroup]]) \
{ \
    threadgroup float red[4]; \
    constexpr uint kSplit = 2u; \
    const uint team = simd_id / kSplit; \
    const uint split = simd_id % kSplit; \
    const uint lane_in_row = split * 32u + simd_lane; \
    const uint row = group_id * 2u + team; \
    float acc = 0.0f; \
    float live[nlive]; \
    for (uint i = 0u; i < nlive; ++i) { \
        live[i] = 0.0f; \
    } \
    uint t = 0u; \
    if (row < rows && group_size == 64u && (cols % 64u) == 0u) { \
        const uint groups_per_row = cols >> 6u; \
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) { \
            const uint group = col >> 6u; \
            const uint local = col & 63u; \
            const uint rgb = row * groups_per_row + group; \
            const float scale = float(scales[rgb]); \
            const float bias = float(biases[rgb]); \
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u)))); \
            acc += affine_q2_unpack8(packed16, scale, bias, input, col); \
            live[t] = acc; \
            t = (t + 1u) % nlive; \
        } \
        for (uint i = 0u; i < nlive; ++i) { \
            acc = keep_live(acc, live[i]); \
        } \
    } \
    ladder_reduce_store(acc, red, simd_lane, simd_id, row, rows, output); \
}

WS_KERNEL(issue_ladder_ws8,  8)
WS_KERNEL(issue_ladder_ws16, 16)
WS_KERNEL(issue_ladder_ws32, 32)

#undef WS_KERNEL

// C2 occupancy/TG sweep: production arithmetic, 2 rows per threadgroup,
// threads_per_row = tg_threads/2 (must be a multiple of 32).
kernel void issue_ladder_tg(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    constant uint& tg_threads       [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[32];
    const uint threads_per_row = tg_threads / 2u;
    const uint simd_per_row = threads_per_row / 32u;
    const uint team = simd_id / simd_per_row;
    const uint split = simd_id % simd_per_row;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint stride = threads_per_row * 8u;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && group_size == 64u && (cols % 64u) == 0u
        && simd_per_row > 0u && (tg_threads % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += stride) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const float scale = float(scales[rgb]);
            const float bias = float(biases[rgb]);
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            acc += affine_q2_unpack8(packed16, scale, bias, input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        float s = 0.0f;
        for (uint i = 0u; i < simd_per_row; ++i) {
            s += red[team * simd_per_row + i];
        }
        output[row] = s;
    }
}
