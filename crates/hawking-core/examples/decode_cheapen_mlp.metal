// Cheapen the MLP affine2 q2 geo_tpr64 decode arithmetic.
//
// Production body is copied from q80_mixed_decode.metal
// (qwen_affine_q2_group32_matvec_geo_tpr64_tg128). These kernels are not
// bound on the decode path. Same geometry as the ALU-roofline probe:
// 128 threads/threadgroup, 2 rows/threadgroup, 8-wide tiles, stride 512.
//
// Attack the 8 dequant FMAs (w = q*scale+bias), not the 8 MACs.
// Inner-loop tax is counted per 8-weight tile / 6 B (2 B codes + 2 B scale
// + 2 B bias), matching receipts/future/MLP_ALU_ROOFLINE.json.
//
//   production     8 dequant FMA + 8 MAC FMA
//   lut4_*         4 dequant FMA (one w per code value) + 8 MAC  — exact
//   vec4           same FMAs, float4 x loads                      — exact
//   unrolled       production association, loop unrolled          — exact
//   fold           s*sum(q*x)+b*sum(x)  — exact over reals, not f32
//   fold_addqx     fold with q*x as adds                          — not f32
//   half_mac       MAC accumulated in half                        — not f32

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

// ── cheapen variants (g64, same addressing as production geo_acc_g64) ──

static inline void cheapen_reduce_store(
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

static inline float lut4_select_w(uint q, float w0, float w1, float w2, float w3) {
    const float w01 = select(w0, w1, q == 1u);
    const float w23 = select(w2, w3, q == 3u);
    return select(w01, w23, q >= 2u);
}

// Four reconstructed weights, production formula, then select. 4 dequant + 8 MAC.
static inline float affine_q2_unpack8_lut4_select(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    const float w0 = float(0u) * scale + bias;
    const float w1 = float(1u) * scale + bias;
    const float w2 = float(2u) * scale + bias;
    const float w3 = float(3u) * scale + bias;
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += lut4_select_w(q, w0, w1, w2, w3) * x[col + i];
    }
    return sum;
}

static inline float affine_q2_unpack8_lut4_index(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float wtab[4];
    wtab[0] = float(0u) * scale + bias;
    wtab[1] = float(1u) * scale + bias;
    wtab[2] = float(2u) * scale + bias;
    wtab[3] = float(3u) * scale + bias;
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += wtab[q] * x[col + i];
    }
    return sum;
}

// Same association as unpack8; x loaded as two float4s.
static inline float affine_q2_unpack8_vec4(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    const float4 x0 = *((device const float4*)(x + col));
    const float4 x1 = *((device const float4*)(x + col + 4u));
    float sum = 0.0f;
    sum += (float((packed16       ) & 3u) * scale + bias) * x0.x;
    sum += (float((packed16 >>  2u) & 3u) * scale + bias) * x0.y;
    sum += (float((packed16 >>  4u) & 3u) * scale + bias) * x0.z;
    sum += (float((packed16 >>  6u) & 3u) * scale + bias) * x0.w;
    sum += (float((packed16 >>  8u) & 3u) * scale + bias) * x1.x;
    sum += (float((packed16 >> 10u) & 3u) * scale + bias) * x1.y;
    sum += (float((packed16 >> 12u) & 3u) * scale + bias) * x1.z;
    sum += (float((packed16 >> 14u) & 3u) * scale + bias) * x1.w;
    return sum;
}

static inline float affine_q2_unpack8_lut4_vec4(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    const float w0 = float(0u) * scale + bias;
    const float w1 = float(1u) * scale + bias;
    const float w2 = float(2u) * scale + bias;
    const float w3 = float(3u) * scale + bias;
    const float4 x0 = *((device const float4*)(x + col));
    const float4 x1 = *((device const float4*)(x + col + 4u));
    float sum = 0.0f;
    sum += lut4_select_w((packed16       ) & 3u, w0, w1, w2, w3) * x0.x;
    sum += lut4_select_w((packed16 >>  2u) & 3u, w0, w1, w2, w3) * x0.y;
    sum += lut4_select_w((packed16 >>  4u) & 3u, w0, w1, w2, w3) * x0.z;
    sum += lut4_select_w((packed16 >>  6u) & 3u, w0, w1, w2, w3) * x0.w;
    sum += lut4_select_w((packed16 >>  8u) & 3u, w0, w1, w2, w3) * x1.x;
    sum += lut4_select_w((packed16 >> 10u) & 3u, w0, w1, w2, w3) * x1.y;
    sum += lut4_select_w((packed16 >> 12u) & 3u, w0, w1, w2, w3) * x1.z;
    sum += lut4_select_w((packed16 >> 14u) & 3u, w0, w1, w2, w3) * x1.w;
    return sum;
}

static inline float affine_q2_unpack8_unrolled(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    sum += (float((packed16       ) & 3u) * scale + bias) * x[col + 0u];
    sum += (float((packed16 >>  2u) & 3u) * scale + bias) * x[col + 1u];
    sum += (float((packed16 >>  4u) & 3u) * scale + bias) * x[col + 2u];
    sum += (float((packed16 >>  6u) & 3u) * scale + bias) * x[col + 3u];
    sum += (float((packed16 >>  8u) & 3u) * scale + bias) * x[col + 4u];
    sum += (float((packed16 >> 10u) & 3u) * scale + bias) * x[col + 5u];
    sum += (float((packed16 >> 12u) & 3u) * scale + bias) * x[col + 6u];
    sum += (float((packed16 >> 14u) & 3u) * scale + bias) * x[col + 7u];
    return sum;
}

// Algebra: sum_i (s*q_i + b)*x_i = s*sum(q_i*x_i) + b*sum(x_i). Exact over reals.
// Sequential f32 (q*s+b)*x is a different rounding, so this is NOT the exact class.
static inline float affine_q2_unpack8_fold(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float acc_qx = 0.0f;
    float acc_x = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        const float xi = x[col + i];
        acc_qx = fma(float(q), xi, acc_qx);
        acc_x += xi;
    }
    return fma(scale, acc_qx, bias * acc_x);
}

static inline float affine_q2_unpack8_fold_addqx(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float acc_qx = 0.0f;
    float acc_x = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        const float xi = x[col + i];
        acc_x += xi;
        if ((q & 2u) != 0u) {
            acc_qx += xi + xi;
        }
        if ((q & 1u) != 0u) {
            acc_qx += xi;
        }
    }
    return fma(scale, acc_qx, bias * acc_x);
}

static inline float affine_q2_unpack8_half_mac(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    half hacc = 0.0h;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        const float w = float(q) * scale + bias;
        hacc += half(w) * half(x[col + i]);
    }
    return float(hacc);
}

#define CHEAPEN_G64_ACC(name, unpack) \
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

CHEAPEN_G64_ACC(acc_lut4_select, affine_q2_unpack8_lut4_select)
CHEAPEN_G64_ACC(acc_lut4_index,  affine_q2_unpack8_lut4_index)
CHEAPEN_G64_ACC(acc_vec4,        affine_q2_unpack8_vec4)
CHEAPEN_G64_ACC(acc_lut4_vec4,   affine_q2_unpack8_lut4_vec4)
CHEAPEN_G64_ACC(acc_unrolled,    affine_q2_unpack8_unrolled)
CHEAPEN_G64_ACC(acc_fold,        affine_q2_unpack8_fold)
CHEAPEN_G64_ACC(acc_fold_addqx,  affine_q2_unpack8_fold_addqx)
CHEAPEN_G64_ACC(acc_half_mac,    affine_q2_unpack8_half_mac)

#undef CHEAPEN_G64_ACC

#define CHEAPEN_KERNEL(kname, accfn) \
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
    cheapen_reduce_store(acc, red, simd_lane, simd_id, row, rows, output); \
}

CHEAPEN_KERNEL(decode_cheapen_lut4_select, acc_lut4_select)
CHEAPEN_KERNEL(decode_cheapen_lut4_index,  acc_lut4_index)
CHEAPEN_KERNEL(decode_cheapen_vec4,        acc_vec4)
CHEAPEN_KERNEL(decode_cheapen_lut4_vec4,   acc_lut4_vec4)
CHEAPEN_KERNEL(decode_cheapen_unrolled,    acc_unrolled)
CHEAPEN_KERNEL(decode_cheapen_fold,        acc_fold)
CHEAPEN_KERNEL(decode_cheapen_fold_addqx,  acc_fold_addqx)
CHEAPEN_KERNEL(decode_cheapen_half_mac,    acc_half_mac)

#undef CHEAPEN_KERNEL
