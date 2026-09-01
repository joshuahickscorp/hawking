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


// ---------------------------------------------------------------------------
// OP-CLASS ABLATION LADDER (G094).
//
// The hoist removed ONE of production's five per-weight ops - the affine FMA -
// and bought nothing on real activations, while removing ALL FIVE (arm_a) buys
// 1.583x. That is not an issue-rate curve, so the cost is not spread evenly
// across the ops. These arms remove exactly one class each, holding byte traffic
// and load count identical to production, to find which class binds.
//
// NONE OF THESE COMPUTE THE RIGHT ANSWER. They are ablations, like arm_a. Their
// outputs are deliberately not compared to production.
//
// per weight:            shift+mask   uint->float   FMA affine   FMA x
//   production                8            8            8          8
//   hoist                     8            8            0          8
//   noaffine                  8            8            0          8   (no chunk fixup)
//   noconv                    8            0            0          0
//   nounpack                  0        1 per chunk      8          8
//   arm_a (stripped)          0            0            0          0
// ---------------------------------------------------------------------------

/// Drops the affine entirely: acc += float(q) * x. Keeps unpack, convert and the
/// x-multiply. This is the hoist WITHOUT its per-chunk fixup, so hoist-vs-this
/// prices the fixup and this-vs-production prices the affine FMA alone.
static inline float affine_q2_acc_g64_noaffine(
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
    float ssink = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        // scales/biases are still LOADED so the byte traffic matches production.
        ssink += float(scales[rgb]) + float(biases[rgb]);
        for (uint i = 0u; i < 8u; ++i) {
            const uint q = (packed16 >> (2u * i)) & 3u;
            acc += float(q) * input[col + i];
        }
    }
    return acc + ssink * 0.0f;
}

/// Removes the uint->float convert AND both FMAs. Keeps shift+mask and every
/// load. noconv-vs-noaffine prices convert plus the x-multiply together;
/// noconv-vs-arm_a prices shift+mask alone.
static inline float affine_q2_acc_g64_noconv(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 6u;
    uint iacc = 0u;
    uint xsink = 0u;
    float ssink = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        ssink += float(scales[rgb]) + float(biases[rgb]);
        for (uint i = 0u; i < 8u; ++i) {
            iacc += (packed16 >> (2u * i)) & 3u;
        }
        const device float* xp = input + col;
        xsink ^= as_type<uint>(xp[0]) ^ as_type<uint>(xp[1])
              ^ as_type<uint>(xp[2]) ^ as_type<uint>(xp[3])
              ^ as_type<uint>(xp[4]) ^ as_type<uint>(xp[5])
              ^ as_type<uint>(xp[6]) ^ as_type<uint>(xp[7]);
    }
    return ssink * 0.0f + float(iacc) * 0.0f + as_type<float>(xsink) * 0.0f;
}

/// Removes shift+mask and 7 of the 8 converts. Keeps BOTH FMAs at full count by
/// reusing one dequantised weight across the chunk. nounpack-vs-production
/// prices the unpack.
static inline float affine_q2_acc_g64_nounpack(
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
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        const float scale = float(scales[rgb]);
        const float bias = float(biases[rgb]);
        const float qf = float(packed16 & 3u);
        for (uint i = 0u; i < 8u; ++i) {
            const float w = qf * scale + bias;
            acc += w * input[col + i];
        }
    }
    return acc;
}


kernel void alu_roofline_affine_q2_geo_tpr64_tg128_noaffine(
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
        acc = affine_q2_acc_g64_noaffine(
            codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_noconv(
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
        acc = affine_q2_acc_g64_noconv(
            codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_nounpack(
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
        acc = affine_q2_acc_g64_nounpack(
            codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}


// ---------------------------------------------------------------------------
// BITCAST DEQUANT (G094 candidate). The op-class ablation put the uint->float
// convert at 44% of this kernel's arithmetic and 15% of its total time - the
// largest single class, and three times the affine FMA the hoist attacked.
//
// A 2-bit code needs no convert. Placing q at bits 21-22 of an f32 whose
// exponent field is 0x40000000 yields EXACTLY f = 2.0 + 0.25*q:
//     q=0 -> 0x40000000 = 2.0       q=2 -> 0x40400000 = 3.0
//     q=1 -> 0x40200000 = 2.5       q=3 -> 0x40600000 = 3.5
// bit 21 is 2^21/2^23 = 0.25 of the significand and the exponent is 2^1, so the
// step is 0.5 per code, NOT 0.25. Getting that factor wrong produced a kernel
// that was 1.225x fast and completely wrong (rel_fro 1.26), which is exactly
// why the output comparison runs on every candidate.
// So q = 2*(f - 2) and the production weight w = q*scale + bias becomes
//     w = (2*scale)*f + (bias - 4*scale)
// with both constants folded ONCE PER GROUP from the same half scale/bias the
// production kernel already loads. Per weight this trades one int-to-float
// convert for one OR. It is exact in the reals; the two FMAs run on different
// constants, so it is NOT expected to be bit-identical and the output is
// compared rather than assumed.
static inline float affine_q2_acc_g64_bitcast(
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
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        const float scale = float(scales[rgb]);
        const float bias = float(biases[rgb]);
        // Folded once per group, not once per weight.
        const float s2 = scale * 2.0f;
        const float b4 = fma(scale, -4.0f, bias);
        for (uint i = 0u; i < 8u; ++i) {
            const uint bits = 0x40000000u | (((packed16 >> (2u * i)) & 3u) << 21u);
            const float f = as_type<float>(bits);
            const float w = fma(s2, f, b4);
            acc = fma(w, input[col + i], acc);
        }
    }
    return acc;
}

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_bitcast(
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
        acc = affine_q2_acc_g64_bitcast(
            codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
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


// DEQUANT HOIST. Same bytes, same access pattern, same loads. The affine is
// applied ONCE PER 8-WEIGHT CHUNK instead of once per weight, using
//
//     sum_i (s*c_i + b) * x_i  ==  s * sum_i(c_i*x_i) + b * sum_i(x_i)
//
// sum_i(x_i) over a chunk is a property of x, not of the output row, so it is
// precomputed once per token and read here rather than recomputed per row. That
// is what makes this 10 FMA per 8 weights instead of 16: 8 for c_i*x_i and 2
// for the affine, against the incumbent's 8 dequant + 8 mac.
//
// NOT bit-identical: the summation order changes.
static inline float affine_q2_unpack8_hoist(
    uint packed16, float scale, float bias,
    device const float* x, uint col, float sumx8)
{
    float sc = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sc += float(q) * x[col + i];
    }
    return scale * sc + bias * sumx8;
}

static inline float affine_q2_geo_acc_g64_hoist(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    device const float* sumx8,
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
        acc += affine_q2_unpack8_hoist(packed16, scale, bias, input, col, sumx8[col >> 3u]);
    }
    return acc;
}

kernel void alu_roofline_affine_q2_geo_tpr64_tg128_hoist(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    device const float* sumx8       [[buffer(8)]],
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
    if (row < rows && group_size == 64u && affine_q2_group_ok(group_size, cols)) {
        acc = affine_q2_geo_acc_g64_hoist(codes, scales, biases, input, sumx8,
                                          row, cols, lane_in_row);
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


// BITCAST DEQUANT for uniform q4 (G094). Same construction that took 3.854 ms
// off the MLP token, mapped to a 4-bit code with a fixed -8 zero point.
//
// A nibble placed at bits 19-22 of an f32 with exponent field 0x40000000 gives
// EXACTLY f = 2.0 + q/8 (the nibble is q*2^19/2^23 = q/16 of the significand,
// doubled by the 2^1 exponent). So q = 8*(f - 2) and the production weight
//     (q - 8) * scale
// becomes
//     (8*scale)*f + (-24*scale)
// with both constants folded ONCE PER GROUP. This removes the int-to-float
// convert AND the -8 subtract, and costs one OR.
//
// The nibble is masked and then shifted to 19: (packed >> 8i) & 0xf, << 19.
// The tempting pre-shift trick - (packed << 19) >> 8i - is WRONG here and was
// measured wrong at rel_fro 0.877: packed is a full 32-bit word of eight
// nibbles, so shifting it left by 19 discards bits 13 and up. It works for the
// q2 kernel only because its packed word is 16 bits.
//
// NOT bit-identical - the multiply becomes an FMA on refolded constants - so
// the output is compared, never assumed.
static inline float q4_unpack8_bitcast(
    uint packed, float s8, float b24,
    device const float* x, uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        const float fl = as_type<float>(0x40000000u | ((byte & 0x0fu) << 19u));
        const float fh = as_type<float>(0x40000000u | ((byte >> 4u) << 19u));
        sum = fma(fma(s8, fl, b24), x[col + 2u * i], sum);
        sum = fma(fma(s8, fh, b24), x[col + 2u * i + 1u], sum);
    }
    return sum;
}

kernel void alu_roofline_q4_geo_tpr64_tg128_bitcast(
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
            acc += q4_unpack8_bitcast(
                packed, scale * 8.0f, scale * -24.0f, input, col);
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
