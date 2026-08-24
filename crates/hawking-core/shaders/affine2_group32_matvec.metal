// HGRAVF01 affine-2 GEMV (codec: unsigned 2-bit + fp16 scale + bias).
//
// Reconstruction (exact, unsigned q, scale may be negative):
//   q = (word >> (2*i)) & 3
//   w = float(q) * scale + bias
//
// Same kernel family as group-32. group_size is a bind-time parameter
// and must be 32 or 64. Codes stay packed. The GEMV dequants in
// registers and never writes a dense W.
//
// Host widens on-disk f16 scale+bias to IEEE f32 before bind so the
// kernel matches the CPU oracle bit-for-bit (q is exact; scale/bias are
// the widened storage values).
//
// Do not confuse per-group `biases` with an nn.Linear output bias.

#include <metal_stdlib>
using namespace metal;

constant uint AFFINE2_CODES_PER_WORD = 16u;

static inline float affine2_w(uint q, float scale, float bias) {
    return float(q) * scale + bias;
}

static inline uint affine2_code(uint word, uint i) {
    return (word >> (2u * i)) & 3u;
}

static inline bool affine2_group_ok(uint group_size, uint cols) {
    return (group_size == 32u || group_size == 64u) && (cols % group_size) == 0u;
}

// Eight consecutive weights (16 packed bits) inside a group.
// `col` is 8-aligned and sits inside one group.
static inline float affine2_unpack8(
    uint packed16,
    float scale,
    float bias,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += affine2_w(q, scale, bias) * x[col + i];
    }
    return sum;
}

// Serial family. One thread per output row. Grid (rows,1,1), TG up to 256.
// Walks 16-code words left to right inside each group.
kernel void affine2_group32_matvec(
    device const uint*  codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows) return;
    if (!affine2_group_ok(group_size, cols)) {
        output[row] = 0.0f;
        return;
    }
    const uint groups_per_row = cols / group_size;
    const uint words_per_group = group_size / AFFINE2_CODES_PER_WORD;
    const uint packed_row = row * (cols / AFFINE2_CODES_PER_WORD);
    float sum = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint rgb = row * groups_per_row + g;
        const float scale = scales[rgb];
        const float bias = biases[rgb];
        const uint word_base = packed_row + g * words_per_group;
        for (uint w = 0u; w < words_per_group; ++w) {
            const uint word = codes[word_base + w];
            const uint col = g * group_size + w * AFFINE2_CODES_PER_WORD;
            for (uint i = 0u; i < AFFINE2_CODES_PER_WORD; ++i) {
                sum += affine2_w(affine2_code(word, i), scale, bias) * input[col + i];
            }
        }
    }
    output[row] = sum;
}

// Compile-time group 32/64 so col/GS is a shift. A runtime group_size on
// this path is a non-constant divide (see qwen_uniform_q4 group-128 note).
static inline float affine2_geo_acc_g32(
    device const uchar* codes,
    device const float* scales,
    device const float* biases,
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
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 8u + (local >> 2u))));
        acc += affine2_unpack8(packed16, scales[rgb], biases[rgb], input, col);
    }
    return acc;
}

static inline float affine2_geo_acc_g64(
    device const uchar* codes,
    device const float* scales,
    device const float* biases,
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
        acc += affine2_unpack8(packed16, scales[rgb], biases[rgb], input, col);
    }
    return acc;
}

// geo_tpr64 occupancy (same thread map as HGRAVU01 geo, different reconstruction).
//   TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row
//   col = lane_in_row * 8, stride 512
//   grid threadgroups = ceil(rows/2), tg = 128
// Every 8-wide tile sits inside one group of 32 or 64. In-register dequant only.
kernel void affine2_group32_matvec_geo_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && affine2_group_ok(group_size, cols)) {
        if (group_size == 32u) {
            acc = affine2_geo_acc_g32(codes, scales, biases, input, row, cols, lane_in_row);
        } else {
            acc = affine2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
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

// Diagnostic: pre-specialization G0 body with runtime col/group_size.
kernel void affine2_group32_matvec_geo_tpr64_tg128_runtime_div(
    device const uchar* codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && affine2_group_ok(group_size, cols)) {
        const uint groups_per_row = cols / group_size;
        const uint bytes_per_group = group_size >> 2u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col / group_size;
            const uint local = col % group_size;
            const uint rgb = row * groups_per_row + group;
            const float scale = scales[rgb];
            const float bias = biases[rgb];
            const uint byte0 = rgb * bytes_per_group + (local >> 2u);
            const uint packed16 = uint(*((device const ushort*)(codes + byte0)));
            acc += affine2_unpack8(packed16, scale, bias, input, col);
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

// Reconstruct w = q*scale + bias into a dense f32 buffer. Used only by the
// standalone parity harness to prove the in-register decode; the GEMV path
// above never materializes this buffer.
kernel void affine2_group32_dequant(
    device const uint*  codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& cols        [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    uint gid                    [[thread_position_in_grid]])
{
    const uint n = rows * cols;
    if (gid >= n) return;
    if (!affine2_group_ok(group_size, cols)) {
        output[gid] = 0.0f;
        return;
    }
    const uint row = gid / cols;
    const uint col = gid - row * cols;
    const uint groups_per_row = cols / group_size;
    const uint group = col / group_size;
    const uint local = col % group_size;
    const uint rgb = row * groups_per_row + group;
    const uint words_per_group = group_size / AFFINE2_CODES_PER_WORD;
    const uint packed_index =
        row * (cols / AFFINE2_CODES_PER_WORD) + group * words_per_group + (local / AFFINE2_CODES_PER_WORD);
    const uint word = codes[packed_index];
    const uint q = affine2_code(word, local & 15u);
    output[gid] = affine2_w(q, scales[rgb], biases[rgb]);
}

// ── Q2F: 4-level LS-fitted 2-bit, group 64, delta only ──
// w = (float(q) - 1.5) * delta. Host widens f16 delta to f32 before bind
// (same as affine2 standalone). Compile-time group 64; no bias buffer.

static inline float q2f_w(uint q, float delta) {
    return (float(q) - 1.5f) * delta;
}

static inline float q2f_unpack8_f32(
    uint packed16,
    float delta,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += q2f_w(q, delta) * x[col + i];
    }
    return sum;
}

static inline float q2f_geo_acc_g64_f32(
    device const uchar* codes,
    device const float* deltas,
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
        acc += q2f_unpack8_f32(packed16, deltas[rgb], input, col);
    }
    return acc;
}

kernel void q2f_group64_matvec(
    device const uchar* codes  [[buffer(0)]],
    device const float* deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& cols        [[buffer(5)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows || (cols % 64u) != 0u) {
        return;
    }
    const uint groups_per_row = cols >> 6u;
    float sum = 0.0f;
    for (uint col = 0u; col + 8u <= cols; col += 8u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        sum += q2f_unpack8_f32(packed16, deltas[rgb], input, col);
    }
    output[row] = sum;
}

kernel void q2f_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const float* deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& cols        [[buffer(5)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        acc = q2f_geo_acc_g64_f32(codes, deltas, input, row, cols, lane_in_row);
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

// Reconstruction into dense f32. Parity harness only; GEMV never writes this.
kernel void q2f_group64_dequant(
    device const uchar* codes  [[buffer(0)]],
    device const float* deltas [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    const uint n = rows * cols;
    if (gid >= n || (cols % 64u) != 0u) {
        return;
    }
    const uint row = gid / cols;
    const uint col = gid - row * cols;
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uint byte = uint(codes[rgb * 16u + (local >> 2u)]);
    const uint q = (byte >> (2u * (local & 3u))) & 3u;
    output[gid] = q2f_w(q, deltas[rgb]);
}

// Fused gate+up(+SwiGLU). Same occupancy as geo_tpr64. Packed codes stay packed.
// Standalone uses f32 deltas (host-widened); production mixed path uses half.
static inline void q2f_unpack8_dual_g64_f32(
    uint packed_g,
    float delta_g,
    uint packed_u,
    float delta_u,
    device const float* x,
    uint col,
    thread float& acc_g,
    thread float& acc_u)
{
    for (uint i = 0u; i < 8u; ++i) {
        const float xv = x[col + i];
        const uint qg = (packed_g >> (2u * i)) & 3u;
        const uint qu = (packed_u >> (2u * i)) & 3u;
        acc_g += q2f_w(qg, delta_g) * xv;
        acc_u += q2f_w(qu, delta_u) * xv;
    }
}

kernel void q2f_group64_matvec_gate_up_geo_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const float* gate_deltas [[buffer(1)]],
    device const uchar* up_codes    [[buffer(2)]],
    device const float* up_deltas   [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float*       gate_out    [[buffer(5)]],
    device float*       up_out      [[buffer(6)]],
    constant uint& rows             [[buffer(7)]],
    constant uint& cols             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            q2f_unpack8_dual_g64_f32(
                gpacked, gate_deltas[rgb],
                upacked, up_deltas[rgb],
                input, col, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[4u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        gate_out[row] = red[t] + red[t + 1u];
        up_out[row] = red[4u + t] + red[t + 1u + 4u];
    }
}

kernel void q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const float* gate_deltas [[buffer(1)]],
    device const uchar* up_codes    [[buffer(2)]],
    device const float* up_deltas   [[buffer(3)]],
    device const float* input       [[buffer(4)]],
    device float*       act_out     [[buffer(5)]],
    constant uint& rows             [[buffer(6)]],
    constant uint& cols             [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            q2f_unpack8_dual_g64_f32(
                gpacked, gate_deltas[rgb],
                upacked, up_deltas[rgb],
                input, col, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[4u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        const float g = red[t] + red[t + 1u];
        const float u = red[4u + t] + red[4u + t + 1u];
        act_out[row] = (g / (1.0f + exp(-g))) * u;
    }
}

// ── g64 kernel-geometry levers (half scale/bias, production byte mix) ──
// Standalone compile unit for isolated GEMV GB/s. Same reconstruction as
// q80_mixed_decode affine2 g64 kernels. No dense W.

static inline float affine2_sa_dot16_f4(
    uint packed, float scale, float bias,
    float4 a, float4 b, float4 c, float4 d)
{
    float s = 0.0f;
    s += (float((packed       ) & 3u) * scale + bias) * a.x;
    s += (float((packed >>  2u) & 3u) * scale + bias) * a.y;
    s += (float((packed >>  4u) & 3u) * scale + bias) * a.z;
    s += (float((packed >>  6u) & 3u) * scale + bias) * a.w;
    s += (float((packed >>  8u) & 3u) * scale + bias) * b.x;
    s += (float((packed >> 10u) & 3u) * scale + bias) * b.y;
    s += (float((packed >> 12u) & 3u) * scale + bias) * b.z;
    s += (float((packed >> 14u) & 3u) * scale + bias) * b.w;
    s += (float((packed >> 16u) & 3u) * scale + bias) * c.x;
    s += (float((packed >> 18u) & 3u) * scale + bias) * c.y;
    s += (float((packed >> 20u) & 3u) * scale + bias) * c.z;
    s += (float((packed >> 22u) & 3u) * scale + bias) * c.w;
    s += (float((packed >> 24u) & 3u) * scale + bias) * d.x;
    s += (float((packed >> 26u) & 3u) * scale + bias) * d.y;
    s += (float((packed >> 28u) & 3u) * scale + bias) * d.z;
    s += (float((packed >> 30u) & 3u) * scale + bias) * d.w;
    return s;
}

static inline void affine2_sa_load_x16(
    device const float* input, uint col,
    thread float4& a, thread float4& b, thread float4& c, thread float4& d)
{
    a = *((device const float4*)(input + col));
    b = *((device const float4*)(input + col + 4u));
    c = *((device const float4*)(input + col + 8u));
    d = *((device const float4*)(input + col + 12u));
}

static inline float affine2_sa_dot16_at(
    device const uchar* codes, device const half* scales, device const half* biases,
    uint row, uint cols, uint col, float4 a, float4 b, float4 c, float4 d)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uint packed = *((device const uint*)(codes + rgb * 16u + (local >> 2u)));
    return affine2_sa_dot16_f4(packed, float(scales[rgb]), float(biases[rgb]), a, b, c, d);
}

static inline float affine2_sa_dot64_at(
    device const uchar* codes, device const half* scales, device const half* biases,
    device const float* input, uint row, uint cols, uint col)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint rgb = row * groups_per_row + group;
    const uint4 packed = *((device const uint4*)(codes + rgb * 16u));
    const float scale = float(scales[rgb]);
    const float bias = float(biases[rgb]);
    float4 a, b, c, d;
    affine2_sa_load_x16(input, col, a, b, c, d);
    float s = affine2_sa_dot16_f4(packed.x, scale, bias, a, b, c, d);
    affine2_sa_load_x16(input, col + 16u, a, b, c, d);
    s += affine2_sa_dot16_f4(packed.y, scale, bias, a, b, c, d);
    affine2_sa_load_x16(input, col + 32u, a, b, c, d);
    s += affine2_sa_dot16_f4(packed.z, scale, bias, a, b, c, d);
    affine2_sa_load_x16(input, col + 48u, a, b, c, d);
    s += affine2_sa_dot16_f4(packed.w, scale, bias, a, b, c, d);
    return s;
}

// No-op / incumbent control: production tpr64 occupancy, half scale/bias, g64.
kernel void affine2_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            acc += affine2_unpack8(packed16, float(scales[rgb]), float(biases[rgb]), input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// Deliberately-bad control: bind-time group_size so col/group_size is a
// non-constant integer divide (the measured 1.37x defect).
kernel void affine2_group64_matvec_geo_tpr64_tg128_runtime_div(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && affine2_group_ok(group_size, cols)) {
        const uint groups_per_row = cols / group_size;
        const uint bytes_per_group = group_size >> 2u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col / group_size;
            const uint local = col % group_size;
            const uint rgb = row * groups_per_row + group;
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * bytes_per_group + (local >> 2u))));
            acc += affine2_unpack8(packed16, float(scales[rgb]), float(biases[rgb]), input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void affine2_group64_matvec_qmvfast_r8tg64(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = group_id * 8u + simd_id * 4u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if ((cols % 64u) == 0u) {
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint col = bk + simd_lane * 16u;
            if (col + 16u > cols) continue;
            float4 a, b, c, d;
            affine2_sa_load_x16(input, col, a, b, c, d);
            if (row0 < rows) acc0 += affine2_sa_dot16_at(codes, scales, biases, row0, cols, col, a, b, c, d);
            if (row0 + 1u < rows) acc1 += affine2_sa_dot16_at(codes, scales, biases, row0 + 1u, cols, col, a, b, c, d);
            if (row0 + 2u < rows) acc2 += affine2_sa_dot16_at(codes, scales, biases, row0 + 2u, cols, col, a, b, c, d);
            if (row0 + 3u < rows) acc3 += affine2_sa_dot16_at(codes, scales, biases, row0 + 3u, cols, col, a, b, c, d);
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1); acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (simd_lane == 0u) {
        if (row0 < rows) output[row0] = acc0;
        if (row0 + 1u < rows) output[row0 + 1u] = acc1;
        if (row0 + 2u < rows) output[row0 + 2u] = acc2;
        if (row0 + 3u < rows) output[row0 + 3u] = acc3;
    }
}

kernel void affine2_group64_matvec_qmvfast_r8tg64_addr_probe(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = group_id * 8u + simd_id * 4u;
    float acc = 0.0f;
    if ((cols % 64u) == 0u && row0 < rows) {
        const uint groups_per_row = cols >> 6u;
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint col = bk + simd_lane * 16u;
            if (col + 16u > cols) continue;
            float4 a, b, c, d;
            affine2_sa_load_x16(input, col, a, b, c, d);
            acc += a.x + d.w;
            for (uint r = 0u; r < 4u; ++r) {
                const uint row = row0 + r;
                if (row >= rows) continue;
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * groups_per_row + group;
                const uint packed = *((device const uint*)(codes + rgb * 16u + (local >> 2u)));
                acc += float(scales[rgb]) + float(biases[rgb]) + as_type<float>(packed);
            }
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u && row0 < rows) output[row0] = acc;
}

kernel void affine2_group64_matvec_wide64_r4tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    const uint row = group_id * 4u + simd_id;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        for (uint col = simd_lane * 64u; col + 64u <= cols; col += 2048u) {
            acc += affine2_sa_dot64_at(codes, scales, biases, input, row, cols, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u && row < rows) output[row] = acc;
}

kernel void affine2_group64_matvec_tgx_r8tg256(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_tile[512];
    const uint row = group_id * 8u + simd_id;
    float acc = 0.0f;
    if ((cols % 64u) == 0u) {
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint load_at = lid * 2u;
            if (bk + load_at + 2u <= cols) {
                *((threadgroup float2*)(x_tile + load_at)) =
                    *((device const float2*)(input + bk + load_at));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row < rows) {
                const uint local = simd_lane * 16u;
                const uint col = bk + local;
                if (col + 16u <= cols) {
                    float4 a = *((threadgroup const float4*)(x_tile + local));
                    float4 b = *((threadgroup const float4*)(x_tile + local + 4u));
                    float4 c = *((threadgroup const float4*)(x_tile + local + 8u));
                    float4 d = *((threadgroup const float4*)(x_tile + local + 12u));
                    acc += affine2_sa_dot16_at(codes, scales, biases, row, cols, col, a, b, c, d);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u && row < rows) output[row] = acc;
}

// ── N024 non-load critical-path levers ──
// Same reconstruction as tpr64: w = q*scale+bias, in-register, no dense W.
// Geometry stays tpr64 occupancy unless a kernel name says otherwise.
// qmvfast / wide64 / tgx are NOT re-tried here (N018: they lost on the
// production path). These arms attack scale/bias traffic, unpack, split-K
// accumulation, and the (q*scale+bias)*x association itself.

constant uint kAffine2SbMaxGroups = 512u;

static inline float affine2_unpack8_vec(
    uint packed16, float scale, float bias, float4 x0, float4 x1)
{
    float s = 0.0f;
    s += (float((packed16       ) & 3u) * scale + bias) * x0.x;
    s += (float((packed16 >>  2u) & 3u) * scale + bias) * x0.y;
    s += (float((packed16 >>  4u) & 3u) * scale + bias) * x0.z;
    s += (float((packed16 >>  6u) & 3u) * scale + bias) * x0.w;
    s += (float((packed16 >>  8u) & 3u) * scale + bias) * x1.x;
    s += (float((packed16 >> 10u) & 3u) * scale + bias) * x1.y;
    s += (float((packed16 >> 12u) & 3u) * scale + bias) * x1.z;
    s += (float((packed16 >> 14u) & 3u) * scale + bias) * x1.w;
    return s;
}

static inline float affine2_unpack8_accfuse_vec(
    uint packed16, float scale, float bias, float4 x0, float4 x1)
{
    float qx = 0.0f;
    float xs = 0.0f;
    const float q0 = float((packed16       ) & 3u);
    const float q1 = float((packed16 >>  2u) & 3u);
    const float q2 = float((packed16 >>  4u) & 3u);
    const float q3 = float((packed16 >>  6u) & 3u);
    const float q4 = float((packed16 >>  8u) & 3u);
    const float q5 = float((packed16 >> 10u) & 3u);
    const float q6 = float((packed16 >> 12u) & 3u);
    const float q7 = float((packed16 >> 14u) & 3u);
    qx += q0 * x0.x; xs += x0.x;
    qx += q1 * x0.y; xs += x0.y;
    qx += q2 * x0.z; xs += x0.z;
    qx += q3 * x0.w; xs += x0.w;
    qx += q4 * x1.x; xs += x1.x;
    qx += q5 * x1.y; xs += x1.y;
    qx += q6 * x1.z; xs += x1.z;
    qx += q7 * x1.w; xs += x1.w;
    return qx * scale + xs * bias;
}

static inline void affine2_sb_stage(
    threadgroup float* sb_scale,
    threadgroup float* sb_bias,
    device const half* scales,
    device const half* biases,
    uint row0,
    uint rows,
    uint gpr,
    uint lid)
{
    for (uint g = lid; g < gpr; g += 128u) {
        if (row0 < rows) {
            const uint rgb = row0 * gpr + g;
            sb_scale[g] = float(scales[rgb]);
            sb_bias[g] = float(biases[rgb]);
        }
        if (row0 + 1u < rows) {
            const uint rgb = (row0 + 1u) * gpr + g;
            sb_scale[kAffine2SbMaxGroups + g] = float(scales[rgb]);
            sb_bias[kAffine2SbMaxGroups + g] = float(biases[rgb]);
        }
    }
}

// Lever: stage per-group scale/bias in threadgroup memory once.
// tpr64 occupancy (TG 128, 2 rows, 64 threads/row). Inner loop does not
// reload scale/bias from device per 8-wide tile.
kernel void affine2_group64_matvec_tgsb_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    threadgroup float sb_scale[2 * kAffine2SbMaxGroups];
    threadgroup float sb_bias[2 * kAffine2SbMaxGroups];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row0 = group_id * 2u;
    const uint row = row0 + team;
    const uint gpr = cols >> 6u;
    float acc = 0.0f;
    if ((cols % 64u) == 0u && gpr <= kAffine2SbMaxGroups) {
        affine2_sb_stage(sb_scale, sb_bias, scales, biases, row0, rows, gpr, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < rows) {
            const uint sb_base = team * kAffine2SbMaxGroups;
            for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * gpr + group;
                const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
                acc += affine2_unpack8(
                    packed16, sb_scale[sb_base + group], sb_bias[sb_base + group], input, col);
            }
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// Lever: software-pipeline the next 8-wide unpack (codes + x) ahead of the
// current FMA. Vectorized x loads. Same (q*scale+bias)*x association as tpr64.
kernel void affine2_group64_matvec_pipe_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        uint packed_n = 0u;
        float scale_n = 0.0f;
        float bias_n = 0.0f;
        float4 x0_n = 0.0f;
        float4 x1_n = 0.0f;
        bool primed = false;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            uint packed;
            float scale, bias;
            float4 x0, x1;
            if (primed) {
                packed = packed_n;
                scale = scale_n;
                bias = bias_n;
                x0 = x0_n;
                x1 = x1_n;
            } else {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * groups_per_row + group;
                packed = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
                scale = float(scales[rgb]);
                bias = float(biases[rgb]);
                x0 = *((device const float4*)(input + col));
                x1 = *((device const float4*)(input + col + 4u));
                primed = true;
            }
            const uint col_n = col + 512u;
            if (col_n + 8u <= cols) {
                const uint group_n = col_n >> 6u;
                const uint local_n = col_n & 63u;
                const uint rgb_n = row * groups_per_row + group_n;
                packed_n = uint(*((device const ushort*)(codes + rgb_n * 16u + (local_n >> 2u))));
                scale_n = float(scales[rgb_n]);
                bias_n = float(biases[rgb_n]);
                x0_n = *((device const float4*)(input + col_n));
                x1_n = *((device const float4*)(input + col_n + 4u));
            }
            acc += affine2_unpack8_vec(packed, scale, bias, x0, x1);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// Lever: 4-way split-K on the reduction (TG 256, 2 rows, 128 threads/row,
// stride 1024). Not tgx: x stays in device memory, occupancy is 2 rows/TG.
kernel void affine2_group64_matvec_splitk4_tg256(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 4u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 1024u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            acc += affine2_unpack8(packed16, float(scales[rgb]), float(biases[rgb]), input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        output[row] = red[t] + red[t + 1u] + red[t + 2u] + red[t + 3u];
    }
}

// Lever: fuse scale/bias into the accumulate (algebraic rewrite):
//   sum_i (q_i*scale + bias)*x_i  =  scale*sum(q_i x_i) + bias*sum(x_i)
// Same tpr64 occupancy. Association differs from tpr64; parity uses 2e-2.
kernel void affine2_group64_matvec_accfuse_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            const float4 x0 = *((device const float4*)(input + col));
            const float4 x1 = *((device const float4*)(input + col + 4u));
            acc += affine2_unpack8_accfuse_vec(
                packed16, float(scales[rgb]), float(biases[rgb]), x0, x1);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// ── N030 non-load fused gate_up_swiglu autopsy (isolated, no 27B) ──
// Same tpr64 occupancy as production. Do not retune load geometry
// (qmvfast / wide64 / tgx stay unused). No dense W.
//
// biasprep: (q*scale)*x in the inner loop; bias*sum(x_group) is applied
// once per row from a precomputed 64-wide x-sum. Algebraically equal to
// (q*scale+bias)*x; association differs. drop skips the bias term
// (deliberately-bad control). decode_probe extracts packed q and poisons
// the acc so the unpack cannot DCE. addr_probe loads the byte stream.

static inline void affine2_sa_unpack8_dual_vec(
    uint packed_g, float scale_g, float bias_g,
    uint packed_u, float scale_u, float bias_u,
    float4 x0, float4 x1,
    thread float& acc_g, thread float& acc_u)
{
    acc_g += affine2_unpack8_vec(packed_g, scale_g, bias_g, x0, x1);
    acc_u += affine2_unpack8_vec(packed_u, scale_u, bias_u, x0, x1);
}

static inline void affine2_sa_unpack8_dual_qx(
    uint packed_g, float scale_g,
    uint packed_u, float scale_u,
    float4 x0, float4 x1,
    thread float& acc_g, thread float& acc_u)
{
    float qx_g = 0.0f;
    float qx_u = 0.0f;
    qx_g += float((packed_g       ) & 3u) * x0.x;
    qx_u += float((packed_u       ) & 3u) * x0.x;
    qx_g += float((packed_g >>  2u) & 3u) * x0.y;
    qx_u += float((packed_u >>  2u) & 3u) * x0.y;
    qx_g += float((packed_g >>  4u) & 3u) * x0.z;
    qx_u += float((packed_u >>  4u) & 3u) * x0.z;
    qx_g += float((packed_g >>  6u) & 3u) * x0.w;
    qx_u += float((packed_u >>  6u) & 3u) * x0.w;
    qx_g += float((packed_g >>  8u) & 3u) * x1.x;
    qx_u += float((packed_u >>  8u) & 3u) * x1.x;
    qx_g += float((packed_g >> 10u) & 3u) * x1.y;
    qx_u += float((packed_u >> 10u) & 3u) * x1.y;
    qx_g += float((packed_g >> 12u) & 3u) * x1.z;
    qx_u += float((packed_u >> 12u) & 3u) * x1.z;
    qx_g += float((packed_g >> 14u) & 3u) * x1.w;
    qx_u += float((packed_u >> 14u) & 3u) * x1.w;
    acc_g += qx_g * scale_g;
    acc_u += qx_u * scale_u;
}

kernel void affine2_xsum64(
    device const float* x    [[buffer(0)]],
    device float* xsum       [[buffer(1)]],
    constant uint& cols      [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]])
{
    if ((cols % 64u) != 0u) return;
    const uint gpr = cols >> 6u;
    if (tid >= gpr) return;
    const uint base = tid << 6u;
    float s = 0.0f;
    for (uint i = 0u; i < 64u; ++i) {
        s += x[base + i];
    }
    xsum[tid] = s;
}

kernel void affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       act_out     [[buffer(7)]],
    constant uint& rows             [[buffer(8)]],
    constant uint& cols             [[buffer(9)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            const float4 x0 = *((device const float4*)(input + col));
            const float4 x1 = *((device const float4*)(input + col + 4u));
            affine2_sa_unpack8_dual_vec(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
                x0, x1, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[4u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        const float g = red[t] + red[t + 1u];
        const float u = red[4u + t] + red[4u + t + 1u];
        act_out[row] = (g / (1.0f + exp(-g))) * u;
    }
}

kernel void affine2_group64_matvec_gate_up_geo_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       gate_out    [[buffer(7)]],
    device float*       up_out      [[buffer(8)]],
    constant uint& rows             [[buffer(9)]],
    constant uint& cols             [[buffer(10)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            const float4 x0 = *((device const float4*)(input + col));
            const float4 x1 = *((device const float4*)(input + col + 4u));
            affine2_sa_unpack8_dual_vec(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
                x0, x1, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[4u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        gate_out[row] = red[t] + red[t + 1u];
        up_out[row] = red[4u + t] + red[4u + t + 1u];
    }
}

kernel void affine2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       act_out     [[buffer(7)]],
    device const float* xsum        [[buffer(8)]],
    constant uint& rows             [[buffer(9)]],
    constant uint& cols             [[buffer(10)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        if (cols == 5120u) {
            for (uint k = 0u; k < 10u; ++k) {
                const uint col = lane_in_row * 8u + k * 512u;
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * groups_per_row + group;
                const uint byte0 = rgb * 16u + (local >> 2u);
                const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
                const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
                const float4 x0 = *((device const float4*)(input + col));
                const float4 x1 = *((device const float4*)(input + col + 4u));
                affine2_sa_unpack8_dual_qx(
                    gpacked, float(gate_scales[rgb]),
                    upacked, float(up_scales[rgb]),
                    x0, x1, acc_g, acc_u);
            }
        } else {
            for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * groups_per_row + group;
                const uint byte0 = rgb * 16u + (local >> 2u);
                const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
                const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
                const float4 x0 = *((device const float4*)(input + col));
                const float4 x1 = *((device const float4*)(input + col + 4u));
                affine2_sa_unpack8_dual_qx(
                    gpacked, float(gate_scales[rgb]),
                    upacked, float(up_scales[rgb]),
                    x0, x1, acc_g, acc_u);
            }
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[4u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        float g = red[t] + red[t + 1u];
        float u = red[4u + t] + red[4u + t + 1u];
        const uint gpr = cols >> 6u;
        if (cols == 5120u) {
            for (uint grp = 0u; grp < 80u; ++grp) {
                const uint rgb = row * 80u + grp;
                const float xs = xsum[grp];
                g += float(gate_biases[rgb]) * xs;
                u += float(up_biases[rgb]) * xs;
            }
        } else {
            for (uint grp = 0u; grp < gpr; ++grp) {
                const uint rgb = row * gpr + grp;
                const float xs = xsum[grp];
                g += float(gate_biases[rgb]) * xs;
                u += float(up_biases[rgb]) * xs;
            }
        }
        act_out[row] = (g / (1.0f + exp(-g))) * u;
    }
}

kernel void affine2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       act_out     [[buffer(7)]],
    device const float* xsum        [[buffer(8)]],
    constant uint& rows             [[buffer(9)]],
    constant uint& cols             [[buffer(10)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            const float4 x0 = *((device const float4*)(input + col));
            const float4 x1 = *((device const float4*)(input + col + 4u));
            affine2_sa_unpack8_dual_qx(
                gpacked, float(gate_scales[rgb]),
                upacked, float(up_scales[rgb]),
                x0, x1, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[4u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        const float g = red[t] + red[t + 1u];
        const float u = red[4u + t] + red[4u + t + 1u];
        act_out[row] = (g / (1.0f + exp(-g))) * u;
        (void)xsum;
        (void)gate_biases;
        (void)up_biases;
    }
}

kernel void affine2_group64_matvec_gate_up_swiglu_decode_probe_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       act_out     [[buffer(7)]],
    constant uint& rows             [[buffer(8)]],
    constant uint& cols             [[buffer(9)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            acc += float((gpacked) & 3u) + float((upacked) & 3u);
            acc += float((gpacked >> 14u) & 3u) + float((upacked >> 14u) & 3u);
            acc += float(gate_scales[rgb]) + float(up_biases[rgb]);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        act_out[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
    (void)input;
    (void)gate_biases;
    (void)up_scales;
}

kernel void affine2_group64_matvec_gate_up_swiglu_addr_probe_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       act_out     [[buffer(7)]],
    constant uint& rows             [[buffer(8)]],
    constant uint& cols             [[buffer(9)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            const float4 x0 = *((device const float4*)(input + col));
            const float4 x1 = *((device const float4*)(input + col + 4u));
            acc += x0.x + x1.w + float(gate_scales[rgb]) + float(up_biases[rgb]);
            acc += float(gpacked & 1u) + float(upacked & 1u);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        act_out[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
    (void)gate_biases;
    (void)up_scales;
}
