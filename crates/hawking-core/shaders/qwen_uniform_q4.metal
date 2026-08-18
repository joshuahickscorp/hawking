// Exact packed uniform-Q4 + FP16 group-scale matvec component for Ascension
// Qwen candidates.
//
// Layout is deliberately frozen and self-contained:
//   * each row is split into contiguous groups of 64 source weights;
//   * every group owns exactly 32 code bytes, including a final short group;
//   * code byte `i` stores the even local weight in its low nibble and the
//     odd local weight in its high nibble;
//   * each nibble is offset-binary signed Q4: `q = nibble - 8`, so q is in
//     [-8, 7]; and
//   * every group has one IEEE FP16 scale, reconstructed as `float(q) * scale`.
//
// This is a bounded component primitive. It is not a Qwen decoder, a token
// loop, a HCLI path, a TG measurement, or a model-TPS claim.

#include <metal_stdlib>
using namespace metal;

constant uint QWEN_UNIFORM_Q4_GROUP_SIZE = 64u;
constant uint QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP = 32u;

// Decode one flat-layout Q4 element (same packing as the matvec body).
static inline float qwen_uniform_q4_value(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uint local = element % group_size;
    const uint code_base = group * (group_size >> 1u);
    const uchar packed = codes[code_base + (local >> 1u)];
    const uchar nibble = (local & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4u);
    const int q = int(nibble) - 8;
    return float(q) * float(scales[group]);
}

kernel void qwen_uniform_q4_group64_matvec(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;

    float sum = 0.0f;
    const uint row_group_base = row * groups_per_row;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
        const uint group_base = row_group_base + group;
        const uint code_base = group_base * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP;
        const float scale = float(scales[group_base]);
        for (uint local_col = 0; local_col < group_len; ++local_col) {
            const uchar packed = codes[code_base + (local_col >> 1u)];
            const uchar nibble = (local_col & 1u) == 0u
                ? (packed & 0x0fu)
                : (packed >> 4u);
            const int q = int(nibble) - 8;
            sum += float(q) * scale * input[group_start + local_col];
        }
    }
    output[row] = sum;
}

// One Q4 weight; same packing as `qwen_uniform_q4_group64_matvec`.
static inline float qwen_uniform_q4_weight_at(
    device const uchar* codes,
    device const half* scales,
    uint row_group_base,
    uint col)
{
    const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
    const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE;
    const uint group_base = row_group_base + group;
    const uchar packed = codes[group_base * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)];
    const uchar nibble = ((local & 1u) == 0u) ? (packed & 0x0fu) : (packed >> 4u);
    return float(int(nibble) - 8) * float(scales[group_base]);
}

// Lever A: 256 threads, each owns R=4 rows with serial left-to-right dots.
// Bit-identical per-row association to `qwen_uniform_q4_group64_matvec`.
// Grid: (ceil(rows / 1024) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_rowblock(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]])
{
    constexpr uint R = 4u;
    constexpr uint kThreads = 256u;
    constexpr uint kRowsPerTg = kThreads * R;
    const uint row0 = group_id * kRowsPerTg + lid * R;
    if (row0 >= rows) return;

    uint rid[4];
    bool has[4];
    rid[0] = row0;
    has[0] = true;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < rows;
        if (!has[r]) rid[r] = row0;
    }
    uint rgb[4];
    for (uint r = 0u; r < R; ++r) {
        rgb[r] = rid[r] * groups_per_row;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n_groups = groups_per_row;
    for (uint group = 0u; group < n_groups; ++group) {
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_end = min(group_start + QWEN_UNIFORM_Q4_GROUP_SIZE, cols);
        for (uint col = group_start; col < group_end; ++col) {
            const float xv = input[col];
            for (uint r = 0u; r < R; ++r) {
                acc[r] += qwen_uniform_q4_weight_at(codes, scales, rgb[r], col) * xv;
            }
        }
    }
    for (uint r = 0u; r < R; ++r) {
        if (has[r]) output[rid[r]] = acc[r];
    }
}

// Lever B: 32 threads/row, 8 rows/TG, simd_sum. Not bit-identical.
// Grid: (ceil(rows / 8) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_simdgroup(
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
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;

    const uint row_group_base = row * groups_per_row;
    float partial = 0.0f;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        partial += qwen_uniform_q4_weight_at(codes, scales, row_group_base, col) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

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

// Geometry-sweep winner for Q4 gate [512, 2048]: 64 threads/row, 128-thread
// TG, 2 rows/TG. Packed decode stays in registers. Grid: ceil(rows/2)*128.
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
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// Group-128 sibling of geo_tpr64_tg128. Same TG / thread map / 8-wide
// unpack. Compile-time 128 so the group-64 kernel above stays untouched
// (a runtime group_size would put a non-constant divide on the G0 path).
// Address by (row, group) in 64-bit so rgb*64 cannot wrap in uint32.
// Grid: ceil(rows/2)*128, TG 128. Caller binds only when cols % 128 == 0.
constant uint QWEN_UNIFORM_Q4_GROUP_SIZE_128 = 128u;
constant uint QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP_128 = 64u;

kernel void qwen_uniform_q4_group128_matvec_geo_tpr64_tg128(
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
        const ulong rgb0 = (ulong)row * (ulong)groups_per_row;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE_128;
            const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE_128;
            const ulong rgb = rgb0 + (ulong)group;
            const float scale = float(scales[rgb]);
            const ulong code_off =
                rgb * (ulong)QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP_128
                + (ulong)(local >> 1u);
            const uint packed = *((device const uint*)(codes + code_off));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
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

// Diagnostic: same launch geometry as geo_tpr64_tg128, but only the
// addressing + DRAM load of scales and packed codes. The loaded values
// are sunk into `acc` so the compiler cannot DCE the traffic. No nibble
// unpack, no input-vector load, no FMA.
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe(
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
            acc += scale + as_type<float>(packed);
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
    (void)input;
}

static inline float qwen_uniform_q4_unpack8_noxin(uint packed, float scale)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        sum += float(int(byte & 0x0fu) - 8) * scale;
        sum += float(int(byte >> 4u) - 8) * scale;
    }
    return sum;
}

// Diagnostic: address + dequant, still no input-vector load / FMA.
// Difference vs addr_probe is the reconstruction ALU.
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe(
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
            acc += qwen_uniform_q4_unpack8_noxin(packed, scale);
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
    (void)input;
}

// Lever B-fast: 4 rows/simdgroup, 32 rows/TG. Not bit-identical.
// Grid: (ceil(rows / 32) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_simdgroup_rowblock4(
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
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) return;
    const uint row1 = row0 + 1u, row2 = row0 + 2u, row3 = row0 + 3u;
    const bool has1 = row1 < rows;
    const bool has2 = row2 < rows;
    const bool has3 = row3 < rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;
    const uint rgb0 = row0 * groups_per_row;
    const uint rgb1 = r1 * groups_per_row;
    const uint rgb2 = r2 * groups_per_row;
    const uint rgb3 = r3 * groups_per_row;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        const float xv = input[col];
        a0 += qwen_uniform_q4_weight_at(codes, scales, rgb0, col) * xv;
        a1 += qwen_uniform_q4_weight_at(codes, scales, rgb1, col) * xv;
        a2 += qwen_uniform_q4_weight_at(codes, scales, rgb2, col) * xv;
        a3 += qwen_uniform_q4_weight_at(codes, scales, rgb3, col) * xv;
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (simd_lane == 0u) {
        output[row0] = a0;
        if (has1) output[row1] = a1;
        if (has2) output[row2] = a2;
        if (has3) output[row3] = a3;
    }
}

// Fused Q+K+V, one thread per concatenated output row, serial L-to-R.
// Bit-identical to three `qwen_uniform_q4_group64_matvec` launches.
// Grid: (q_rows + k_rows + v_rows, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_qkv(
    device const uchar* q_codes     [[buffer(0)]],
    device const half*  q_scales    [[buffer(1)]],
    device const uchar* k_codes     [[buffer(2)]],
    device const half*  k_scales    [[buffer(3)]],
    device const uchar* v_codes     [[buffer(4)]],
    device const half*  v_scales    [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       q_output    [[buffer(7)]],
    device float*       k_output    [[buffer(8)]],
    device float*       v_output    [[buffer(9)]],
    constant uint& q_rows           [[buffer(10)]],
    constant uint& k_rows           [[buffer(11)]],
    constant uint& v_rows           [[buffer(12)]],
    constant uint& cols             [[buffer(13)]],
    constant uint& groups_per_row   [[buffer(14)]],
    uint tid                         [[thread_position_in_grid]])
{
    const uint total = q_rows + k_rows + v_rows;
    if (tid >= total) return;

    device const uchar* codes;
    device const half* scales;
    device float* out;
    uint local;
    if (tid < q_rows) {
        codes = q_codes; scales = q_scales; out = q_output; local = tid;
    } else if (tid < q_rows + k_rows) {
        codes = k_codes; scales = k_scales; out = k_output; local = tid - q_rows;
    } else {
        codes = v_codes; scales = v_scales; out = v_output; local = tid - q_rows - k_rows;
    }

    float sum = 0.0f;
    const uint row_group_base = local * groups_per_row;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
        const uint group_base = row_group_base + group;
        const uint code_base = group_base * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP;
        const float scale = float(scales[group_base]);
        for (uint local_col = 0; local_col < group_len; ++local_col) {
            const uchar packed = codes[code_base + (local_col >> 1u)];
            const uchar nibble = (local_col & 1u) == 0u
                ? (packed & 0x0fu)
                : (packed >> 4u);
            const int q = int(nibble) - 8;
            sum += float(q) * scale * input[group_start + local_col];
        }
    }
    out[local] = sum;
}

// Fused Q+K+V simdgroup (Lever B). Not bit-identical.
// Grid: (ceil((q+k+v)/8) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_qkv_simdgroup(
    device const uchar* q_codes     [[buffer(0)]],
    device const half*  q_scales    [[buffer(1)]],
    device const uchar* k_codes     [[buffer(2)]],
    device const half*  k_scales    [[buffer(3)]],
    device const uchar* v_codes     [[buffer(4)]],
    device const half*  v_scales    [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       q_output    [[buffer(7)]],
    device float*       k_output    [[buffer(8)]],
    device float*       v_output    [[buffer(9)]],
    constant uint& q_rows           [[buffer(10)]],
    constant uint& k_rows           [[buffer(11)]],
    constant uint& v_rows           [[buffer(12)]],
    constant uint& cols             [[buffer(13)]],
    constant uint& groups_per_row   [[buffer(14)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint tid = group_id * kSimdgroupsPerThreadgroup + simd_id;
    const uint total = q_rows + k_rows + v_rows;
    if (tid >= total) return;

    device const uchar* codes;
    device const half* scales;
    device float* out;
    uint local;
    if (tid < q_rows) {
        codes = q_codes; scales = q_scales; out = q_output; local = tid;
    } else if (tid < q_rows + k_rows) {
        codes = k_codes; scales = k_scales; out = k_output; local = tid - q_rows;
    } else {
        codes = v_codes; scales = v_scales; out = v_output; local = tid - q_rows - k_rows;
    }

    const uint row_group_base = local * groups_per_row;
    float partial = 0.0f;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        partial += qwen_uniform_q4_weight_at(codes, scales, row_group_base, col) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) out[local] = partial;
}

// Lever B-fast: 8 rows/simdgroup, 64 rows/TG. Not bit-identical.
// Grid: (ceil(rows / 64) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_simdgroup_rowblock8(
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
    constexpr uint R = 8u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) return;

    uint rid[8];
    bool has[8];
    uint rgb[8];
    rid[0] = row0;
    has[0] = true;
    rgb[0] = row0 * groups_per_row;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < rows;
        if (!has[r]) rid[r] = row0;
        rgb[r] = rid[r] * groups_per_row;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        const float xv = input[col];
        for (uint r = 0u; r < R; ++r) {
            acc[r] += qwen_uniform_q4_weight_at(codes, scales, rgb[r], col) * xv;
        }
    }
    for (uint r = 0u; r < R; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_lane == 0u) {
        for (uint r = 0u; r < R; ++r) {
            if (has[r]) output[rid[r]] = acc[r];
        }
    }
}

// 64 threads/row (two simdgroups), 4 rows/TG. Longer reduction, fewer
// iterations on the 2048-wide lm_head. Not bit-identical.
// Grid: (ceil(rows / 4) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_simdgroup_x64(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kRowsPerTg = 4u;
    constexpr uint kThreadsPerRow = 64u;
    const uint row = group_id * kRowsPerTg + (simd_id >> 1u);
    if (row >= rows) return;
    const uint sg_half = simd_id & 1u;
    const uint row_group_base = row * groups_per_row;
    float partial = 0.0f;
    for (uint base = 0u; base < cols; base += kThreadsPerRow) {
        const uint col = base + sg_half * 32u + simd_lane;
        if (col >= cols) continue;
        partial += qwen_uniform_q4_weight_at(codes, scales, row_group_base, col) * input[col];
    }
    partial = simd_sum(partial);
    threadgroup float sh[8];
    if (simd_lane == 0u) sh[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0u && sg_half == 0u) {
        output[row] = sh[simd_id] + sh[simd_id + 1u];
    }
    (void)lid;
}

// Decode a checked compact Q4 vector into a persistent f32 control buffer.
// Used for RMSNorm weights only; matrix bodies stay packed.
kernel void qwen_uniform_q4_decode_vector(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& elements    [[buffer(3)]],
    constant uint& group_size  [[buffer(4)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= elements) return;
    output[id] = qwen_uniform_q4_value(codes, scales, id, group_size);
}

// Direct packed Q4 embedding lookup — no host f32 embedding table.
kernel void qwen_uniform_q4_embedding_lookup(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& token       [[buffer(3)]],
    constant uint& hidden      [[buffer(4)]],
    constant uint& vocab       [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= hidden || token >= vocab) return;
    const uint element = token * hidden + id;
    output[id] = qwen_uniform_q4_value(codes, scales, element, group_size);
}

// Device-resident autoregressive feedback: previous step's argmax id stays in
// a device buffer (`sampled_token`) and is gathered without a host round-trip.
// Decode is identical to `qwen_uniform_q4_embedding_lookup` so host-token and
// device-token gathers stay bit-identical for the same id.
kernel void qwen_uniform_q4_embedding_lookup_device_token(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    device const uint* token_id [[buffer(3)]],
    constant uint& hidden      [[buffer(4)]],
    constant uint& vocab       [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    uint id                     [[thread_position_in_grid]])
{
    const uint token = token_id[0];
    if (id >= hidden || token >= vocab) return;
    const uint element = token * hidden + id;
    output[id] = qwen_uniform_q4_value(codes, scales, element, group_size);
}

// Fused final RMSNorm + uniform-Q4 lm_head. Each TG independently
// reconstructs the 2048-wide RMSNorm into threadgroup memory (same
// 256-thread tree + sqrt + 1/rms as rmsnorm_f32) then owns 64 vocab
// rows with the simdgroup8 Q4 reduction. This is the 151936×2048
// MAC bucket. Not bit-identical to the serial oracle.
//
// threadgroup(0): 256 reduce floats + 2048 x_hat floats.
// Grid: (ceil(rows / 64) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8(
    device const float* x            [[buffer(0)]],
    device const float* norm_weight  [[buffer(1)]],
    device const uchar* codes        [[buffer(2)]],
    device const half*  scales       [[buffer(3)]],
    device float*       x_norm       [[buffer(4)]],
    device float*       logits       [[buffer(5)]],
    constant uint& rows              [[buffer(6)]],
    constant uint& cols              [[buffer(7)]],
    constant uint& groups_per_row    [[buffer(8)]],
    constant float& eps              [[buffer(9)]],
    threadgroup float* sh            [[threadgroup(0)]],
    uint group_id                     [[threadgroup_position_in_grid]],
    uint lid                          [[thread_index_in_threadgroup]],
    uint simd_lane                    [[thread_index_in_simdgroup]],
    uint simd_id                      [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kThreads = 256u;
    constexpr uint R = 8u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;

    threadgroup float* reduce = sh;
    threadgroup float* x_hat = sh + kThreads;

    float partial = 0.0f;
    for (uint i = lid; i < cols; i += kThreads) {
        const float v = x[i];
        partial += v * v;
    }
    reduce[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) reduce[lid] += reduce[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms = sqrt(reduce[0] / float(cols) + eps);
    const float inv = 1.0f / rms;
    for (uint i = lid; i < cols; i += kThreads) {
        const float hat = x[i] * inv * norm_weight[i];
        x_hat[i] = hat;
        if (group_id == 0u) {
            x_norm[i] = hat;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) return;

    uint rid[8];
    bool has[8];
    uint rgb[8];
    rid[0] = row0;
    has[0] = true;
    rgb[0] = row0 * groups_per_row;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < rows;
        if (!has[r]) rid[r] = row0;
        rgb[r] = rid[r] * groups_per_row;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) continue;
        const float xv = x_hat[col];
        for (uint r = 0u; r < R; ++r) {
            acc[r] += qwen_uniform_q4_weight_at(codes, scales, rgb[r], col) * xv;
        }
    }
    for (uint r = 0u; r < R; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_lane == 0u) {
        for (uint r = 0u; r < R; ++r) {
            if (has[r]) logits[rid[r]] = acc[r];
        }
    }
}

// Per-group [fp16 scale | 32 code bytes]. Same q = nibble-8 decode as the
// split-buffer kernel. Grid: (rows, 1, 1), TG 256.
kernel void qwen_uniform_q4_group64_matvec_interleaved(
    device const uchar* records     [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant uint& rows             [[buffer(3)]],
    constant uint& cols             [[buffer(4)]],
    constant uint& groups_per_row   [[buffer(5)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint stride = 2u + QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP;
    float sum = 0.0f;
    const uint row_base = row * groups_per_row;
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint rec = (row_base + group) * stride;
        const float scale = float(*((device const half*)(records + rec)));
        device const uchar* codes = records + rec + 2u;
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
        for (uint local_col = 0u; local_col < group_len; ++local_col) {
            const uchar packed = codes[local_col >> 1u];
            const uchar nibble = (local_col & 1u) == 0u
                ? (packed & 0x0fu)
                : (packed >> 4u);
            sum += float(int(nibble) - 8) * scale * input[group_start + local_col];
        }
    }
    output[row] = sum;
}

// One simdgroup per row. Each lane owns one code byte (two weights) of the
// 64-wide group, so a group is one iteration instead of 64 scalar weight_at
// calls. Grid: (ceil(rows / 8) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_vecgroup(
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
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) return;
    const uint row_group_base = row * groups_per_row;
    const uint local0 = simd_lane << 1u;
    const uint local1 = local0 + 1u;
    float partial = 0.0f;
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
        if (local0 >= group_len) continue;
        const uint group_base = row_group_base + group;
        const float scale = float(scales[group_base]);
        const uchar packed = codes[group_base * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + simd_lane];
        const float q0 = float(int(packed & 0x0fu) - 8);
        partial += q0 * scale * input[group_start + local0];
        if (local1 < group_len) {
            const float q1 = float(int(packed >> 4u) - 8);
            partial += q1 * scale * input[group_start + local1];
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

// vecgroup plus one cooperative load of X into threadgroup memory.
// Tile is 2048 floats (8 KiB). Grid: (ceil(rows / 8) * 256, 1, 1), TG 256.
kernel void qwen_uniform_q4_group64_matvec_vecgroup_x(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kXTile = 2048u;
    threadgroup float x_tg[kXTile];
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    const bool live = row < rows;
    const uint row_group_base = live ? row * groups_per_row : 0u;
    const uint local0 = simd_lane << 1u;
    const uint local1 = local0 + 1u;
    float partial = 0.0f;
    for (uint tile = 0u; tile < cols; tile += kXTile) {
        const uint tile_n = min(kXTile, cols - tile);
        for (uint i = lid; i < tile_n; i += 256u) {
            x_tg[i] = input[tile + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            const uint g0 = tile / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint g1 = min(groups_per_row, (tile + tile_n + QWEN_UNIFORM_Q4_GROUP_SIZE - 1u)
                / QWEN_UNIFORM_Q4_GROUP_SIZE);
            for (uint group = g0; group < g1; ++group) {
                const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
                const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
                if (local0 >= group_len) continue;
                const uint group_base = row_group_base + group;
                const float scale = float(scales[group_base]);
                const uchar packed =
                    codes[group_base * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + simd_lane];
                const uint x0 = group_start + local0 - tile;
                const float q0 = float(int(packed & 0x0fu) - 8);
                partial += q0 * scale * x_tg[x0];
                if (local1 < group_len) {
                    const float q1 = float(int(packed >> 4u) - 8);
                    partial += q1 * scale * x_tg[x0 + 1u];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

// 4 rows / simdgroup, 32 rows / TG. Vectorized group decode, X from device
// (one pair of floats reused across the four rows). Grid: ceil(rows/32)*256.
kernel void qwen_uniform_q4_group64_matvec_vecgroup_r4(
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
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) return;
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const bool has1 = row1 < rows;
    const bool has2 = row2 < rows;
    const bool has3 = row3 < rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;
    const uint rgb0 = row0 * groups_per_row;
    const uint rgb1 = r1 * groups_per_row;
    const uint rgb2 = r2 * groups_per_row;
    const uint rgb3 = r3 * groups_per_row;
    const uint local0 = simd_lane << 1u;
    const uint local1 = local0 + 1u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
        if (local0 >= group_len) continue;
        const float xv0 = input[group_start + local0];
        const float xv1 = (local1 < group_len) ? input[group_start + local1] : 0.0f;
        const uint code_off = simd_lane;
        const uchar p0 = codes[(rgb0 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
        const uchar p1 = codes[(rgb1 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
        const uchar p2 = codes[(rgb2 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
        const uchar p3 = codes[(rgb3 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
        const float s0 = float(scales[rgb0 + group]);
        const float s1 = float(scales[rgb1 + group]);
        const float s2 = float(scales[rgb2 + group]);
        const float s3 = float(scales[rgb3 + group]);
        a0 += float(int(p0 & 0x0fu) - 8) * s0 * xv0;
        a1 += float(int(p1 & 0x0fu) - 8) * s1 * xv0;
        a2 += float(int(p2 & 0x0fu) - 8) * s2 * xv0;
        a3 += float(int(p3 & 0x0fu) - 8) * s3 * xv0;
        if (local1 < group_len) {
            a0 += float(int(p0 >> 4u) - 8) * s0 * xv1;
            a1 += float(int(p1 >> 4u) - 8) * s1 * xv1;
            a2 += float(int(p2 >> 4u) - 8) * s2 * xv1;
            a3 += float(int(p3 >> 4u) - 8) * s3 * xv1;
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (simd_lane == 0u) {
        output[row0] = a0;
        if (has1) output[row1] = a1;
        if (has2) output[row2] = a2;
        if (has3) output[row3] = a3;
    }
}

// 4 rows / simdgroup + X tile in threadgroup memory.
// Grid: (ceil(rows / 32) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_vecgroup_r4_x(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    constexpr uint kXTile = 2048u;
    threadgroup float x_tg[kXTile];
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    const bool live = row0 < rows;
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const bool has1 = live && row1 < rows;
    const bool has2 = live && row2 < rows;
    const bool has3 = live && row3 < rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;
    const uint rgb0 = live ? row0 * groups_per_row : 0u;
    const uint rgb1 = live ? r1 * groups_per_row : 0u;
    const uint rgb2 = live ? r2 * groups_per_row : 0u;
    const uint rgb3 = live ? r3 * groups_per_row : 0u;
    const uint local0 = simd_lane << 1u;
    const uint local1 = local0 + 1u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint tile = 0u; tile < cols; tile += kXTile) {
        const uint tile_n = min(kXTile, cols - tile);
        for (uint i = lid; i < tile_n; i += 256u) {
            x_tg[i] = input[tile + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            const uint g0 = tile / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint g1 = min(groups_per_row, (tile + tile_n + QWEN_UNIFORM_Q4_GROUP_SIZE - 1u)
                / QWEN_UNIFORM_Q4_GROUP_SIZE);
            for (uint group = g0; group < g1; ++group) {
                const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
                const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
                if (local0 >= group_len) continue;
                const uint x0 = group_start + local0 - tile;
                const float xv0 = x_tg[x0];
                const float xv1 = (local1 < group_len) ? x_tg[x0 + 1u] : 0.0f;
                const uint code_off = simd_lane;
                const uchar p0 =
                    codes[(rgb0 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
                const uchar p1 =
                    codes[(rgb1 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
                const uchar p2 =
                    codes[(rgb2 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
                const uchar p3 =
                    codes[(rgb3 + group) * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + code_off];
                const float s0 = float(scales[rgb0 + group]);
                const float s1 = float(scales[rgb1 + group]);
                const float s2 = float(scales[rgb2 + group]);
                const float s3 = float(scales[rgb3 + group]);
                a0 += float(int(p0 & 0x0fu) - 8) * s0 * xv0;
                a1 += float(int(p1 & 0x0fu) - 8) * s1 * xv0;
                a2 += float(int(p2 & 0x0fu) - 8) * s2 * xv0;
                a3 += float(int(p3 & 0x0fu) - 8) * s3 * xv0;
                if (local1 < group_len) {
                    a0 += float(int(p0 >> 4u) - 8) * s0 * xv1;
                    a1 += float(int(p1 >> 4u) - 8) * s1 * xv1;
                    a2 += float(int(p2 >> 4u) - 8) * s2 * xv1;
                    a3 += float(int(p3 >> 4u) - 8) * s3 * xv1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (simd_lane == 0u) {
        output[row0] = a0;
        if (has1) output[row1] = a1;
        if (has2) output[row2] = a2;
        if (has3) output[row3] = a3;
    }
}

// Vectorized group decode at the winning x64 occupancy: 64 threads/row,
// 4 rows/TG. The two simdgroups of a row take even/odd groups.
// Grid: (ceil(rows / 4) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_uniform_q4_group64_matvec_vecgroup_x64(
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
    constexpr uint kRowsPerTg = 4u;
    const uint row = group_id * kRowsPerTg + (simd_id >> 1u);
    if (row >= rows) return;
    const uint sg_half = simd_id & 1u;
    const uint row_group_base = row * groups_per_row;
    const uint local0 = simd_lane << 1u;
    const uint local1 = local0 + 1u;
    float partial = 0.0f;
    for (uint group = sg_half; group < groups_per_row; group += 2u) {
        const uint group_start = group * QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint group_len = min(QWEN_UNIFORM_Q4_GROUP_SIZE, cols - group_start);
        if (local0 >= group_len) continue;
        const uint group_base = row_group_base + group;
        const float scale = float(scales[group_base]);
        const uchar packed = codes[group_base * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + simd_lane];
        const float q0 = float(int(packed & 0x0fu) - 8);
        partial += q0 * scale * input[group_start + local0];
        if (local1 < group_len) {
            const float q1 = float(int(packed >> 4u) - 8);
            partial += q1 * scale * input[group_start + local1];
        }
    }
    partial = simd_sum(partial);
    threadgroup float sh[8];
    if (simd_lane == 0u) sh[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0u && sg_half == 0u) {
        output[row] = sh[simd_id] + sh[simd_id + 1u];
    }
}

// ── K-column geo_tpr64: amortize the weight sweep over K positions ────────
//
// Decode is bandwidth-bound at K=1: the measured single-GEMV roof is
// 699.57 GB/s on 13.6 GB of Q4 codes, and decode+FMA are a 4.7% tax on top
// (HONEST_ROOF_WEIGHT_ADDRESSING.json). One full weight sweep per token puts
// 100 TPS out of reach for every representation that has shown capability
// (NX_TPS_FRONTIER.json). The only axis that moves that wall is emitting K
// positions per sweep.
//
// Thread map, launch geometry and per-weight arithmetic are IDENTICAL to
// qwen_uniform_q4_group64_matvec_geo_tpr64_tg128. The single change: each
// decoded weight is multiplied into K accumulators instead of one, so the
// same code byte serves K positions. Bytes stay flat, FLOPs scale K.
//
// Activations are position-interleaved -- input[col * K + k] -- so the K
// values a thread needs for one column are contiguous. Output matches:
// output[row * K + k]. At K == 1 both collapse to the matvec layout and the
// arithmetic is bit-identical to it (same (q*scale)*x association, same
// accumulation order, same two-stage reduction).
//
// Grid: ceil(rows / 2) * 128, TG (128, 1, 1). Same as the matvec.

template <uint K>
static inline void qwen_uniform_q4_unpack8_mac_k(
    uint packed,
    float scale,
    device const float* x,
    uint col,
    thread float* acc)
{
    for (uint i = 0u; i < 4u; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        const float w_even = float(int(byte & 0x0fu) - 8) * scale;
        const float w_odd = float(int(byte >> 4u) - 8) * scale;
        const uint c_even = (col + 2u * i) * K;
        const uint c_odd = c_even + K;
        for (uint k = 0u; k < K; ++k) {
            acc[k] += w_even * x[c_even + k];
            acc[k] += w_odd * x[c_odd + k];
        }
    }
}

template <uint K>
static inline void qwen_uniform_q4_geo_tpr64_matmul_k_body(
    device const uchar* codes,
    device const half* scales,
    device const float* input,
    device float* output,
    uint rows,
    uint cols,
    uint groups_per_row,
    threadgroup float* red,
    uint group_id,
    uint simd_lane,
    uint simd_id)
{
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;

    float acc[K];
    for (uint k = 0u; k < K; ++k) {
        acc[k] = 0.0f;
    }
    if (row < rows) {
        const uint rgb0 = row * groups_per_row;
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint rgb = rgb0 + group;
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            qwen_uniform_q4_unpack8_mac_k<K>(packed, scale, input, col, acc);
        }
    }
    for (uint k = 0u; k < K; ++k) {
        const float summed = simd_sum(acc[k]);
        if (simd_lane == 0u) {
            red[k * 4u + simd_id] = summed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        for (uint k = 0u; k < K; ++k) {
            output[row * K + k] =
                red[k * 4u + team * kSplit] + red[k * 4u + team * kSplit + 1u];
        }
    }
}

#define QWEN_UNIFORM_Q4_MATMUL_K(KVAL)                                        \
kernel void qwen_uniform_q4_group64_matmul_k##KVAL##_geo_tpr64_tg128(         \
    device const uchar* codes       [[buffer(0)]],                            \
    device const half* scales       [[buffer(1)]],                            \
    device const float* input       [[buffer(2)]],                            \
    device float* output            [[buffer(3)]],                            \
    constant uint& rows             [[buffer(4)]],                            \
    constant uint& cols             [[buffer(5)]],                            \
    constant uint& groups_per_row   [[buffer(6)]],                            \
    uint group_id                    [[threadgroup_position_in_grid]],        \
    uint simd_lane                   [[thread_index_in_simdgroup]],           \
    uint simd_id                     [[simdgroup_index_in_threadgroup]])      \
{                                                                             \
    threadgroup float red[4u * KVAL];                                         \
    qwen_uniform_q4_geo_tpr64_matmul_k_body<KVAL>(                            \
        codes, scales, input, output, rows, cols, groups_per_row,             \
        red, group_id, simd_lane, simd_id);                                   \
}

QWEN_UNIFORM_Q4_MATMUL_K(1)
QWEN_UNIFORM_Q4_MATMUL_K(2)
QWEN_UNIFORM_Q4_MATMUL_K(4)
QWEN_UNIFORM_Q4_MATMUL_K(8)

#undef QWEN_UNIFORM_Q4_MATMUL_K

// ── R x K tiled geo_tpr64: fix the activation:code ratio ──────────────────
//
// NX_MATMUL_K_AMORTIZATION.json (first pass) measured the naive K-column
// kernel above and REFUTED it: K=4 amortized only 1.19x and K=8 was a net
// loss. The cause is visible in the byte ratios -- a Q4 code byte holds two
// weights, so it consumes 8 bytes of f32 activation. Activation traffic is
// 8x code traffic at K=1 and 8K at K=1 rows/thread:
//
//   K=1  code  633 GB/s   activation  5065 GB/s   ratio  8:1
//   K=4  code  188 GB/s   activation  6020 GB/s   ratio 32:1
//   K=8  code   65 GB/s   activation  4183 GB/s   ratio 64:1
//
// Adding accumulators does not make the sweep cheaper if each accumulator
// drags its own activation stream. The ratio is fixed by R, the number of
// ROWS a thread serves from one activation load: 8K/R bytes of activation
// per byte of code. R == K restores the K=1 ratio, so the code stream should
// return to its K=1 rate while serving K positions.
//
// Each thread holds R*K accumulators and loads 8*K activations per 4*R code
// bytes. Launch geometry, thread map and per-weight arithmetic are otherwise
// unchanged from the matvec. Rows per TG = 2 * R.
//
// Grid: ceil(rows / (2*R)) * 128, TG (128, 1, 1).

template <uint R, uint K>
static inline void qwen_uniform_q4_geo_tpr64_matmul_rk_body(
    device const uchar* codes,
    device const half* scales,
    device const float* input,
    device float* output,
    uint rows,
    uint cols,
    uint groups_per_row,
    threadgroup float* red,
    uint group_id,
    uint simd_lane,
    uint simd_id)
{
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row0 = (group_id * 2u + team) * R;

    // Live registers: R packed + R scales + R*K accumulators + 2*K staged
    // activations. Staging all 8*K activations at once (the first version of
    // this kernel) spilled at K=4 and cost more than it saved -- R=4 K=4
    // amortized 1.33x against R=2 K=2's 1.73x. Only one code-byte's worth of
    // activations is held live here.
    float acc[R * K];
    for (uint i = 0u; i < R * K; ++i) {
        acc[i] = 0.0f;
    }

    for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
        const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint local_byte = (col - group * QWEN_UNIFORM_Q4_GROUP_SIZE) >> 1u;

        uint packed[R];
        float scale[R];
        for (uint r = 0u; r < R; ++r) {
            const uint row = row0 + r;
            const uint safe = row < rows ? row : (rows - 1u);
            const uint rgb = safe * groups_per_row + group;
            scale[r] = row < rows ? float(scales[rgb]) : 0.0f;
            packed[r] = *((device const uint*)(codes
                + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + local_byte));
        }

        for (uint i = 0u; i < 4u; ++i) {
            // One activation pair load feeds all R rows.
            float xe[K];
            float xo[K];
            const uint base_e = (col + 2u * i) * K;
            for (uint k = 0u; k < K; ++k) {
                xe[k] = input[base_e + k];
                xo[k] = input[base_e + K + k];
            }
            for (uint r = 0u; r < R; ++r) {
                const uint byte = (packed[r] >> (8u * i)) & 0xffu;
                const float w_even = float(int(byte & 0x0fu) - 8) * scale[r];
                const float w_odd = float(int(byte >> 4u) - 8) * scale[r];
                for (uint k = 0u; k < K; ++k) {
                    acc[r * K + k] += w_even * xe[k];
                    acc[r * K + k] += w_odd * xo[k];
                }
            }
        }
    }

    for (uint i = 0u; i < R * K; ++i) {
        const float summed = simd_sum(acc[i]);
        if (simd_lane == 0u) {
            red[i * 4u + simd_id] = summed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        for (uint r = 0u; r < R; ++r) {
            const uint row = row0 + r;
            if (row >= rows) {
                break;
            }
            for (uint k = 0u; k < K; ++k) {
                const uint i = r * K + k;
                output[row * K + k] =
                    red[i * 4u + team * kSplit] + red[i * 4u + team * kSplit + 1u];
            }
        }
    }
}

#define QWEN_UNIFORM_Q4_MATMUL_RK(RVAL, KVAL)                                 \
kernel void qwen_uniform_q4_group64_matmul_r##RVAL##k##KVAL##_geo_tpr64_tg128(\
    device const uchar* codes       [[buffer(0)]],                            \
    device const half* scales       [[buffer(1)]],                            \
    device const float* input       [[buffer(2)]],                            \
    device float* output            [[buffer(3)]],                            \
    constant uint& rows             [[buffer(4)]],                            \
    constant uint& cols             [[buffer(5)]],                            \
    constant uint& groups_per_row   [[buffer(6)]],                            \
    uint group_id                    [[threadgroup_position_in_grid]],        \
    uint simd_lane                   [[thread_index_in_simdgroup]],           \
    uint simd_id                     [[simdgroup_index_in_threadgroup]])      \
{                                                                             \
    threadgroup float red[4u * RVAL * KVAL];                                  \
    qwen_uniform_q4_geo_tpr64_matmul_rk_body<RVAL, KVAL>(                     \
        codes, scales, input, output, rows, cols, groups_per_row,             \
        red, group_id, simd_lane, simd_id);                                   \
}

QWEN_UNIFORM_Q4_MATMUL_RK(2, 2)
QWEN_UNIFORM_Q4_MATMUL_RK(4, 4)
QWEN_UNIFORM_Q4_MATMUL_RK(8, 4)
QWEN_UNIFORM_Q4_MATMUL_RK(8, 8)
QWEN_UNIFORM_Q4_MATMUL_RK(2, 4)
QWEN_UNIFORM_Q4_MATMUL_RK(4, 2)
QWEN_UNIFORM_Q4_MATMUL_RK(4, 8)
QWEN_UNIFORM_Q4_MATMUL_RK(16, 4)

#undef QWEN_UNIFORM_Q4_MATMUL_RK

// ── binary-plane matvec: W ~ s1*P1 + s2*P2 + ..., each Pi a sign plane ────
//
// G033 measured this family winning the low-bit end offline: one plane at
// 1.2500 b/elem holds 0.796776 against flat q2's 0.772929 at 2.2500, and two
// planes reach 0.933975 at 2.5000. CODEC_ALU_COST then set the bar every codec
// has to clear -- 0.810 ps/element, where q4 sits at 88% of the bandwidth roof
// and q3 already fails at 0.855. So the family lives or dies on decode ALU.
//
// The reason the prior is favourable here: a sign plane needs a bit test and a
// select, not a field extract that crosses byte boundaries, and its weight is
// applied by choosing +s or -s rather than converting an integer and
// multiplying. The K per-plane contributions are summed FIRST and the
// activation is touched ONCE, so the cost is K selects plus K adds plus one
// FMA per weight, against q4's shift, mask, convert, multiply and FMA.
//
// Layout, matching the accounting in tools/gravity_planes_ladder.py exactly:
//   codes  [(row * groups_per_row + group) * K + k] * 8 + byte_in_group
//          one bit per weight per plane, 8 weights per byte, 8 bytes per
//          group of 64 per plane
//   scales [(row * groups_per_row + group) * K + k]   one f16 per group per plane
// so a plane costs 1 + 16/64 bits/elem and K planes cost K * 1.25.
//
// Grid: ceil(rows/2)*128, TG 128. Same thread map as geo_tpr64.

template <uint K>
static inline void qwen_binary_planes_geo_tpr64_body(
    device const uchar* codes,
    device const half* scales,
    device const float* input,
    device float* output,
    uint rows,
    uint cols,
    uint groups_per_row,
    threadgroup float* red,
    uint group_id,
    uint simd_lane,
    uint simd_id)
{
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint local = col - group * QWEN_UNIFORM_Q4_GROUP_SIZE;
            const uint gbase = (row * groups_per_row + group) * K;
            uchar plane[K];
            float s[K];
            for (uint k = 0u; k < K; ++k) {
                plane[k] = codes[(gbase + k) * 8u + (local >> 3u)];
                s[k] = float(scales[gbase + k]);
            }
            for (uint e = 0u; e < 8u; ++e) {
                float w = 0.0f;
                for (uint k = 0u; k < K; ++k) {
                    w += ((plane[k] >> e) & 1u) ? s[k] : -s[k];
                }
                acc += w * input[col + e];
            }
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

#define QWEN_BINARY_PLANES(KVAL)                                              \
kernel void qwen_binary_planes_k##KVAL##_matvec_geo_tpr64_tg128(              \
    device const uchar* codes       [[buffer(0)]],                            \
    device const half* scales       [[buffer(1)]],                            \
    device const float* input       [[buffer(2)]],                            \
    device float* output            [[buffer(3)]],                            \
    constant uint& rows             [[buffer(4)]],                            \
    constant uint& cols             [[buffer(5)]],                            \
    constant uint& groups_per_row   [[buffer(6)]],                            \
    uint group_id                    [[threadgroup_position_in_grid]],        \
    uint simd_lane                   [[thread_index_in_simdgroup]],           \
    uint simd_id                     [[simdgroup_index_in_threadgroup]])      \
{                                                                             \
    threadgroup float red[4];                                                 \
    qwen_binary_planes_geo_tpr64_body<KVAL>(                                  \
        codes, scales, input, output, rows, cols, groups_per_row,             \
        red, group_id, simd_lane, simd_id);                                   \
}

QWEN_BINARY_PLANES(1)
QWEN_BINARY_PLANES(2)
QWEN_BINARY_PLANES(3)

#undef QWEN_BINARY_PLANES
