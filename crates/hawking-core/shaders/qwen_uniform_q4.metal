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
