// Q80 mixed-representation decode kernels.
//
// Contract: packed bytes are read directly. A value is decoded in registers
// and consumed in the matvec in the same kernel. These kernels must never
// write a dense (rows × cols) weight reconstruction.
//
// Pack-lane authority (kernel-facing bodies; JSON envelopes are stripped
// before bind):
//
//   gate_proj  binary_group
//     magic HGRAVB01 / hawking.gravity.binary_sign_scale.v1
//     body  = fp16 scales[groups] || sign bits (LSB-first)
//     group_size = 128, scale = stored fp16 (codec uses mean-abs)
//
//   up_proj    binary + rice_q1_rms @ 2%
//     magic HGRAVR02 / hawking.gravity.binary_outlier_residual.v2
//     body  = fp16 scales || signs || u32 first_index || rice(diffs)
//             || fp16 rms_scale || 1-bit residual signs
//     rice  = unary quotient (1-bits) + 0 + k LSBs, LSB-first bitstream
//
//   down_proj  hgravs01_r160_b3
//     magic HGRAVS01 factor bodies (left then right)
//     each  = fp16 scales[groups] || packed unsigned codes
//     bits  = 3, group_size = 64, q = code - 3, value = q * scale
//     execute y = L @ (R @ x); mid[rank] is the only temporary.

#include <metal_stdlib>
using namespace metal;

// ── binary_group ──────────────────────────────────────────────────────────

static inline float q80_binary_group_serial_row(
    device const uchar* signs,
    device const half* scales,
    device const float* input,
    uint row,
    uint cols,
    uint group_size,
    uint groups_per_row)
{
    return gk_binary_group_serial_row(
        signs, scales, input, row, cols, group_size, groups_per_row);
}

// Grid: (rows, 1, 1), threadgroup: (256, 1, 1).
kernel void q80_binary_group_matvec(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    output[row] = q80_binary_group_serial_row(
        signs, scales, input, row, cols, group_size, groups_per_row);
}

// ── rice_q1 residual (no dense W) ─────────────────────────────────────────

struct Q80RiceReader {
    device const uchar* data;
    uint byte_count;
    uint byte_i;
    uint bit_i;
};

static inline uint q80_rice_read_bit(thread Q80RiceReader& r)
{
    if (r.byte_i >= r.byte_count) {
        return 0u;
    }
    const uint bit = (uint(r.data[r.byte_i]) >> r.bit_i) & 1u;
    r.bit_i += 1u;
    if (r.bit_i == 8u) {
        r.bit_i = 0u;
        r.byte_i += 1u;
    }
    return bit;
}

static inline uint q80_rice_read_lsbs(thread Q80RiceReader& r, uint k)
{
    uint value = 0u;
    for (uint i = 0u; i < k; ++i) {
        value |= q80_rice_read_bit(r) << i;
    }
    return value;
}

static inline uint q80_rice_read_value(thread Q80RiceReader& r, uint k)
{
    uint q = 0u;
    while (q80_rice_read_bit(r) == 1u) {
        q += 1u;
        if (q > 0x00ffffffu) {
            break;
        }
    }
    const uint rem = (k == 0u) ? 0u : q80_rice_read_lsbs(r, k);
    return (q << k) | rem;
}

static inline float q80_residual_q1_value(
    device const uchar* signs,
    uint outlier_index,
    float scale)
{
    const uchar byte = signs[outlier_index >> 3u];
    const bool positive = ((byte >> (outlier_index & 7u)) & 1u) != 0u;
    return positive ? scale : -scale;
}

// Serial rice decode + scatter-add into y. One lane does the whole stream so
// the add order matches the CPU oracle (increasing packed index). Grid may be
// any non-zero size; only thread 0 works. Temporary: a few registers.
kernel void q80_rice_q1_residual_apply(
    device const uchar* rice_bytes      [[buffer(0)]],
    device const uchar* residual_signs  [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    constant uint& first_index          [[buffer(4)]],
    constant uint& rice_k               [[buffer(5)]],
    constant uint& rice_byte_count      [[buffer(6)]],
    constant uint& outlier_count        [[buffer(7)]],
    constant uint& cols                 [[buffer(8)]],
    constant uint& residual_scale_bits  [[buffer(9)]],
    uint tid                             [[thread_position_in_grid]])
{
    if (tid != 0u || outlier_count == 0u || cols == 0u) {
        return;
    }
    const float scale = float(as_type<half>(ushort(residual_scale_bits)));
    uint index = first_index;
    {
        const float v = q80_residual_q1_value(residual_signs, 0u, scale);
        output[index / cols] += v * input[index % cols];
    }
    Q80RiceReader reader;
    reader.data = rice_bytes;
    reader.byte_count = rice_byte_count;
    reader.byte_i = 0u;
    reader.bit_i = 0u;
    for (uint n = 1u; n < outlier_count; ++n) {
        index += q80_rice_read_value(reader, rice_k);
        const float v = q80_residual_q1_value(residual_signs, n, scale);
        output[index / cols] += v * input[index % cols];
    }
}

// Per-token residual apply from bind-time expanded, sorted indices.
// `row_ptr` is CSR over those indices (already row-major). One thread per
// output row; add order matches serial rice apply. Grid: (rows,1,1), TG 256.
kernel void q80_sparse_q1_apply_csr(
    device const uint* indices          [[buffer(0)]],
    device const uint* row_ptr          [[buffer(1)]],
    device const uchar* residual_signs  [[buffer(2)]],
    device const float* input           [[buffer(3)]],
    device float* output                [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& cols                 [[buffer(6)]],
    constant uint& residual_scale_bits  [[buffer(7)]],
    uint row                             [[thread_position_in_grid]])
{
    if (row >= rows || cols == 0u) {
        return;
    }
    const float scale = float(as_type<half>(ushort(residual_scale_bits)));
    const uint begin = row_ptr[row];
    const uint end = row_ptr[row + 1u];
    float acc = output[row];
    for (uint n = begin; n < end; ++n) {
        const uint col = indices[n] % cols;
        acc += q80_residual_q1_value(residual_signs, n, scale) * input[col];
    }
    output[row] = acc;
}

// Bind-time rice expand: writes uint32 indices, never a dense W.
// Same serial decoder as the apply kernel.
kernel void q80_rice_q1_expand_indices(
    device const uchar* rice_bytes      [[buffer(0)]],
    device uint* indices                [[buffer(1)]],
    constant uint& first_index          [[buffer(2)]],
    constant uint& rice_k               [[buffer(3)]],
    constant uint& rice_byte_count      [[buffer(4)]],
    constant uint& outlier_count        [[buffer(5)]],
    uint tid                             [[thread_position_in_grid]])
{
    if (tid != 0u || outlier_count == 0u) {
        return;
    }
    uint index = first_index;
    indices[0] = index;
    Q80RiceReader reader;
    reader.data = rice_bytes;
    reader.byte_count = rice_byte_count;
    reader.byte_i = 0u;
    reader.bit_i = 0u;
    for (uint n = 1u; n < outlier_count; ++n) {
        index += q80_rice_read_value(reader, rice_k);
        indices[n] = index;
    }
}

// ── hgravs01 uniform factor (3-bit group-64) ──────────────────────────────

static inline uint q80_uniform_extract(
    device const uchar* codes,
    uint element,
    uint bits)
{
    return gk_uniform_extract(codes, element, bits);
}

static inline float q80_uniform_value(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    return gk_uniform_value(codes, scales, element, group_size, bits, bound);
}

// Serial left-to-right f32 association. Grid: (rows, 1, 1), TG: (256, 1, 1).
// Groups are along the flattened factor (row-major), not necessarily
// row-aligned — down_proj L is [2048, 160] and 160 % 64 != 0.
kernel void q80_hgravs01_factor_matvec(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float sum = 0.0f;
    const uint row_base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        const uint element = row_base + col;
        sum += q80_uniform_value(codes, scales, element, group_size, bits, bound)
            * input[col];
    }
    output[row] = sum;
}

// ── throughput path (decode in registers / simdgroup, never dense W) ──────
//
// Serial kernels above are the association-preserving baseline. These kernels
// raise occupancy: one simdgroup owns one (or R) output row(s), columns walk
// in 32-wide tiles, simd_sum reduces. A value is still decoded into a register
// and consumed in the same FMA. Nothing writes a (rows × cols) reconstruction.

static inline uint q80_uniform_extract_wide(
    device const uchar* codes,
    uint element,
    uint bits)
{
    return gk_uniform_extract_wide(codes, element, bits);
}

static inline float q80_uniform_value_wide(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    return gk_uniform_value_wide(codes, scales, element, group_size, bits, bound);
}

static inline float q80_binary_lane_term(
    device const uchar* signs,
    device const half* scales,
    device const float* input,
    uint row_base,
    uint scale_base,
    uint col,
    uint group_size)
{
    return gk_binary_lane_term(
        signs, scales, input, row_base, scale_base, col, group_size);
}

// Eight consecutive columns from one sign byte. Col must be 8-aligned and
// lie inside a single scale group (true for group_size=128 and col%8==0).
static inline float q80_binary_byte_dot(
    device const uchar* signs,
    device const half* scales,
    device const float* input,
    uint row_base,
    uint scale_base,
    uint col,
    uint group_size)
{
    const float scale = float(scales[scale_base + col / group_size]);
    const uchar byte = signs[(row_base + col) >> 3u];
    float sum = 0.0f;
    sum += ((byte & 0x01u) ? scale : -scale) * input[col];
    sum += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
    sum += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
    sum += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
    sum += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
    sum += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
    sum += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
    sum += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
    return sum;
}

// One simdgroup per row, 8 rows / 256-thread TG. Grid: ceil(rows/8)*256.
kernel void q80_binary_group_matvec_simd(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        partial += q80_binary_lane_term(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// 4 rows / simdgroup, 32 rows / TG. Grid: ceil(rows/32)*256.
kernel void q80_binary_group_matvec_rowblock4(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) {
        return;
    }
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const bool has1 = row1 < rows;
    const bool has2 = row2 < rows;
    const bool has3 = row3 < rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;
    const uint rb0 = row0 * cols, rb1 = r1 * cols, rb2 = r2 * cols, rb3 = r3 * cols;
    const uint sb0 = row0 * groups_per_row, sb1 = r1 * groups_per_row;
    const uint sb2 = r2 * groups_per_row, sb3 = r3 * groups_per_row;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        a0 += q80_binary_lane_term(signs, scales, input, rb0, sb0, col, group_size);
        a1 += q80_binary_lane_term(signs, scales, input, rb1, sb1, col, group_size);
        a2 += q80_binary_lane_term(signs, scales, input, rb2, sb2, col, group_size);
        a3 += q80_binary_lane_term(signs, scales, input, rb3, sb3, col, group_size);
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

// Binary simd + CSR residual in the same dispatch. Residual add order is
// serial on lane 0 after the binary reduction (same as shipped CSR apply).
kernel void q80_binary_group_csr_matvec(
    device const uchar* signs           [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device const uint* indices          [[buffer(4)]],
    device const uint* row_ptr          [[buffer(5)]],
    device const uchar* residual_signs  [[buffer(6)]],
    constant uint& rows                 [[buffer(7)]],
    constant uint& cols                 [[buffer(8)]],
    constant uint& group_size           [[buffer(9)]],
    constant uint& groups_per_row       [[buffer(10)]],
    constant uint& residual_scale_bits  [[buffer(11)]],
    uint group_id                        [[threadgroup_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || cols == 0u) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        partial += q80_binary_lane_term(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        const float rscale = float(as_type<half>(ushort(residual_scale_bits)));
        const uint begin = row_ptr[row];
        const uint end = row_ptr[row + 1u];
        float acc = partial;
        for (uint n = begin; n < end; ++n) {
            const uint col = indices[n] % cols;
            acc += q80_residual_q1_value(residual_signs, n, rscale) * input[col];
        }
        output[row] = acc;
    }
}

// Cooperative CSR-only apply (used if binary is already in `output`).
// 32 lanes split the row's outliers, then simd_sum. Grid: ceil(rows/8)*256.
kernel void q80_sparse_q1_apply_csr_simd(
    device const uint* indices          [[buffer(0)]],
    device const uint* row_ptr          [[buffer(1)]],
    device const uchar* residual_signs  [[buffer(2)]],
    device const float* input           [[buffer(3)]],
    device float* output                [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& cols                 [[buffer(6)]],
    constant uint& residual_scale_bits  [[buffer(7)]],
    uint group_id                        [[threadgroup_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || cols == 0u) {
        return;
    }
    const float scale = float(as_type<half>(ushort(residual_scale_bits)));
    const uint begin = row_ptr[row];
    const uint end = row_ptr[row + 1u];
    float partial = 0.0f;
    for (uint n = begin + simd_lane; n < end; n += 32u) {
        const uint col = indices[n] % cols;
        partial += q80_residual_q1_value(residual_signs, n, scale) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] += partial;
    }
}

// One simdgroup per factor row. Grid: ceil(rows/8)*256, TG 256.
kernel void q80_hgravs01_factor_matvec_simd(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const uint element = row_base + col;
        partial += q80_uniform_value_wide(
            codes, scales, element, group_size, bits, bound) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// Fused y = L @ (R @ x) in one dispatch. Each threadgroup recomputes mid[rank]
// into threadgroup memory (640 B, not dense W), then its L-rows consume it.
// Rank is the shipped hgravs01_r160 codec; larger rank is refused.
// Grid: ceil(left_rows/8)*256, TG 256.
kernel void q80_hgravs01_two_stage_matvec(
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
                partial += q80_uniform_value_wide(
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
        partial += q80_uniform_value_wide(
            left_codes, left_scales, row_base + col, group_size, bits, bound)
            * mid[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[lrow] = partial;
    }
}

// Byte-unpack simd: each lane consumes 8 consecutive weights per tile.
// Grid: ceil(rows/8)*256, TG 256.
kernel void q80_binary_group_matvec_simd_bytes(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < cols; base += 256u) {
        const uint col = base + simd_lane * 8u;
        if (col + 8u > cols) {
            continue;
        }
        partial += q80_binary_byte_dot(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// Contiguous 64-col chunks (2048/32), left-to-right inside the chunk, then
// simd_sum. Closer association to the serial oracle than strided tiles.
// Grid: ceil(rows/8)*256, TG 256.
kernel void q80_binary_group_matvec_chunk(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || cols == 0u) {
        return;
    }
    const uint chunk = (cols + 31u) / 32u;
    const uint start = simd_lane * chunk;
    const uint end = min(start + chunk, cols);
    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    uint col = start;
    while (col + 8u <= end && (col & 7u) == 0u) {
        partial += q80_binary_byte_dot(
            signs, scales, input, row_base, scale_base, col, group_size);
        col += 8u;
    }
    while (col < end) {
        partial += q80_binary_lane_term(
            signs, scales, input, row_base, scale_base, col, group_size);
        col += 1u;
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// One 256-thread TG per row; each lane dots 8 columns (one sign byte), then
// 8-simdgroup reduce. Grid: rows*256, TG 256.
kernel void q80_binary_group_matvec_tg256(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    if (row >= rows) {
        return;
    }
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    const uint col = lid * 8u;
    float partial = 0.0f;
    if (col + 8u <= cols) {
        partial = q80_binary_byte_dot(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        output[row] = acc;
    }
}

// tg256 binary + serial CSR residual. Grid: rows*256, TG 256.
kernel void q80_binary_group_csr_matvec_tg256(
    device const uchar* signs           [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device const uint* indices          [[buffer(4)]],
    device const uint* row_ptr          [[buffer(5)]],
    device const uchar* residual_signs  [[buffer(6)]],
    constant uint& rows                 [[buffer(7)]],
    constant uint& cols                 [[buffer(8)]],
    constant uint& group_size           [[buffer(9)]],
    constant uint& groups_per_row       [[buffer(10)]],
    constant uint& residual_scale_bits  [[buffer(11)]],
    uint group_id                        [[threadgroup_position_in_grid]],
    uint lid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    if (row >= rows || cols == 0u) {
        return;
    }
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    const uint col = lid * 8u;
    float partial = 0.0f;
    if (col + 8u <= cols) {
        partial = q80_binary_byte_dot(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        const float rscale = float(as_type<half>(ushort(residual_scale_bits)));
        const uint begin = row_ptr[row];
        const uint end = row_ptr[row + 1u];
        for (uint n = begin; n < end; ++n) {
            const uint c = indices[n] % cols;
            acc += q80_residual_q1_value(residual_signs, n, rscale) * input[c];
        }
        output[row] = acc;
    }
}

// Fused byte-unpack binary + serial CSR residual. Grid: ceil(rows/8)*256.
kernel void q80_binary_group_csr_matvec_bytes(
    device const uchar* signs           [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device const uint* indices          [[buffer(4)]],
    device const uint* row_ptr          [[buffer(5)]],
    device const uchar* residual_signs  [[buffer(6)]],
    constant uint& rows                 [[buffer(7)]],
    constant uint& cols                 [[buffer(8)]],
    constant uint& group_size           [[buffer(9)]],
    constant uint& groups_per_row       [[buffer(10)]],
    constant uint& residual_scale_bits  [[buffer(11)]],
    uint group_id                        [[threadgroup_position_in_grid]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || cols == 0u) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < cols; base += 256u) {
        const uint col = base + simd_lane * 8u;
        if (col + 8u > cols) {
            continue;
        }
        partial += q80_binary_byte_dot(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        const float rscale = float(as_type<half>(ushort(residual_scale_bits)));
        const uint begin = row_ptr[row];
        const uint end = row_ptr[row + 1u];
        float acc = partial;
        for (uint n = begin; n < end; ++n) {
            const uint col = indices[n] % cols;
            acc += q80_residual_q1_value(residual_signs, n, rscale) * input[col];
        }
        output[row] = acc;
    }
}

// 3-bit specialized factor: 8 codes / lane, 256-col tiles.
// Grid: ceil(rows/8)*256, TG 256. bits must be 3, bound 3.
kernel void q80_hgravs01_factor_matvec_simd3(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || bits != 3u) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    // 8-wide whenever 8 codes fit, not only on 256-col tiles. down L is
    // 2048x160: the 256-tile path did zero 8-wide work and fell through
    // to the 1-wide remainder.
    for (uint col = simd_lane * 8u; col + 8u <= cols; col += 256u) {
        const uint byte0 = gk_packed_lsb_byte(row_base + col, 3u);
        const uint b0 = uint(codes[byte0]);
        const uint b1 = uint(codes[byte0 + 1u]);
        const uint b2 = uint(codes[byte0 + 2u]);
        const int q0 = int(b0 & 7u) - 3;
        const int q1 = int((b0 >> 3u) & 7u) - 3;
        const int q2 = int(((b0 >> 6u) | (b1 << 2u)) & 7u) - 3;
        const int q3 = int((b1 >> 1u) & 7u) - 3;
        const int q4 = int((b1 >> 4u) & 7u) - 3;
        const int q5 = int(((b1 >> 7u) | (b2 << 1u)) & 7u) - 3;
        const int q6 = int((b2 >> 2u) & 7u) - 3;
        const int q7 = int((b2 >> 5u) & 7u) - 3;
        const float s0 = float(scales[(row_base + col) / group_size]);
        const float s1 = float(scales[(row_base + col + 1u) / group_size]);
        const float s2 = float(scales[(row_base + col + 2u) / group_size]);
        const float s3 = float(scales[(row_base + col + 3u) / group_size]);
        const float s4 = float(scales[(row_base + col + 4u) / group_size]);
        const float s5 = float(scales[(row_base + col + 5u) / group_size]);
        const float s6 = float(scales[(row_base + col + 6u) / group_size]);
        const float s7 = float(scales[(row_base + col + 7u) / group_size]);
        partial += float(q0) * s0 * input[col];
        partial += float(q1) * s1 * input[col + 1u];
        partial += float(q2) * s2 * input[col + 2u];
        partial += float(q3) * s3 * input[col + 3u];
        partial += float(q4) * s4 * input[col + 4u];
        partial += float(q5) * s5 * input[col + 5u];
        partial += float(q6) * s6 * input[col + 6u];
        partial += float(q7) * s7 * input[col + 7u];
    }
    const uint rem = (cols / 8u) * 8u;
    for (uint col = rem + simd_lane; col < cols; col += 32u) {
        partial += q80_uniform_value_wide(
            codes, scales, row_base + col, group_size, bits, bound) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// ── HGRAVU01 geo_tpr64 (G0 launch class, existing packed bytes) ───────────
//
// Same thread mapping as qwen_uniform_q4_group64_matvec_geo_tpr64_tg128:
//   TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row, col = lane*8, stride 512.
// Consumes the HGRAVU01 body already on disk. No repack. No dense W.
//
//   bits=3: 24 code bytes/group, LSB 3-bit, q = code - bound (bound=3)
//   bits=4: 32 code bytes/group, even nibble low, q = nibble - bound (bound=7)
//
// groups_per_row is derived as cols/group. Caller must bind only when
// group_size matches the specialized kernel (64 or 128) and cols is a
// multiple of that group. Packed bytes stay packed. No dense W.

static inline float hgravu01_q3_unpack8(
    device const uchar* codes,
    uint byte0,
    float scale,
    int bound,
    device const float* x,
    uint col)
{
    const uint b0 = uint(codes[byte0]);
    const uint b1 = uint(codes[byte0 + 1u]);
    const uint b2 = uint(codes[byte0 + 2u]);
    float sum = 0.0f;
    sum += float(int(b0 & 7u) - bound) * scale * x[col];
    sum += float(int((b0 >> 3u) & 7u) - bound) * scale * x[col + 1u];
    sum += float(int(((b0 >> 6u) | (b1 << 2u)) & 7u) - bound) * scale * x[col + 2u];
    sum += float(int((b1 >> 1u) & 7u) - bound) * scale * x[col + 3u];
    sum += float(int((b1 >> 4u) & 7u) - bound) * scale * x[col + 4u];
    sum += float(int(((b1 >> 7u) | (b2 << 1u)) & 7u) - bound) * scale * x[col + 5u];
    sum += float(int((b2 >> 2u) & 7u) - bound) * scale * x[col + 6u];
    sum += float(int((b2 >> 5u) & 7u) - bound) * scale * x[col + 7u];
    return sum;
}

static inline float hgravu01_q4_unpack8(
    uint packed,
    float scale,
    int bound,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const uint byte = (packed >> (8u * i)) & 0xffu;
        sum += float(int(byte & 0x0fu) - bound) * scale * x[col + 2u * i];
        sum += float(int(byte >> 4u) - bound) * scale * x[col + 2u * i + 1u];
    }
    return sum;
}

// Grid: ceil(rows/2)*128, TG 128. bits must be 3, group_size 64, cols % 64 == 0.
kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
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
    if (row < rows && bits == 3u && group_size == 64u && (cols & 63u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        const int qbound = int(bound);
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const float scale = float(scales[rgb]);
            const uint byte0 = rgb * 24u + ((local * 3u) >> 3u);
            acc += hgravu01_q3_unpack8(codes, byte0, scale, qbound, input, col);
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

// Grid: ceil(rows/2)*128, TG 128. bits must be 3, group_size 128, cols % 128 == 0.
// Same 8-wide LSB-3 unpack as the g64 sibling. 48 code bytes/group (128*3/8).
// Address by (row, group) so lm_head (248320×5120) does not overflow uint32
// element*bits. group is a compile-time shift, not a bind-time divide.
kernel void qwen_uniform_q3_group128_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
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
    if (row < rows && bits == 3u && group_size == 128u && (cols & 127u) == 0u) {
        const uint groups_per_row = cols >> 7u;
        const int qbound = int(bound);
        const ulong rgb0 = (ulong)row * (ulong)groups_per_row;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 7u;
            const uint local = col & 127u;
            const ulong rgb = rgb0 + (ulong)group;
            const float scale = float(scales[rgb]);
            const ulong byte0 = rgb * 48ul + (ulong)((local * 3u) >> 3u);
            acc += hgravu01_q3_unpack8(codes, uint(byte0), scale, qbound, input, col);
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

// Grid: ceil(rows/2)*128, TG 128. bits must be 4, group_size 64, cols % 64 == 0.
// Same nibble packing as HQ30UQ4; offset is bound=7, not nibble-8.
// Address by (row, group), not element*bits: uint32 element*4 overflows
// on lm_head (248320×5120) at row 209715.
kernel void qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
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
    if (row < rows && bits == 4u && group_size == 64u && (cols & 63u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        const int qbound = int(bound);
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * 32u + (local >> 1u)));
            acc += hgravu01_q4_unpack8(packed, scale, qbound, input, col);
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

// ── HGRAVF01 affine q2 (w = q * scale + bias, q in {0,1,2,3}) ──
// Same geo_tpr64 occupancy as HGRAVU01. Different reconstruction. No bound.
// group_size is 32 or 64; 8-wide tiles sit inside one group either way.
// Kernel names keep the group32 family; bind-time group_size selects 32/64.

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

// fold_addqx sibling of production unpack8. Algebra:
//   sum_i (s*q_i + b)*x_i = s*sum(q_i*x_i) + b*sum(x_i)
// with q*x as adds (q in {0,1,2,3}) instead of float(q)*x.
// Opt-in via Affine2Geo::FoldAddqx / HAWKING_AFFINE2_GEO=fold_addqx.
// Default production stays affine_q2_unpack8. Empirically bit-identical
// on sealed-3.14 MLP (receipts/future/MLP_DECODE_CHEAPEN.json).
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

// Compile-time group 32/64 so col/GS is a shift. A runtime group_size on
// this path is a non-constant divide (see qwen_uniform_q4 group-128 note).
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

static inline float affine_q2_geo_acc_g32_fold_addqx(
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
        acc += affine_q2_unpack8_fold_addqx(packed16, scale, bias, input, col);
    }
    return acc;
}

static inline float affine_q2_geo_acc_g64_fold_addqx(
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
        acc += affine_q2_unpack8_fold_addqx(packed16, scale, bias, input, col);
    }
    return acc;
}

// Serial family (HAWKING_QWEN38_RECON_FUSE=0). One thread per row.
// Grid (rows,1,1), TG (256,1,1).
kernel void qwen_affine_q2_group32_matvec(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows || !affine_q2_group_ok(group_size, cols)) {
        return;
    }
    const uint groups_per_row = cols / group_size;
    const uint bytes_per_group = group_size >> 2u;
    float acc = 0.0f;
    for (uint col = 0u; col + 8u <= cols; col += 8u) {
        const uint group = col / group_size;
        const uint local = col % group_size;
        const uint rgb = row * groups_per_row + group;
        const float scale = float(scales[rgb]);
        const float bias = float(biases[rgb]);
        const uint byte0 = rgb * bytes_per_group + (local >> 2u);
        const uint packed16 = uint(*((device const ushort*)(codes + byte0)));
        acc += affine_q2_unpack8(packed16, scale, bias, input, col);
    }
    output[row] = acc;
}

// G0 occupancy. Grid ceil(rows/2)*128, TG 128.
// Specializes 32 vs 64 so the inner loop never emits a runtime divide.
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

// Same launch geometry and binds as geo_tpr64. Unpack is fold_addqx.
// Opt-in sibling; production decode stays on geo_tpr64 unless selected.
kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128_fold_addqx(
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
            acc = affine_q2_geo_acc_g32_fold_addqx(
                codes, scales, biases, input, row, cols, lane_in_row);
        } else {
            acc = affine_q2_geo_acc_g64_fold_addqx(
                codes, scales, biases, input, row, cols, lane_in_row);
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

// Diagnostic: the pre-specialization G0 body. Runtime `col / group_size`
// is the suspected cost of affine2 g64 vs q4 geo_tpr64. Not bound in
// production decode.
kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128_runtime_div(
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
        const uint groups_per_row = cols / group_size;
        const uint bytes_per_group = group_size >> 2u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col / group_size;
            const uint local = col % group_size;
            const uint rgb = row * groups_per_row + group;
            const float scale = float(scales[rgb]);
            const float bias = float(biases[rgb]);
            const uint byte0 = rgb * bytes_per_group + (local >> 2u);
            const uint packed16 = uint(*((device const ushort*)(codes + byte0)));
            acc += affine_q2_unpack8(packed16, scale, bias, input, col);
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

static inline void affine_q2_unpack8_dual_g64(
    uint packed_g,
    float scale_g,
    float bias_g,
    uint packed_u,
    float scale_u,
    float bias_u,
    device const float* x,
    uint col,
    thread float& acc_g,
    thread float& acc_u)
{
    for (uint i = 0u; i < 8u; ++i) {
        const float xv = x[col + i];
        const uint qg = (packed_g >> (2u * i)) & 3u;
        const uint qu = (packed_u >> (2u * i)) & 3u;
        acc_g += (float(qg) * scale_g + bias_g) * xv;
        acc_u += (float(qu) * scale_u + bias_u) * xv;
    }
}

static inline void affine_q2_unpack8_dual_g64_fold_addqx(
    uint packed_g,
    float scale_g,
    float bias_g,
    uint packed_u,
    float scale_u,
    float bias_u,
    device const float* x,
    uint col,
    thread float& acc_g,
    thread float& acc_u)
{
    float acc_qx_g = 0.0f;
    float acc_qx_u = 0.0f;
    float acc_x = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const float xv = x[col + i];
        const uint qg = (packed_g >> (2u * i)) & 3u;
        const uint qu = (packed_u >> (2u * i)) & 3u;
        acc_x += xv;
        if ((qg & 2u) != 0u) {
            acc_qx_g += xv + xv;
        }
        if ((qg & 1u) != 0u) {
            acc_qx_g += xv;
        }
        if ((qu & 2u) != 0u) {
            acc_qx_u += xv + xv;
        }
        if ((qu & 1u) != 0u) {
            acc_qx_u += xv;
        }
    }
    acc_g += fma(scale_g, acc_qx_g, bias_g * acc_x);
    acc_u += fma(scale_u, acc_qx_u, bias_u * acc_x);
}

// Dual-accumulate gate+up on the production geo_tpr64 map. Compile-time
// group 64. Packed codes stay packed. Grid: ceil(rows/2)*128, TG 128.
kernel void qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128(
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
            affine_q2_unpack8_dual_g64(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
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
        up_out[row] = red[4u + t] + red[4u + t + 1u];
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128(
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
            affine_q2_unpack8_dual_g64(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
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

kernel void qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128_fold_addqx(
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
            affine_q2_unpack8_dual_g64_fold_addqx(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
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
        up_out[row] = red[4u + t] + red[4u + t + 1u];
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_fold_addqx(
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
            affine_q2_unpack8_dual_g64_fold_addqx(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
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

// ── Affine2 g64 kernel-geometry levers (do not reuse q4 tpr64 tile) ──
// All three keep in-register dequant: no dense W. Compile-time group 64.
// Association matches affine_q2_unpack8: (q*scale+bias)*x per element.

static inline float affine2_dot16_f4(
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

static inline void affine2_load_x16(
    device const float* input, uint col,
    thread float4& a, thread float4& b, thread float4& c, thread float4& d)
{
    a = *((device const float4*)(input + col));
    b = *((device const float4*)(input + col + 4u));
    c = *((device const float4*)(input + col + 8u));
    d = *((device const float4*)(input + col + 12u));
}

static inline float affine2_dot16_at(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    uint row, uint cols, uint col,
    float4 a, float4 b, float4 c, float4 d)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uint packed = *((device const uint*)(codes + rgb * 16u + (local >> 2u)));
    return affine2_dot16_f4(packed, float(scales[rgb]), float(biases[rgb]), a, b, c, d);
}

static inline float affine2_dot64_at(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
    device const float* input,
    uint row, uint cols, uint col)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint rgb = row * groups_per_row + group;
    const uint4 packed = *((device const uint4*)(codes + rgb * 16u));
    const float scale = float(scales[rgb]);
    const float bias = float(biases[rgb]);
    float4 a, b, c, d;
    affine2_load_x16(input, col, a, b, c, d);
    float s = affine2_dot16_f4(packed.x, scale, bias, a, b, c, d);
    affine2_load_x16(input, col + 16u, a, b, c, d);
    s += affine2_dot16_f4(packed.y, scale, bias, a, b, c, d);
    affine2_load_x16(input, col + 32u, a, b, c, d);
    s += affine2_dot16_f4(packed.z, scale, bias, a, b, c, d);
    affine2_load_x16(input, col + 48u, a, b, c, d);
    s += affine2_dot16_f4(packed.w, scale, bias, a, b, c, d);
    return s;
}

// Lever 1: qmv_fast tile. 2 simdgroups, 4 rows/simdgroup, TG 64.
// One simdgroup loads x once (16 values/thread, K-block 512) and reuses it
// across 4 output rows. uint32 code loads. No threadgroup reduction.
// Grid threads = ceil(rows/8)*64, TG 64.
kernel void qwen_affine_q2_group64_matvec_qmvfast_r8tg64(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = group_id * 8u + simd_id * 4u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if ((cols % 64u) == 0u) {
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint col = bk + simd_lane * 16u;
            if (col + 16u > cols) continue;
            float4 a, b, c, d;
            affine2_load_x16(input, col, a, b, c, d);
            if (row0 < rows) {
                acc0 += affine2_dot16_at(codes, scales, biases, row0, cols, col, a, b, c, d);
            }
            if (row0 + 1u < rows) {
                acc1 += affine2_dot16_at(codes, scales, biases, row0 + 1u, cols, col, a, b, c, d);
            }
            if (row0 + 2u < rows) {
                acc2 += affine2_dot16_at(codes, scales, biases, row0 + 2u, cols, col, a, b, c, d);
            }
            if (row0 + 3u < rows) {
                acc3 += affine2_dot16_at(codes, scales, biases, row0 + 3u, cols, col, a, b, c, d);
            }
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (simd_lane == 0u) {
        if (row0 < rows) output[row0] = acc0;
        if (row0 + 1u < rows) output[row0 + 1u] = acc1;
        if (row0 + 2u < rows) output[row0 + 2u] = acc2;
        if (row0 + 3u < rows) output[row0 + 3u] = acc3;
    }
}

// Same geometry, load codes+scale+bias+x, skip the (q*scale+bias)*x FMA.
// Addresses the byte stream so the compute kernel can be compared to a
// load-only ceiling. Writes a poison sum so the loads cannot DCE.
kernel void qwen_affine_q2_group64_matvec_qmvfast_r8tg64_addr_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = group_id * 8u + simd_id * 4u;
    float acc = 0.0f;
    if ((cols % 64u) == 0u && row0 < rows) {
        const uint groups_per_row = cols >> 6u;
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint col = bk + simd_lane * 16u;
            if (col + 16u > cols) continue;
            float4 a, b, c, d;
            affine2_load_x16(input, col, a, b, c, d);
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

// Lever 2: whole-group 64-wide vector loads. 32 threads/row (one simdgroup),
// 4 rows/TG, TG 128. Each thread owns one g64 group (uint4 codes = 16 B),
// so scale/bias load once per 64 weights and stay off the 8-wide inner path.
// K-block 2048. Grid threads = ceil(rows/4)*128, TG 128.
kernel void qwen_affine_q2_group64_matvec_wide64_r4tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    const uint row = group_id * 4u + simd_id;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        for (uint col = simd_lane * 64u; col + 64u <= cols; col += 2048u) {
            acc += affine2_dot64_at(codes, scales, biases, input, row, cols, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u && row < rows) output[row] = acc;
}

// Lever 3: threadgroup-staged x (K-tile 512) + 8 rows/TG, TG 256.
// 8 simdgroups, one row each. x is cooperatively loaded once per K-tile
// and reused by all 8 rows (split-K is the 32-lane simd_sum).
// Grid threads = ceil(rows/8)*256, TG 256.
kernel void qwen_affine_q2_group64_matvec_tgx_r8tg256(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
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
                acc += affine2_dot16_at(codes, scales, biases, row, cols, col, a, b, c, d);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u && row < rows) output[row] = acc;
}

static inline void affine2_dot16_dual_at(
    device const uchar* gate_codes, device const half* gate_scales, device const half* gate_biases,
    device const uchar* up_codes, device const half* up_scales, device const half* up_biases,
    uint row, uint cols, uint col,
    float4 a, float4 b, float4 c, float4 d,
    thread float& acc_g, thread float& acc_u)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uint byte0 = rgb * 16u + (local >> 2u);
    const uint pg = *((device const uint*)(gate_codes + byte0));
    const uint pu = *((device const uint*)(up_codes + byte0));
    acc_g += affine2_dot16_f4(pg, float(gate_scales[rgb]), float(gate_biases[rgb]), a, b, c, d);
    acc_u += affine2_dot16_f4(pu, float(up_scales[rgb]), float(up_biases[rgb]), a, b, c, d);
}

static inline void affine2_dot64_dual_at(
    device const uchar* gate_codes, device const half* gate_scales, device const half* gate_biases,
    device const uchar* up_codes, device const half* up_scales, device const half* up_biases,
    device const float* input,
    uint row, uint cols, uint col,
    thread float& acc_g, thread float& acc_u)
{
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint rgb = row * groups_per_row + group;
    const uint4 pg = *((device const uint4*)(gate_codes + rgb * 16u));
    const uint4 pu = *((device const uint4*)(up_codes + rgb * 16u));
    const float sg = float(gate_scales[rgb]);
    const float bg = float(gate_biases[rgb]);
    const float su = float(up_scales[rgb]);
    const float bu = float(up_biases[rgb]);
    float4 a, b, c, d;
    affine2_load_x16(input, col, a, b, c, d);
    acc_g += affine2_dot16_f4(pg.x, sg, bg, a, b, c, d);
    acc_u += affine2_dot16_f4(pu.x, su, bu, a, b, c, d);
    affine2_load_x16(input, col + 16u, a, b, c, d);
    acc_g += affine2_dot16_f4(pg.y, sg, bg, a, b, c, d);
    acc_u += affine2_dot16_f4(pu.y, su, bu, a, b, c, d);
    affine2_load_x16(input, col + 32u, a, b, c, d);
    acc_g += affine2_dot16_f4(pg.z, sg, bg, a, b, c, d);
    acc_u += affine2_dot16_f4(pu.z, su, bu, a, b, c, d);
    affine2_load_x16(input, col + 48u, a, b, c, d);
    acc_g += affine2_dot16_f4(pg.w, sg, bg, a, b, c, d);
    acc_u += affine2_dot16_f4(pu.w, su, bu, a, b, c, d);
}

kernel void qwen_affine_q2_group64_matvec_gate_up_qmvfast_r8tg64(
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
    const uint row0 = group_id * 8u + simd_id * 4u;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    if ((cols % 64u) == 0u) {
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint col = bk + simd_lane * 16u;
            if (col + 16u > cols) continue;
            float4 a, b, c, d;
            affine2_load_x16(input, col, a, b, c, d);
            if (row0 < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0, cols, col, a, b, c, d, g0, u0);
            }
            if (row0 + 1u < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0 + 1u, cols, col, a, b, c, d, g1, u1);
            }
            if (row0 + 2u < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0 + 2u, cols, col, a, b, c, d, g2, u2);
            }
            if (row0 + 3u < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0 + 3u, cols, col, a, b, c, d, g3, u3);
            }
        }
    }
    g0 = simd_sum(g0); g1 = simd_sum(g1); g2 = simd_sum(g2); g3 = simd_sum(g3);
    u0 = simd_sum(u0); u1 = simd_sum(u1); u2 = simd_sum(u2); u3 = simd_sum(u3);
    if (simd_lane == 0u) {
        if (row0 < rows) { gate_out[row0] = g0; up_out[row0] = u0; }
        if (row0 + 1u < rows) { gate_out[row0 + 1u] = g1; up_out[row0 + 1u] = u1; }
        if (row0 + 2u < rows) { gate_out[row0 + 2u] = g2; up_out[row0 + 2u] = u2; }
        if (row0 + 3u < rows) { gate_out[row0 + 3u] = g3; up_out[row0 + 3u] = u3; }
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_qmvfast_r8tg64(
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
    const uint row0 = group_id * 8u + simd_id * 4u;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    if ((cols % 64u) == 0u) {
        for (uint bk = 0u; bk < cols; bk += 512u) {
            const uint col = bk + simd_lane * 16u;
            if (col + 16u > cols) continue;
            float4 a, b, c, d;
            affine2_load_x16(input, col, a, b, c, d);
            if (row0 < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0, cols, col, a, b, c, d, g0, u0);
            }
            if (row0 + 1u < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0 + 1u, cols, col, a, b, c, d, g1, u1);
            }
            if (row0 + 2u < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0 + 2u, cols, col, a, b, c, d, g2, u2);
            }
            if (row0 + 3u < rows) {
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row0 + 3u, cols, col, a, b, c, d, g3, u3);
            }
        }
    }
    g0 = simd_sum(g0); g1 = simd_sum(g1); g2 = simd_sum(g2); g3 = simd_sum(g3);
    u0 = simd_sum(u0); u1 = simd_sum(u1); u2 = simd_sum(u2); u3 = simd_sum(u3);
    if (simd_lane == 0u) {
        if (row0 < rows) act_out[row0] = (g0 / (1.0f + exp(-g0))) * u0;
        if (row0 + 1u < rows) act_out[row0 + 1u] = (g1 / (1.0f + exp(-g1))) * u1;
        if (row0 + 2u < rows) act_out[row0 + 2u] = (g2 / (1.0f + exp(-g2))) * u2;
        if (row0 + 3u < rows) act_out[row0 + 3u] = (g3 / (1.0f + exp(-g3))) * u3;
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_wide64_r4tg128(
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
    const uint row = group_id * 4u + simd_id;
    float acc_g = 0.0f, acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        for (uint col = simd_lane * 64u; col + 64u <= cols; col += 2048u) {
            affine2_dot64_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                input, row, cols, col, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u && row < rows) {
        gate_out[row] = acc_g;
        up_out[row] = acc_u;
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_wide64_r4tg128(
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
    const uint row = group_id * 4u + simd_id;
    float acc_g = 0.0f, acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        for (uint col = simd_lane * 64u; col + 64u <= cols; col += 2048u) {
            affine2_dot64_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                input, row, cols, col, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u && row < rows) {
        act_out[row] = (acc_g / (1.0f + exp(-acc_g))) * acc_u;
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_tgx_r8tg256(
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
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_tile[512];
    const uint row = group_id * 8u + simd_id;
    float acc_g = 0.0f, acc_u = 0.0f;
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
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row, cols, col, a, b, c, d, acc_g, acc_u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u && row < rows) {
        gate_out[row] = acc_g;
        up_out[row] = acc_u;
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_tgx_r8tg256(
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
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float x_tile[512];
    const uint row = group_id * 8u + simd_id;
    float acc_g = 0.0f, acc_u = 0.0f;
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
                affine2_dot16_dual_at(gate_codes, gate_scales, gate_biases, up_codes, up_scales, up_biases,
                    row, cols, col, a, b, c, d, acc_g, acc_u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u && row < rows) {
        act_out[row] = (acc_g / (1.0f + exp(-acc_g))) * acc_u;
    }
}

// ── N024 non-load critical-path levers (production decode) ──
// Same reconstruction as tpr64. In-register dequant, no dense W.
// qmvfast / wide64 / tgx are not re-tried (N018 lost them on this path).

constant uint kQwenAffine2SbMaxGroups = 512u;

static inline float affine2_prod_unpack8_vec(
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

static inline float affine2_prod_unpack8_accfuse_vec(
    uint packed16, float scale, float bias, float4 x0, float4 x1)
{
    float qx = 0.0f;
    float xs = 0.0f;
    qx += float((packed16       ) & 3u) * x0.x; xs += x0.x;
    qx += float((packed16 >>  2u) & 3u) * x0.y; xs += x0.y;
    qx += float((packed16 >>  4u) & 3u) * x0.z; xs += x0.z;
    qx += float((packed16 >>  6u) & 3u) * x0.w; xs += x0.w;
    qx += float((packed16 >>  8u) & 3u) * x1.x; xs += x1.x;
    qx += float((packed16 >> 10u) & 3u) * x1.y; xs += x1.y;
    qx += float((packed16 >> 12u) & 3u) * x1.z; xs += x1.z;
    qx += float((packed16 >> 14u) & 3u) * x1.w; xs += x1.w;
    return qx * scale + xs * bias;
}

static inline void affine2_prod_unpack8_dual_vec(
    uint packed_g, float scale_g, float bias_g,
    uint packed_u, float scale_u, float bias_u,
    float4 x0, float4 x1,
    thread float& acc_g, thread float& acc_u)
{
    acc_g += affine2_prod_unpack8_vec(packed_g, scale_g, bias_g, x0, x1);
    acc_u += affine2_prod_unpack8_vec(packed_u, scale_u, bias_u, x0, x1);
}

static inline void affine2_prod_unpack8_dual_accfuse(
    uint packed_g, float scale_g, float bias_g,
    uint packed_u, float scale_u, float bias_u,
    float4 x0, float4 x1,
    thread float& acc_g, thread float& acc_u)
{
    float xs = 0.0f;
    float qx_g = 0.0f;
    float qx_u = 0.0f;
    xs += x0.x; qx_g += float((packed_g       ) & 3u) * x0.x; qx_u += float((packed_u       ) & 3u) * x0.x;
    xs += x0.y; qx_g += float((packed_g >>  2u) & 3u) * x0.y; qx_u += float((packed_u >>  2u) & 3u) * x0.y;
    xs += x0.z; qx_g += float((packed_g >>  4u) & 3u) * x0.z; qx_u += float((packed_u >>  4u) & 3u) * x0.z;
    xs += x0.w; qx_g += float((packed_g >>  6u) & 3u) * x0.w; qx_u += float((packed_u >>  6u) & 3u) * x0.w;
    xs += x1.x; qx_g += float((packed_g >>  8u) & 3u) * x1.x; qx_u += float((packed_u >>  8u) & 3u) * x1.x;
    xs += x1.y; qx_g += float((packed_g >> 10u) & 3u) * x1.y; qx_u += float((packed_u >> 10u) & 3u) * x1.y;
    xs += x1.z; qx_g += float((packed_g >> 12u) & 3u) * x1.z; qx_u += float((packed_u >> 12u) & 3u) * x1.z;
    xs += x1.w; qx_g += float((packed_g >> 14u) & 3u) * x1.w; qx_u += float((packed_u >> 14u) & 3u) * x1.w;
    acc_g += qx_g * scale_g + xs * bias_g;
    acc_u += qx_u * scale_u + xs * bias_u;
}

// N030: defer bias. Inner loop is scale*sum(q x) only; bias*sum(x_group)
// is applied once per row from a precomputed x-sum. Same tpr64 occupancy.
static inline void affine2_prod_unpack8_dual_qx(
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

static inline void affine2_prod_sb_stage(
    threadgroup float* sb_scale,
    threadgroup float* sb_bias,
    device const half* scales,
    device const half* biases,
    uint row0,
    uint rows,
    uint gpr,
    uint lid,
    uint tg)
{
    for (uint g = lid; g < gpr; g += tg) {
        if (row0 < rows) {
            const uint rgb = row0 * gpr + g;
            sb_scale[g] = float(scales[rgb]);
            sb_bias[g] = float(biases[rgb]);
        }
        if (row0 + 1u < rows) {
            const uint rgb = (row0 + 1u) * gpr + g;
            sb_scale[kQwenAffine2SbMaxGroups + g] = float(scales[rgb]);
            sb_bias[kQwenAffine2SbMaxGroups + g] = float(biases[rgb]);
        }
    }
}

kernel void qwen_affine_q2_group64_matvec_tgsb_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    threadgroup float sb_scale[2 * kQwenAffine2SbMaxGroups];
    threadgroup float sb_bias[2 * kQwenAffine2SbMaxGroups];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row0 = group_id * 2u;
    const uint row = row0 + team;
    const uint gpr = cols >> 6u;
    float acc = 0.0f;
    if ((cols % 64u) == 0u && gpr <= kQwenAffine2SbMaxGroups) {
        affine2_prod_sb_stage(sb_scale, sb_bias, scales, biases, row0, rows, gpr, lid, 128u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < rows) {
            const uint sb_base = team * kQwenAffine2SbMaxGroups;
            for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * gpr + group;
                const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
                acc += affine_q2_unpack8(
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

kernel void qwen_affine_q2_group64_matvec_pipe_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
                packed = packed_n; scale = scale_n; bias = bias_n; x0 = x0_n; x1 = x1_n;
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
            acc += affine2_prod_unpack8_vec(packed, scale, bias, x0, x1);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void qwen_affine_q2_group64_matvec_splitk4_tg256(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
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
            acc += affine_q2_unpack8(packed16, float(scales[rgb]), float(biases[rgb]), input, col);
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

kernel void qwen_affine_q2_group64_matvec_accfuse_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            const float4 x0 = *((device const float4*)(input + col));
            const float4 x1 = *((device const float4*)(input + col + 4u));
            acc += affine2_prod_unpack8_accfuse_vec(
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

kernel void qwen_affine_q2_group64_matvec_gate_up_tgsb_tpr64_tg128(
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
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    threadgroup float gs[2 * kQwenAffine2SbMaxGroups];
    threadgroup float gb[2 * kQwenAffine2SbMaxGroups];
    threadgroup float us[2 * kQwenAffine2SbMaxGroups];
    threadgroup float ub[2 * kQwenAffine2SbMaxGroups];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row0 = group_id * 2u;
    const uint row = row0 + team;
    const uint gpr = cols >> 6u;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if ((cols % 64u) == 0u && gpr <= kQwenAffine2SbMaxGroups) {
        affine2_prod_sb_stage(gs, gb, gate_scales, gate_biases, row0, rows, gpr, lid, 128u);
        affine2_prod_sb_stage(us, ub, up_scales, up_biases, row0, rows, gpr, lid, 128u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < rows) {
            const uint sb_base = team * kQwenAffine2SbMaxGroups;
            for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * gpr + group;
                const uint byte0 = rgb * 16u + (local >> 2u);
                const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
                const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
                affine_q2_unpack8_dual_g64(
                    gpacked, gs[sb_base + group], gb[sb_base + group],
                    upacked, us[sb_base + group], ub[sb_base + group],
                    input, col, acc_g, acc_u);
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
        gate_out[row] = red[t] + red[t + 1u];
        up_out[row] = red[4u + t] + red[4u + t + 1u];
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_tgsb_tpr64_tg128(
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
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    threadgroup float gs[2 * kQwenAffine2SbMaxGroups];
    threadgroup float gb[2 * kQwenAffine2SbMaxGroups];
    threadgroup float us[2 * kQwenAffine2SbMaxGroups];
    threadgroup float ub[2 * kQwenAffine2SbMaxGroups];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row0 = group_id * 2u;
    const uint row = row0 + team;
    const uint gpr = cols >> 6u;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if ((cols % 64u) == 0u && gpr <= kQwenAffine2SbMaxGroups) {
        affine2_prod_sb_stage(gs, gb, gate_scales, gate_biases, row0, rows, gpr, lid, 128u);
        affine2_prod_sb_stage(us, ub, up_scales, up_biases, row0, rows, gpr, lid, 128u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < rows) {
            const uint sb_base = team * kQwenAffine2SbMaxGroups;
            for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * gpr + group;
                const uint byte0 = rgb * 16u + (local >> 2u);
                const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
                const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
                affine_q2_unpack8_dual_g64(
                    gpacked, gs[sb_base + group], gb[sb_base + group],
                    upacked, us[sb_base + group], ub[sb_base + group],
                    input, col, acc_g, acc_u);
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
        const float g = red[t] + red[t + 1u];
        const float u = red[4u + t] + red[4u + t + 1u];
        act_out[row] = (g / (1.0f + exp(-g))) * u;
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_pipe_tpr64_tg128(
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
        uint gpack_n = 0u, upack_n = 0u;
        float gs_n = 0.0f, gb_n = 0.0f, us_n = 0.0f, ub_n = 0.0f;
        float4 x0_n = 0.0f, x1_n = 0.0f;
        bool primed = false;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            uint gpack, upack;
            float gs, gb, us, ub;
            float4 x0, x1;
            if (primed) {
                gpack = gpack_n; upack = upack_n;
                gs = gs_n; gb = gb_n; us = us_n; ub = ub_n;
                x0 = x0_n; x1 = x1_n;
            } else {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * groups_per_row + group;
                const uint byte0 = rgb * 16u + (local >> 2u);
                gpack = uint(*((device const ushort*)(gate_codes + byte0)));
                upack = uint(*((device const ushort*)(up_codes + byte0)));
                gs = float(gate_scales[rgb]); gb = float(gate_biases[rgb]);
                us = float(up_scales[rgb]); ub = float(up_biases[rgb]);
                x0 = *((device const float4*)(input + col));
                x1 = *((device const float4*)(input + col + 4u));
                primed = true;
            }
            const uint col_n = col + 512u;
            if (col_n + 8u <= cols) {
                const uint group_n = col_n >> 6u;
                const uint local_n = col_n & 63u;
                const uint rgb_n = row * groups_per_row + group_n;
                const uint byte_n = rgb_n * 16u + (local_n >> 2u);
                gpack_n = uint(*((device const ushort*)(gate_codes + byte_n)));
                upack_n = uint(*((device const ushort*)(up_codes + byte_n)));
                gs_n = float(gate_scales[rgb_n]); gb_n = float(gate_biases[rgb_n]);
                us_n = float(up_scales[rgb_n]); ub_n = float(up_biases[rgb_n]);
                x0_n = *((device const float4*)(input + col_n));
                x1_n = *((device const float4*)(input + col_n + 4u));
            }
            affine2_prod_unpack8_dual_vec(gpack, gs, gb, upack, us, ub, x0, x1, acc_g, acc_u);
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

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_pipe_tpr64_tg128(
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
        uint gpack_n = 0u, upack_n = 0u;
        float gs_n = 0.0f, gb_n = 0.0f, us_n = 0.0f, ub_n = 0.0f;
        float4 x0_n = 0.0f, x1_n = 0.0f;
        bool primed = false;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            uint gpack, upack;
            float gs, gb, us, ub;
            float4 x0, x1;
            if (primed) {
                gpack = gpack_n; upack = upack_n;
                gs = gs_n; gb = gb_n; us = us_n; ub = ub_n;
                x0 = x0_n; x1 = x1_n;
            } else {
                const uint group = col >> 6u;
                const uint local = col & 63u;
                const uint rgb = row * groups_per_row + group;
                const uint byte0 = rgb * 16u + (local >> 2u);
                gpack = uint(*((device const ushort*)(gate_codes + byte0)));
                upack = uint(*((device const ushort*)(up_codes + byte0)));
                gs = float(gate_scales[rgb]); gb = float(gate_biases[rgb]);
                us = float(up_scales[rgb]); ub = float(up_biases[rgb]);
                x0 = *((device const float4*)(input + col));
                x1 = *((device const float4*)(input + col + 4u));
                primed = true;
            }
            const uint col_n = col + 512u;
            if (col_n + 8u <= cols) {
                const uint group_n = col_n >> 6u;
                const uint local_n = col_n & 63u;
                const uint rgb_n = row * groups_per_row + group_n;
                const uint byte_n = rgb_n * 16u + (local_n >> 2u);
                gpack_n = uint(*((device const ushort*)(gate_codes + byte_n)));
                upack_n = uint(*((device const ushort*)(up_codes + byte_n)));
                gs_n = float(gate_scales[rgb_n]); gb_n = float(gate_biases[rgb_n]);
                us_n = float(up_scales[rgb_n]); ub_n = float(up_biases[rgb_n]);
                x0_n = *((device const float4*)(input + col_n));
                x1_n = *((device const float4*)(input + col_n + 4u));
            }
            affine2_prod_unpack8_dual_vec(gpack, gs, gb, upack, us, ub, x0, x1, acc_g, acc_u);
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

kernel void qwen_affine_q2_group64_matvec_gate_up_splitk4_tg256(
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
    threadgroup float red[16];
    constexpr uint kSplit = 4u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 1024u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            affine_q2_unpack8_dual_g64(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
                input, col, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[8u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        gate_out[row] = red[t] + red[t + 1u] + red[t + 2u] + red[t + 3u];
        up_out[row] = red[8u + t] + red[8u + t + 1u] + red[8u + t + 2u] + red[8u + t + 3u];
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_splitk4_tg256(
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
    threadgroup float red[16];
    constexpr uint kSplit = 4u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        const uint groups_per_row = cols >> 6u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 1024u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * groups_per_row + group;
            const uint byte0 = rgb * 16u + (local >> 2u);
            const uint gpacked = uint(*((device const ushort*)(gate_codes + byte0)));
            const uint upacked = uint(*((device const ushort*)(up_codes + byte0)));
            affine_q2_unpack8_dual_g64(
                gpacked, float(gate_scales[rgb]), float(gate_biases[rgb]),
                upacked, float(up_scales[rgb]), float(up_biases[rgb]),
                input, col, acc_g, acc_u);
        }
    }
    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);
    if (simd_lane == 0u) {
        red[simd_id] = acc_g;
        red[8u + simd_id] = acc_u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const uint t = team * kSplit;
        const float g = red[t] + red[t + 1u] + red[t + 2u] + red[t + 3u];
        const float u = red[8u + t] + red[8u + t + 1u] + red[8u + t + 2u] + red[8u + t + 3u];
        act_out[row] = (g / (1.0f + exp(-g))) * u;
    }
}

kernel void qwen_affine_q2_group64_matvec_gate_up_accfuse_tpr64_tg128(
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
            affine2_prod_unpack8_dual_accfuse(
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

kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_accfuse_tpr64_tg128(
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
            affine2_prod_unpack8_dual_accfuse(
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

// N030: fused gate_up_swiglu with deferred group-64 bias. Same tpr64
// occupancy as the incumbent. xsum is 80 floats for hidden=5120.
// Do not retune load geometry (N024 / N018). No dense W.
kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128(
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
                affine2_prod_unpack8_dual_qx(
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
                affine2_prod_unpack8_dual_qx(
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

kernel void qwen_affine_q2_group64_matvec_gate_up_biasprep_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_scales [[buffer(1)]],
    device const half*  gate_biases [[buffer(2)]],
    device const uchar* up_codes    [[buffer(3)]],
    device const half*  up_scales   [[buffer(4)]],
    device const half*  up_biases   [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       gate_out    [[buffer(7)]],
    device float*       up_out      [[buffer(8)]],
    device const float* xsum        [[buffer(9)]],
    constant uint& rows             [[buffer(10)]],
    constant uint& cols             [[buffer(11)]],
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
            affine2_prod_unpack8_dual_qx(
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
        float g = red[t] + red[t + 1u];
        float u = red[4u + t] + red[4u + t + 1u];
        const uint gpr = cols >> 6u;
        for (uint grp = 0u; grp < gpr; ++grp) {
            const uint rgb = row * gpr + grp;
            const float xs = xsum[grp];
            g += float(gate_biases[rgb]) * xs;
            u += float(up_biases[rgb]) * xs;
        }
        gate_out[row] = g;
        up_out[row] = u;
    }
}

// Deliberately-bad control: same occupancy and inner loop as biasprep,
// but the group-64 bias term is dropped. Token ids must change.
kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128(
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
            affine2_prod_unpack8_dual_qx(
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

// One-row gather of HGRAVF01 embed. Never a dense W.
kernel void qwen38_hgrafv_embedding_lookup(
    device const uchar* codes     [[buffer(0)]],
    device const half*  scales    [[buffer(1)]],
    device const half*  biases    [[buffer(2)]],
    device float* hidden          [[buffer(3)]],
    constant uint& token          [[buffer(4)]],
    constant uint& hidden_size    [[buffer(5)]],
    constant uint& vocab          [[buffer(6)]],
    constant uint& group_size     [[buffer(7)]],
    uint dim                       [[thread_position_in_grid]])
{
    if (dim >= hidden_size || token >= vocab || !affine_q2_group_ok(group_size, hidden_size)) {
        return;
    }
    const uint groups_per_row = hidden_size / group_size;
    const uint bytes_per_group = group_size >> 2u;
    const uint group = dim / group_size;
    const uint local = dim % group_size;
    const uint rgb = token * groups_per_row + group;
    const float scale = float(scales[rgb]);
    const float bias = float(biases[rgb]);
    const uint byte = uint(codes[rgb * bytes_per_group + (local >> 2u)]);
    const uint q = (byte >> (2u * (local & 3u))) & 3u;
    hidden[dim] = float(q) * scale + bias;
}

// ── Q2F: 4-level LS-fitted 2-bit, group 64, delta only ──
// Reconstruction: w = (float(q) - 1.5) * delta, q in {0,1,2,3}.
// Same occupancy as affine2 geo_tpr64. No bias buffer. Group is a
// compile-time 64 (col>>6, col&63); a bind-time group_size here would
// be the 1.37x integer-divide defect. Packed codes stay packed.

static inline float q2f_unpack8(
    uint packed16, float delta,
    device const float* x, uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += ((float(q) - 1.5f) * delta) * x[col + i];
    }
    return sum;
}

static inline float q2f_geo_acc_g64(
    device const uchar* codes,
    device const half* deltas,
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
        const float delta = float(deltas[rgb]);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += q2f_unpack8(packed16, delta, input, col);
    }
    return acc;
}

// Serial family. One thread per row. Grid (rows,1,1), TG 256.
// Compile-time group 64: col>>6 / col&63, no runtime divide.
kernel void qwen_q2f_group64_matvec(
    device const uchar* codes       [[buffer(0)]],
    device const half*  deltas      [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows || (cols % 64u) != 0u) {
        return;
    }
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = 0u; col + 8u <= cols; col += 8u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const float delta = float(deltas[rgb]);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += q2f_unpack8(packed16, delta, input, col);
    }
    output[row] = acc;
}

// G0 occupancy. Grid ceil(rows/2)*128, TG 128. Group 64 is a literal.
kernel void qwen_q2f_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  deltas      [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = q2f_geo_acc_g64(codes, deltas, input, row, cols, lane_in_row);
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

static inline void q2f_unpack8_dual_g64(
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
        acc_g += ((float(qg) - 1.5f) * delta_g) * xv;
        acc_u += ((float(qu) - 1.5f) * delta_u) * xv;
    }
}

kernel void qwen_q2f_group64_matvec_gate_up_geo_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_deltas [[buffer(1)]],
    device const uchar* up_codes    [[buffer(2)]],
    device const half*  up_deltas   [[buffer(3)]],
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
            q2f_unpack8_dual_g64(
                gpacked, float(gate_deltas[rgb]),
                upacked, float(up_deltas[rgb]),
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
        up_out[row] = red[4u + t] + red[4u + t + 1u];
    }
}

kernel void qwen_q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128(
    device const uchar* gate_codes  [[buffer(0)]],
    device const half*  gate_deltas [[buffer(1)]],
    device const uchar* up_codes    [[buffer(2)]],
    device const half*  up_deltas   [[buffer(3)]],
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
            q2f_unpack8_dual_g64(
                gpacked, float(gate_deltas[rgb]),
                upacked, float(up_deltas[rgb]),
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

// HGRAVU01 q8: one code is one byte. Serial family extract walks 8 bits
// of that byte. These kernels load the byte, subtract bound, FMA — no
// bit loop, no dense W, no threadgroup weight staging.
static inline float q80_uniform8_byte_dot(
    device const uchar* codes,
    device const half* scales,
    device const float* input,
    uint row_base,
    uint col,
    uint group_size,
    int bound)
{
    const uint element = row_base + col;
    const float scale = float(scales[element / group_size]);
    device const uchar* p = codes + element;
    float sum = 0.0f;
    sum += float(int(p[0]) - bound) * scale * input[col];
    sum += float(int(p[1]) - bound) * scale * input[col + 1u];
    sum += float(int(p[2]) - bound) * scale * input[col + 2u];
    sum += float(int(p[3]) - bound) * scale * input[col + 3u];
    sum += float(int(p[4]) - bound) * scale * input[col + 4u];
    sum += float(int(p[5]) - bound) * scale * input[col + 5u];
    sum += float(int(p[6]) - bound) * scale * input[col + 6u];
    sum += float(int(p[7]) - bound) * scale * input[col + 7u];
    return sum;
}

// One simdgroup per row, 8-wide byte tiles. Grid: ceil(rows/8)*256, TG 256.
// bits must be 8. 8 consecutive cols stay inside one group-64.
kernel void q80_uniform8_matvec_simd_bytes(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || bits != 8u || group_size == 0u) {
        return;
    }
    float partial = 0.0f;
    const uint row_base = row * cols;
    const int ibound = int(bound);
    const bool packed8 = group_size >= 8u && (group_size & 7u) == 0u;
    for (uint base = 0u; base < cols; base += 256u) {
        const uint col = base + simd_lane * 8u;
        if (col + 8u > cols) {
            continue;
        }
        if (packed8) {
            partial += q80_uniform8_byte_dot(
                codes, scales, input, row_base, col, group_size, ibound);
        } else {
            for (uint k = 0u; k < 8u; ++k) {
                const uint element = row_base + col + k;
                const float scale = float(scales[element / group_size]);
                partial += float(int(codes[element]) - ibound) * scale * input[col + k];
            }
        }
    }
    const uint rem = (cols / 8u) * 8u;
    for (uint col = rem + simd_lane; col < cols; col += 32u) {
        const uint element = row_base + col;
        const float scale = float(scales[element / group_size]);
        partial += float(int(codes[element]) - ibound) * scale * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// One 256-thread TG per row; each lane dots 8 Q8 codes, then 8-SG reduce.
// Loops 2048-col tiles so 4096-col out_proj is covered. Grid: rows*256.
kernel void q80_uniform8_matvec_tg256(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    if (row >= rows || bits != 8u || group_size == 0u) {
        return;
    }
    const uint row_base = row * cols;
    const int ibound = int(bound);
    const bool packed8 = group_size >= 8u && (group_size & 7u) == 0u;
    float partial = 0.0f;
    for (uint tile = 0u; tile < cols; tile += 2048u) {
        const uint col = tile + lid * 8u;
        if (col + 8u > cols) {
            continue;
        }
        if (packed8) {
            partial += q80_uniform8_byte_dot(
                codes, scales, input, row_base, col, group_size, ibound);
        } else {
            for (uint k = 0u; k < 8u; ++k) {
                const uint element = row_base + col + k;
                const float scale = float(scales[element / group_size]);
                partial += float(int(codes[element]) - ibound) * scale * input[col + k];
            }
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        output[row] = acc;
    }
}

// Same fused contract, 4 L-rows per simdgroup (32 L-rows / TG). Fewer
// threadgroups recompute R; four independent FMA chains hide decode latency.
// Grid: ceil(left_rows/32)*256, TG 256.
kernel void q80_hgravs01_two_stage_matvec_rowblock4(
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
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
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
                partial += q80_uniform_value_wide(
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

    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= left_rows) {
        return;
    }
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const bool has1 = row1 < left_rows;
    const bool has2 = row2 < left_rows;
    const bool has3 = row3 < left_rows;
    const uint r1 = has1 ? row1 : row0;
    const uint r2 = has2 ? row2 : row0;
    const uint r3 = has3 ? row3 : row0;
    const uint rb0 = row0 * left_cols;
    const uint rb1 = r1 * left_cols;
    const uint rb2 = r2 * left_cols;
    const uint rb3 = r3 * left_cols;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint base = 0u; base < left_cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= left_cols) {
            continue;
        }
        const float mv = mid[col];
        a0 += q80_uniform_value_wide(
            left_codes, left_scales, rb0 + col, group_size, bits, bound) * mv;
        a1 += q80_uniform_value_wide(
            left_codes, left_scales, rb1 + col, group_size, bits, bound) * mv;
        a2 += q80_uniform_value_wide(
            left_codes, left_scales, rb2 + col, group_size, bits, bound) * mv;
        a3 += q80_uniform_value_wide(
            left_codes, left_scales, rb3 + col, group_size, bits, bound) * mv;
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

// ── DRAM-row execution-order layouts ──────────────────────────────────────
// Per-group [fp16 scale | sign bytes]. Same decoded values as the split
// scales+signs pair. Grid: (rows, 1, 1), TG 256.

kernel void q80_binary_group_matvec_interleaved(
    device const uchar* records     [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device float* output            [[buffer(2)]],
    constant uint& rows             [[buffer(3)]],
    constant uint& cols             [[buffer(4)]],
    constant uint& group_size       [[buffer(5)]],
    constant uint& groups_per_row   [[buffer(6)]],
    uint row                         [[thread_position_in_grid]])
{
    if (row >= rows || group_size == 0u || (group_size & 7u) != 0u) return;
    const uint sign_bytes = group_size >> 3u;
    const uint stride = 2u + sign_bytes;
    float sum = 0.0f;
    const uint row_base = row * groups_per_row;
    for (uint group = 0u; group < groups_per_row; ++group) {
        const uint rec = (row_base + group) * stride;
        const float scale = float(*((device const half*)(records + rec)));
        device const uchar* signs = records + rec + 2u;
        const uint group_start = group * group_size;
        const uint group_end = min(group_start + group_size, cols);
        uint col = group_start;
        while (col + 8u <= group_end) {
            const uchar byte = signs[(col - group_start) >> 3u];
            sum += ((byte & 0x01u) ? scale : -scale) * input[col];
            sum += ((byte & 0x02u) ? scale : -scale) * input[col + 1u];
            sum += ((byte & 0x04u) ? scale : -scale) * input[col + 2u];
            sum += ((byte & 0x08u) ? scale : -scale) * input[col + 3u];
            sum += ((byte & 0x10u) ? scale : -scale) * input[col + 4u];
            sum += ((byte & 0x20u) ? scale : -scale) * input[col + 5u];
            sum += ((byte & 0x40u) ? scale : -scale) * input[col + 6u];
            sum += ((byte & 0x80u) ? scale : -scale) * input[col + 7u];
            col += 8u;
        }
        while (col < group_end) {
            const uint local = col - group_start;
            const uchar byte = signs[local >> 3u];
            const bool positive = ((byte >> (local & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
    }
    output[row] = sum;
}

// Sequential vs row-conflict read. `stride` is the per-thread step in bytes.
// Sequential probe uses stride = nthreads * 16; conflict probe uses 8192+64.
// Each thread issues `iters` float4 loads. Writes acc so the loads stay live.
kernel void dram_row_locality_read_reduce(
    device const uchar* data        [[buffer(0)]],
    device float* out               [[buffer(1)]],
    constant uint& nbytes           [[buffer(2)]],
    constant uint& stride           [[buffer(3)]],
    constant uint& iters            [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]],
    uint nthreads                    [[threads_per_grid]])
{
    if (tid >= nthreads || stride < 16u || nbytes < 16u) {
        return;
    }
    float acc = 0.0f;
    uint off = (tid * 16u) % (nbytes - 15u);
    for (uint i = 0u; i < iters; ++i) {
        const float4 v = *((device const float4*)(data + off));
        acc += v.x + v.y + v.z + v.w;
        off += stride;
        if (off + 16u > nbytes) {
            off = off % (nbytes - 15u);
        }
    }
    out[tid] = acc;
}

// Unique-bytes-once sequential read. Each thread owns a disjoint 16-byte-aligned
// slice and walks it once. No wrap, no decode, no model math. This is the honest
// batch=1 decode traffic shape: every active packed byte is touched exactly once.
kernel void q80_decode_shape_unique_once(
    device const uchar* data        [[buffer(0)]],
    device float* out               [[buffer(1)]],
    constant uint& nbytes           [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]],
    uint nthreads                    [[threads_per_grid]])
{
    if (tid >= nthreads || nbytes < 16u) {
        return;
    }
    const uint nvec = nbytes / 16u;
    const uint mine = nvec / nthreads;
    const uint extra = nvec % nthreads;
    const uint start_vec = tid * mine + min(tid, extra);
    const uint count = mine + (tid < extra ? 1u : 0u);
    float acc = 0.0f;
    uint off = start_vec * 16u;
    for (uint i = 0u; i < count; ++i) {
        const float4 v = *((device const float4*)(data + off));
        acc += v.x + v.y + v.z + v.w;
        off += 16u;
    }
    out[tid] = acc;
}

// Gathered organ read. `organ_offsets[i]` is a byte offset into `data`.
// threads_per_organ lanes walk one organ sequentially. No decode.
kernel void q80_decode_shape_gather(
    device const uchar* data        [[buffer(0)]],
    device const uint* organ_offsets [[buffer(1)]],
    device float* out               [[buffer(2)]],
    constant uint& n_organs         [[buffer(3)]],
    constant uint& organ_bytes      [[buffer(4)]],
    constant uint& threads_per_organ [[buffer(5)]],
    uint tid                         [[thread_position_in_grid]])
{
    if (threads_per_organ == 0u) {
        return;
    }
    const uint organ = tid / threads_per_organ;
    const uint lane = tid % threads_per_organ;
    if (organ >= n_organs || organ_bytes < 16u) {
        return;
    }
    const uint base = organ_offsets[organ];
    const uint nvec = organ_bytes / 16u;
    const uint mine = nvec / threads_per_organ;
    const uint extra = nvec % threads_per_organ;
    const uint start_vec = lane * mine + min(lane, extra);
    const uint count = mine + (lane < extra ? 1u : 0u);
    float acc = 0.0f;
    uint off = base + start_vec * 16u;
    for (uint i = 0u; i < count; ++i) {
        const float4 v = *((device const float4*)(data + off));
        acc += v.x + v.y + v.z + v.w;
        off += 16u;
    }
    out[tid] = acc;
}

// Minimum live store so a dispatch cannot be DCE'd. Dispatch-tax probe.
kernel void q80_decode_shape_nop(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = float(tid);
}

// Dense f32 row-dot at one-thread-per-row. Same launch geometry as a serial
// decode organ, but the arithmetic is a fused multiply-add, not a codec.
kernel void q80_decode_shape_fma(
    device const float* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& cols   [[buffer(3)]],
    uint tid               [[thread_position_in_grid]])
{
    float acc = 0.0f;
    const uint row_off = tid * cols;
    for (uint c = 0u; c < cols; ++c) {
        acc += w[row_off + c] * x[c];
    }
    out[tid] = acc;
}

// ---------------------------------------------------------------------------
// TOKEN_NS diagnostic probes. Same launch geometry as the production
// recon-fuse kernels. Addr loads packed bytes + scales and keeps the
// loads live. Decode unpacks to a register accumulator and does not
// touch the input vector. Difference vs full is FMA with x.
// ---------------------------------------------------------------------------

kernel void q80_binary_group_matvec_tg256_addr_probe(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    float partial = 0.0f;
    if (row < rows) {
        const uint col = lid * 8u;
        if (col + 8u <= cols) {
            const float scale = float(scales[row * groups_per_row + col / group_size]);
            const uchar byte = signs[(row * cols + col) >> 3u];
            partial = scale + float(byte);
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row < rows) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        output[row] = acc;
    }
    (void)input;
}

kernel void q80_binary_group_matvec_tg256_decode_probe(
    device const uchar* signs       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& groups_per_row   [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    float partial = 0.0f;
    if (row < rows) {
        const uint col = lid * 8u;
        if (col + 8u <= cols) {
            const float scale = float(scales[row * groups_per_row + col / group_size]);
            const uchar byte = signs[(row * cols + col) >> 3u];
            partial += (byte & 0x01u) ? scale : -scale;
            partial += (byte & 0x02u) ? scale : -scale;
            partial += (byte & 0x04u) ? scale : -scale;
            partial += (byte & 0x08u) ? scale : -scale;
            partial += (byte & 0x10u) ? scale : -scale;
            partial += (byte & 0x20u) ? scale : -scale;
            partial += (byte & 0x40u) ? scale : -scale;
            partial += (byte & 0x80u) ? scale : -scale;
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row < rows) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        output[row] = acc;
    }
    (void)input;
}

kernel void q80_uniform8_matvec_tg256_addr_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    float partial = 0.0f;
    if (row < rows && group_size != 0u) {
        const uint row_base = row * cols;
        for (uint tile = 0u; tile < cols; tile += 2048u) {
            const uint col = tile + lid * 8u;
            if (col + 8u > cols) {
                continue;
            }
            const uint element = row_base + col;
            const float scale = float(scales[element / group_size]);
            partial += scale + float(codes[element]) + float(codes[element + 7u]);
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row < rows) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        output[row] = acc;
    }
    (void)input;
    (void)bits;
    (void)bound;
}

kernel void q80_uniform8_matvec_tg256_decode_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    const int ibound = int(bound);
    float partial = 0.0f;
    if (row < rows && bits == 8u && group_size != 0u) {
        const uint row_base = row * cols;
        for (uint tile = 0u; tile < cols; tile += 2048u) {
            const uint col = tile + lid * 8u;
            if (col + 8u > cols) {
                continue;
            }
            const uint element = row_base + col;
            const float scale = float(scales[element / group_size]);
            for (uint k = 0u; k < 8u; ++k) {
                partial += float(int(codes[element + k]) - ibound) * scale;
            }
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row < rows) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        output[row] = acc;
    }
    (void)input;
}

kernel void q80_uniform8_matvec_simd_bytes_addr_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    float partial = 0.0f;
    if (row < rows && group_size != 0u) {
        const uint row_base = row * cols;
        for (uint base = 0u; base < cols; base += 256u) {
            const uint col = base + simd_lane * 8u;
            if (col + 8u > cols) {
                continue;
            }
            const uint element = row_base + col;
            const float scale = float(scales[element / group_size]);
            partial += scale + float(codes[element]) + float(codes[element + 7u]);
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u && row < rows) {
        output[row] = partial;
    }
    (void)input;
    (void)bits;
    (void)bound;
}

kernel void q80_uniform8_matvec_simd_bytes_decode_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    const int ibound = int(bound);
    float partial = 0.0f;
    if (row < rows && bits == 8u && group_size != 0u) {
        const uint row_base = row * cols;
        for (uint base = 0u; base < cols; base += 256u) {
            const uint col = base + simd_lane * 8u;
            if (col + 8u > cols) {
                continue;
            }
            const uint element = row_base + col;
            const float scale = float(scales[element / group_size]);
            for (uint k = 0u; k < 8u; ++k) {
                partial += float(int(codes[element + k]) - ibound) * scale;
            }
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u && row < rows) {
        output[row] = partial;
    }
    (void)input;
}

kernel void q80_hgravs01_factor_matvec_simd3_addr_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    float partial = 0.0f;
    if (row < rows && group_size != 0u) {
        const uint row_base = row * cols;
        for (uint col = simd_lane * 8u; col + 8u <= cols; col += 256u) {
            const uint byte0 = gk_packed_lsb_byte(row_base + col, 3u);
            const uint b0 = uint(codes[byte0]);
            const uint b2 = uint(codes[byte0 + 2u]);
            const float s0 = float(scales[(row_base + col) / group_size]);
            partial += float(b0) + float(b2) + s0;
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u && row < rows) {
        output[row] = partial;
    }
    (void)input;
    (void)bits;
    (void)bound;
}

kernel void q80_hgravs01_factor_matvec_simd3_decode_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    float partial = 0.0f;
    if (row < rows && bits == 3u && group_size != 0u) {
        const uint row_base = row * cols;
        for (uint col = simd_lane * 8u; col + 8u <= cols; col += 256u) {
            const uint byte0 = gk_packed_lsb_byte(row_base + col, 3u);
            const uint b0 = uint(codes[byte0]);
            const uint b1 = uint(codes[byte0 + 1u]);
            const uint b2 = uint(codes[byte0 + 2u]);
            const int q0 = int(b0 & 7u) - 3;
            const int q1 = int((b0 >> 3u) & 7u) - 3;
            const int q2 = int(((b0 >> 6u) | (b1 << 2u)) & 7u) - 3;
            const int q3 = int((b1 >> 1u) & 7u) - 3;
            const int q4 = int((b1 >> 4u) & 7u) - 3;
            const int q5 = int(((b1 >> 7u) | (b2 << 1u)) & 7u) - 3;
            const int q6 = int((b2 >> 2u) & 7u) - 3;
            const int q7 = int((b2 >> 5u) & 7u) - 3;
            const float s0 = float(scales[(row_base + col) / group_size]);
            partial += float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7) * s0;
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u && row < rows) {
        output[row] = partial;
    }
    (void)input;
    (void)bound;
}

kernel void q80_binary_group_csr_matvec_tg256_addr_probe(
    device const uchar* signs           [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device const uint* indices          [[buffer(4)]],
    device const uint* row_ptr          [[buffer(5)]],
    device const uchar* residual_signs  [[buffer(6)]],
    constant uint& rows                 [[buffer(7)]],
    constant uint& cols                 [[buffer(8)]],
    constant uint& group_size           [[buffer(9)]],
    constant uint& groups_per_row       [[buffer(10)]],
    constant uint& residual_scale_bits  [[buffer(11)]],
    uint group_id                        [[threadgroup_position_in_grid]],
    uint lid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    float partial = 0.0f;
    if (row < rows) {
        const uint col = lid * 8u;
        if (col + 8u <= cols) {
            const float scale = float(scales[row * groups_per_row + col / group_size]);
            const uchar byte = signs[(row * cols + col) >> 3u];
            partial = scale + float(byte);
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row < rows) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        const uint begin = row_ptr[row];
        const uint end = row_ptr[row + 1u];
        if (begin < end) {
            acc += float(indices[begin] % cols) + float(residual_signs[begin]);
        }
        acc += float(as_type<half>(ushort(residual_scale_bits)));
        output[row] = acc;
    }
    (void)input;
}

kernel void q80_binary_group_csr_matvec_tg256_decode_probe(
    device const uchar* signs           [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const float* input           [[buffer(2)]],
    device float* output                [[buffer(3)]],
    device const uint* indices          [[buffer(4)]],
    device const uint* row_ptr          [[buffer(5)]],
    device const uchar* residual_signs  [[buffer(6)]],
    constant uint& rows                 [[buffer(7)]],
    constant uint& cols                 [[buffer(8)]],
    constant uint& group_size           [[buffer(9)]],
    constant uint& groups_per_row       [[buffer(10)]],
    constant uint& residual_scale_bits  [[buffer(11)]],
    uint group_id                        [[threadgroup_position_in_grid]],
    uint lid                             [[thread_index_in_threadgroup]],
    uint simd_lane                       [[thread_index_in_simdgroup]],
    uint simd_id                         [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    float partial = 0.0f;
    if (row < rows) {
        const uint col = lid * 8u;
        if (col + 8u <= cols) {
            const float scale = float(scales[row * groups_per_row + col / group_size]);
            const uchar byte = signs[(row * cols + col) >> 3u];
            partial += (byte & 0x01u) ? scale : -scale;
            partial += (byte & 0x02u) ? scale : -scale;
            partial += (byte & 0x04u) ? scale : -scale;
            partial += (byte & 0x08u) ? scale : -scale;
            partial += (byte & 0x10u) ? scale : -scale;
            partial += (byte & 0x20u) ? scale : -scale;
            partial += (byte & 0x40u) ? scale : -scale;
            partial += (byte & 0x80u) ? scale : -scale;
        }
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        red[simd_id] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row < rows) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            acc += red[i];
        }
        const float rscale = float(as_type<half>(ushort(residual_scale_bits)));
        const uint begin = row_ptr[row];
        const uint end = row_ptr[row + 1u];
        for (uint n = begin; n < end; ++n) {
            acc += q80_residual_q1_value(residual_signs, n, rscale);
        }
        output[row] = acc;
    }
    (void)input;
    (void)indices;
}

// ── N031 mlp_down: tpr64 g64 probes + residual-store fusions ──────────────
// down_proj is [hidden=5120, intermediate=17408]. That is the transpose of
// gate/up ([17408, 5120]): fewer threadgroups, more work per row. Gate/up
// conclusions do not transfer. Compile-time group 64; no bind-time divide.
//
// Store fusions (buffer 7 = residual):
//   add         output[row] = residual[row] + y
//   add_plain   output[row] = y                 (BAD control: drops residual)
//   add_ssq     add, and ssq[row] = v*v         (next-norm prep)
//   pairstore   add via one float2 store/TG     (output geometry)
// Probes keep the incumbent launch map (ceil(rows/2)*128, TG 128).

static inline float affine_q2_geo_addr_g64(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
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
        acc += float(scales[rgb]) + float(biases[rgb]) + float(packed16);
    }
    return acc;
}

static inline float affine_q2_geo_decode_g64(
    device const uchar* codes,
    device const half* scales,
    device const half* biases,
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
        for (uint i = 0u; i < 8u; ++i) {
            const uint q = (packed16 >> (2u * i)) & 3u;
            acc += float(q) * scale + bias;
        }
    }
    return acc;
}

kernel void qwen_affine_q2_group64_matvec_geo_tpr64_tg128_addr_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = affine_q2_geo_addr_g64(codes, scales, biases, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
    (void)input;
}

kernel void qwen_affine_q2_group64_matvec_geo_tpr64_tg128_decode_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = affine_q2_geo_decode_g64(codes, scales, biases, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
    (void)input;
}

kernel void qwen_affine_q2_group64_matvec_geo_tpr64_tg128_output_probe(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
}

kernel void qwen_affine_q2_group64_matvec_add_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    device const float* residual    [[buffer(7)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = residual[row] + red[team * kSplit] + red[team * kSplit + 1u];
    }
}

kernel void qwen_affine_q2_group64_matvec_add_tpr64_tg128_plain(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    device const float* residual    [[buffer(7)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
    (void)residual;
}

kernel void qwen_affine_q2_group64_matvec_add_ssq_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    device const float* residual    [[buffer(7)]],
    device float*       ssq         [[buffer(8)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        const float v = residual[row] + red[team * kSplit] + red[team * kSplit + 1u];
        output[row] = v;
        ssq[row] = v * v;
    }
}

kernel void qwen_affine_q2_group64_matvec_add_tpr64_tg128_pairstore(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    device const float* residual    [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint lid                         [[thread_index_in_threadgroup]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row0 = group_id * 2u;
    const uint row = row0 + team;
    float acc = 0.0f;
    if (row < rows && (cols % 64u) == 0u) {
        acc = affine_q2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u && row0 + 1u < rows) {
        const float2 y = float2(red[0] + red[1], red[2] + red[3]);
        const float2 r = *((device const float2*)(residual + row0));
        *((device float2*)(output + row0)) = r + y;
    } else if (lid == 0u && row0 < rows) {
        output[row0] = residual[row0] + red[0] + red[1];
    }
}

kernel void qwen38_ssq_prepared_rmsnorm_tg(
    device const float* hidden       [[buffer(0)]],
    device const float* ssq          [[buffer(1)]],
    device const float* weight       [[buffer(2)]],
    device float*       x_norm       [[buffer(3)]],
    constant uint& hidden_n          [[buffer(4)]],
    constant float& eps              [[buffer(5)]],
    threadgroup float* scratch       [[threadgroup(0)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]])
{
    float sum = 0.0f;
    for (uint index = tid; index < hidden_n; index += tg_size) {
        sum += ssq[index];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = 1.0f / sqrt(scratch[0] / float(hidden_n) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < hidden_n; index += tg_size) {
        x_norm[index] = hidden[index] * inverse_rms * (1.0f + weight[index]);
    }
}
