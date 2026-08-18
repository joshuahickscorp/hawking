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
// groups_per_row is derived as cols/64. Caller must bind only when
// group_size==64 and cols%64==0 (every Qwen3.8 Uniform GEMV K).

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
