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
    float sum = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint group_start = group * group_size;
        const uint group_end = min(group_start + group_size, cols);
        const float scale = float(scales[scale_base + group]);
        uint col = group_start;
        while (col < group_end && ((row_base + col) & 7u) != 0u) {
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
        while (col + 8u <= group_end) {
            const uchar byte = signs[(row_base + col) >> 3u];
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
            const uint flat = row_base + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            sum += (positive ? scale : -scale) * input[col];
            col += 1u;
        }
    }
    return sum;
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
    const uint bit0 = element * bits;
    uint value = 0u;
    for (uint b = 0u; b < bits; ++b) {
        const uint bit_index = bit0 + b;
        const uchar byte = codes[bit_index >> 3u];
        const uint bit = (byte >> (bit_index & 7u)) & 1u;
        value |= (bit << b);
    }
    return value;
}

static inline float q80_uniform_value(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    const uint group = element / group_size;
    const uint code = q80_uniform_extract(codes, element, bits);
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
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
