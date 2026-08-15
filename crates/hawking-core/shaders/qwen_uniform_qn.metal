// Uniform Qn (bits=2 or 3) + FP16 group-scale matvec for Lane-N bisection.
//
// Layout (flat elements, little-endian bit stream):
//   * groups of `group_size` source weights (128 for Lane-N q2/q3 arms)
//   * one FP16 scale per group
//   * each weight is `bits` bits, offset-binary: q = code - bound where
//     bound = (1<<(bits-1))-1, packed little-endian across the padded flat
//     element stream (same as numpy packbits bitorder=little of bit matrix)
//
// Codes stay packed in DRAM. Reconstruction is float(q) * scale.

#include <metal_stdlib>
using namespace metal;

static inline uint qwen_uniform_qn_extract(
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

static inline float qwen_uniform_qn_value(
    device const uchar* codes,
    device const half* scales,
    uint element,
    uint group_size,
    uint bits,
    uint bound)
{
    const uint group = element / group_size;
    const uint code = qwen_uniform_qn_extract(codes, element, bits);
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
}

kernel void qwen_uniform_qn_matvec(
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
        sum += qwen_uniform_qn_value(codes, scales, element, group_size, bits, bound)
            * input[col];
    }
    output[row] = sum;
}

kernel void qwen_uniform_qn_decode_vector(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& elements    [[buffer(3)]],
    constant uint& group_size  [[buffer(4)]],
    constant uint& bits        [[buffer(5)]],
    constant uint& bound        [[buffer(6)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= elements) return;
    output[id] = qwen_uniform_qn_value(codes, scales, id, group_size, bits, bound);
}

kernel void qwen_uniform_qn_embedding_lookup(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& token       [[buffer(3)]],
    constant uint& hidden      [[buffer(4)]],
    constant uint& vocab       [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    constant uint& bits        [[buffer(7)]],
    constant uint& bound        [[buffer(8)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= hidden || token >= vocab) return;
    const uint element = token * hidden + id;
    output[id] = qwen_uniform_qn_value(codes, scales, element, group_size, bits, bound);
}
