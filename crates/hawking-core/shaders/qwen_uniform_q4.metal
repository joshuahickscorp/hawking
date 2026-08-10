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
