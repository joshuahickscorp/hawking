// Diagnostic kernels only. Not on the token path.
// Separate reconstruction cost from 1-thread-per-row access cost.

#include <metal_stdlib>
using namespace metal;

kernel void disc_stream_control(
    device const uchar* data [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& nbytes    [[buffer(2)]],
    constant uint& iters     [[buffer(3)]],
    uint tid                 [[thread_position_in_grid]],
    uint nthreads            [[threads_per_grid]])
{
    if (tid >= nthreads || nbytes < 16u || iters == 0u) return;
    const uint span = nbytes - 15u;
    const uint stride = nthreads * 16u;
    float acc = 0.0f;
    uint off = (tid * 16u) % span;
    for (uint i = 0u; i < iters; ++i) {
        const float4 v = *((device const float4*)(data + off));
        acc += v.x + v.y + v.z + v.w;
        off += stride;
        if (off + 16u > nbytes) off = off % span;
    }
    out[tid] = acc;
}

// Same launch as shipped mixed matvec: 1 thread / row, TG 256.
kernel void disc_f32_serial(
    device const float* weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& cols         [[buffer(4)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float sum = 0.0f;
    const uint base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        sum += weights[base + col] * input[col];
    }
    output[row] = sum;
}

// 1 simdgroup / row, 8 rows / TG. Coalesced column walk.
kernel void disc_f32_simd(
    device const float* weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& cols         [[buffer(4)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSg = 8u;
    const uint row = group_id * kSg + simd_id;
    if (row >= rows) return;
    float partial = 0.0f;
    const uint base = row * cols;
    for (uint col = simd_lane; col < cols; col += 32u) {
        partial += weights[base + col] * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

// 256 threads / row. Grid = rows * 256.
kernel void disc_f32_tg256(
    device const float* weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& cols         [[buffer(4)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint lid                    [[thread_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    const uint row = group_id;
    if (row >= rows) return;
    float partial = 0.0f;
    const uint base = row * cols;
    for (uint col = lid; col < cols; col += 256u) {
        partial += weights[base + col] * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) red[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) acc += red[i];
        output[row] = acc;
    }
}

// Weight traffic removed. Same 1-thread/row loop over x.
kernel void disc_x_only_serial(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint row                  [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint col = 0u; col < cols; ++col) sum += input[col];
    output[row] = sum;
}

// Shipped Q8 extract: 8 bit-serial iterations per element.
static inline uint disc_bit_extract(device const uchar* codes, uint element, uint bits)
{
    const uint bit0 = element * bits;
    uint value = 0u;
    for (uint b = 0u; b < bits; ++b) {
        const uint bit_index = bit0 + b;
        const uchar byte = codes[bit_index >> 3u];
        value |= ((uint(byte) >> (bit_index & 7u)) & 1u) << b;
    }
    return value;
}

kernel void disc_q8_bit_serial(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    constant uint& cols       [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    constant uint& bits       [[buffer(7)]],
    constant uint& bound      [[buffer(8)]],
    uint row                  [[thread_position_in_grid]])
{
    if (row >= rows || group_size == 0u) return;
    float sum = 0.0f;
    const uint base = row * cols;
    const int ibound = int(bound);
    for (uint col = 0u; col < cols; ++col) {
        const uint element = base + col;
        const float scale = float(scales[element / group_size]);
        const int q = int(disc_bit_extract(codes, element, bits)) - ibound;
        sum += float(q) * scale * input[col];
    }
    output[row] = sum;
}

// Same launch, byte load. Isolates the 8-iteration extract.
kernel void disc_q8_byte_serial(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    constant uint& cols       [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    constant uint& bits       [[buffer(7)]],
    constant uint& bound      [[buffer(8)]],
    uint row                  [[thread_position_in_grid]])
{
    if (row >= rows || bits != 8u || group_size == 0u) return;
    float sum = 0.0f;
    const uint base = row * cols;
    const int ibound = int(bound);
    for (uint col = 0u; col < cols; ++col) {
        const uint element = base + col;
        const float scale = float(scales[element / group_size]);
        sum += float(int(codes[element]) - ibound) * scale * input[col];
    }
    output[row] = sum;
}

// Cheap Q8 decode, coalesced 1-SG/row.
kernel void disc_q8_byte_simd(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    constant uint& cols       [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    constant uint& bits       [[buffer(7)]],
    constant uint& bound      [[buffer(8)]],
    uint group_id             [[threadgroup_position_in_grid]],
    uint simd_lane            [[thread_index_in_simdgroup]],
    uint simd_id              [[simdgroup_index_in_threadgroup]])
{
    if (bits != 8u || group_size == 0u) return;
    constexpr uint kSg = 8u;
    const uint row = group_id * kSg + simd_id;
    if (row >= rows) return;
    float partial = 0.0f;
    const uint base = row * cols;
    const int ibound = int(bound);
    for (uint col = simd_lane; col < cols; col += 32u) {
        const uint element = base + col;
        const float scale = float(scales[element / group_size]);
        partial += float(int(codes[element]) - ibound) * scale * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) output[row] = partial;
}

// Cheap Q8 decode, 256 threads / row.
kernel void disc_q8_byte_tg256(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    constant uint& cols       [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    constant uint& bits       [[buffer(7)]],
    constant uint& bound      [[buffer(8)]],
    uint group_id             [[threadgroup_position_in_grid]],
    uint lid                  [[thread_index_in_threadgroup]],
    uint simd_lane            [[thread_index_in_simdgroup]],
    uint simd_id              [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[8];
    if (bits != 8u || group_size == 0u) return;
    const uint row = group_id;
    if (row >= rows) return;
    float partial = 0.0f;
    const uint base = row * cols;
    const int ibound = int(bound);
    for (uint col = lid; col < cols; col += 256u) {
        const uint element = base + col;
        const float scale = float(scales[element / group_size]);
        partial += float(int(codes[element]) - ibound) * scale * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) red[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float acc = 0.0f;
        for (uint i = 0u; i < 8u; ++i) acc += red[i];
        output[row] = acc;
    }
}

// Load packed bytes as floats, no decode. Same serial launch.
kernel void disc_load_only_serial(
    device const uchar* codes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant uint& rows       [[buffer(3)]],
    constant uint& cols       [[buffer(4)]],
    uint row                  [[thread_position_in_grid]])
{
    if (row >= rows) return;
    float sum = 0.0f;
    const uint base = row * cols;
    for (uint col = 0u; col < cols; ++col) {
        sum += float(codes[base + col]) * input[col];
    }
    output[row] = sum;
}
