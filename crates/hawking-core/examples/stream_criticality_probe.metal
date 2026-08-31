// Per-stream marginal cost probe for executable_economics.
//
// Same geo_tpr64_tg128 addressing as MLP ARM A / stream_count_probe.
// Trip count is held at the production loop (col = lane*8; col += 512).
// Independent keep-limits drop ONE stream's loads at a time; the other
// two streams stay at full production traffic. Arithmetic is stripped
// (XOR/add sink) so a time delta is a streaming delta, not an ALU delta.
// Production shaders are not bound; this file is probe-only.
//
// Counted payload per iteration at baseline is 38 B: 2 (codes, unique)
// + 2 (scale, broadcast) + 2 (bias, broadcast) + 32 (x, row-reused).

#include <metal_stdlib>
using namespace metal;

constant uint kTg = 128u;
constant uint kSplit = 2u;

static inline uint geo_row(uint group_id, uint simd_id) {
    return group_id * 2u + (simd_id / kSplit);
}
static inline uint geo_lane(uint simd_id, uint simd_lane) {
    return (simd_id % kSplit) * 32u + simd_lane;
}

static inline void geo_store(
    threadgroup float* red,
    device float* output,
    float acc,
    uint row,
    uint rows,
    uint simd_lane,
    uint simd_id)
{
    acc = simd_sum(acc);
    if (simd_lane == 0u) red[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

static inline uint xor8(device const float* xp) {
    return as_type<uint>(xp[0]) ^ as_type<uint>(xp[1])
         ^ as_type<uint>(xp[2]) ^ as_type<uint>(xp[3])
         ^ as_type<uint>(xp[4]) ^ as_type<uint>(xp[5])
         ^ as_type<uint>(xp[6]) ^ as_type<uint>(xp[7]);
}

// Gated three-stream consumer. Limits are runtime uniforms so the
// compiler cannot DCE a stream whose keep-fraction is not a literal 0.
// Loop always walks the full column range; a stream whose col >= limit
// is simply not loaded. That is the controlled quantity.
kernel void stream_crit_gated(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& code_limit       [[buffer(7)]],
    constant uint& aux_limit        [[buffer(8)]],
    constant uint& x_limit          [[buffer(9)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint row = geo_row(group_id, simd_id);
    const uint lane = geo_lane(simd_id, simd_lane);
    float acc = 0.0f;
    if (row < rows) {
        const uint gpr = cols >> 6u;
        uint csink = 0u;
        uint xsink = 0u;
        float ssink = 0.0f;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * gpr + group;
            if (col < code_limit) {
                const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
                csink ^= packed16;
            }
            if (col < aux_limit) {
                ssink += float(scales[rgb]) + float(biases[rgb]);
            }
            if (col < x_limit) {
                xsink ^= xor8(input + col);
            }
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
    (void)kTg;
}

// Launch + reduction floor. No weight/x loads, no inner loop.
kernel void stream_crit_empty(
    device float* output            [[buffer(0)]],
    constant uint& rows             [[buffer(1)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint row = geo_row(group_id, simd_id);
    geo_store(red, output, 0.0f, row, rows, simd_lane, simd_id);
}
