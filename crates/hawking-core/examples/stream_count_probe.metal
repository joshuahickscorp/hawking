// Stream-count ladder + alignment/stride discriminator.
// Arithmetic stripped (XOR/add sink). Bytes per thread-iteration are 38 on
// every rung. Production shaders are not bound; this file is probe-only.
//
// geo_tpr64_tg128, group 64, same loop as ARM A:
//   col = lane_in_row * 8; col += 512
//   64 threads/row, 2 rows/threadgroup.
//
// Counted payload per iteration is always 38. Pad bytes in storage are not
// loaded. Alignment of each load is natural for its type.

#include <metal_stdlib>
using namespace metal;

constant uint kTg = 128u;
constant uint kSplit = 2u;
constant uint kRecord6 = 8u;   // 6 B payload + 2 B pad
constant uint kRecord38 = 40u; // 6 B operand + 2 B pad + 32 B x

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

// ── 4 streams: 2+2+2+32  (MLP ARM A shape) ────────────────────────────────

kernel void stream_count_mlp_2_2_2_32(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& work_cols        [[buffer(7)]],
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
        const uint limit = work_cols == 0u ? cols : work_cols;
        uint csink = 0u;
        uint xsink = 0u;
        float ssink = 0.0f;
        for (uint col = lane * 8u; col + 8u <= limit; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * gpr + group;
            const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            csink ^= packed16;
            ssink += float(scales[rgb]) + float(biases[rgb]);
            xsink ^= xor8(input + col);
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
    (void)kTg;
}

// ── 3 streams: 4+2+32  (DeltaNet ARM A shape) ─────────────────────────────
// packed4[rgb*8 + (local>>3)] = uint(code | scale<<16). bias is the 2 B stream.

kernel void stream_count_dn_4_2_32(
    device const uint*  packed4     [[buffer(0)]],
    device const half*  biases      [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
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
            csink ^= packed4[rgb * 8u + (local >> 3u)];
            ssink += float(biases[rgb]);
            xsink ^= xor8(input + col);
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// ── 3 streams: 2+4+32  (size-swap of DeltaNet; between-rung) ──────────────
// codes stay 2 B per thread. packed_sb[rgb] = uint(scale | bias<<16).

kernel void stream_count_mid_2_4_32(
    device const uchar* codes       [[buffer(0)]],
    device const uint*  packed_sb   [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       output      [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
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
        uint ssink = 0u;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * gpr + group;
            csink ^= uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
            ssink ^= packed_sb[rgb];
            xsink ^= xor8(input + col);
        }
        acc = float(csink) + float(ssink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// ── 2 streams: 6+32  (codes+scale+bias interleaved) ───────────────────────
// record = 8 B: uint(code|scale<<16) at 0, ushort bias at 4, pad at 6.
// Load 6 B payload, not the pad.

kernel void stream_count_pack_6_32(
    device const uchar* packed6     [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant uint& rows             [[buffer(3)]],
    constant uint& cols             [[buffer(4)]],
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
        uint bsink = 0u;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 6u;
            const uint local = col & 63u;
            const uint rgb = row * gpr + group;
            device const uchar* rec = packed6 + (rgb * 8u + (local >> 3u)) * kRecord6;
            csink ^= *((device const uint*)rec);
            bsink ^= uint(*((device const ushort*)(rec + 4)));
            xsink ^= xor8(input + col);
        }
        acc = float(csink) + float(bsink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// ── 1 stream: 38  (operand + x in one packed record) ──────────────────────
// record = 40 B: uint(code|scale<<16) at 0, ushort bias at 4, pad at 6,
// 8 floats at 8. Load 6+32=38. x is NOT reused across rows.

kernel void stream_count_pack_38(
    device const uchar* packed38    [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant uint& rows             [[buffer(2)]],
    constant uint& cols             [[buffer(3)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint row = geo_row(group_id, simd_id);
    const uint lane = geo_lane(simd_id, simd_lane);
    float acc = 0.0f;
    if (row < rows) {
        const uint trips = cols / 512u;
        uint csink = 0u;
        uint xsink = 0u;
        uint bsink = 0u;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint trip = col / 512u;
            const uint slot = row * (64u * trips) + trip * 64u + lane;
            device const uchar* rec = packed38 + slot * kRecord38;
            csink ^= *((device const uint*)rec);
            bsink ^= uint(*((device const ushort*)(rec + 4)));
            xsink ^= xor8((device const float*)(rec + 8));
        }
        acc = float(csink) + float(bsink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// ── alignment: 3 operand streams + x, 2 B payloads in 4 B slots ───────────

kernel void stream_count_align_4(
    device const uchar* codes4      [[buffer(0)]],
    device const uchar* scales4     [[buffer(1)]],
    device const uchar* biases4     [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
            csink ^= uint(*((device const ushort*)(codes4 + rgb * 32u + (local >> 2u) * 4u)));
            ssink += float(*((device const half*)(scales4 + rgb * 4u)))
                   + float(*((device const half*)(biases4 + rgb * 4u)));
            xsink ^= xor8(input + col);
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// ── alignment: 3 operand streams + x, 2 B payloads in 16 B slots ──────────

kernel void stream_count_align_16(
    device const uchar* codes16     [[buffer(0)]],
    device const uchar* scales16    [[buffer(1)]],
    device const uchar* biases16    [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
            csink ^= uint(*((device const ushort*)(codes16 + rgb * 128u + (local >> 2u) * 16u)));
            ssink += float(*((device const half*)(scales16 + rgb * 16u)))
                   + float(*((device const half*)(biases16 + rgb * 16u)));
            xsink ^= xor8(input + col);
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// ── stride: codes contiguous per thread; scale/bias stay group-broadcast ─

kernel void stream_count_stride_contig(
    device const ushort* codes_c    [[buffer(0)]],
    device const half*   scales     [[buffer(1)]],
    device const half*   biases     [[buffer(2)]],
    device const float*  input      [[buffer(3)]],
    device float*        output     [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
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
        const uint trips = cols / 512u;
        uint csink = 0u;
        uint xsink = 0u;
        float ssink = 0.0f;
        for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
            const uint trip = col / 512u;
            const uint group = col >> 6u;
            const uint rgb = row * gpr + group;
            const uint idx = row * (64u * trips) + lane * trips + trip;
            csink ^= uint(codes_c[idx]);
            ssink += float(scales[rgb]) + float(biases[rgb]);
            xsink ^= xor8(input + col);
        }
        acc = ssink + float(csink) + as_type<float>(xsink);
    }
    geo_store(red, output, acc, row, rows, simd_lane, simd_id);
}

// Launch+reduction floor. No weight/x loads.

kernel void stream_count_zero(
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
    const uint row = geo_row(group_id, simd_id);
    const uint split = simd_id % kSplit;
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = float(group_id);
    }
    (void)codes;
    (void)scales;
    (void)biases;
    (void)input;
    (void)cols;
}
