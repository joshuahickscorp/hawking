// N033 — competent native operator for K=2 shared binary bases.
//
// Representation: W[r,c] ≈ s0[r,c>>6] * sign0[r,c] + s1[r,c>>6] * sign1[r,c]
//   signs shared across layers; f16 group-64 scales per layer.
// Two-pass N032 (group-dots then scale-contract) is 384 dispatches with a
// threadgroup_barrier per group (80 or 272). That is the defect, not the math.
//
// This file is ONE operator: y = sum_k scale_k * (sign_k ⊙ x).
// Tile geometry the representation wants (S022 §10), not q2f tpr64:
//   32 threads / row (one simdgroup), 8 rows / TG of 256
//   x (and for c5120, 8 rows of K=2 signs) staged in threadgroup memory once
//   reduction in simd_sum; no per-group barrier; no bind-time shape
//   K=2 and group 64 are literals (shift, not divide)
//
// Shapes are exact: 17408 and 5120 both divide the launch geometry, so there
// is no `if (row < rows)` and no `constant uint& rows`.

#include <metal_stdlib>
using namespace metal;

static inline float k2_mac8(
    uchar p0,
    uchar p1,
    float a,
    float b,
    float x0, float x1, float x2, float x3,
    float x4, float x5, float x6, float x7)
{
    float acc = 0.0f;
    acc += ((p0 & 0x01u) != 0u ? a : -a) * x0;
    acc += ((p0 & 0x02u) != 0u ? a : -a) * x1;
    acc += ((p0 & 0x04u) != 0u ? a : -a) * x2;
    acc += ((p0 & 0x08u) != 0u ? a : -a) * x3;
    acc += ((p0 & 0x10u) != 0u ? a : -a) * x4;
    acc += ((p0 & 0x20u) != 0u ? a : -a) * x5;
    acc += ((p0 & 0x40u) != 0u ? a : -a) * x6;
    acc += ((p0 & 0x80u) != 0u ? a : -a) * x7;
    acc += ((p1 & 0x01u) != 0u ? b : -b) * x0;
    acc += ((p1 & 0x02u) != 0u ? b : -b) * x1;
    acc += ((p1 & 0x04u) != 0u ? b : -b) * x2;
    acc += ((p1 & 0x08u) != 0u ? b : -b) * x3;
    acc += ((p1 & 0x10u) != 0u ? b : -b) * x4;
    acc += ((p1 & 0x20u) != 0u ? b : -b) * x5;
    acc += ((p1 & 0x40u) != 0u ? b : -b) * x6;
    acc += ((p1 & 0x80u) != 0u ? b : -b) * x7;
    return acc;
}

// ── gate/up, c5120: stage x + 8 rows of signs in TGM (30 KiB) ─────────────
// TG 256, 8 simdgroups, 1 row / simdgroup. 17408/8 = 2176 TGs exactly.
// Scale layout: [k=0|1][orow][group], k1 offset = 17408 * 80.

kernel void shared_binary_k2_fused_xsign_c5120_r8_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[5120];
    threadgroup uchar packed0[5120];
    threadgroup uchar packed1[5120];

    const uint orow = group_id * 8u + simd_id;
    const uint sign_off = group_id * 5120u;
    for (uint i = lid; i < 5120u; i += 256u) {
        xs[i] = input[i];
        packed0[i] = signs0[sign_off + i];
        packed1[i] = signs1[sign_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    const uint sc0 = orow * 80u;
    const uint sc1 = 1392640u + sc0;
    const uint sp = simd_id * 640u;
    for (uint ocol = simd_lane * 8u; ocol + 8u <= 5120u; ocol += 256u) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uchar p0 = packed0[sp + (ocol >> 3u)];
        const uchar p1 = packed1[sp + (ocol >> 3u)];
        acc += k2_mac8(
            p0, p1, a, b,
            xs[ocol], xs[ocol + 1u], xs[ocol + 2u], xs[ocol + 3u],
            xs[ocol + 4u], xs[ocol + 5u], xs[ocol + 6u], xs[ocol + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        output[orow] = acc;
    }
}

// Same TGM loads, no FMA. Load-cost control: if this overlaps the fused
// body, the FMAs were not the work.
kernel void shared_binary_k2_fused_xsign_c5120_r8_tg256_noop(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[5120];
    threadgroup uchar packed0[5120];
    threadgroup uchar packed1[5120];

    const uint orow = group_id * 8u + simd_id;
    const uint sign_off = group_id * 5120u;
    for (uint i = lid; i < 5120u; i += 256u) {
        xs[i] = input[i];
        packed0[i] = signs0[sign_off + i];
        packed1[i] = signs1[sign_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (void)scales;
    if (simd_lane == 0u) {
        output[orow] = xs[0] * 0.0f + float(packed0[0]) * 0.0f + float(packed1[0]) * 0.0f;
    }
}

// Streamed ablation: same 8-row / 32-tpr launch, x and signs from device.
kernel void shared_binary_k2_fused_stream_c5120_tpr32_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint orow = group_id * 8u + simd_id;
    float acc = 0.0f;
    const uint sc0 = orow * 80u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 5120u;
    for (uint ocol = simd_lane * 8u; ocol + 8u <= 5120u; ocol += 256u) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uint flat = row_base + ocol;
        const uchar p0 = signs0[flat >> 3u];
        const uchar p1 = signs1[flat >> 3u];
        acc += k2_mac8(
            p0, p1, a, b,
            input[ocol], input[ocol + 1u], input[ocol + 2u], input[ocol + 3u],
            input[ocol + 4u], input[ocol + 5u], input[ocol + 6u], input[ocol + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        output[orow] = acc;
    }
}

// q2f-like occupancy: TG 128, 2 rows, 64 threads/row. One TG barrier for the
// 2-simdgroup reduce, never per group. Included so the search can reject it.
kernel void shared_binary_k2_fused_stream_c5120_tpr64_tg128(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    float acc = 0.0f;
    const uint sc0 = orow * 80u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 5120u;
    for (uint ocol = lane_in_row * 8u; ocol + 8u <= 5120u; ocol += 512u) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uint flat = row_base + ocol;
        const uchar p0 = signs0[flat >> 3u];
        const uchar p1 = signs1[flat >> 3u];
        acc += k2_mac8(
            p0, p1, a, b,
            input[ocol], input[ocol + 1u], input[ocol + 2u], input[ocol + 3u],
            input[ocol + 4u], input[ocol + 5u], input[ocol + 6u], input[ocol + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        const uint t = team * 2u;
        output[orow] = red[t] + red[t + 1u];
    }
}

// Bad control: one thread walks every column.
kernel void shared_binary_k2_fused_serial_c5120(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint orow                  [[thread_position_in_grid]])
{
    float acc = 0.0f;
    const uint sc0 = orow * 80u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 5120u;
    for (uint ocol = 0u; ocol < 5120u; ++ocol) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uint flat = row_base + ocol;
        const uint bit = flat & 7u;
        const float xv = input[ocol];
        acc += ((((signs0[flat >> 3u] >> bit) & 1u) != 0u) ? a : -a) * xv;
        acc += ((((signs1[flat >> 3u] >> bit) & 1u) != 0u) ? b : -b) * xv;
    }
    output[orow] = acc;
}

// 64-layer amortize: signs + x in TGM once, 64 scale tensors applied in
// registers. One dispatch per organ. Same x in the harness (decode x changes
// per layer; this arm measures the representation's reuse, not a real token).
kernel void shared_binary_k2_fused_xsign_layers64_c5120_r8_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[5120];
    threadgroup uchar packed0[5120];
    threadgroup uchar packed1[5120];

    const uint orow = group_id * 8u + simd_id;
    const uint sign_off = group_id * 5120u;
    for (uint i = lid; i < 5120u; i += 256u) {
        xs[i] = input[i];
        packed0[i] = signs0[sign_off + i];
        packed1[i] = signs1[sign_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sp = simd_id * 640u;
    for (uint layer = 0u; layer < 64u; ++layer) {
        float acc = 0.0f;
        const uint layer_off = layer * 2785280u;
        const uint sc0 = layer_off + orow * 80u;
        const uint sc1 = layer_off + 1392640u + orow * 80u;
        for (uint ocol = simd_lane * 8u; ocol + 8u <= 5120u; ocol += 256u) {
            const uint g = ocol >> 6u;
            const float a = float(scales[sc0 + g]);
            const float b = float(scales[sc1 + g]);
            const uchar p0 = packed0[sp + (ocol >> 3u)];
            const uchar p1 = packed1[sp + (ocol >> 3u)];
            acc += k2_mac8(
                p0, p1, a, b,
                xs[ocol], xs[ocol + 1u], xs[ocol + 2u], xs[ocol + 3u],
                xs[ocol + 4u], xs[ocol + 5u], xs[ocol + 6u], xs[ocol + 7u]);
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            output[layer * 17408u + orow] = acc;
        }
    }
}

// ── down_proj, c17408: x is 69 KiB, will not fit TGM. Tile 1024 floats. ──
// 17 tiles, 8 rows / TG of 256. 5120/8 = 640 TGs exactly.
// Scale k1 offset = 5120 * 272 = 1,392,640 (same number, different gpr).

kernel void shared_binary_k2_fused_xtile_c17408_r8_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[1024];

    const uint orow = group_id * 8u + simd_id;
    float acc = 0.0f;
    const uint sc0 = orow * 272u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 17408u;
    for (uint tile = 0u; tile < 17u; ++tile) {
        const uint base = tile * 1024u;
        for (uint i = lid; i < 1024u; i += 256u) {
            xs[i] = input[base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ocol = simd_lane * 8u; ocol + 8u <= 1024u; ocol += 256u) {
            const uint gcol = base + ocol;
            const uint g = gcol >> 6u;
            const float a = float(scales[sc0 + g]);
            const float b = float(scales[sc1 + g]);
            const uint flat = row_base + gcol;
            const uchar p0 = signs0[flat >> 3u];
            const uchar p1 = signs1[flat >> 3u];
            acc += k2_mac8(
                p0, p1, a, b,
                xs[ocol], xs[ocol + 1u], xs[ocol + 2u], xs[ocol + 3u],
                xs[ocol + 4u], xs[ocol + 5u], xs[ocol + 6u], xs[ocol + 7u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        output[orow] = acc;
    }
}

kernel void shared_binary_k2_fused_stream_c17408_tpr32_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint orow = group_id * 8u + simd_id;
    float acc = 0.0f;
    const uint sc0 = orow * 272u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 17408u;
    for (uint ocol = simd_lane * 8u; ocol + 8u <= 17408u; ocol += 256u) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uint flat = row_base + ocol;
        const uchar p0 = signs0[flat >> 3u];
        const uchar p1 = signs1[flat >> 3u];
        acc += k2_mac8(
            p0, p1, a, b,
            input[ocol], input[ocol + 1u], input[ocol + 2u], input[ocol + 3u],
            input[ocol + 4u], input[ocol + 5u], input[ocol + 6u], input[ocol + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        output[orow] = acc;
    }
}

kernel void shared_binary_k2_fused_stream_c17408_tpr64_tg128(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    float acc = 0.0f;
    const uint sc0 = orow * 272u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 17408u;
    for (uint ocol = lane_in_row * 8u; ocol + 8u <= 17408u; ocol += 512u) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uint flat = row_base + ocol;
        const uchar p0 = signs0[flat >> 3u];
        const uchar p1 = signs1[flat >> 3u];
        acc += k2_mac8(
            p0, p1, a, b,
            input[ocol], input[ocol + 1u], input[ocol + 2u], input[ocol + 3u],
            input[ocol + 4u], input[ocol + 5u], input[ocol + 6u], input[ocol + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        const uint t = team * 2u;
        output[orow] = red[t] + red[t + 1u];
    }
}

kernel void shared_binary_k2_fused_serial_c17408(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint orow                  [[thread_position_in_grid]])
{
    float acc = 0.0f;
    const uint sc0 = orow * 272u;
    const uint sc1 = 1392640u + sc0;
    const uint row_base = orow * 17408u;
    for (uint ocol = 0u; ocol < 17408u; ++ocol) {
        const uint g = ocol >> 6u;
        const float a = float(scales[sc0 + g]);
        const float b = float(scales[sc1 + g]);
        const uint flat = row_base + ocol;
        const uint bit = flat & 7u;
        const float xv = input[ocol];
        acc += ((((signs0[flat >> 3u] >> bit) & 1u) != 0u) ? a : -a) * xv;
        acc += ((((signs1[flat >> 3u] >> bit) & 1u) != 0u) ? b : -b) * xv;
    }
    output[orow] = acc;
}

kernel void shared_binary_k2_fused_xtile_layers64_c17408_r8_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[1024];

    const uint orow = group_id * 8u + simd_id;
    const uint row_base = orow * 17408u;
    for (uint layer = 0u; layer < 64u; ++layer) {
        float acc = 0.0f;
        const uint layer_off = layer * 2785280u;
        const uint sc0 = layer_off + orow * 272u;
        const uint sc1 = layer_off + 1392640u + orow * 272u;
        for (uint tile = 0u; tile < 17u; ++tile) {
            const uint base = tile * 1024u;
            for (uint i = lid; i < 1024u; i += 256u) {
                xs[i] = input[base + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ocol = simd_lane * 8u; ocol + 8u <= 1024u; ocol += 256u) {
                const uint gcol = base + ocol;
                const uint g = gcol >> 6u;
                const float a = float(scales[sc0 + g]);
                const float b = float(scales[sc1 + g]);
                const uint flat = row_base + gcol;
                const uchar p0 = signs0[flat >> 3u];
                const uchar p1 = signs1[flat >> 3u];
                acc += k2_mac8(
                    p0, p1, a, b,
                    xs[ocol], xs[ocol + 1u], xs[ocol + 2u], xs[ocol + 3u],
                    xs[ocol + 4u], xs[ocol + 5u], xs[ocol + 6u], xs[ocol + 7u]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            output[layer * 5120u + orow] = acc;
        }
    }
}

// K=4 fused stream, gate/up. Same launch as K=2 tpr32. Scales [k=0..3][orow][g]
// with k-stride 17408*80. Used for the bytes/token_ns curve, not the 0.53-bpw arm.
kernel void shared_binary_k4_fused_stream_c5120_tpr32_tg256(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const uchar* signs2 [[buffer(2)]],
    device const uchar* signs3 [[buffer(3)]],
    device const half*  scales [[buffer(4)]],
    device const float* input  [[buffer(5)]],
    device float*       output [[buffer(6)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint orow = group_id * 8u + simd_id;
    float acc = 0.0f;
    const uint sc0 = orow * 80u;
    const uint kst = 1392640u;
    const uint row_base = orow * 5120u;
    for (uint ocol = simd_lane * 8u; ocol + 8u <= 5120u; ocol += 256u) {
        const uint g = ocol >> 6u;
        const float a0 = float(scales[sc0 + g]);
        const float a1 = float(scales[kst + sc0 + g]);
        const float a2 = float(scales[2u * kst + sc0 + g]);
        const float a3 = float(scales[3u * kst + sc0 + g]);
        const uint flat = row_base + ocol;
        const uchar p0 = signs0[flat >> 3u];
        const uchar p1 = signs1[flat >> 3u];
        const uchar p2 = signs2[flat >> 3u];
        const uchar p3 = signs3[flat >> 3u];
        acc += k2_mac8(
            p0, p1, a0, a1,
            input[ocol], input[ocol + 1u], input[ocol + 2u], input[ocol + 3u],
            input[ocol + 4u], input[ocol + 5u], input[ocol + 6u], input[ocol + 7u]);
        acc += k2_mac8(
            p2, p3, a2, a3,
            input[ocol], input[ocol + 1u], input[ocol + 2u], input[ocol + 3u],
            input[ocol + 4u], input[ocol + 5u], input[ocol + 6u], input[ocol + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        output[orow] = acc;
    }
}
