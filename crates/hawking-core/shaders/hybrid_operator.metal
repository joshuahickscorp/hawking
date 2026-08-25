// N038 — hybrid MLP operator: binary bulk + DISTRIBUTED correction, ONE fuse.
//
// Binary g64 (1.25 bpw) is physically faster than q2f (N032: 23.43 ms vs 27.55)
// and generation-dead. N036: the injury is UNIFORM, so sparse / protected-island
// correction cannot heal it. This file is the structurally distinct combination
// that had not been run as a single operator:
//
//   y = binary_g64(x) + distributed_correction(x)
//
// Two corrections, both distributed by construction, both billed:
//   1. low-rank  y += U * (V^T x)     rank is a literal (r8 / r32)
//   2. shared-K  y += sum_k s_k * (B_k ⊙ x)   K=2 extra bases, signs shared
//
// Group 64 is `col >> 6`. Shapes 5120 / 17408 are literals. No bind-time
// rows/cols/rank, no dense W.
//
// Low-rank geometry the representation wants (S022 §10): V^T x is the SAME
// vector for every row, so it is computed ONCE per threadgroup. Binary body
// keeps the measured-competent tpr64 occupancy (2 rows / TG of 128, 64
// lanes/row). c5120 stages x in TGM; c17408 streams. Shared-K extra bases
// are per-row and stay on tpr64.

#include <metal_stdlib>
using namespace metal;

static inline float mac8_binary(
    uchar byte,
    float scale,
    float x0, float x1, float x2, float x3,
    float x4, float x5, float x6, float x7)
{
    float acc = 0.0f;
    acc += ((byte & 0x01u) != 0u ? scale : -scale) * x0;
    acc += ((byte & 0x02u) != 0u ? scale : -scale) * x1;
    acc += ((byte & 0x04u) != 0u ? scale : -scale) * x2;
    acc += ((byte & 0x08u) != 0u ? scale : -scale) * x3;
    acc += ((byte & 0x10u) != 0u ? scale : -scale) * x4;
    acc += ((byte & 0x20u) != 0u ? scale : -scale) * x5;
    acc += ((byte & 0x40u) != 0u ? scale : -scale) * x6;
    acc += ((byte & 0x80u) != 0u ? scale : -scale) * x7;
    return acc;
}

static inline float mac8_signed_scale(
    uchar byte,
    float a,
    float x0, float x1, float x2, float x3,
    float x4, float x5, float x6, float x7)
{
    return mac8_binary(byte, a, x0, x1, x2, x3, x4, x5, x6, x7);
}

// ── binary + rank-8, c5120, tpr64: x in TGM, V^T x once per TG ────────────
// 17408/2 = 8704 TGs exactly. 128 threads reduce V^T x, then 2 rows binary.

kernel void binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float proj[8];
    threadgroup float pred[32];
    threadgroup float red[4];

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    float p4 = 0.0f, p5 = 0.0f, p6 = 0.0f, p7 = 0.0f;
    for (uint col = lid; col < 5120u; col += 128u) {
        const float xv = input[col];
        const uint vb = col * 8u;
        p0 += float(V[vb + 0u]) * xv;
        p1 += float(V[vb + 1u]) * xv;
        p2 += float(V[vb + 2u]) * xv;
        p3 += float(V[vb + 3u]) * xv;
        p4 += float(V[vb + 4u]) * xv;
        p5 += float(V[vb + 5u]) * xv;
        p6 += float(V[vb + 6u]) * xv;
        p7 += float(V[vb + 7u]) * xv;
    }
    p0 = simd_sum(p0); p1 = simd_sum(p1); p2 = simd_sum(p2); p3 = simd_sum(p3);
    p4 = simd_sum(p4); p5 = simd_sum(p5); p6 = simd_sum(p6); p7 = simd_sum(p7);
    if (simd_lane == 0u) {
        const uint o = simd_id * 8u;
        pred[o + 0u] = p0; pred[o + 1u] = p1; pred[o + 2u] = p2; pred[o + 3u] = p3;
        pred[o + 4u] = p4; pred[o + 5u] = p5; pred[o + 6u] = p6; pred[o + 7u] = p7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8u) {
        proj[lid] = pred[lid] + pred[8u + lid] + pred[16u + lid] + pred[24u + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * 5120u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= 5120u; col += 512u) {
        const float scale = float(scales[orow * 80u + (col >> 6u)]);
        const uchar byte = signs[(row_base + col) >> 3u];
        acc += mac8_binary(
            byte, scale,
            input[col], input[col + 1u], input[col + 2u], input[col + 3u],
            input[col + 4u], input[col + 5u], input[col + 6u], input[col + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        const uint t = team * 2u;
        float y = red[t] + red[t + 1u];
        const uint ua = orow * 8u;
        y += float(U[ua + 0u]) * proj[0];
        y += float(U[ua + 1u]) * proj[1];
        y += float(U[ua + 2u]) * proj[2];
        y += float(U[ua + 3u]) * proj[3];
        y += float(U[ua + 4u]) * proj[4];
        y += float(U[ua + 5u]) * proj[5];
        y += float(U[ua + 6u]) * proj[6];
        y += float(U[ua + 7u]) * proj[7];
        output[orow] = y;
    }
}

kernel void binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128_noop(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint orow = group_id * 2u + team;
    float sink = input[lid] + float(signs[orow]) + float(scales[orow * 80u])
        + float(U[orow * 8u]) + float(V[lid * 8u]);
    sink = simd_sum(sink);
    if (split == 0u && simd_lane == 0u) {
        output[orow] = sink * 0.0f;
    }
}

// ── binary + rank-8, c17408 tpr64: stream x, V^T x once per TG ────────────
// 5120/2 = 2560 TGs exactly.

kernel void binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float proj[8];
    threadgroup float pred[32];
    threadgroup float red[4];
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    float p4 = 0.0f, p5 = 0.0f, p6 = 0.0f, p7 = 0.0f;
    for (uint col = lid; col < 17408u; col += 128u) {
        const float xv = input[col];
        const uint vb = col * 8u;
        p0 += float(V[vb + 0u]) * xv;
        p1 += float(V[vb + 1u]) * xv;
        p2 += float(V[vb + 2u]) * xv;
        p3 += float(V[vb + 3u]) * xv;
        p4 += float(V[vb + 4u]) * xv;
        p5 += float(V[vb + 5u]) * xv;
        p6 += float(V[vb + 6u]) * xv;
        p7 += float(V[vb + 7u]) * xv;
    }
    p0 = simd_sum(p0); p1 = simd_sum(p1); p2 = simd_sum(p2); p3 = simd_sum(p3);
    p4 = simd_sum(p4); p5 = simd_sum(p5); p6 = simd_sum(p6); p7 = simd_sum(p7);
    if (simd_lane == 0u) {
        const uint o = simd_id * 8u;
        pred[o + 0u] = p0; pred[o + 1u] = p1; pred[o + 2u] = p2; pred[o + 3u] = p3;
        pred[o + 4u] = p4; pred[o + 5u] = p5; pred[o + 6u] = p6; pred[o + 7u] = p7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8u) {
        proj[lid] = pred[lid] + pred[8u + lid] + pred[16u + lid] + pred[24u + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * 17408u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= 17408u; col += 512u) {
        const float scale = float(scales[orow * 272u + (col >> 6u)]);
        const uchar byte = signs[(row_base + col) >> 3u];
        acc += mac8_binary(
            byte, scale,
            input[col], input[col + 1u], input[col + 2u], input[col + 3u],
            input[col + 4u], input[col + 5u], input[col + 6u], input[col + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        const uint t = team * 2u;
        float y = red[t] + red[t + 1u];
        const uint ua = orow * 8u;
        y += float(U[ua + 0u]) * proj[0];
        y += float(U[ua + 1u]) * proj[1];
        y += float(U[ua + 2u]) * proj[2];
        y += float(U[ua + 3u]) * proj[3];
        y += float(U[ua + 4u]) * proj[4];
        y += float(U[ua + 5u]) * proj[5];
        y += float(U[ua + 6u]) * proj[6];
        y += float(U[ua + 7u]) * proj[7];
        output[orow] = y;
    }
}

kernel void binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128_noop(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint orow = group_id * 2u + team;
    float sink = 0.0f;
    for (uint col = lid; col < 17408u; col += 128u) {
        sink += input[col] + float(V[col * 8u]);
    }
    sink += float(signs[orow]) + float(scales[orow * 272u]) + float(U[orow * 8u]);
    sink = simd_sum(sink);
    if (split == 0u && simd_lane == 0u) {
        output[orow] = sink * 0.0f;
    }
}

// ── binary + rank-32, tpr64: each simdgroup owns 8 of 32 V^T x components ─

kernel void binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float proj[32];
    threadgroup float red[4];

    const uint k0 = simd_id * 8u;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    float q4 = 0.0f, q5 = 0.0f, q6 = 0.0f, q7 = 0.0f;
    for (uint col = simd_lane; col < 5120u; col += 32u) {
        const float xv = input[col];
        const uint vb = col * 32u + k0;
        q0 += float(V[vb + 0u]) * xv;
        q1 += float(V[vb + 1u]) * xv;
        q2 += float(V[vb + 2u]) * xv;
        q3 += float(V[vb + 3u]) * xv;
        q4 += float(V[vb + 4u]) * xv;
        q5 += float(V[vb + 5u]) * xv;
        q6 += float(V[vb + 6u]) * xv;
        q7 += float(V[vb + 7u]) * xv;
    }
    q0 = simd_sum(q0); q1 = simd_sum(q1); q2 = simd_sum(q2); q3 = simd_sum(q3);
    q4 = simd_sum(q4); q5 = simd_sum(q5); q6 = simd_sum(q6); q7 = simd_sum(q7);
    if (simd_lane == 0u) {
        proj[k0 + 0u] = q0; proj[k0 + 1u] = q1; proj[k0 + 2u] = q2; proj[k0 + 3u] = q3;
        proj[k0 + 4u] = q4; proj[k0 + 5u] = q5; proj[k0 + 6u] = q6; proj[k0 + 7u] = q7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * 5120u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= 5120u; col += 512u) {
        const float scale = float(scales[orow * 80u + (col >> 6u)]);
        const uchar byte = signs[(row_base + col) >> 3u];
        acc += mac8_binary(
            byte, scale,
            input[col], input[col + 1u], input[col + 2u], input[col + 3u],
            input[col + 4u], input[col + 5u], input[col + 6u], input[col + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        const uint t = team * 2u;
        float y = red[t] + red[t + 1u];
        const uint ua = orow * 32u;
        for (uint k = 0u; k < 32u; ++k) {
            y += float(U[ua + k]) * proj[k];
        }
        output[orow] = y;
    }
}

kernel void binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float proj[32];
    threadgroup float red[4];
    const uint k0 = simd_id * 8u;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    float q4 = 0.0f, q5 = 0.0f, q6 = 0.0f, q7 = 0.0f;
    for (uint col = simd_lane; col < 17408u; col += 32u) {
        const float xv = input[col];
        const uint vb = col * 32u + k0;
        q0 += float(V[vb + 0u]) * xv;
        q1 += float(V[vb + 1u]) * xv;
        q2 += float(V[vb + 2u]) * xv;
        q3 += float(V[vb + 3u]) * xv;
        q4 += float(V[vb + 4u]) * xv;
        q5 += float(V[vb + 5u]) * xv;
        q6 += float(V[vb + 6u]) * xv;
        q7 += float(V[vb + 7u]) * xv;
    }
    q0 = simd_sum(q0); q1 = simd_sum(q1); q2 = simd_sum(q2); q3 = simd_sum(q3);
    q4 = simd_sum(q4); q5 = simd_sum(q5); q6 = simd_sum(q6); q7 = simd_sum(q7);
    if (simd_lane == 0u) {
        proj[k0 + 0u] = q0; proj[k0 + 1u] = q1; proj[k0 + 2u] = q2; proj[k0 + 3u] = q3;
        proj[k0 + 4u] = q4; proj[k0 + 5u] = q5; proj[k0 + 6u] = q6; proj[k0 + 7u] = q7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * 17408u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= 17408u; col += 512u) {
        const float scale = float(scales[orow * 272u + (col >> 6u)]);
        const uchar byte = signs[(row_base + col) >> 3u];
        acc += mac8_binary(
            byte, scale,
            input[col], input[col + 1u], input[col + 2u], input[col + 3u],
            input[col + 4u], input[col + 5u], input[col + 6u], input[col + 7u]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u) {
        const uint t = team * 2u;
        float y = red[t] + red[t + 1u];
        const uint ua = orow * 32u;
        for (uint k = 0u; k < 32u; ++k) {
            y += float(U[ua + k]) * proj[k];
        }
        output[orow] = y;
    }
}

kernel void binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128_noop(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint orow = group_id * 2u + team;
    float sink = input[lid] + float(signs[orow]) + float(scales[orow * 80u])
        + float(U[orow * 32u]) + float(V[lid * 32u]);
    sink = simd_sum(sink);
    if (split == 0u && simd_lane == 0u) {
        output[orow] = sink * 0.0f;
    }
}

kernel void binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128_noop(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const half*  U      [[buffer(2)]],
    device const half*  V      [[buffer(3)]],
    device const float* input  [[buffer(4)]],
    device float*       output [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint orow = group_id * 2u + team;
    float sink = 0.0f;
    for (uint col = lid; col < 17408u; col += 128u) {
        sink += input[col] + float(V[col * 32u]);
    }
    sink += float(signs[orow]) + float(scales[orow * 272u]) + float(U[orow * 32u]);
    sink = simd_sum(sink);
    if (split == 0u && simd_lane == 0u) {
        output[orow] = sink * 0.0f;
    }
}

// ── binary bulk + K=2 shared-basis coefficient correction ─────────────────
// Extra bases B0, B1 are ±1 packed the same way as the bulk; their signs are
// stored once (amortized across layers). Per-layer f16 group-64 scales for
// the two extra bases sit in `escale`, k1 offset = rows * (cols>>6).
// Extra bases are per-row, so they cannot share a V^T x-style proj.

kernel void binary_shared_k2_fused_geo_c5120_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const uchar* signs0 [[buffer(2)]],
    device const uchar* signs1 [[buffer(3)]],
    device const half*  escale [[buffer(4)]],
    device const float* input  [[buffer(5)]],
    device float*       output [[buffer(6)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint COLS = 5120u;
    constexpr uint GPR = 80u;
    constexpr uint K1OFF = 17408u * 80u;
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * COLS;
    const uint sc = orow * GPR;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= COLS; col += 512u) {
        const uint g = col >> 6u;
        const float sb = float(scales[sc + g]);
        const float a0 = float(escale[sc + g]);
        const float a1 = float(escale[K1OFF + sc + g]);
        const uint flat = row_base + col;
        const uint bi = flat >> 3u;
        const float x0 = input[col];
        const float x1 = input[col + 1u];
        const float x2 = input[col + 2u];
        const float x3 = input[col + 3u];
        const float x4 = input[col + 4u];
        const float x5 = input[col + 5u];
        const float x6 = input[col + 6u];
        const float x7 = input[col + 7u];
        acc += mac8_signed_scale(signs[bi], sb, x0, x1, x2, x3, x4, x5, x6, x7);
        acc += mac8_signed_scale(signs0[bi], a0, x0, x1, x2, x3, x4, x5, x6, x7);
        acc += mac8_signed_scale(signs1[bi], a1, x0, x1, x2, x3, x4, x5, x6, x7);
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

kernel void binary_shared_k2_fused_geo_c17408_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const uchar* signs0 [[buffer(2)]],
    device const uchar* signs1 [[buffer(3)]],
    device const half*  escale [[buffer(4)]],
    device const float* input  [[buffer(5)]],
    device float*       output [[buffer(6)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint COLS = 17408u;
    constexpr uint GPR = 272u;
    constexpr uint K1OFF = 5120u * 272u;
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * COLS;
    const uint sc = orow * GPR;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= COLS; col += 512u) {
        const uint g = col >> 6u;
        const float sb = float(scales[sc + g]);
        const float a0 = float(escale[sc + g]);
        const float a1 = float(escale[K1OFF + sc + g]);
        const uint flat = row_base + col;
        const uint bi = flat >> 3u;
        const float x0 = input[col];
        const float x1 = input[col + 1u];
        const float x2 = input[col + 2u];
        const float x3 = input[col + 3u];
        const float x4 = input[col + 4u];
        const float x5 = input[col + 5u];
        const float x6 = input[col + 6u];
        const float x7 = input[col + 7u];
        acc += mac8_signed_scale(signs[bi], sb, x0, x1, x2, x3, x4, x5, x6, x7);
        acc += mac8_signed_scale(signs0[bi], a0, x0, x1, x2, x3, x4, x5, x6, x7);
        acc += mac8_signed_scale(signs1[bi], a1, x0, x1, x2, x3, x4, x5, x6, x7);
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

kernel void binary_shared_k2_fused_geo_c5120_tpr64_tg128_noop(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const uchar* signs0 [[buffer(2)]],
    device const uchar* signs1 [[buffer(3)]],
    device const half*  escale [[buffer(4)]],
    device const float* input  [[buffer(5)]],
    device float*       output [[buffer(6)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    constexpr uint COLS = 5120u;
    constexpr uint GPR = 80u;
    constexpr uint K1OFF = 17408u * 80u;
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * COLS;
    const uint sc = orow * GPR;
    float sink = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= COLS; col += 512u) {
        const uint g = col >> 6u;
        const uint bi = (row_base + col) >> 3u;
        sink += float(signs[bi]) + float(signs0[bi]) + float(signs1[bi]);
        sink += float(scales[sc + g]) + float(escale[sc + g]) + float(escale[K1OFF + sc + g]);
        sink += input[col];
    }
    sink = simd_sum(sink);
    if (simd_lane == 0u && split == 0u) {
        output[orow] = sink * 0.0f;
    }
}

kernel void binary_shared_k2_fused_geo_c17408_tpr64_tg128_noop(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const uchar* signs0 [[buffer(2)]],
    device const uchar* signs1 [[buffer(3)]],
    device const half*  escale [[buffer(4)]],
    device const float* input  [[buffer(5)]],
    device float*       output [[buffer(6)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    constexpr uint COLS = 17408u;
    constexpr uint GPR = 272u;
    constexpr uint K1OFF = 5120u * 272u;
    const uint team = simd_id >> 1u;
    const uint split = simd_id & 1u;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint orow = group_id * 2u + team;
    const uint row_base = orow * COLS;
    const uint sc = orow * GPR;
    float sink = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= COLS; col += 512u) {
        const uint g = col >> 6u;
        const uint bi = (row_base + col) >> 3u;
        sink += float(signs[bi]) + float(signs0[bi]) + float(signs1[bi]);
        sink += float(scales[sc + g]) + float(escale[sc + g]) + float(escale[K1OFF + sc + g]);
        sink += input[col];
    }
    sink = simd_sum(sink);
    if (simd_lane == 0u && split == 0u) {
        output[orow] = sink * 0.0f;
    }
}
