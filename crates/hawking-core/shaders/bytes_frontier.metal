// N032 bytes-frontier native operators.
//
// Representations that move fewer ACTIVE bytes/token than q2f g64 (2.25 bpw)
// without reconstructing dense W:
//   1. ternary {-1,0,+1} packed 5-trits-in-8-bits + f16 group-64 scale
//   2. K=2 shared binary bases (signs loaded once) + per-layer group scales
//   3. binary plane + sparse CSR correction, one fused dispatch
//
// Group 64 is a compile-time shift (col >> 6). Shapes are specialized so the
// inner loop never divides by a bind-time group_size/cols (N003 / S022 §5).
// Zero trits contribute a 0 FMA; packed bytes are still loaded (active=stored
// for ternary). Shared-basis is the one that actually amortizes DRAM.

#include <metal_stdlib>
using namespace metal;

// trit 0 -> -1, 1 -> 0, 2 -> +1. byte = t0 + 3 t1 + 9 t2 + 27 t3 + 81 t4.
static inline void ternary5_mac(
    uchar packed,
    device const float* x,
    device const half* scales,
    uint row,
    uint col0,
    uint cols,
    uint gpr,
    thread float& acc)
{
    uint v = uint(packed);
    for (uint i = 0u; i < 5u; ++i) {
        const uint col = col0 + i;
        if (col >= cols) {
            break;
        }
        const uint t = v % 3u;
        v /= 3u;
        const float w = float(int(t) - 1);
        const float scale = float(scales[row * gpr + (col >> 6u)]);
        acc += (w * scale) * x[col];
    }
}

static inline float ternary_acc_c5120(
    device const uchar* codes,
    device const half* scales,
    device const float* x,
    uint row,
    uint lane)
{
    constexpr uint COLS = 5120u;
    constexpr uint BPR = 1024u;
    constexpr uint GPR = 80u;
    float acc = 0.0f;
    const uint row_bytes = row * BPR;
    for (uint b = lane; b < BPR; b += 64u) {
        ternary5_mac(codes[row_bytes + b], x, scales, row, b * 5u, COLS, GPR, acc);
    }
    return acc;
}

static inline float ternary_acc_c17408(
    device const uchar* codes,
    device const half* scales,
    device const float* x,
    uint row,
    uint lane)
{
    constexpr uint COLS = 17408u;
    constexpr uint FULL_BYTES = 3481u;
    constexpr uint BPR = 3482u;
    constexpr uint GPR = 272u;
    float acc = 0.0f;
    const uint row_bytes = row * BPR;
    for (uint b = lane; b < FULL_BYTES; b += 64u) {
        ternary5_mac(codes[row_bytes + b], x, scales, row, b * 5u, COLS, GPR, acc);
    }
    if (lane == 0u) {
        ternary5_mac(codes[row_bytes + FULL_BYTES], x, scales, row, FULL_BYTES * 5u, COLS, GPR, acc);
    }
    return acc;
}

// Grid: ceil(rows/2)*128, TG 128. Two rows, 64 lanes each (q2f geo occupancy).
kernel void ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = ternary_acc_c5120(codes, scales, input, row, lane_in_row);
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

kernel void ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = ternary_acc_c17408(codes, scales, input, row, lane_in_row);
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

// BAD control: one thread per row walks every packed byte (not the geo stride).
kernel void ternary_5in8_g64_matvec_serial_c5120(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint row                   [[thread_position_in_grid]])
{
    constexpr uint COLS = 5120u;
    constexpr uint BPR = 1024u;
    constexpr uint GPR = 80u;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    const uint row_bytes = row * BPR;
    for (uint b = 0u; b < BPR; ++b) {
        ternary5_mac(codes[row_bytes + b], input, scales, row, b * 5u, COLS, GPR, acc);
    }
    output[row] = acc;
}

kernel void ternary_5in8_g64_matvec_serial_c17408(
    device const uchar* codes  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint row                   [[thread_position_in_grid]])
{
    constexpr uint COLS = 17408u;
    constexpr uint FULL_BYTES = 3481u;
    constexpr uint BPR = 3482u;
    constexpr uint GPR = 272u;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    const uint row_bytes = row * BPR;
    for (uint b = 0u; b < BPR; ++b) {
        ternary5_mac(codes[row_bytes + b], input, scales, row, b * 5u, COLS, GPR, acc);
    }
    output[row] = acc;
}

// ── binary sign+scale, group 64 compile-time ──────────────────────────────

static inline float binary_acc_g64(
    device const uchar* signs,
    device const half* scales,
    device const float* x,
    uint row,
    uint cols,
    uint lane)
{
    const uint gpr = cols >> 6u;
    const uint row_base = row * cols;
    float acc = 0.0f;
    for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
        const float scale = float(scales[row * gpr + (col >> 6u)]);
        const uint flat = row_base + col;
        const uchar byte = signs[flat >> 3u];
        acc += ((byte & 0x01u) ? scale : -scale) * x[col];
        acc += ((byte & 0x02u) ? scale : -scale) * x[col + 1u];
        acc += ((byte & 0x04u) ? scale : -scale) * x[col + 2u];
        acc += ((byte & 0x08u) ? scale : -scale) * x[col + 3u];
        acc += ((byte & 0x10u) ? scale : -scale) * x[col + 4u];
        acc += ((byte & 0x20u) ? scale : -scale) * x[col + 5u];
        acc += ((byte & 0x40u) ? scale : -scale) * x[col + 6u];
        acc += ((byte & 0x80u) ? scale : -scale) * x[col + 7u];
    }
    return acc;
}

kernel void binary_g64_matvec_geo_c5120_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 5120u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = binary_acc_g64(signs, scales, input, row, COLS, lane_in_row);
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

kernel void binary_g64_matvec_geo_c17408_tpr64_tg128(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 17408u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = binary_acc_g64(signs, scales, input, row, COLS, lane_in_row);
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

static inline float binary_acc_g64_serial(
    device const uchar* signs,
    device const half* scales,
    device const float* x,
    uint row,
    uint cols)
{
    const uint gpr = cols >> 6u;
    const uint row_base = row * cols;
    float acc = 0.0f;
    for (uint col = 0u; col + 8u <= cols; col += 8u) {
        const float scale = float(scales[row * gpr + (col >> 6u)]);
        const uint flat = row_base + col;
        const uchar byte = signs[flat >> 3u];
        acc += ((byte & 0x01u) ? scale : -scale) * x[col];
        acc += ((byte & 0x02u) ? scale : -scale) * x[col + 1u];
        acc += ((byte & 0x04u) ? scale : -scale) * x[col + 2u];
        acc += ((byte & 0x08u) ? scale : -scale) * x[col + 3u];
        acc += ((byte & 0x10u) ? scale : -scale) * x[col + 4u];
        acc += ((byte & 0x20u) ? scale : -scale) * x[col + 5u];
        acc += ((byte & 0x40u) ? scale : -scale) * x[col + 6u];
        acc += ((byte & 0x80u) ? scale : -scale) * x[col + 7u];
    }
    return acc;
}

kernel void binary_g64_matvec_serial_c5120(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint row                   [[thread_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    output[row] = binary_acc_g64_serial(signs, scales, input, row, 5120u);
}

kernel void binary_g64_matvec_serial_c17408(
    device const uchar* signs  [[buffer(0)]],
    device const half*  scales [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint row                   [[thread_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    output[row] = binary_acc_g64_serial(signs, scales, input, row, 17408u);
}

// ── shared K=2 binary bases: group dots once, then per-layer scale ────────
// dots layout: [k=0|1][row][group]  (K major)

kernel void shared_binary_k2_group_dots_c5120_g64_tpr64_tg128(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       dots   [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red0[4];
    threadgroup float red1[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 5120u;
    constexpr uint GPR = 80u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    const uint row_base = row * COLS;
    for (uint g = 0u; g < GPR; ++g) {
        float a0 = 0.0f;
        float a1 = 0.0f;
        if (row < rows) {
            const uint col = g * 64u + lane_in_row;
            const uint flat = row_base + col;
            const uchar b0 = signs0[flat >> 3u];
            const uchar b1 = signs1[flat >> 3u];
            const uint bit = flat & 7u;
            const float xv = input[col];
            a0 = (((b0 >> bit) & 1u) != 0u) ? xv : -xv;
            a1 = (((b1 >> bit) & 1u) != 0u) ? xv : -xv;
        }
        a0 = simd_sum(a0);
        a1 = simd_sum(a1);
        if (simd_lane == 0u) {
            red0[simd_id] = a0;
            red1[simd_id] = a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && row < rows) {
            const uint t = team * kSplit;
            const uint off = row * GPR + g;
            dots[off] = red0[t] + red0[t + 1u];
            dots[rows * GPR + off] = red1[t] + red1[t + 1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void shared_binary_k2_group_dots_c17408_g64_tpr64_tg128(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       dots   [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red0[4];
    threadgroup float red1[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 17408u;
    constexpr uint GPR = 272u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    const uint row_base = row * COLS;
    for (uint g = 0u; g < GPR; ++g) {
        float a0 = 0.0f;
        float a1 = 0.0f;
        if (row < rows) {
            const uint col = g * 64u + lane_in_row;
            const uint flat = row_base + col;
            const uchar b0 = signs0[flat >> 3u];
            const uchar b1 = signs1[flat >> 3u];
            const uint bit = flat & 7u;
            const float xv = input[col];
            a0 = (((b0 >> bit) & 1u) != 0u) ? xv : -xv;
            a1 = (((b1 >> bit) & 1u) != 0u) ? xv : -xv;
        }
        a0 = simd_sum(a0);
        a1 = simd_sum(a1);
        if (simd_lane == 0u) {
            red0[simd_id] = a0;
            red1[simd_id] = a1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split == 0u && simd_lane == 0u && row < rows) {
            const uint t = team * kSplit;
            const uint off = row * GPR + g;
            dots[off] = red0[t] + red0[t + 1u];
            dots[rows * GPR + off] = red1[t] + red1[t + 1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// y[r] = sum_k sum_g s[k,r,g] * d[k,r,g]. gpr is 80 or 272 (specialized).
kernel void shared_binary_k2_scale_contract_gpr80(
    device const half*  scales [[buffer(0)]],
    device const float* dots   [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    uint row                   [[thread_position_in_grid]])
{
    constexpr uint GPR = 80u;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    const uint off = row * GPR;
    const uint k1 = rows * GPR;
    for (uint g = 0u; g < GPR; ++g) {
        acc += float(scales[off + g]) * dots[off + g];
        acc += float(scales[k1 + off + g]) * dots[k1 + off + g];
    }
    output[row] = acc;
}

kernel void shared_binary_k2_scale_contract_gpr272(
    device const half*  scales [[buffer(0)]],
    device const float* dots   [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    uint row                   [[thread_position_in_grid]])
{
    constexpr uint GPR = 272u;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    const uint off = row * GPR;
    const uint k1 = rows * GPR;
    for (uint g = 0u; g < GPR; ++g) {
        acc += float(scales[off + g]) * dots[off + g];
        acc += float(scales[k1 + off + g]) * dots[k1 + off + g];
    }
    output[row] = acc;
}

// BAD control: re-load both bases every layer (no amortisation).
kernel void shared_binary_k2_reload_every_layer_c5120(
    device const uchar* signs0 [[buffer(0)]],
    device const uchar* signs1 [[buffer(1)]],
    device const half*  scales [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 5120u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = binary_acc_g64(signs0, scales, input, row, COLS, lane_in_row);
        acc += binary_acc_g64(signs1, scales + (rows * (COLS >> 6u)), input, row, COLS, lane_in_row);
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

// ── binary + sparse CSR fused (the residual plane) ────────────────────────
// CSR: row_ptr[rows+1], col_idx[nnz] u32, corr_f16[nnz] (signed correction).

static inline float csr_add(
    device const uint* row_ptr,
    device const uint* col_idx,
    device const half* corr,
    device const float* x,
    uint row)
{
    float acc = 0.0f;
    const uint begin = row_ptr[row];
    const uint end = row_ptr[row + 1u];
    for (uint j = begin; j < end; ++j) {
        acc += float(corr[j]) * x[col_idx[j]];
    }
    return acc;
}

kernel void binary_sparse_fused_geo_c5120_tpr64_tg128(
    device const uchar* signs   [[buffer(0)]],
    device const half*  scales  [[buffer(1)]],
    device const uint*  row_ptr [[buffer(2)]],
    device const uint*  col_idx [[buffer(3)]],
    device const half*  corr    [[buffer(4)]],
    device const float* input   [[buffer(5)]],
    device float*       output  [[buffer(6)]],
    constant uint& rows         [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 5120u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = binary_acc_g64(signs, scales, input, row, COLS, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u]
            + csr_add(row_ptr, col_idx, corr, input, row);
    }
}

kernel void binary_sparse_fused_geo_c17408_tpr64_tg128(
    device const uchar* signs   [[buffer(0)]],
    device const half*  scales  [[buffer(1)]],
    device const uint*  row_ptr [[buffer(2)]],
    device const uint*  col_idx [[buffer(3)]],
    device const half*  corr    [[buffer(4)]],
    device const float* input   [[buffer(5)]],
    device float*       output  [[buffer(6)]],
    constant uint& rows         [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 17408u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = binary_acc_g64(signs, scales, input, row, COLS, lane_in_row);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u]
            + csr_add(row_ptr, col_idx, corr, input, row);
    }
}

kernel void binary_sparse_fused_serial_c5120(
    device const uchar* signs   [[buffer(0)]],
    device const half*  scales  [[buffer(1)]],
    device const uint*  row_ptr [[buffer(2)]],
    device const uint*  col_idx [[buffer(3)]],
    device const half*  corr    [[buffer(4)]],
    device const float* input   [[buffer(5)]],
    device float*       output  [[buffer(6)]],
    constant uint& rows         [[buffer(7)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows) {
        return;
    }
    output[row] = binary_acc_g64_serial(signs, scales, input, row, 5120u)
        + csr_add(row_ptr, col_idx, corr, input, row);
}

// NO-OP control: binary plane only, ignore CSR (same binary bytes, drop correction).
// ── q2f g64 baseline (N021): w = (q-1.5)*delta, 4 codes/byte, half delta ──

static inline float q2f_acc_g64(
    device const uchar* codes,
    device const half* deltas,
    device const float* x,
    uint row,
    uint cols,
    uint lane)
{
    const uint gpr = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * gpr + group;
        const float delta = float(deltas[rgb]);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        for (uint i = 0u; i < 8u; ++i) {
            const uint q = (packed16 >> (2u * i)) & 3u;
            acc += ((float(q) - 1.5f) * delta) * x[col + i];
        }
    }
    return acc;
}

kernel void q2f_g64_matvec_geo_c5120_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 5120u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = q2f_acc_g64(codes, deltas, input, row, COLS, lane_in_row);
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

kernel void q2f_g64_matvec_geo_c17408_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const half*  deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint group_id              [[threadgroup_position_in_grid]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 17408u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows) {
        acc = q2f_acc_g64(codes, deltas, input, row, COLS, lane_in_row);
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

kernel void q2f_g64_matvec_serial_c5120(
    device const uchar* codes  [[buffer(0)]],
    device const half*  deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    uint row                   [[thread_position_in_grid]])
{
    constexpr uint COLS = 5120u;
    if (row >= rows) {
        return;
    }
    const uint gpr = COLS >> 6u;
    float acc = 0.0f;
    for (uint col = 0u; col + 8u <= COLS; col += 8u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * gpr + group;
        const float delta = float(deltas[rgb]);
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        for (uint i = 0u; i < 8u; ++i) {
            const uint q = (packed16 >> (2u * i)) & 3u;
            acc += ((float(q) - 1.5f) * delta) * input[col + i];
        }
    }
    output[row] = acc;
}

kernel void binary_sparse_noop_drop_csr_c5120(
    device const uchar* signs   [[buffer(0)]],
    device const half*  scales  [[buffer(1)]],
    device const uint*  row_ptr [[buffer(2)]],
    device const uint*  col_idx [[buffer(3)]],
    device const half*  corr    [[buffer(4)]],
    device const float* input   [[buffer(5)]],
    device float*       output  [[buffer(6)]],
    constant uint& rows         [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    constexpr uint COLS = 5120u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    (void)row_ptr;
    (void)col_idx;
    (void)corr;
    float acc = 0.0f;
    if (row < rows) {
        acc = binary_acc_g64(signs, scales, input, row, COLS, lane_in_row);
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
