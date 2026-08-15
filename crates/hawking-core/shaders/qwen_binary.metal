// Packed binary sign + FP16 group-scale matvec component for Ascension Qwen
// candidates.
//
// Layout: row-major sign bitstream (LSB-first within each byte) and one
// half-precision absmax scale per contiguous group of `group_size` columns.
// Qwen30 admits group_size=128; cols are multiples of the group size.
//
// Entry-point policy (Q80 L0 regression, commit 8d451127):
// - `qwen_binary_sign_scale_matvec` is the association-preserving serial
//   kernel. The shared TCB (`qwen_binary_sign_scale_matvec_component_tcb`)
//   and Q80 structural kernel traces dispatch this name with serial geometry
//   (grid = rows, TG = 256). Restoring serial under this name keeps Q80
//   numerics at O(1e-6) residual error without widening tolerance.
// - `qwen_binary_sign_scale_matvec_tiled` is the Q30 speed path: 8 rows per
//   256-thread TG, contiguous 32-col tiles, simd_sum reduction. Requires
//   tiled geometry (grid = ceil(rows/8)*256, TG = 256). Association differs
//   from serial; Q30 admits it via bit-identical decode on mt8.
// - `qwen_binary_sign_scale_matvec_serial` is an explicit serial alias for
//   Q30 SerialControl A/B and component parity.

#include <metal_stdlib>
using namespace metal;

// Shared serial accumulation: left-to-right f32 order, one thread per row.
// Byte-wise unpack preserves col = group_start .. group_end-1 order.
static inline float qwen_binary_sign_scale_matvec_serial_row(
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

// Association-preserving default for the shared TCB / Q80 path.
// Grid: (rows, 1, 1), threadgroup: (256, 1, 1).
kernel void qwen_binary_sign_scale_matvec(
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
    output[row] = qwen_binary_sign_scale_matvec_serial_row(
        signs, scales, input, row, cols, group_size, groups_per_row);
}

// Explicit serial alias (Q30 SerialControl / component A/B).
kernel void qwen_binary_sign_scale_matvec_serial(
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
    output[row] = qwen_binary_sign_scale_matvec_serial_row(
        signs, scales, input, row, cols, group_size, groups_per_row);
}

// Q30 live speed path: coalesced 32-col tiles + simdgroup reduction.
// Geometry: one simdgroup (32 lanes) owns one output row; eight rows share a
// 256-thread threadgroup. Association differs from serial.
// Grid: (ceil(rows / 8) * 256, 1, 1), threadgroup: (256, 1, 1).
kernel void qwen_binary_sign_scale_matvec_tiled(
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
    if (row >= rows) return;

    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;

    // Tile columns in contiguous 32-wide blocks. For group_size multiple of 32
    // (Qwen30 admits 128), every tile lies inside a single scale group, so the
    // scale is identical for all lanes and is loaded once per tile.
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float scale = float(scales[scale_base + col / group_size]);
        const uint flat = row_base + col;
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        partial += (positive ? scale : -scale) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// Prior opt-in candidate: strided column ownership (lane handles col,
// col+32, ...). Retained so existing `--packed-matvec-kernel simdgroup-candidate`
// receipts keep a distinct entry point. Prefer the tiled entry above for Q30.
kernel void qwen_binary_sign_scale_matvec_simdgroup_candidate(
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
    if (row >= rows) return;

    float sum = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint col = simd_lane; col < cols; col += 32u) {
        const uint flat = row_base + col;
        const float scale = float(scales[scale_base + col / group_size]);
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        sum += (positive ? scale : -scale) * input[col];
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) output[row] = sum;
}

// ---------------------------------------------------------------------------
// Register-blocked row variants (LANE Q).
//
// Each simdgroup owns R consecutive output rows. Every lane keeps R independent
// float accumulators so column-loop FMAs are independent chains (ILP for DRAM
// latency hiding). The input activation is loaded once per column tile and
// reused across the R rows.
//
// Within each row the column order is identical to qwen_binary_sign_scale_matvec
// (contiguous 32-wide tiles, lane L multiplies base+L, then simd_sum). So each
// output element is bit-identical to the R=1 default by construction.
//
// Geometry: 8 simdgroups / TG → 8*R rows per TG.
// Grid: (ceil(rows / (8*R)) * 256, 1, 1), TG (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void qwen_binary_sign_scale_matvec_rowblock2(
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
    constexpr uint R = 2u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) {
        return;
    }
    const uint row1 = row0 + 1u;
    const bool has1 = row1 < rows;
    const uint r1 = has1 ? row1 : row0;

    float a0 = 0.0f;
    float a1 = 0.0f;
    const uint rb0 = row0 * cols;
    const uint rb1 = r1 * cols;
    const uint sb0 = row0 * groups_per_row;
    const uint sb1 = r1 * groups_per_row;

    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float x = input[col];
        const uint g = col / group_size;
        {
            const float scale = float(scales[sb0 + g]);
            const uint flat = rb0 + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            a0 += (positive ? scale : -scale) * x;
        }
        {
            const float scale = float(scales[sb1 + g]);
            const uint flat = rb1 + col;
            const uchar byte = signs[flat >> 3u];
            const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
            a1 += (positive ? scale : -scale) * x;
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    if (simd_lane == 0u) {
        output[row0] = a0;
        if (has1) {
            output[row1] = a1;
        }
    }
}

kernel void qwen_binary_sign_scale_matvec_rowblock4(
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

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint rb0 = row0 * cols, rb1 = r1 * cols, rb2 = r2 * cols, rb3 = r3 * cols;
    const uint sb0 = row0 * groups_per_row, sb1 = r1 * groups_per_row;
    const uint sb2 = r2 * groups_per_row, sb3 = r3 * groups_per_row;

    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float x = input[col];
        const uint g = col / group_size;
        {
            const float scale = float(scales[sb0 + g]);
            const uint flat = rb0 + col;
            const uchar byte = signs[flat >> 3u];
            a0 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
        {
            const float scale = float(scales[sb1 + g]);
            const uint flat = rb1 + col;
            const uchar byte = signs[flat >> 3u];
            a1 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
        {
            const float scale = float(scales[sb2 + g]);
            const uint flat = rb2 + col;
            const uchar byte = signs[flat >> 3u];
            a2 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
        {
            const float scale = float(scales[sb3 + g]);
            const uint flat = rb3 + col;
            const uchar byte = signs[flat >> 3u];
            a3 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
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

kernel void qwen_binary_sign_scale_matvec_rowblock8(
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
    constexpr uint R = 8u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint row0 = group_id * kRowsPerTg + simd_id * R;
    if (row0 >= rows) {
        return;
    }
    uint rid[8];
    bool has[8];
    rid[0] = row0;
    has[0] = true;
    for (uint r = 1u; r < R; ++r) {
        rid[r] = row0 + r;
        has[r] = rid[r] < rows;
        if (!has[r]) {
            rid[r] = row0;
        }
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint rb[8];
    uint sb[8];
    for (uint r = 0u; r < R; ++r) {
        rb[r] = rid[r] * cols;
        sb[r] = rid[r] * groups_per_row;
    }

    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float x = input[col];
        const uint g = col / group_size;
        // Explicit unrolled-style body: eight independent FMA chains.
        for (uint r = 0u; r < R; ++r) {
            const float scale = float(scales[sb[r] + g]);
            const uint flat = rb[r] + col;
            const uchar byte = signs[flat >> 3u];
            acc[r] += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
    }
    for (uint r = 0u; r < R; ++r) {
        acc[r] = simd_sum(acc[r]);
    }
    if (simd_lane == 0u) {
        for (uint r = 0u; r < R; ++r) {
            if (has[r]) {
                output[row0 + r] = acc[r];
            }
        }
    }
}

// ---- LANE K kernels grafted onto LANE Q's row-blocked base ----

// Fused Q+K+V packed matvec (Lane K / K1).
//
// Q, K, and V share the same activation vector and write disjoint outputs.
// Concatenating their rows into one dispatch yields
//   ceil((q_rows + k_rows + v_rows) / 8)
// threadgroups — 640 for Qwen30 (512+64+64) — so the starved K/V projections
// no longer run as 64-TG and 64-TG solo launches on an ~80-core GPU.
//
// Per-row accumulation is identical to `qwen_binary_sign_scale_matvec`: each
// simdgroup owns one logical row of exactly one projection, so association
// order is unchanged. Weight tensors stay separate (three sign/scale pairs);
// the artifact is not repacked.
//
// Grid: (ceil(total_rows / 8) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_binary_sign_scale_matvec_qkv(
    device const uchar* q_signs     [[buffer(0)]],
    device const half*  q_scales    [[buffer(1)]],
    device const uchar* k_signs     [[buffer(2)]],
    device const half*  k_scales    [[buffer(3)]],
    device const uchar* v_signs     [[buffer(4)]],
    device const half*  v_scales    [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       q_output    [[buffer(7)]],
    device float*       k_output    [[buffer(8)]],
    device float*       v_output    [[buffer(9)]],
    constant uint& q_rows           [[buffer(10)]],
    constant uint& k_rows           [[buffer(11)]],
    constant uint& v_rows           [[buffer(12)]],
    constant uint& cols             [[buffer(13)]],
    constant uint& group_size       [[buffer(14)]],
    constant uint& groups_per_row   [[buffer(15)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint global_row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (global_row >= total_rows) return;

    device const uchar* signs;
    device const half* scales;
    device float* output;
    uint row;
    if (global_row < q_rows) {
        signs = q_signs;
        scales = q_scales;
        output = q_output;
        row = global_row;
    } else if (global_row < q_rows + k_rows) {
        signs = k_signs;
        scales = k_scales;
        output = k_output;
        row = global_row - q_rows;
    } else {
        signs = v_signs;
        scales = v_scales;
        output = v_output;
        row = global_row - q_rows - k_rows;
    }

    float partial = 0.0f;
    const uint row_base = row * cols;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float scale = float(scales[scale_base + col / group_size]);
        const uint flat = row_base + col;
        const uchar byte = signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        partial += (positive ? scale : -scale) * input[col];
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        output[row] = partial;
    }
}

// Fused Q+K+V packed matvec with R=4 row ownership (RowBlock4 + Lane K / K1).
//
// Same surface as `qwen_binary_sign_scale_matvec_qkv`, but each simdgroup owns
// R=4 consecutive global rows with the same tiled-column association as
// `qwen_binary_sign_scale_matvec_rowblock4`. Concatenating Q/K/V yields
//   ceil((q_rows + k_rows + v_rows) / 32)
// threadgroups — 20 for Qwen30 (512+64+64) — instead of three separate
// rowblock4 launches (16+2+2).
//
// Per-row accumulation is bit-identical to the split rowblock4 path: each of
// the R rows is an independent f32 chain over contiguous 32-col tiles with
// simd_sum reduction. Qwen30 row counts (512/64/64) are multiples of
// rows_per_tg=32, so no TG spans a Q/K/V boundary; the general path still
// resolves each global row to its projection independently.
//
// Grid: (ceil(total_rows / 32) * 256, 1, 1), TG (256, 1, 1).
kernel void qwen_binary_sign_scale_matvec_qkv_rowblock4(
    device const uchar* q_signs     [[buffer(0)]],
    device const half*  q_scales    [[buffer(1)]],
    device const uchar* k_signs     [[buffer(2)]],
    device const half*  k_scales    [[buffer(3)]],
    device const uchar* v_signs     [[buffer(4)]],
    device const half*  v_scales    [[buffer(5)]],
    device const float* input       [[buffer(6)]],
    device float*       q_output    [[buffer(7)]],
    device float*       k_output    [[buffer(8)]],
    device float*       v_output    [[buffer(9)]],
    constant uint& q_rows           [[buffer(10)]],
    constant uint& k_rows           [[buffer(11)]],
    constant uint& v_rows           [[buffer(12)]],
    constant uint& cols             [[buffer(13)]],
    constant uint& group_size       [[buffer(14)]],
    constant uint& groups_per_row   [[buffer(15)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint R = 4u;
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kRowsPerTg = kSimdgroupsPerThreadgroup * R;
    const uint total_rows = q_rows + k_rows + v_rows;
    const uint global_row0 = group_id * kRowsPerTg + simd_id * R;
    if (global_row0 >= total_rows) {
        return;
    }

    // Resolve each of the R global rows to a projection + local row.
    // Inactive slots (past total_rows) alias global_row0 so the hot path
    // stays uniform; they are never written.
    const bool has1 = (global_row0 + 1u) < total_rows;
    const bool has2 = (global_row0 + 2u) < total_rows;
    const bool has3 = (global_row0 + 3u) < total_rows;
    const uint g0 = global_row0;
    const uint g1 = has1 ? (global_row0 + 1u) : global_row0;
    const uint g2 = has2 ? (global_row0 + 2u) : global_row0;
    const uint g3 = has3 ? (global_row0 + 3u) : global_row0;

    device const uchar* s0 = (g0 < q_rows) ? q_signs
        : (g0 < q_rows + k_rows) ? k_signs : v_signs;
    device const uchar* s1 = (g1 < q_rows) ? q_signs
        : (g1 < q_rows + k_rows) ? k_signs : v_signs;
    device const uchar* s2 = (g2 < q_rows) ? q_signs
        : (g2 < q_rows + k_rows) ? k_signs : v_signs;
    device const uchar* s3 = (g3 < q_rows) ? q_signs
        : (g3 < q_rows + k_rows) ? k_signs : v_signs;
    device const half* sc0 = (g0 < q_rows) ? q_scales
        : (g0 < q_rows + k_rows) ? k_scales : v_scales;
    device const half* sc1 = (g1 < q_rows) ? q_scales
        : (g1 < q_rows + k_rows) ? k_scales : v_scales;
    device const half* sc2 = (g2 < q_rows) ? q_scales
        : (g2 < q_rows + k_rows) ? k_scales : v_scales;
    device const half* sc3 = (g3 < q_rows) ? q_scales
        : (g3 < q_rows + k_rows) ? k_scales : v_scales;
    device float* o0 = (g0 < q_rows) ? q_output
        : (g0 < q_rows + k_rows) ? k_output : v_output;
    device float* o1 = (g1 < q_rows) ? q_output
        : (g1 < q_rows + k_rows) ? k_output : v_output;
    device float* o2 = (g2 < q_rows) ? q_output
        : (g2 < q_rows + k_rows) ? k_output : v_output;
    device float* o3 = (g3 < q_rows) ? q_output
        : (g3 < q_rows + k_rows) ? k_output : v_output;
    const uint local0 = (g0 < q_rows) ? g0
        : (g0 < q_rows + k_rows) ? (g0 - q_rows) : (g0 - q_rows - k_rows);
    const uint local1 = (g1 < q_rows) ? g1
        : (g1 < q_rows + k_rows) ? (g1 - q_rows) : (g1 - q_rows - k_rows);
    const uint local2 = (g2 < q_rows) ? g2
        : (g2 < q_rows + k_rows) ? (g2 - q_rows) : (g2 - q_rows - k_rows);
    const uint local3 = (g3 < q_rows) ? g3
        : (g3 < q_rows + k_rows) ? (g3 - q_rows) : (g3 - q_rows - k_rows);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint rb0 = local0 * cols, rb1 = local1 * cols;
    const uint rb2 = local2 * cols, rb3 = local3 * cols;
    const uint sb0 = local0 * groups_per_row, sb1 = local1 * groups_per_row;
    const uint sb2 = local2 * groups_per_row, sb3 = local3 * groups_per_row;

    // Same column tiling + independent FMA chains as rowblock4.
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= cols) {
            continue;
        }
        const float x = input[col];
        const uint g = col / group_size;
        {
            const float scale = float(sc0[sb0 + g]);
            const uint flat = rb0 + col;
            const uchar byte = s0[flat >> 3u];
            a0 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
        {
            const float scale = float(sc1[sb1 + g]);
            const uint flat = rb1 + col;
            const uchar byte = s1[flat >> 3u];
            a1 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
        {
            const float scale = float(sc2[sb2 + g]);
            const uint flat = rb2 + col;
            const uchar byte = s2[flat >> 3u];
            a2 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
        {
            const float scale = float(sc3[sb3 + g]);
            const uint flat = rb3 + col;
            const uchar byte = s3[flat >> 3u];
            a3 += ((((byte >> (flat & 7u)) & 1u) != 0u) ? scale : -scale) * x;
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (simd_lane == 0u) {
        o0[local0] = a0;
        if (has1) o1[local1] = a1;
        if (has2) o2[local2] = a2;
        if (has3) o3[local3] = a3;
    }
}

// Fused post-attention RMSNorm + router packed matvec (Lane K / K3).
//
// The router is only 128 rows = 16 threadgroups and cannot fill an M3 Ultra.
// Folding the preceding post-attention RMSNorm into the same dispatch removes
// one launch and recomputes the (cheap) norm inside each router TG so the
// matvec still sees the same x_norm values. Per-row matvec association matches
// `qwen_binary_sign_scale_matvec`. RMSNorm reduction matches `rmsnorm_f32`
// (same 256-thread tree). Only group_id 0 writes x_norm; every TG applies the
// same inv_rms on the fly for its router rows.
//
// Grid: (ceil(router_rows / 8) * 256, 1, 1), TG (256, 1, 1).
// threadgroup(0): 256 floats for variance reduction.
kernel void qwen_binary_postnorm_router_matvec(
    device const float* x              [[buffer(0)]],
    device const float* norm_weight    [[buffer(1)]],
    device float*       x_norm         [[buffer(2)]],
    device const uchar* router_signs   [[buffer(3)]],
    device const half*  router_scales  [[buffer(4)]],
    device float*       router_logits  [[buffer(5)]],
    constant uint& hidden              [[buffer(6)]],
    constant uint& router_rows         [[buffer(7)]],
    constant uint& group_size          [[buffer(8)]],
    constant uint& groups_per_row      [[buffer(9)]],
    constant float& eps                [[buffer(10)]],
    uint group_id                       [[threadgroup_position_in_grid]],
    uint tid                            [[thread_position_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint tg_size                        [[threads_per_threadgroup]],
    threadgroup float* shmem            [[threadgroup(0)]])
{
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;

    // Phase 1: RMSNorm variance, identical association to rmsnorm_f32.
    float partial_var = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = x[i];
        partial_var += v * v;
    }
    shmem[tid] = partial_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms = sqrt(shmem[0] / (float)hidden + eps);
    const float inv = 1.0f / rms;

    // Phase 2: materialize x_norm once (group 0 only) for the expert wave.
    if (group_id == 0u) {
        for (uint i = tid; i < hidden; i += tg_size) {
            x_norm[i] = x[i] * inv * norm_weight[i];
        }
    }
    // No grid barrier available; matvec below uses on-the-fly scaled inputs
    // rather than reading x_norm, so other TGs never wait on group 0's stores.

    // Phase 3: router rows, same geometry as qwen_binary_sign_scale_matvec.
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= router_rows) return;

    float partial = 0.0f;
    const uint row_base = row * hidden;
    const uint scale_base = row * groups_per_row;
    for (uint base = 0u; base < hidden; base += kSimdWidth) {
        const uint col = base + simd_lane;
        if (col >= hidden) {
            continue;
        }
        const float scale = float(router_scales[scale_base + col / group_size]);
        const uint flat = row_base + col;
        const uchar byte = router_signs[flat >> 3u];
        const bool positive = ((byte >> (flat & 7u)) & 1u) != 0u;
        const float xcol = x[col] * inv * norm_weight[col];
        partial += (positive ? scale : -scale) * xcol;
    }
    partial = simd_sum(partial);
    if (simd_lane == 0u) {
        router_logits[row] = partial;
    }
}
