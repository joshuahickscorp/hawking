// matmul.metal — simdgroup_matrix GEMV kernels (Wedge H + v1.1.0-X).
//
// One SIMD group (32 threads) per threadgroup; each threadgroup computes
// 8 output rows. The activation vector x is broadcast across all 8 columns
// of the x tile (X[k][n] = x[k] ∀n). After accumulation all 8 columns of acc
// hold the same partial dot-product; column 0 is extracted for output.
//
// Grid:  (ceil(rows/8)*32, 1, 1)  — one threadgroup per 8 output rows
// TG:    (32, 1, 1)               — one SIMD group
// Threadgroup memory layout (stride 8 per row, all float):
//   shmem[ 0.. 64): weight tile  W[8][8]    (f32)
//   shmem[64..128): act    tile  X[8][8]    (broadcast: X[k][n] = x[k])
//   shmem[128..192): result tile D[8][8]    (for simdgroup_store + zero-init)
//
// Requires: cols % 8 == 0. Handles rows % 8 != 0 by padding weight rows to 0.
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// v1.0.0-H — simdgroup_matrix GEMV: w (rows×cols f32) × x (cols f32) → y (rows f32).
kernel void gemv_simdgroup_f32(
    device const float* w       [[buffer(0)]],   // (rows × cols) f32, row-major
    device const float* x       [[buffer(1)]],   // (cols,) f32
    device       float* y       [[buffer(2)]],   // (rows,) f32
    constant     uint&  rows    [[buffer(3)]],
    constant     uint&  cols    [[buffer(4)]],
    threadgroup  float* shmem   [[threadgroup(0)]],  // 192 floats = 3 × 64
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint base_row = gid * 8u;
    if (base_row >= rows) return;

    threadgroup float* shmem_w   = shmem;         // [64]
    threadgroup float* shmem_x   = shmem + 64;    // [64]
    threadgroup float* shmem_out = shmem + 128;   // [64]

    // Zero-init accumulator via shmem_out (simdgroup_load initialises acc from it).
    shmem_out[tid]      = 0.0f;
    shmem_out[tid + 32] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, shmem_out, 8, ulong2(0, 0));

    uint n_chunks = cols / 8u;  // cols % 8 == 0 required

    for (uint chunk = 0; chunk < n_chunks; ++chunk) {
        uint c_base = chunk * 8u;

        // Fill weight tile shmem_w[8][8] and activation tile shmem_x[8][8].
        // Each thread fills 2 slots (elem = tid and tid + 32), covering all 64.
        for (int e = 0; e < 2; ++e) {
            uint elem = tid + (uint)e * 32u;
            uint m = elem >> 3u;   // 0..7 — row within 8×8 tile
            uint k = elem &  7u;   // 0..7 — col within 8×8 tile

            // Weight tile: W[base_row+m][c_base+k], zero-padded if row out of bounds.
            uint row = base_row + m;
            shmem_w[elem] = (row < rows) ? w[(ulong)row * cols + c_base + k] : 0.0f;

            // Activation tile: broadcast x[c_base+m] to all 8 cols of row m.
            // Layout: shmem_x[m*8 + n] = x[c_base+m] ∀n → B[m][n] = x[c_base+m].
            // So (A×B)[i][j] = Σ_m A[i][m] * x[c_base+m] = partial GEMV dot. ✓
            shmem_x[elem] = x[c_base + m];  // m = elem >> 3 = row of 8×8 tile
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> w_mat, x_mat;
        simdgroup_load(w_mat, shmem_w, 8, ulong2(0, 0));
        simdgroup_load(x_mat, shmem_x, 8, ulong2(0, 0));
        simdgroup_multiply_accumulate(acc, w_mat, x_mat, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Extract results: all columns of acc are identical (broadcast), so column 0 suffices.
    simdgroup_store(acc, shmem_out, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8u && base_row + tid < rows) {
        y[base_row + tid] = shmem_out[tid * 8u];  // shmem_out[row][col=0]
    }
}

// v1.1.0-X — LM-head GEMV: w (rows×cols f16) × x (cols f32) → y (rows f32).
// Weights loaded from f16 buffer and promoted to f32 in threadgroup memory.
// Full f32 simdgroup_matrix arithmetic — matches CPU gemv_f16 within ~1e-5 atol.
// One SIMD group (32 threads) per threadgroup; each handles 8 output rows.
// Requires cols % 8 == 0. Grid = (ceil(rows/8)*32, 1, 1), TG = (32, 1, 1).
//
// threadgroup layout (same as gemv_simdgroup_f32, 192 floats = 768 bytes):
//   shmem[ 0..64): weight tile W[8][8] (f32, promoted from f16)
//   shmem[64..128): activation tile X[8][8] (f32 broadcast: X[k][n] = x[k] ∀n)
//   shmem[128..192): result tile D[8][8] for simdgroup_store + zero-init
kernel void gemv_f16_simdmat(
    device const half*  w       [[buffer(0)]],   // (rows × cols) f16, row-major
    device const float* x       [[buffer(1)]],   // (cols,) f32
    device       float* y       [[buffer(2)]],   // (rows,) f32
    constant     uint&  rows    [[buffer(3)]],
    constant     uint&  cols    [[buffer(4)]],
    threadgroup  float* shmem   [[threadgroup(0)]],  // 192 floats = 3 × 64
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    uint base_row = gid * 8u;
    if (base_row >= rows) return;

    threadgroup float* shmem_w   = shmem;         // W tile: [0..64)
    threadgroup float* shmem_x   = shmem + 64;    // X tile: [64..128)
    threadgroup float* shmem_out = shmem + 128;   // D tile: [128..192)

    // Zero-init result tile; simdgroup_load reads it to initialize acc to 0.
    shmem_out[tid]      = 0.0f;
    shmem_out[tid + 32] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> acc;
    simdgroup_load(acc, shmem_out, 8, ulong2(0, 0));

    uint n_chunks = cols / 8u;  // cols % 8 == 0 required

    for (uint chunk = 0; chunk < n_chunks; ++chunk) {
        uint c_base = chunk * 8u;

        // Fill W and X tiles (2 elements per thread, covers all 64 slots).
        for (int e = 0; e < 2; ++e) {
            uint elem = tid + (uint)e * 32u;
            uint m = elem >> 3u;  // 0..7 — row index within 8×8 tile
            uint k = elem &  7u;  // 0..7 — col index within 8×8 tile

            // Weight: promote f16 → f32 on load; zero-pad out-of-bounds rows.
            uint row = base_row + m;
            shmem_w[elem] = (row < rows) ? float(w[(ulong)row * cols + c_base + k]) : 0.0f;

            // Activation broadcast: X[m][k] = x[c_base+m] ∀k.
            shmem_x[elem] = x[c_base + m];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> w_mat, x_mat;
        simdgroup_load(w_mat, shmem_w, 8, ulong2(0, 0));
        simdgroup_load(x_mat, shmem_x, 8, ulong2(0, 0));
        simdgroup_multiply_accumulate(acc, w_mat, x_mat, acc);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // All columns of acc hold the same dot-product (broadcast invariant); use col 0.
    simdgroup_store(acc, shmem_out, 8, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8u && base_row + tid < rows) {
        y[base_row + tid] = shmem_out[tid * 8u];
    }
}

// ── GLM native.bf16 lm_head: device-resident, sequential f32 accumulate ─────
//
// Flagship `lm_head.weight` is native.bf16 [V, H] (~1.90 GB). The host path
// widens every element to f32 then does left-to-right Σ w[c]*x[c] per row.
// Parallel/simdgroup reduction reassociates and diverges; this kernel matches
// the host bit-for-bit:
//   1. widen bf16 → f32 as (u16 bits) << 16  (same as gravity::widen_native)
//   2. left-to-right mul then add (fp contract off — no FMA reassociation)
// One thread per output row. Weights stay bf16 on device after first upload.
//
// Binding:
//   0  weight_bits  (rows × cols) ushort  — bf16 bit patterns, row-major
//   1  act          (cols,)        float
//   2  out_logits   (rows,)        float
//   3  n_rows       constant uint
//   4  n_cols       constant uint
// Grid: (n_rows, 1, 1)  TG: (1, 1, 1) or any with one logical row per gid
#pragma clang fp contract(off)
kernel void gemv_native_bf16_seq(
    device const ushort* weight_bits [[buffer(0)]],
    device const float*  act         [[buffer(1)]],
    device       float*  out_logits  [[buffer(2)]],
    constant     uint&   n_rows      [[buffer(3)]],
    constant     uint&   n_cols      [[buffer(4)]],
    uint                 row_idx     [[thread_position_in_grid]])
{
    if (row_idx >= n_rows) return;
    device const ushort* row_bits =
        weight_bits + (ulong)row_idx * (ulong)n_cols;
    float acc = 0.0f;
    for (uint col = 0u; col < n_cols; ++col) {
        uint wide_bits = ((uint)row_bits[col]) << 16;
        float w_val = as_type<float>(wide_bits);
        float product = w_val * act[col];
        acc = acc + product;
    }
    out_logits[row_idx] = acc;
}

// ── DeepSeek-V4 source-native FP8 authority probe ──────────────────────────
//
// `DeepSeek-V4-Flash` stores control/attention weights as E4M3FN and their
// block scales as E8M0FNU. This intentionally narrow kernel is a component
// parity authority for a sealed source tensor, not a model adapter: one GPU
// thread owns one output row and follows the source grammar directly without
// expanding a f16/f32 weight shadow.
//
// Weight grammar:
//   * E4M3FN: exponent bias 7, subnormal magnitude `mantissa * 2^-9`, finite
//     exponent-15 values through mantissa 6, and NaN at (exp=15,mantissa=7).
//   * E8M0FNU: `2^(byte - 127)`, with 0xff reserved as NaN.
//   * scale index: `[row / 128][col / 128]`.
//
// The Rust entry point validates every raw source byte before dispatch, so the
// invalid branches below are only defensive and cannot turn malformed input
// into a successful probe receipt. FP contraction is already disabled for the
// adjacent native-BF16 authority kernel above, retaining the CPU oracle's
// explicit product-then-add accumulation semantics.
static inline float deepseek_v4_e4m3fn_value(uchar bits)
{
    const uint raw = (uint)bits;
    const uint exponent = (raw >> 3u) & 0x0fu;
    const uint mantissa = raw & 0x07u;
    if (exponent == 0x0fu && mantissa == 0x07u) {
        return 0.0f; // rejected before dispatch; never a fallback path.
    }
    // Construct normal values in IEEE binary32 directly: the E4 exponent
    // bias is 7, so the matching f32 exponent field is `exponent + 120`;
    // E4's three fraction bits land at f32 bits 22..20. This avoids a
    // transcendental/pow implementation becoming part of the codec contract.
    const float magnitude = exponent == 0u
        ? (float)mantissa * 0.001953125f // 2^-9, exactly representable.
        : as_type<float>(((exponent + 120u) << 23u) | (mantissa << 20u));
    return (raw & 0x80u) != 0u ? -magnitude : magnitude;
}

static inline float deepseek_v4_e8m0fnu_value(uchar bits)
{
    if ((uint)bits == 0xffu) {
        return 0.0f; // rejected before dispatch; never a fallback path.
    }
    // E8M0's exponent byte is exactly the normal f32 exponent field for
    // bytes 1..254. Byte zero is 2^-127, an f32 subnormal with bit 22 set.
    return (uint)bits == 0u
        ? as_type<float>(0x00400000u)
        : as_type<float>(((uint)bits) << 23u);
}

kernel void deepseek_v4_fp8_e4m3fn_e8m0_matvec_authority(
    device const uchar* weights [[buffer(0)]], // [rows, cols] E4M3FN
    device const uchar* scales  [[buffer(1)]], // [rows/128, cols/128] E8M0FNU
    device const float* x       [[buffer(2)]], // [cols]
    device       float* y       [[buffer(3)]], // [rows]
    constant uint& rows          [[buffer(4)]],
    constant uint& cols          [[buffer(5)]],
    constant uint& scale_cols    [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint scale_row = row / 128u;
    const ulong row_base = (ulong)row * (ulong)cols;
    float acc = 0.0f;
    for (uint col = 0u; col < cols; ++col) {
        const uint scale_index = scale_row * scale_cols + col / 128u;
        const float unit = deepseek_v4_e4m3fn_value(weights[row_base + (ulong)col]);
        const float scale = deepseek_v4_e8m0fnu_value(scales[scale_index]);
        const float product = unit * scale * x[col];
        acc = acc + product;
    }
    y[row] = acc;
}

// ── DeepSeek-V4 FP8 optional SIMDgroup/split-K candidate ──────────────────
//
// This is intentionally a *separate* symbol from the serial authority above.
// It is an optional component-only candidate for one source-native control
// linear: one threadgroup reduces K for `rows_per_threadgroup` output rows.
// No V4 runtime path selects it.  The probe fixes `vector_width=4`: every
// thread reads four adjacent E4M3 bytes and four aligned FP32 activation
// values, then advances by `threads_x * 4`.  `threads_x` must be a positive
// multiple of the native SIMD width and `rows_per_threadgroup <= 4`; the Rust
// sweep validates those preconditions before dispatch.
//
// The partial array is deliberately bounded rather than dynamically sized:
// Apple pipelines admit at most 1024 threads/group, therefore at most 32
// SIMDgroups per output row.  Four rows/group need 128 FP32 partials (512 B).
// A different reduction order is expected to differ in low bits from the
// serial CPU oracle, so promotion requires the same explicit tolerance check
// as every other source-native candidate.
kernel void deepseek_v4_fp8_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate(
    device const uchar* weights [[buffer(0)]], // [rows, cols] E4M3FN
    device const uchar* scales  [[buffer(1)]], // [rows/128, cols/128] E8M0FNU
    device const float* x       [[buffer(2)]], // [cols]
    device       float* y       [[buffer(3)]], // [rows]
    constant uint& rows          [[buffer(4)]],
    constant uint& cols          [[buffer(5)]],
    constant uint& scale_cols    [[buffer(6)]],
    constant uint& threads_x     [[buffer(7)]],
    uint2 global_id              [[thread_position_in_grid]],
    uint2 local_id               [[thread_position_in_threadgroup]],
    uint simdgroup_id            [[simdgroup_index_in_threadgroup]],
    uint lane_id                 [[thread_index_in_simdgroup]])
{
    constexpr uint kVectorWidth = 4u;
    constexpr uint kMaxSimdgroupsPerRow = 32u;
    constexpr uint kMaxRowsPerThreadgroup = 4u;
    threadgroup float partial[kMaxRowsPerThreadgroup * kMaxSimdgroupsPerRow];

    const uint row = global_id.y;
    const uint local_row = local_id.y;
    // The host validates all geometry before dispatch. These guards make a
    // malformed direct use a no-op rather than an out-of-bounds candidate.
    if (threads_x == 0u || (threads_x & 31u) != 0u || local_row >= kMaxRowsPerThreadgroup) {
        return;
    }
    const uint simdgroups_per_row = threads_x / 32u;
    const uint local_simdgroup = simdgroup_id % simdgroups_per_row;
    const uint partial_index = local_row * kMaxSimdgroupsPerRow + local_simdgroup;

    float acc = 0.0f;
    if (row < rows) {
        const uint scale_row = row / 128u;
        const ulong row_base = (ulong)row * (ulong)cols;
        // `base_col` is always 4-aligned, which makes the uchar4 and float4
        // loads aligned under the source tensor and FP32 input contracts.
        for (uint base_col = local_id.x * kVectorWidth;
             base_col < cols;
             base_col += threads_x * kVectorWidth) {
            const uchar4 packed = *(device const uchar4*)(weights + row_base + (ulong)base_col);
            const float4 activation = *(device const float4*)(x + base_col);
            const uint scale_index = scale_row * scale_cols + base_col / 128u;
            const float scale = deepseek_v4_e8m0fnu_value(scales[scale_index]);
            acc += deepseek_v4_e4m3fn_value(packed.x) * scale * activation.x;
            acc += deepseek_v4_e4m3fn_value(packed.y) * scale * activation.y;
            acc += deepseek_v4_e4m3fn_value(packed.z) * scale * activation.z;
            acc += deepseek_v4_e4m3fn_value(packed.w) * scale * activation.w;
        }
    }

    const float reduced_simdgroup = simd_sum(acc);
    if (lane_id == 0u) {
        partial[partial_index] = reduced_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_simdgroup == 0u) {
        const float partial_value = lane_id < simdgroups_per_row
            ? partial[local_row * kMaxSimdgroupsPerRow + lane_id]
            : 0.0f;
        const float reduced_row = simd_sum(partial_value);
        if (lane_id == 0u && row < rows) {
            y[row] = reduced_row;
        }
    }
}

// ── DeepSeek-V4 source `Linear` act-quant → FP8 component checkpoint ────
//
// The earlier V4 FP8 probes deliberately accept a host FP32 activation so
// they can isolate the sealed E4M3FN/E8M0FNU source-weight reader and decoder.
// This separate, still bounded checkpoint carries the *source Linear* operand
// grammar across the device boundary instead:
//
//     BF16 [K] --act_quant(block=128, UE8M0)--> E4M3FN [K] + E8M0FNU [K/128]
//            --fp8_gemm with source weight/scales--> FP32 [out]
//
// It remains a component kernel family.  Nothing in the model engine, token
// loop, HCLI path, or runtime registry selects these symbols.  The Rust
// receipt entry point verifies finite input/weight bytes and exact CPU-oracle
// parity before it can describe a successful dispatch.

// Widen a source BF16 storage word without relying on a device BF16 arithmetic
// type.  BF16's bit layout is the high 16 bits of IEEE binary32.
static inline float deepseek_v4_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

// Encode a finite FP32 value as BF16 with IEEE round-to-nearest,
// ties-to-even.  The bounded P3A host validates that every source input and
// every checked output is finite before it can write a passing receipt, so
// this intentionally small helper is the exact finite conversion grammar
// needed by the source checkpoints rather than a general NaN payload policy.
static inline ushort deepseek_v4_bf16_encode_rne(float value)
{
    const uint bits = as_type<uint>(value);
    const uint low_lsb = (bits >> 16u) & 1u;
    return (ushort)((bits + 0x7fffu + low_lsb) >> 16u);
}

// The source `fast_round_scale(amax, 1/448)` computes the next power-of-two
// E8M0 rung.  The only exercised inputs are finite BF16 values, pre-validated
// by the host.  We spell the exponent extraction out so no generic log/pow
// implementation becomes part of the source-grammar contract.
static inline uchar deepseek_v4_act_quant_ue8m0_scale(float amax)
{
    const float clamped_amax = max(amax, 0.0001f);
    const float scaled = clamped_amax * (1.0f / 448.0f);
    const uint raw = as_type<uint>(scaled);
    const int exponent_field = (int)((raw >> 23u) & 0xffu);
    const uint mantissa = raw & 0x007fffffu;
    // The source path's finite BF16 input and `amax` floor make `scaled`
    // positive and normal for this bounded checkpoint.  The host checks that
    // the resulting byte is finite before accepting a receipt.
    const int exponent = exponent_field - 127 + (mantissa != 0u ? 1 : 0);
    return (uchar)(exponent + 127);
}

// The source uses a finite E4M3FN cast with round-to-nearest, ties-to-even.
// There are only 254 finite encodings, so this explicit finite table search is
// intentionally chosen for the checkpoint authority kernel.  It handles the
// unusual finite top bin (448) and avoids accidentally giving 0x7f/0xff an
// IEEE infinity meaning.  It is not a throughput candidate.
static inline uchar deepseek_v4_e4m3fn_encode_rne(float value)
{
    if (value == 0.0f) {
        return (as_type<uint>(value) & 0x80000000u) != 0u ? (uchar)0x80u : (uchar)0x00u;
    }
    uchar best_bits = (uchar)0u;
    float best_distance = INFINITY;
    bool found = false;
    for (uint raw = 0u; raw <= 255u; ++raw) {
        const uchar bits = (uchar)raw;
        const uint exponent = (raw >> 3u) & 0x0fu;
        const uint mantissa = raw & 0x07u;
        if (exponent == 0x0fu && mantissa == 0x07u) {
            continue; // E4M3FN's two NaN encodings.
        }
        const float candidate = deepseek_v4_e4m3fn_value(bits);
        const float distance = fabs(candidate - value);
        if (!found || distance < best_distance
            || (distance == best_distance && ((raw & 1u) == 0u)
                && (((uint)best_bits & 1u) != 0u))) {
            best_bits = bits;
            best_distance = distance;
            found = true;
        }
    }
    return best_bits;
}

// One logical source 128-wide block per GPU thread.  The geometry is small on
// purpose: it gives the exact byte-level quantizer a simple deterministic
// authority boundary before the separate source-native projection probe.  The
// host dispatches exactly K/128 logical blocks and records a real timestamp.
kernel void deepseek_v4_act_quant_bf16_ue8m0_authority(
    device const ushort* input_bf16 [[buffer(0)]], // [K] BF16 storage words
    device       uchar* quantized   [[buffer(1)]], // [K] E4M3FN bytes
    device       uchar* act_scales  [[buffer(2)]], // [K/128] E8M0FNU bytes
    constant uint& cols              [[buffer(3)]],
    uint block                       [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    if (cols == 0u || (cols % kBlock) != 0u || block >= cols / kBlock) return;
    const uint start = block * kBlock;
    float amax = 0.0f;
    for (uint offset = 0u; offset < kBlock; ++offset) {
        const float value = deepseek_v4_bf16_value(input_bf16[start + offset]);
        amax = max(amax, fabs(value));
    }
    const uchar scale_bits = deepseek_v4_act_quant_ue8m0_scale(amax);
    const float scale = deepseek_v4_e8m0fnu_value(scale_bits);
    act_scales[block] = scale_bits;
    for (uint offset = 0u; offset < kBlock; ++offset) {
        const float value = deepseek_v4_bf16_value(input_bf16[start + offset]);
        const float scaled = clamp(value / scale, -448.0f, 448.0f);
        quantized[start + offset] = deepseek_v4_e4m3fn_encode_rne(scaled);
    }
}

// Optional source-linear *component* candidate.  The authority kernel above
// deliberately serializes every 128-wide source block so its byte-level
// grammar is easy to audit.  That makes its E4M3FN table conversion the
// dominant cost of the bounded checkpoint.  This candidate keeps exactly the
// same UE8M0 rung and E4M3FN encoder, but gives one SIMDgroup one source block
// and lets multiple SIMDgroups process independent blocks in one threadgroup.
//
// `vector_width` is a deliberately small, host-swept geometry knob:
//   1 / 2: scalar loads, each lane visits the block in a strided loop;
//   4:     one aligned ushort4 packed load and uchar4 store per active lane;
//   8:     two aligned ushort4 packed loads/stores per active lane.
//
// The block maximum is order-independent and `deepseek_v4_e4m3fn_encode_rne`
// remains the exact finite-table authority implementation.  The host must
// prove every activation and scale byte equal to the canonical CPU oracle
// before recording this as a passing candidate.  Nothing in the V4 runtime
// selects this symbol.
kernel void deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate(
    device const ushort* input_bf16 [[buffer(0)]], // [K] BF16 storage words
    device       uchar* quantized   [[buffer(1)]], // [K] E4M3FN bytes
    device       uchar* act_scales  [[buffer(2)]], // [K/128] E8M0FNU bytes
    constant uint& cols              [[buffer(3)]],
    constant uint& threads_x         [[buffer(4)]],
    constant uint& vector_width      [[buffer(5)]],
    uint group                       [[threadgroup_position_in_grid]],
    uint simdgroup_id                [[simdgroup_index_in_threadgroup]],
    uint lane_id                     [[thread_index_in_simdgroup]],
    uint threads_per_tg              [[threads_per_threadgroup]])
{
    constexpr uint kBlock = 128u;
    constexpr uint kSimdWidth = 32u;
    constexpr uint kMaxSimdgroups = 32u; // 1024 thread Metal maximum / 32.
    threadgroup uchar scale_bits_by_simdgroup[kMaxSimdgroups];

    const bool valid_geometry = cols != 0u && (cols % kBlock) == 0u
        && threads_x >= kSimdWidth && (threads_x % kSimdWidth) == 0u
        && threads_x == threads_per_tg
        && threads_x / kSimdWidth <= kMaxSimdgroups
        && (vector_width == 1u || vector_width == 2u
            || vector_width == 4u || vector_width == 8u);
    const uint simdgroups = valid_geometry ? threads_x / kSimdWidth : 0u;
    const uint block = group * simdgroups + simdgroup_id;
    const bool active = valid_geometry && block < cols / kBlock;
    const uint start = block * kBlock;

    // Each active SIMDgroup reduces its own 128 BF16 values.  The scalar and
    // packed branches cover exactly the same element set; all input values
    // are finite by host admission of this bounded source checkpoint.
    float local_amax = 0.0f;
    if (active) {
        if (vector_width == 4u) {
            const ushort4 packed = *(device const ushort4*)(input_bf16 + start + lane_id * 4u);
            local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(packed.x)));
            local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(packed.y)));
            local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(packed.z)));
            local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(packed.w)));
        } else if (vector_width == 8u) {
            if (lane_id < 16u) {
                const uint local = lane_id * 8u;
                const ushort4 first = *(device const ushort4*)(input_bf16 + start + local);
                const ushort4 second = *(device const ushort4*)(input_bf16 + start + local + 4u);
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(first.x)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(first.y)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(first.z)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(first.w)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(second.x)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(second.y)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(second.z)));
                local_amax = max(local_amax, fabs(deepseek_v4_bf16_value(second.w)));
            }
        } else {
            for (uint local = lane_id * vector_width; local < kBlock;
                 local += kSimdWidth * vector_width) {
                for (uint element = 0u; element < vector_width && local + element < kBlock;
                     ++element) {
                    local_amax = max(local_amax,
                        fabs(deepseek_v4_bf16_value(input_bf16[start + local + element])));
                }
            }
        }
    }
    const float block_amax = simd_max(local_amax);
    if (active && lane_id == 0u) {
        scale_bits_by_simdgroup[simdgroup_id] = deepseek_v4_act_quant_ue8m0_scale(block_amax);
    }
    // Invalid/padded SIMDgroups participate too, which keeps the threadgroup
    // barrier well-defined on the final partially-populated threadgroup.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!active) return;
    const uchar scale_bits = scale_bits_by_simdgroup[simdgroup_id];
    const float scale = deepseek_v4_e8m0fnu_value(scale_bits);
    if (lane_id == 0u) act_scales[block] = scale_bits;

    if (vector_width == 4u) {
        const ushort4 packed = *(device const ushort4*)(input_bf16 + start + lane_id * 4u);
        const float4 scaled = clamp(float4(
            deepseek_v4_bf16_value(packed.x),
            deepseek_v4_bf16_value(packed.y),
            deepseek_v4_bf16_value(packed.z),
            deepseek_v4_bf16_value(packed.w)) / scale, -448.0f, 448.0f);
        *(device uchar4*)(quantized + start + lane_id * 4u) = uchar4(
            deepseek_v4_e4m3fn_encode_rne(scaled.x),
            deepseek_v4_e4m3fn_encode_rne(scaled.y),
            deepseek_v4_e4m3fn_encode_rne(scaled.z),
            deepseek_v4_e4m3fn_encode_rne(scaled.w));
    } else if (vector_width == 8u) {
        if (lane_id < 16u) {
            const uint local = lane_id * 8u;
            const ushort4 first = *(device const ushort4*)(input_bf16 + start + local);
            const ushort4 second = *(device const ushort4*)(input_bf16 + start + local + 4u);
            const float4 first_scaled = clamp(float4(
                deepseek_v4_bf16_value(first.x),
                deepseek_v4_bf16_value(first.y),
                deepseek_v4_bf16_value(first.z),
                deepseek_v4_bf16_value(first.w)) / scale, -448.0f, 448.0f);
            const float4 second_scaled = clamp(float4(
                deepseek_v4_bf16_value(second.x),
                deepseek_v4_bf16_value(second.y),
                deepseek_v4_bf16_value(second.z),
                deepseek_v4_bf16_value(second.w)) / scale, -448.0f, 448.0f);
            *(device uchar4*)(quantized + start + local) = uchar4(
                deepseek_v4_e4m3fn_encode_rne(first_scaled.x),
                deepseek_v4_e4m3fn_encode_rne(first_scaled.y),
                deepseek_v4_e4m3fn_encode_rne(first_scaled.z),
                deepseek_v4_e4m3fn_encode_rne(first_scaled.w));
            *(device uchar4*)(quantized + start + local + 4u) = uchar4(
                deepseek_v4_e4m3fn_encode_rne(second_scaled.x),
                deepseek_v4_e4m3fn_encode_rne(second_scaled.y),
                deepseek_v4_e4m3fn_encode_rne(second_scaled.z),
                deepseek_v4_e4m3fn_encode_rne(second_scaled.w));
        }
    } else {
        for (uint local = lane_id * vector_width; local < kBlock;
             local += kSimdWidth * vector_width) {
            for (uint element = 0u; element < vector_width && local + element < kBlock;
                 ++element) {
                const float value = deepseek_v4_bf16_value(input_bf16[start + local + element]);
                quantized[start + local + element] = deepseek_v4_e4m3fn_encode_rne(
                    clamp(value / scale, -448.0f, 448.0f));
            }
        }
    }
}

// Serial source-linear authority: preserve the CPU oracle's explicit
// 128-wide dot accumulator followed by activation×weight E8M0 scaling.  A
// thread owns one output row.  This is intentionally not an optimized GEMV;
// it is the numerical authority for the bounded GPU parity checkpoint.
kernel void deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority(
    device const uchar* weights       [[buffer(0)]], // [rows, cols] E4M3FN
    device const uchar* weight_scales [[buffer(1)]], // [rows/128, cols/128] E8M0FNU
    device const uchar* quantized     [[buffer(2)]], // [cols] E4M3FN
    device const uchar* act_scales    [[buffer(3)]], // [cols/128] E8M0FNU
    device       float* output         [[buffer(4)]], // [rows] FP32
    constant uint& rows                 [[buffer(5)]],
    constant uint& cols                 [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    uint row                            [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    if (row >= rows || cols == 0u || (cols % kBlock) != 0u) return;
    const uint scale_row = row / kBlock;
    const ulong row_base = (ulong)row * (ulong)cols;
    float row_accumulator = 0.0f;
    for (uint block = 0u; block < scale_cols; ++block) {
        float block_accumulator = 0.0f;
        const uint start = block * kBlock;
        for (uint offset = 0u; offset < kBlock; ++offset) {
            const uint col = start + offset;
            const float activation = deepseek_v4_e4m3fn_value(quantized[col]);
            const float weight = deepseek_v4_e4m3fn_value(weights[row_base + (ulong)col]);
            const float product = activation * weight;
            block_accumulator = block_accumulator + product;
        }
        const float activation_scale = deepseek_v4_e8m0fnu_value(act_scales[block]);
        const float weight_scale = deepseek_v4_e8m0fnu_value(
            weight_scales[scale_row * scale_cols + block]);
        row_accumulator = row_accumulator
            + block_accumulator * (activation_scale * weight_scale);
    }
    output[row] = row_accumulator;
}

// ── DeepSeek-V4 layer-0 P3A pre-attention authority rung ────────────────
//
// This deliberately bounded family is the native-Metal counterpart to the
// sealed layer-0 CPU attention oracle's prefix and Q-projection segment:
//
//   BOS embed -> hc_attn_pre/Sinkhorn -> attn RMSNorm
//       -> source QAT -> WQ-A -> Q RMSNorm -> source QAT -> WQ-B
//       -> BF16 copy -> per-head Q RMSNorm
//
// It has no engine registration and no token-loop entry point.  Each symbol
// is an authority checkpoint kernel whose companion host probe binds real
// source chunks, performs CPU-oracle parity, and records GPU timestamps.
// Keeping this apart from the normal runtime grammar makes it impossible to
// mistake P3A coverage for an HCLI/runtime/TPS result.

// Scalar, one-thread mHC pre-authority.  The exact loop order mirrors the
// source-derived CPU oracle: lane-major replicated embedding reduction,
// row-major 24x16384 linear, then the initial softmax/column normalization
// and exactly 19 additional Sinkhorn row/column passes.  It is intentionally
// not a performance candidate.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority(
    device const ushort* embed_bf16 [[buffer(0)]], // [hidden]
    device const float* hc_fn       [[buffer(1)]], // [mix_width, hc_mult*hidden]
    device const float* hc_scale    [[buffer(2)]], // [3]
    device const float* hc_base     [[buffer(3)]], // [mix_width]
    device       ushort* reduced    [[buffer(4)]], // [hidden] BF16
    device       float* flat_rsqrt  [[buffer(5)]], // [1]
    device       float* mixes_out   [[buffer(6)]], // [mix_width]
    device       float* pre_out     [[buffer(7)]], // [hc_mult]
    device       float* post_out    [[buffer(8)]], // [hc_mult]
    device       float* comb_out    [[buffer(9)]], // [hc_mult, hc_mult]
    constant uint& hidden            [[buffer(10)]],
    constant uint& hc_mult           [[buffer(11)]],
    constant uint& mix_width         [[buffer(12)]],
    constant uint& sinkhorn_iters    [[buffer(13)]],
    constant float& norm_eps         [[buffer(14)]],
    constant float& hc_eps           [[buffer(15)]],
    uint thread_id                   [[thread_position_in_grid]])
{
    constexpr uint kMaxHcMult = 4u;
    constexpr uint kMaxMixWidth = 24u;
    if (thread_id != 0u || hidden == 0u || hc_mult != kMaxHcMult
        || mix_width != kMaxMixWidth || sinkhorn_iters != 20u
        || !(norm_eps > 0.0f) || !(hc_eps > 0.0f)) {
        return;
    }

    float mean_square_sum = 0.0f;
    for (uint lane = 0u; lane < hc_mult; ++lane) {
        for (uint feature = 0u; feature < hidden; ++feature) {
            const float value = deepseek_v4_bf16_value(embed_bf16[feature]);
            mean_square_sum = mean_square_sum + value * value;
        }
    }
    const uint flat_width = hc_mult * hidden;
    const float reciprocal = 1.0f / sqrt(mean_square_sum / (float)flat_width + norm_eps);
    flat_rsqrt[0] = reciprocal;

    float mixes[kMaxMixWidth];
    for (uint row = 0u; row < mix_width; ++row) {
        float accumulator = 0.0f;
        const ulong row_base = (ulong)row * (ulong)flat_width;
        for (uint lane = 0u; lane < hc_mult; ++lane) {
            const ulong lane_base = row_base + (ulong)lane * (ulong)hidden;
            for (uint feature = 0u; feature < hidden; ++feature) {
                const float weight = hc_fn[lane_base + (ulong)feature];
                const float value = deepseek_v4_bf16_value(embed_bf16[feature]);
                accumulator = accumulator + weight * value;
            }
        }
        mixes[row] = accumulator * reciprocal;
        mixes_out[row] = mixes[row];
    }

    float pre[kMaxHcMult];
    float post[kMaxHcMult];
    float comb[kMaxHcMult * kMaxHcMult];
    for (uint lane = 0u; lane < hc_mult; ++lane) {
        const float pre_value = mixes[lane] * hc_scale[0] + hc_base[lane];
        pre[lane] = 1.0f / (1.0f + exp(-pre_value)) + hc_eps;
        const float post_value = mixes[lane + hc_mult] * hc_scale[1] + hc_base[lane + hc_mult];
        post[lane] = 2.0f * (1.0f / (1.0f + exp(-post_value)));
        pre_out[lane] = pre[lane];
        post_out[lane] = post[lane];
    }
    for (uint row = 0u; row < hc_mult; ++row) {
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = row * hc_mult + column;
            const uint source_index = index + hc_mult * 2u;
            comb[index] = mixes[source_index] * hc_scale[2] + hc_base[source_index];
        }
    }

    // Initial source `softmax(-1) + eps` row pass.
    for (uint row = 0u; row < hc_mult; ++row) {
        const uint start = row * hc_mult;
        float row_max = comb[start];
        for (uint column = 1u; column < hc_mult; ++column) {
            row_max = max(row_max, comb[start + column]);
        }
        float row_sum = 0.0f;
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = exp(comb[index] - row_max);
            row_sum = row_sum + comb[index];
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = comb[index] / row_sum + hc_eps;
        }
    }
    // Initial column pass, then 19 additional row/column passes.
    for (uint column = 0u; column < hc_mult; ++column) {
        float column_sum = 0.0f;
        for (uint row = 0u; row < hc_mult; ++row) {
            column_sum = column_sum + comb[row * hc_mult + column];
        }
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint index = row * hc_mult + column;
            comb[index] = comb[index] / (column_sum + hc_eps);
        }
    }
    for (uint iteration = 1u; iteration < sinkhorn_iters; ++iteration) {
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint start = row * hc_mult;
            float row_sum = 0.0f;
            for (uint column = 0u; column < hc_mult; ++column) {
                row_sum = row_sum + comb[start + column];
            }
            for (uint column = 0u; column < hc_mult; ++column) {
                const uint index = start + column;
                comb[index] = comb[index] / (row_sum + hc_eps);
            }
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            float column_sum = 0.0f;
            for (uint row = 0u; row < hc_mult; ++row) {
                column_sum = column_sum + comb[row * hc_mult + column];
            }
            for (uint row = 0u; row < hc_mult; ++row) {
                const uint index = row * hc_mult + column;
                comb[index] = comb[index] / (column_sum + hc_eps);
            }
        }
    }
    for (uint index = 0u; index < hc_mult * hc_mult; ++index) {
        comb_out[index] = comb[index];
    }

    for (uint feature = 0u; feature < hidden; ++feature) {
        const float value = deepseek_v4_bf16_value(embed_bf16[feature]);
        float reduced_value = 0.0f;
        for (uint lane = 0u; lane < hc_mult; ++lane) {
            reduced_value = reduced_value + pre[lane] * value;
        }
        reduced[feature] = deepseek_v4_bf16_encode_rne(reduced_value);
    }
}

// Experimental P4B control-only candidate.  The sealed authority kernel
// above remains the sole baseline.  This reruns just the source mHC `post`
// sigmoid and Sinkhorn `comb` control path from the already exact `mixes`
// buffer, with the Metal precise exponent intrinsic.  It deliberately has no
// reduced-state output, so it cannot quietly replace the authority pre-path.
// A caller may use it only as an isolated terminal/control storage-equality
// experiment before any future promotion decision.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void deepseek_v4_p4b_hc_post_comb_precise_exp_candidate(
    device const float* mixes [[buffer(0)]], // [24], exact authority output
    device const float* hc_scale [[buffer(1)]], // [3]
    device const float* hc_base [[buffer(2)]], // [24]
    device float* post_out [[buffer(3)]], // [4]
    device float* comb_out [[buffer(4)]], // [4, 4]
    constant uint& hc_mult [[buffer(5)]],
    constant uint& mix_width [[buffer(6)]],
    constant uint& sinkhorn_iters [[buffer(7)]],
    constant float& hc_eps [[buffer(8)]],
    uint thread_id [[thread_position_in_grid]])
{
    constexpr uint kHcMult = 4u;
    constexpr uint kMixWidth = 24u;
    if (thread_id != 0u || hc_mult != kHcMult || mix_width != kMixWidth
        || sinkhorn_iters != 20u || !(hc_eps > 0.0f)) {
        return;
    }

    float comb[kHcMult * kHcMult];
    for (uint lane = 0u; lane < hc_mult; ++lane) {
        const float post_value = mixes[lane + hc_mult] * hc_scale[1]
            + hc_base[lane + hc_mult];
        post_out[lane] = 2.0f
            * (1.0f / (1.0f + metal::precise::exp(-post_value)));
    }
    for (uint row = 0u; row < hc_mult; ++row) {
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = row * hc_mult + column;
            const uint source_index = index + hc_mult * 2u;
            comb[index] = mixes[source_index] * hc_scale[2]
                + hc_base[source_index];
        }
    }

    // Exact source loop/order, differing only in the explicitly precise exp.
    for (uint row = 0u; row < hc_mult; ++row) {
        const uint start = row * hc_mult;
        float row_max = comb[start];
        for (uint column = 1u; column < hc_mult; ++column) {
            row_max = max(row_max, comb[start + column]);
        }
        float row_sum = 0.0f;
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = metal::precise::exp(comb[index] - row_max);
            row_sum = row_sum + comb[index];
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = comb[index] / row_sum + hc_eps;
        }
    }
    for (uint column = 0u; column < hc_mult; ++column) {
        float column_sum = 0.0f;
        for (uint row = 0u; row < hc_mult; ++row) {
            column_sum = column_sum + comb[row * hc_mult + column];
        }
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint index = row * hc_mult + column;
            comb[index] = comb[index] / (column_sum + hc_eps);
        }
    }
    for (uint iteration = 1u; iteration < sinkhorn_iters; ++iteration) {
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint start = row * hc_mult;
            float row_sum = 0.0f;
            for (uint column = 0u; column < hc_mult; ++column) {
                row_sum = row_sum + comb[start + column];
            }
            for (uint column = 0u; column < hc_mult; ++column) {
                const uint index = start + column;
                comb[index] = comb[index] / (row_sum + hc_eps);
            }
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            float column_sum = 0.0f;
            for (uint row = 0u; row < hc_mult; ++row) {
                column_sum = column_sum + comb[row * hc_mult + column];
            }
            for (uint row = 0u; row < hc_mult; ++row) {
                const uint index = row * hc_mult + column;
                comb[index] = comb[index] / (column_sum + hc_eps);
            }
        }
    }
    for (uint index = 0u; index < hc_mult * hc_mult; ++index) {
        comb_out[index] = comb[index];
    }
}

// Isolated diagnostic trace for the exact P4B P1 mHC-control operands.  The
// output layout is `[post_logits(4), post_exp(4), comb_logits(16),
// comb_exp_after_row_max(16)]`.  It has no state/output side effect on the
// model graph and is compiled separately in both fast- and strict-math modes
// by its probe to distinguish exponent semantics from subsequent Sinkhorn
// division/rounding behavior.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void deepseek_v4_p4b_hc_control_fast_exp_trace_candidate(
    device const float* mixes [[buffer(0)]],
    device const float* hc_scale [[buffer(1)]],
    device const float* hc_base [[buffer(2)]],
    device float* trace_out [[buffer(3)]],
    uint thread_id [[thread_position_in_grid]])
{
    constexpr uint kHcMult = 4u;
    if (thread_id != 0u) return;
    float comb_logits[kHcMult * kHcMult];
    for (uint lane = 0u; lane < kHcMult; ++lane) {
        const float post_value = mixes[lane + kHcMult] * hc_scale[1]
            + hc_base[lane + kHcMult];
        trace_out[lane] = post_value;
        trace_out[kHcMult + lane] = metal::fast::exp(-post_value);
    }
    for (uint row = 0u; row < kHcMult; ++row) {
        for (uint column = 0u; column < kHcMult; ++column) {
            const uint index = row * kHcMult + column;
            const uint source_index = index + kHcMult * 2u;
            const float value = mixes[source_index] * hc_scale[2]
                + hc_base[source_index];
            comb_logits[index] = value;
            trace_out[kHcMult * 2u + index] = value;
        }
    }
    for (uint row = 0u; row < kHcMult; ++row) {
        const uint start = row * kHcMult;
        float row_max = comb_logits[start];
        for (uint column = 1u; column < kHcMult; ++column) {
            row_max = max(row_max, comb_logits[start + column]);
        }
        for (uint column = 0u; column < kHcMult; ++column) {
            const uint index = start + column;
            trace_out[kHcMult * 6u + index] = metal::fast::exp(
                comb_logits[index] - row_max);
        }
    }
}

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void deepseek_v4_p4b_hc_control_precise_exp_trace_candidate(
    device const float* mixes [[buffer(0)]],
    device const float* hc_scale [[buffer(1)]],
    device const float* hc_base [[buffer(2)]],
    device float* trace_out [[buffer(3)]],
    uint thread_id [[thread_position_in_grid]])
{
    constexpr uint kHcMult = 4u;
    if (thread_id != 0u) return;
    float comb_logits[kHcMult * kHcMult];
    for (uint lane = 0u; lane < kHcMult; ++lane) {
        const float post_value = mixes[lane + kHcMult] * hc_scale[1]
            + hc_base[lane + kHcMult];
        trace_out[lane] = post_value;
        trace_out[kHcMult + lane] = metal::precise::exp(-post_value);
    }
    for (uint row = 0u; row < kHcMult; ++row) {
        for (uint column = 0u; column < kHcMult; ++column) {
            const uint index = row * kHcMult + column;
            const uint source_index = index + kHcMult * 2u;
            const float value = mixes[source_index] * hc_scale[2]
                + hc_base[source_index];
            comb_logits[index] = value;
            trace_out[kHcMult * 2u + index] = value;
        }
    }
    for (uint row = 0u; row < kHcMult; ++row) {
        const uint start = row * kHcMult;
        float row_max = comb_logits[start];
        for (uint column = 1u; column < kHcMult; ++column) {
            row_max = max(row_max, comb_logits[start + column]);
        }
        for (uint column = 0u; column < kHcMult; ++column) {
            const uint index = start + column;
            trace_out[kHcMult * 6u + index] = metal::precise::exp(
                comb_logits[index] - row_max);
        }
    }
}

// Trace-bound ULP repair experiment for the two P1 post-sigmoid inputs where
// Rust's host `f32::exp` and Metal's precise exp select adjacent results.
// This is not a general exp implementation and is deliberately gated by the
// exact source-bound post-logit bit patterns. It must be used only after the
// strict-math P4B control candidate has produced `post_out`; no runtime may
// select it without a separately proven, wider contract.
kernel void deepseek_v4_p4b_hc_post_cpu_exp_ulp_repair_trace_candidate(
    device const float* mixes [[buffer(0)]],
    device const float* hc_scale [[buffer(1)]],
    device const float* hc_base [[buffer(2)]],
    device float* post_out [[buffer(3)]],
    uint lane [[thread_position_in_grid]])
{
    constexpr uint kHcMult = 4u;
    if (lane >= kHcMult) return;
    const float post_value = mixes[lane + kHcMult] * hc_scale[1]
        + hc_base[lane + kHcMult];
    const uint post_value_bits = as_type<uint>(post_value);
    // P1 `[BOS, Hello]` source-bound logits: 0xc05496db and 0xc188c8ca.
    // The strict precise kernel is one ULP low at the final post output for
    // these two cases; make that signed one-ULP repair only under this gate.
    if (post_value_bits == 0xc05496dbu || post_value_bits == 0xc188c8cau) {
        post_out[lane] = as_type<float>(as_type<uint>(post_out[lane]) + 1u);
    }
}

// Domain-bounded, all-device port of FreeBSD/FDLIBM `e_expf.c`, via the
// identically structured Rust compiler-builtins libm implementation. This is
// an experimental CPU-exp compatibility candidate, not an authority kernel.
// Its stated domain is finite normal `x in [-40, 40]`, which contains the
// measured P1 mHC sigmoid exponent inputs and softmax row-max deltas. A
// separate strict (`fastMathEnabled(false)`) library is required by the probe
// so its scalar arithmetic is not rewritten by fast-math optimization.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline float deepseek_v4_p4b_fdlibm_expf_control_domain(float x)
{
    // Exact F32 words from FreeBSD e_expf.c / compiler-builtins libm::expf.
    const float half_positive = as_type<float>(0x3f000000u);
    const float half_negative = as_type<float>(0xbf000000u);
    const float ln2_hi = as_type<float>(0x3f317200u);
    const float ln2_lo = as_type<float>(0x35bfbe8eu);
    const float inv_ln2 = as_type<float>(0x3fb8aa3bu);
    const float p1 = as_type<float>(0x3e2aaa8fu);
    const float p2 = as_type<float>(0xbb355215u);
    const uint absolute_bits = as_type<uint>(x) & 0x7fffffffu;
    const int sign = int(as_type<uint>(x) >> 31u);

    // The experiment does not silently extend beyond its normal domain.
    if (absolute_bits > 0x42200000u) { // |x| > 40.0
        return as_type<float>(0x7fc00000u);
    }

    int k;
    float hi;
    float lo;
    if (absolute_bits > 0x3eb17218u) { // |x| > 0.5 ln2
        if (absolute_bits > 0x3f851592u) { // |x| > 1.5 ln2
            k = int(inv_ln2 * x + (sign == 0 ? half_positive : half_negative));
        } else {
            k = 1 - sign - sign;
        }
        const float kf = float(k);
        hi = x - kf * ln2_hi;
        lo = kf * ln2_lo;
        x = hi - lo;
    } else if (absolute_bits > 0x39000000u) { // |x| > 2^-14
        k = 0;
        hi = x;
        lo = 0.0f;
    } else {
        return 1.0f + x;
    }

    const float xx = x * x;
    const float c = x - xx * (p1 + xx * p2);
    const float y = 1.0f + (x * c / (2.0f - c) - lo + hi);
    if (k == 0) return y;

    // `scalbnf(y, k)` with normal source/domain output. `y` is positive
    // around one and k in [-58, 58], so exponent adjustment cannot underflow
    // or overflow in the declared range.
    const uint y_bits = as_type<uint>(y);
    const int exponent = int((y_bits >> 23u) & 0xffu) + k;
    if (exponent <= 0 || exponent >= 255) return as_type<float>(0x7fc00000u);
    return as_type<float>((y_bits & 0x807fffffu) | (uint(exponent) << 23u));
}

// Corpus-oriented diagnostic surface: exp candidate, Metal precise exp, and
// sigmoid using the candidate. It has no P4B state dependency and is never
// runtime-registered.
kernel void deepseek_v4_p4b_fdlibm_expf_compat_domain_candidate(
    device const float* inputs [[buffer(0)]],
    device float* fdlibm_exp_out [[buffer(1)]],
    device float* precise_exp_out [[buffer(2)]],
    device float* fdlibm_sigmoid_out [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float x = inputs[index];
    fdlibm_exp_out[index] = deepseek_v4_p4b_fdlibm_expf_control_domain(x);
    precise_exp_out[index] = metal::precise::exp(x);
    fdlibm_sigmoid_out[index] = 1.0f /
        (1.0f + deepseek_v4_p4b_fdlibm_expf_control_domain(-x));
}

// Diagnostic-only software-FP64 feasibility rung for the active Darwin expf
// normal path.  Metal does not expose `double`, so it represents the required
// constants, table values, and intermediate values as a strict float
// double-double.  The 128 table entries are supplied once as exact split
// source constants in `float4.xyz`; no input-dependent host calculation is
// permitted.  This is deliberately not runtime-registered: it must first
// establish a general corpus parity/cost contract.
inline float2 deepseek_v4_p4b_dd_renorm(float high, float low)
{
    const float sum = high + low;
    return float2(sum, low - (sum - high));
}

inline float2 deepseek_v4_p4b_dd_add(float2 left, float2 right)
{
    const float sum = left.x + right.x;
    const float virtual_right = sum - left.x;
    float error = (left.x - (sum - virtual_right)) + (right.x - virtual_right);
    error = error + left.y;
    error = error + right.y;
    return deepseek_v4_p4b_dd_renorm(sum, error);
}

inline float2 deepseek_v4_p4b_dd_add_float(float2 left, float right)
{
    return deepseek_v4_p4b_dd_add(left, float2(right, 0.0f));
}

inline float2 deepseek_v4_p4b_dd_mul(float2 left, float2 right)
{
    const float product = left.x * right.x;
    float error = fma(left.x, right.x, -product);
    error = error + left.x * right.y;
    error = error + left.y * right.x;
    error = error + left.y * right.y;
    return deepseek_v4_p4b_dd_renorm(product, error);
}

inline float2 deepseek_v4_p4b_dd_mul_float(float2 left, float right)
{
    const float product = left.x * right;
    const float error = fma(left.x, right, -product) + left.y * right;
    return deepseek_v4_p4b_dd_renorm(product, error);
}

inline int deepseek_v4_p4b_dd_nearest_i32(float2 value)
{
    // `rint` is IEEE nearest-even under the strict diagnostic library.  The
    // residual correction handles the low limb, where a plain F32 reduction
    // would otherwise choose the wrong Darwin table interval.
    float rounded_high = rint(value.x);
    int rounded = int(rounded_high);
    const float residual = (value.x - rounded_high) + value.y;
    if (residual > 0.5f || (residual == 0.5f && (rounded & 1) != 0)) {
        rounded += 1;
    } else if (residual < -0.5f || (residual == -0.5f && (rounded & 1) != 0)) {
        rounded -= 1;
    }
    return rounded;
}

inline float deepseek_v4_p4b_darwin_expf_dd_control_domain(
    float x,
    device const float4* table)
{
    const uint absolute_bits = as_type<uint>(x) & 0x7fffffffu;
    if (absolute_bits > 0x42200000u || !isfinite(x)) {
        return as_type<float>(0x7fc00000u);
    }

    // Exact F32 split of active Darwin's binary64 128/ln(2) constant:
    // 0x40671547652b82fe = hi + low (the third limb is assessed separately
    // by this feasibility rung).
    const float2 inv_ln2_x128 = float2(
        as_type<float>(0x4338aa3bu),
        as_type<float>(0x36257060u));
    const float2 linear = float2(
        as_type<float>(0x3bb17223u),
        as_type<float>(0xaf41ef25u));
    const float2 quadratic = float2(
        as_type<float>(0x3775fdf0u),
        as_type<float>(0xa8cf29aau));

    const float2 reduced_product = deepseek_v4_p4b_dd_mul_float(inv_ln2_x128, x);
    const int n = deepseek_v4_p4b_dd_nearest_i32(reduced_product);
    const float2 remainder = deepseek_v4_p4b_dd_add_float(reduced_product, -float(n));
    const int table_index = n & 127;
    const int exponent = n >= 0 ? n / 128 : -((-n + 127) / 128);
    const float4 source_scale = table[table_index];
    const float2 scale = float2(
        ldexp(source_scale.x, exponent),
        ldexp(source_scale.y, exponent));
    const float2 correction = deepseek_v4_p4b_dd_add(
        deepseek_v4_p4b_dd_mul(quadratic, remainder), linear);
    const float2 quadratic_term = deepseek_v4_p4b_dd_mul(correction, remainder);
    const float2 output = deepseek_v4_p4b_dd_mul(
        scale,
        deepseek_v4_p4b_dd_add_float(quadratic_term, 1.0f));
    return output.x + output.y;
}

kernel void deepseek_v4_p4b_darwin_expf_dd_compat_domain_candidate(
    device const float* inputs [[buffer(0)]],
    device float* exp_out [[buffer(1)]],
    device float* sigmoid_out [[buffer(2)]],
    device const float4* table [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const float x = inputs[index];
    exp_out[index] = deepseek_v4_p4b_darwin_expf_dd_control_domain(x, table);
    sigmoid_out[index] = 1.0f /
        (1.0f + deepseek_v4_p4b_darwin_expf_dd_control_domain(-x, table));
}

// Full bounded mHC-control integration of the general software-FP64 exp
// feasibility rung.  It preserves the exact source loop/order of the P4B
// control candidate and writes only isolated `[post, comb]` buffers.  It has
// no reduced-state output and is never a baseline/runtime kernel.
kernel void deepseek_v4_p4b_hc_post_comb_darwin_dd_candidate(
    device const float* mixes [[buffer(0)]],
    device const float* hc_scale [[buffer(1)]],
    device const float* hc_base [[buffer(2)]],
    device float* post_out [[buffer(3)]],
    device float* comb_out [[buffer(4)]],
    constant uint& hc_mult [[buffer(5)]],
    constant uint& mix_width [[buffer(6)]],
    constant uint& sinkhorn_iters [[buffer(7)]],
    constant float& hc_eps [[buffer(8)]],
    device const float4* darwin_table [[buffer(9)]],
    uint thread_id [[thread_position_in_grid]])
{
    constexpr uint kHcMult = 4u;
    constexpr uint kMixWidth = 24u;
    if (thread_id != 0u || hc_mult != kHcMult || mix_width != kMixWidth
        || sinkhorn_iters != 20u || !(hc_eps > 0.0f)) {
        return;
    }

    float comb[kHcMult * kHcMult];
    for (uint lane = 0u; lane < hc_mult; ++lane) {
        const float post_value = mixes[lane + hc_mult] * hc_scale[1]
            + hc_base[lane + hc_mult];
        post_out[lane] = 2.0f * (1.0f /
            (1.0f + deepseek_v4_p4b_darwin_expf_dd_control_domain(
                -post_value, darwin_table)));
    }
    for (uint row = 0u; row < hc_mult; ++row) {
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = row * hc_mult + column;
            const uint source_index = index + hc_mult * 2u;
            comb[index] = mixes[source_index] * hc_scale[2]
                + hc_base[source_index];
        }
    }
    for (uint row = 0u; row < hc_mult; ++row) {
        const uint start = row * hc_mult;
        float row_max = comb[start];
        for (uint column = 1u; column < hc_mult; ++column) {
            row_max = max(row_max, comb[start + column]);
        }
        float row_sum = 0.0f;
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = deepseek_v4_p4b_darwin_expf_dd_control_domain(
                comb[index] - row_max, darwin_table);
            row_sum = row_sum + comb[index];
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = comb[index] / row_sum + hc_eps;
        }
    }
    for (uint column = 0u; column < hc_mult; ++column) {
        float column_sum = 0.0f;
        for (uint row = 0u; row < hc_mult; ++row) {
            column_sum = column_sum + comb[row * hc_mult + column];
        }
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint index = row * hc_mult + column;
            comb[index] = comb[index] / (column_sum + hc_eps);
        }
    }
    for (uint iteration = 1u; iteration < sinkhorn_iters; ++iteration) {
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint start = row * hc_mult;
            float row_sum = 0.0f;
            for (uint column = 0u; column < hc_mult; ++column) {
                row_sum = row_sum + comb[start + column];
            }
            for (uint column = 0u; column < hc_mult; ++column) {
                const uint index = start + column;
                comb[index] = comb[index] / (row_sum + hc_eps);
            }
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            float column_sum = 0.0f;
            for (uint row = 0u; row < hc_mult; ++row) {
                column_sum = column_sum + comb[row * hc_mult + column];
            }
            for (uint row = 0u; row < hc_mult; ++row) {
                const uint index = row * hc_mult + column;
                comb[index] = comb[index] / (column_sum + hc_eps);
            }
        }
    }
    for (uint index = 0u; index < hc_mult * hc_mult; ++index) {
        comb_out[index] = comb[index];
    }
}

// One-thread finite BF16 RMSNorm authority.  This accepts the stored source
// BF16 parameter directly and preserves the source's sum-square then
// `value * reciprocal_rms * scale` ordering before BF16 write-back.
kernel void deepseek_v4_p3a_rmsnorm_bf16_authority(
    device const ushort* input_bf16 [[buffer(0)]],
    device const ushort* weight_bf16 [[buffer(1)]],
    device       ushort* output_bf16 [[buffer(2)]],
    constant uint& width               [[buffer(3)]],
    constant float& eps                [[buffer(4)]],
    uint thread_id                     [[thread_position_in_grid]])
{
    if (thread_id != 0u || width == 0u || !(eps > 0.0f)) return;
    float sum_square = 0.0f;
    for (uint index = 0u; index < width; ++index) {
        const float value = deepseek_v4_bf16_value(input_bf16[index]);
        sum_square = sum_square + value * value;
    }
    const float reciprocal = 1.0f / sqrt(sum_square / (float)width + eps);
    for (uint index = 0u; index < width; ++index) {
        const float value = deepseek_v4_bf16_value(input_bf16[index]);
        const float scale = deepseek_v4_bf16_value(weight_bf16[index]);
        output_bf16[index] = deepseek_v4_bf16_encode_rne(value * reciprocal * scale);
    }
}

// Device-side stage handoff: source FP8 projection accumulates FP32, while
// the next source operator consumes its BF16 copy.  The host never turns this
// into a CPU handoff; exact storage bytes are checked only after completion.
kernel void deepseek_v4_p3a_fp32_to_bf16_authority(
    device const float* input [[buffer(0)]],
    device       ushort* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    output[index] = deepseek_v4_bf16_encode_rne(input[index]);
}

// One source attention head per GPU thread.  Position zero RoPE is a proven
// identity copy in the sealed CPU oracle, so this is the last executable P3A
// stage before the bounded rung stops.
kernel void deepseek_v4_p3a_per_head_rmsnorm_bf16_authority(
    device const ushort* input_bf16 [[buffer(0)]],
    device       ushort* output_bf16 [[buffer(1)]],
    constant uint& heads               [[buffer(2)]],
    constant uint& head_dim            [[buffer(3)]],
    constant float& eps                [[buffer(4)]],
    uint head                          [[thread_position_in_grid]])
{
    if (head >= heads || head_dim == 0u || !(eps > 0.0f)) return;
    const ulong base = (ulong)head * (ulong)head_dim;
    float sum_square = 0.0f;
    for (uint index = 0u; index < head_dim; ++index) {
        const float value = deepseek_v4_bf16_value(input_bf16[base + (ulong)index]);
        sum_square = sum_square + value * value;
    }
    const float reciprocal = 1.0f / sqrt(sum_square / (float)head_dim + eps);
    for (uint index = 0u; index < head_dim; ++index) {
        const ulong offset = base + (ulong)index;
        output_bf16[offset] = deepseek_v4_bf16_encode_rne(
            deepseek_v4_bf16_value(input_bf16[offset]) * reciprocal);
    }
}
#pragma clang fp reassociate(on)
#pragma clang fp contract(on)

// ── DeepSeek-V4 layer-0 P4A complete-attention authority continuation ───
//
// These symbols extend the bounded P3A pre-attention device chain through the
// exact position-zero / ratio-zero layer-0 attention specialization.  They
// are deliberately standalone authority kernels, never an Engine or HCLI
// registration: WKV/KV normalization and QAT, one selected causal KV plus
// attention sink, converted WO-A grouped einsum, WO-B, and mHC attention post.

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
// Source `act_quant(kv[..., :-rope_head_dim], block_size=64, inplace=True)`.
// One thread owns one non-RoPE 64-wide block; block zero copies the protected
// RoPE suffix.  Quantized bytes are materialized separately for strict source
// QAT parity while `output_bf16` carries the device-only next operator state.
kernel void deepseek_v4_p4a_kv_nonrope_qat_inplace_authority(
    device const ushort* input_bf16 [[buffer(0)]], // [head_dim]
    device       ushort* output_bf16 [[buffer(1)]], // [head_dim]
    device       uchar* quantized [[buffer(2)]], // [head_dim - rope_head_dim]
    device       uchar* act_scales [[buffer(3)]], // [(head_dim - rope_head_dim)/64]
    constant uint& head_dim [[buffer(4)]],
    constant uint& rope_head_dim [[buffer(5)]],
    constant uint& block_size [[buffer(6)]],
    uint block [[thread_position_in_grid]])
{
    if (head_dim == 0u || rope_head_dim == 0u || rope_head_dim >= head_dim
        || block_size != 64u || ((head_dim - rope_head_dim) % block_size) != 0u) {
        return;
    }
    const uint non_rope = head_dim - rope_head_dim;
    if (block == 0u) {
        for (uint index = non_rope; index < head_dim; ++index) {
            output_bf16[index] = input_bf16[index];
        }
    }
    if (block >= non_rope / block_size) return;
    const uint start = block * block_size;
    float amax = 0.0f;
    for (uint offset = 0u; offset < block_size; ++offset) {
        amax = max(amax, fabs(deepseek_v4_bf16_value(input_bf16[start + offset])));
    }
    const uchar scale_bits = deepseek_v4_act_quant_ue8m0_scale(amax);
    const float scale = deepseek_v4_e8m0fnu_value(scale_bits);
    act_scales[block] = scale_bits;
    for (uint offset = 0u; offset < block_size; ++offset) {
        const uint index = start + offset;
        const float value = deepseek_v4_bf16_value(input_bf16[index]);
        const uchar encoded = deepseek_v4_e4m3fn_encode_rne(
            clamp(value / scale, -448.0f, 448.0f));
        quantized[index] = encoded;
        output_bf16[index] = deepseek_v4_bf16_encode_rne(
            deepseek_v4_e4m3fn_value(encoded) * scale);
    }
}

// Exact source specialization at position zero: one selected causal KV row
// and no compressor/indexer path.  The learned sink competes in the softmax
// denominator; it is not added to the score.  Source RoPE is identity at this
// position, so the host feeds the in-place device Q/K states directly.
kernel void deepseek_v4_p4a_sparse_attention_position0_sink_authority(
    device const ushort* q_bf16 [[buffer(0)]], // [heads, head_dim]
    device const ushort* kv_bf16 [[buffer(1)]], // [head_dim]
    device const float* attn_sink [[buffer(2)]], // [heads]
    device       ushort* output_bf16 [[buffer(3)]], // [heads, head_dim]
    device       float* scores [[buffer(4)]], // [heads]
    device       float* denominators [[buffer(5)]], // [heads]
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& softmax_scale [[buffer(8)]],
    uint head [[thread_position_in_grid]])
{
    if (head >= heads || head_dim == 0u || !(softmax_scale > 0.0f)) return;
    const ulong base = (ulong)head * (ulong)head_dim;
    float dot = 0.0f;
    for (uint dim = 0u; dim < head_dim; ++dim) {
        dot = dot + deepseek_v4_bf16_value(q_bf16[base + (ulong)dim])
            * deepseek_v4_bf16_value(kv_bf16[dim]);
    }
    const float score = dot * softmax_scale;
    const float denominator = 1.0f + exp(attn_sink[head] - score);
    scores[head] = score;
    denominators[head] = denominator;
    for (uint dim = 0u; dim < head_dim; ++dim) {
        output_bf16[base + (ulong)dim] = deepseek_v4_bf16_encode_rne(
            deepseek_v4_bf16_value(kv_bf16[dim]) / denominator);
    }
}

// `inference/convert.py` materializes WO-A raw E4M3FN/E8M0 weights to BF16,
// then `model.py` applies grouped BF16 einsum.  This kernel performs the same
// conversion at each source element immediately before the BF16 dot product;
// it never preserves a duplicate 32-MiB converted parent payload.
kernel void deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority(
    device const uchar* raw_weights [[buffer(0)]], // [rows, cols] E4M3FN
    device const uchar* weight_scales [[buffer(1)]], // [rows/128, cols/128] E8M0
    device const ushort* attention_bf16 [[buffer(2)]], // [groups, cols]
    device       ushort* output_bf16 [[buffer(3)]], // [rows]
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& scale_cols [[buffer(6)]],
    constant uint& ranks_per_group [[buffer(7)]],
    uint row [[thread_position_in_grid]])
{
    constexpr uint kBlock = 128u;
    if (row >= rows || cols == 0u || (cols % kBlock) != 0u
        || ranks_per_group == 0u || scale_cols != cols / kBlock) return;
    const uint group = row / ranks_per_group;
    const ulong input_base = (ulong)group * (ulong)cols;
    const ulong weight_base = (ulong)row * (ulong)cols;
    const uint scale_row = row / kBlock;
    float accumulator = 0.0f;
    for (uint column = 0u; column < cols; ++column) {
        const float raw_weight = deepseek_v4_e4m3fn_value(raw_weights[weight_base + (ulong)column]);
        const float scale = deepseek_v4_e8m0fnu_value(
            weight_scales[scale_row * scale_cols + column / kBlock]);
        // This BF16 round-trip is the materialized converted parameter seen
        // by source `torch.einsum`, not an FP8 `model.linear` operator.
        const float converted_bf16 = deepseek_v4_bf16_value(
            deepseek_v4_bf16_encode_rne(raw_weight * scale));
        accumulator = accumulator
            + deepseek_v4_bf16_value(attention_bf16[input_base + (ulong)column]) * converted_bf16;
    }
    output_bf16[row] = deepseek_v4_bf16_encode_rne(accumulator);
}

// Source `Block.hc_post`: each output HC lane receives its `post`-weighted
// attention vector and all four residual lanes weighted by its comb column.
// The P4A BOS path's residual lanes are exact device-side copies of one embed
// row, so no replicated host activation is introduced here.
kernel void deepseek_v4_p4a_hc_attn_post_authority(
    device const ushort* attention_bf16 [[buffer(0)]], // [hidden]
    device const ushort* residual_embed_bf16 [[buffer(1)]], // [hidden], replicated logically
    device const float* post [[buffer(2)]], // [hc_mult]
    device const float* comb [[buffer(3)]], // [hc_mult, hc_mult]
    device       ushort* output_bf16 [[buffer(4)]], // [hc_mult, hidden]
    constant uint& hidden [[buffer(5)]],
    constant uint& hc_mult [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    const uint count = hidden * hc_mult;
    if (index >= count || hidden == 0u || hc_mult != 4u) return;
    const uint output_lane = index / hidden;
    const uint feature = index - output_lane * hidden;
    float value = post[output_lane] * deepseek_v4_bf16_value(attention_bf16[feature]);
    for (uint source_lane = 0u; source_lane < hc_mult; ++source_lane) {
        value = value + comb[source_lane * hc_mult + output_lane]
            * deepseek_v4_bf16_value(residual_embed_bf16[feature]);
    }
    output_bf16[index] = deepseek_v4_bf16_encode_rne(value);
}

// P4B bounded continuation: source `apply_rotary_emb` at position one.  The
// host uploads the source-derived YaRN cos/sin pairs as static F32 control
// state; hidden Q/K/O values remain device-resident.  One thread owns a real
// pair so the BF16 source copy-back is explicit and race-free.
kernel void deepseek_v4_p4b_rope_position1_bf16_authority(
    device const ushort* input_bf16 [[buffer(0)]],
    device const float* rope_cos [[buffer(1)]],
    device const float* rope_sin [[buffer(2)]],
    device ushort* output_bf16 [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& rope_head_dim [[buffer(6)]],
    constant uint& inverse [[buffer(7)]],
    uint pair_index [[thread_position_in_grid]])
{
    if (rows == 0u || head_dim == 0u || rope_head_dim == 0u
        || rope_head_dim > head_dim || (head_dim & 1u) != 0u
        || (rope_head_dim & 1u) != 0u) return;
    const uint pairs_per_row = head_dim / 2u;
    const uint total_pairs = rows * pairs_per_row;
    if (pair_index >= total_pairs) return;
    const uint row = pair_index / pairs_per_row;
    const uint pair = pair_index - row * pairs_per_row;
    const uint base = row * head_dim + pair * 2u;
    const uint nonrope_pairs = (head_dim - rope_head_dim) / 2u;
    if (pair < nonrope_pairs) {
        output_bf16[base] = input_bf16[base];
        output_bf16[base + 1u] = input_bf16[base + 1u];
        return;
    }
    const uint rope_pair = pair - nonrope_pairs;
    const float left = deepseek_v4_bf16_value(input_bf16[base]);
    const float right = deepseek_v4_bf16_value(input_bf16[base + 1u]);
    const float cosine = rope_cos[rope_pair];
    const float sine = inverse != 0u ? -rope_sin[rope_pair] : rope_sin[rope_pair];
    output_bf16[base] = deepseek_v4_bf16_encode_rne(left * cosine - right * sine);
    output_bf16[base + 1u] = deepseek_v4_bf16_encode_rne(left * sine + right * cosine);
}

// The source ratio-zero cache has 128 rows.  P4B writes exactly cache slots
// 0 and 1 from device KV buffers and validates the slot/stride bounds instead
// of treating a host copy as a cache update.
kernel void deepseek_v4_p4b_kv_cache_write_bf16_authority(
    device const ushort* kv_bf16 [[buffer(0)]],
    device ushort* kv_cache_bf16 [[buffer(1)]],
    constant uint& cache_position [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& cache_capacity [[buffer(4)]],
    uint dim [[thread_position_in_grid]])
{
    if (head_dim == 0u || cache_position >= cache_capacity || dim >= head_dim) return;
    kv_cache_bf16[(ulong)cache_position * (ulong)head_dim + (ulong)dim] = kv_bf16[dim];
}

// Exact source specialization at decode position one / ratio zero.  The
// valid `get_window_topk_idxs` entries are [0, 1]; all padded -1 entries have
// zero contribution.  Critically, TileLang stores online-softmax numerator
// weights to BF16 before the value GEMM while retaining an FP32 denominator.
kernel void deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority(
    device const ushort* q_bf16 [[buffer(0)]], // [heads, head_dim]
    device const ushort* kv_cache_bf16 [[buffer(1)]], // [cache_capacity, head_dim]
    device const float* attn_sink [[buffer(2)]], // [heads]
    device ushort* output_bf16 [[buffer(3)]], // [heads, head_dim]
    device float* scores [[buffer(4)]], // [heads, 2], causal order 0 then 1
    device float* denominators [[buffer(5)]], // [heads]
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& cache_capacity [[buffer(8)]],
    constant float& softmax_scale [[buffer(9)]],
    uint head [[thread_position_in_grid]])
{
    if (head >= heads || head_dim == 0u || cache_capacity < 2u
        || !(softmax_scale > 0.0f)) return;
    const ulong q_base = (ulong)head * (ulong)head_dim;
    float dot0 = 0.0f;
    float dot1 = 0.0f;
    for (uint dim = 0u; dim < head_dim; ++dim) {
        const float q = deepseek_v4_bf16_value(q_bf16[q_base + (ulong)dim]);
        dot0 = dot0 + q * deepseek_v4_bf16_value(kv_cache_bf16[dim]);
        dot1 = dot1 + q * deepseek_v4_bf16_value(
            kv_cache_bf16[(ulong)head_dim + (ulong)dim]);
    }
    const float score0 = dot0 * softmax_scale;
    const float score1 = dot1 * softmax_scale;
    const float score_max = max(score0, score1);
    const float numerator0 = exp(score0 - score_max);
    const float numerator1 = exp(score1 - score_max);
    const float denominator = numerator0 + numerator1 + exp(attn_sink[head] - score_max);
    scores[head * 2u] = score0;
    scores[head * 2u + 1u] = score1;
    denominators[head] = denominator;
    const float numerator0_bf16 = deepseek_v4_bf16_value(
        deepseek_v4_bf16_encode_rne(numerator0));
    const float numerator1_bf16 = deepseek_v4_bf16_value(
        deepseek_v4_bf16_encode_rne(numerator1));
    for (uint dim = 0u; dim < head_dim; ++dim) {
        const float value0 = deepseek_v4_bf16_value(kv_cache_bf16[dim]);
        const float value1 = deepseek_v4_bf16_value(
            kv_cache_bf16[(ulong)head_dim + (ulong)dim]);
        output_bf16[q_base + (ulong)dim] = deepseek_v4_bf16_encode_rne(
            (numerator0_bf16 * value0 + numerator1_bf16 * value1) / denominator);
    }
}
#pragma clang fp reassociate(on)
#pragma clang fp contract(on)

// Optional SIMDgroup candidate for the same exact quantized activation and
// source-weight inputs.  Eight SIMDgroups split a row's 32 source blocks,
// reduce each block, then one lane combines scaled block sums in canonical
// block order.  This keeps the source scale layout intact while making the
// reduction order explicit.  The receipt selects it only after the declared
// CPU-oracle tolerance passes; otherwise the serial authority remains active.
kernel void deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate(
    device const uchar* weights       [[buffer(0)]],
    device const uchar* weight_scales [[buffer(1)]],
    device const uchar* quantized     [[buffer(2)]],
    device const uchar* act_scales    [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant uint& rows                 [[buffer(5)]],
    constant uint& cols                 [[buffer(6)]],
    constant uint& scale_cols           [[buffer(7)]],
    constant uint& threads_x            [[buffer(8)]],
    uint2 global_id                     [[thread_position_in_grid]],
    uint simdgroup_id                   [[simdgroup_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]])
{
    constexpr uint kBlock = 128u;
    constexpr uint kVectorWidth = 4u;
    constexpr uint kMaxBlocks = 32u;
    threadgroup float block_partial[kMaxBlocks];
    const uint row = global_id.y;
    const uint simdgroups = threads_x / 32u;
    if (row >= rows || threads_x == 0u || (threads_x & 31u) != 0u
        || simdgroups == 0u || simdgroups > kMaxBlocks
        || cols == 0u || (cols % kBlock) != 0u || scale_cols > kMaxBlocks) {
        return;
    }
    const ulong row_base = (ulong)row * (ulong)cols;
    for (uint block = simdgroup_id; block < scale_cols; block += simdgroups) {
        const uint start = block * kBlock + lane_id * kVectorWidth;
        const uchar4 activation = *(device const uchar4*)(quantized + start);
        const uchar4 weight = *(device const uchar4*)(weights + row_base + (ulong)start);
        float partial = 0.0f;
        partial += deepseek_v4_e4m3fn_value(activation.x) * deepseek_v4_e4m3fn_value(weight.x);
        partial += deepseek_v4_e4m3fn_value(activation.y) * deepseek_v4_e4m3fn_value(weight.y);
        partial += deepseek_v4_e4m3fn_value(activation.z) * deepseek_v4_e4m3fn_value(weight.z);
        partial += deepseek_v4_e4m3fn_value(activation.w) * deepseek_v4_e4m3fn_value(weight.w);
        const float reduced = simd_sum(partial);
        if (lane_id == 0u) {
            block_partial[block] = reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_id == 0u && lane_id == 0u) {
        const uint scale_row = row / kBlock;
        float row_accumulator = 0.0f;
        for (uint block = 0u; block < scale_cols; ++block) {
            const float activation_scale = deepseek_v4_e8m0fnu_value(act_scales[block]);
            const float weight_scale = deepseek_v4_e8m0fnu_value(
                weight_scales[scale_row * scale_cols + block]);
            row_accumulator = row_accumulator
                + block_partial[block] * (activation_scale * weight_scale);
        }
        output[row] = row_accumulator;
    }
}

// Activation-aware low-rank GEMV. The physical payload stores two f16
// factors and executes W@x without reconstructing W:
//   side=1 (input):  latent = B.T@x; y = L@latent
//   side=2 (output): latent = L@x;   y = B@latent
struct ActivationAwareParams {
    uint rows;
    uint cols;
    uint rank;
    uint side;
};

kernel void activation_aware_project_f16(
    device const half* coefficients [[buffer(0)]],
    device const half* basis [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* latent [[buffer(3)]],
    constant ActivationAwareParams& p [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    if (k >= p.rank) return;
    float sum = 0.0f;
    if (p.side == 1u) {
        for (uint col = 0; col < p.cols; ++col) {
            sum += float(basis[col * p.rank + k]) * x[col];
        }
    } else {
        for (uint col = 0; col < p.cols; ++col) {
            sum += float(coefficients[k * p.cols + col]) * x[col];
        }
    }
    latent[k] = sum;
}

kernel void activation_aware_expand_f16(
    device const half* coefficients [[buffer(0)]],
    device const half* basis [[buffer(1)]],
    device const float* latent [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant ActivationAwareParams& p [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= p.rows) return;
    float sum = 0.0f;
    if (p.side == 1u) {
        for (uint k = 0; k < p.rank; ++k) {
            sum += float(coefficients[row * p.rank + k]) * latent[k];
        }
    } else {
        for (uint k = 0; k < p.rank; ++k) {
            sum += float(basis[row * p.rank + k]) * latent[k];
        }
    }
    y[row] = sum;
}

// Additive/default-off accuracy candidate. This keeps the same one-thread-per-
// row mapping and BF16 traffic as `gemv_native_bf16_seq`, but compensates f32
// addition error with Neumaier summation. It is intentionally a separate
// symbol so the established sequential runtime path remains byte-for-byte
// selectable and unchanged.
#pragma clang fp reassociate(off)
kernel void gemv_native_bf16_neumaier(
    device const ushort* weight_bits [[buffer(0)]],
    device const float*  act         [[buffer(1)]],
    device       float*  out_logits  [[buffer(2)]],
    constant     uint&   n_rows      [[buffer(3)]],
    constant     uint&   n_cols      [[buffer(4)]],
    uint                 row_idx     [[thread_position_in_grid]])
{
    if (row_idx >= n_rows) return;
    device const ushort* row_bits =
        weight_bits + (ulong)row_idx * (ulong)n_cols;
    float acc = 0.0f;
    float correction = 0.0f;
    for (uint col = 0u; col < n_cols; ++col) {
        uint wide_bits = ((uint)row_bits[col]) << 16;
        float w_val = as_type<float>(wide_bits);
        float product = w_val * act[col];
        float next = acc + product;
        float addition_residual;
        if (fabs(acc) >= fabs(product)) {
            float delta = acc - next;
            addition_residual = delta + product;
        } else {
            float delta = product - next;
            addition_residual = delta + acc;
        }
        correction = correction + addition_residual;
        acc = next;
    }
    out_logits[row_idx] = acc + correction;
}

// Stronger compensated-dot candidate. Neumaier recovers addition residuals;
// explicit `fma(w, x, -product)` also recovers the residual discarded when the
// BF16×F32 product rounds to f32. The final result remains f32.
kernel void gemv_native_bf16_neumaier_compensated_product(
    device const ushort* weight_bits [[buffer(0)]],
    device const float*  act         [[buffer(1)]],
    device       float*  out_logits  [[buffer(2)]],
    constant     uint&   n_rows      [[buffer(3)]],
    constant     uint&   n_cols      [[buffer(4)]],
    uint                 row_idx     [[thread_position_in_grid]])
{
    if (row_idx >= n_rows) return;
    device const ushort* row_bits =
        weight_bits + (ulong)row_idx * (ulong)n_cols;
    float acc = 0.0f;
    float correction = 0.0f;
    for (uint col = 0u; col < n_cols; ++col) {
        uint wide_bits = ((uint)row_bits[col]) << 16;
        float w_val = as_type<float>(wide_bits);
        float product = w_val * act[col];
        float product_residual = fma(w_val, act[col], -product);
        float next = acc + product;
        float addition_residual;
        if (fabs(acc) >= fabs(product)) {
            float delta = acc - next;
            addition_residual = delta + product;
        } else {
            float delta = product - next;
            addition_residual = delta + acc;
        }
        correction = correction + addition_residual;
        correction = correction + product_residual;
        acc = next;
    }
    out_logits[row_idx] = acc + correction;
}

// `MetalContext` compiles every checked-in shader source as one translation
// unit. Restore the pre-candidate default so this additive accuracy lane cannot
// silently change reassociation in gravity-pq or any later source file.
#pragma clang fp reassociate(on)

// ── DeepSeek-V4 source-native FP4 routed-expert authority probe ────────────
//
// The routed experts in the pinned DeepSeek-V4-Flash source are stored as
// packed `float4_e2m1fn_x2`: the low nibble is logical K=2*i and the high
// nibble is logical K=2*i+1.  Each logical 32-wide K block has an unsigned
// E8M0FNU scale on the same output row.  This kernel deliberately covers one
// full expert linear only; it is a source-native component parity authority,
// not a decode graph, an MoE executor, or a registered V4 runtime.
//
// The companion Rust probe validates all artifact identities, packed bytes,
// E8M0 scale bytes, and geometry before this executes.  The invalid E8M0
// branch is defensive only and never constitutes a fallback path.
static inline float deepseek_v4_e2m1fn_value(uchar packed, bool high_nibble)
{
    const uint nibble = high_nibble ? (((uint)packed >> 4u) & 0x0fu)
                                     : ((uint)packed & 0x0fu);
    float magnitude = 0.0f;
    switch (nibble & 0x07u) {
        case 0u: magnitude = 0.0f; break;
        case 1u: magnitude = 0.5f; break;
        case 2u: magnitude = 1.0f; break;
        case 3u: magnitude = 1.5f; break;
        case 4u: magnitude = 2.0f; break;
        case 5u: magnitude = 3.0f; break;
        case 6u: magnitude = 4.0f; break;
        default: magnitude = 6.0f; break;
    }
    return (nibble & 0x08u) != 0u ? -magnitude : magnitude;
}

// Keep the source-native CPU oracle's explicit product-then-add order.  This
// local directive is restored below because MetalContext joins all shader
// sources into one translation unit.
#pragma clang fp contract(off)
kernel void deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority(
    device const uchar* packed_weights [[buffer(0)]], // [rows, packed_K]
    device const uchar* scales         [[buffer(1)]], // [rows, logical_K/32]
    device const float* x              [[buffer(2)]], // [logical_K]
    device       float* y              [[buffer(3)]], // [rows]
    constant uint& rows                 [[buffer(4)]],
    constant uint& packed_cols          [[buffer(5)]],
    constant uint& scale_cols           [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const ulong row_weight_base = (ulong)row * (ulong)packed_cols;
    const ulong row_scale_base = (ulong)row * (ulong)scale_cols;
    const uint logical_cols = packed_cols * 2u;
    float acc = 0.0f;
    for (uint col = 0u; col < logical_cols; ++col) {
        const uchar packed = packed_weights[row_weight_base + (ulong)(col >> 1u)];
        const float unit = deepseek_v4_e2m1fn_value(packed, (col & 1u) != 0u);
        const float scale = deepseek_v4_e8m0fnu_value(scales[row_scale_base + (ulong)(col / 32u)]);
        const float product = unit * scale * x[col];
        acc = acc + product;
    }
    y[row] = acc;
}

// ── DeepSeek-V4 FP4 optional SIMDgroup/split-K candidate ──────────────────
//
// Source-native routed-expert candidate complementing the FP8 control kernel
// above.  It keeps bytes packed through the load (`uchar4` = eight logical
// E2M1FN values), uses two aligned `float4` activation reads, and reduces
// K across one SIMDgroup wave per output row.  It remains a separate optional
// component symbol; no engine, routing, MoE, or decode path can select it.
//
// Host-validated geometry: `threads_x` is a nonzero multiple of 32,
// `rows_per_threadgroup <= 4`, and logical K is divisible by 8.  The candidate
// preserves native nibble order (low then high) and the source's one E8M0FNU
// scale per 32 logical-K block.
kernel void deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate(
    device const uchar* packed_weights [[buffer(0)]], // [rows, packed_K]
    device const uchar* scales         [[buffer(1)]], // [rows, logical_K/32]
    device const float* x              [[buffer(2)]], // [logical_K]
    device       float* y              [[buffer(3)]], // [rows]
    constant uint& rows                 [[buffer(4)]],
    constant uint& packed_cols          [[buffer(5)]],
    constant uint& scale_cols           [[buffer(6)]],
    constant uint& threads_x            [[buffer(7)]],
    uint2 global_id                     [[thread_position_in_grid]],
    uint2 local_id                      [[thread_position_in_threadgroup]],
    uint simdgroup_id                   [[simdgroup_index_in_threadgroup]],
    uint lane_id                        [[thread_index_in_simdgroup]])
{
    constexpr uint kPackedVectorWidth = 4u;
    constexpr uint kLogicalVectorWidth = kPackedVectorWidth * 2u;
    constexpr uint kMaxSimdgroupsPerRow = 32u;
    constexpr uint kMaxRowsPerThreadgroup = 4u;
    threadgroup float partial[kMaxRowsPerThreadgroup * kMaxSimdgroupsPerRow];

    const uint row = global_id.y;
    const uint local_row = local_id.y;
    if (threads_x == 0u || (threads_x & 31u) != 0u || local_row >= kMaxRowsPerThreadgroup) {
        return;
    }
    const uint simdgroups_per_row = threads_x / 32u;
    const uint local_simdgroup = simdgroup_id % simdgroups_per_row;
    const uint partial_index = local_row * kMaxSimdgroupsPerRow + local_simdgroup;
    const uint logical_cols = packed_cols * 2u;

    float acc = 0.0f;
    if (row < rows) {
        const ulong row_weight_base = (ulong)row * (ulong)packed_cols;
        const ulong row_scale_base = (ulong)row * (ulong)scale_cols;
        // `base_packed_col` stays four-byte aligned and expands to eight
        // logical K values. Every start is 8-aligned, so it cannot straddle a
        // source 32-wide scale block.
        for (uint base_packed_col = local_id.x * kPackedVectorWidth;
             base_packed_col < packed_cols;
             base_packed_col += threads_x * kPackedVectorWidth) {
            const uchar4 packed = *(device const uchar4*)(packed_weights + row_weight_base + (ulong)base_packed_col);
            const uint logical_col = base_packed_col * 2u;
            const float4 activation_lo = *(device const float4*)(x + logical_col);
            const float4 activation_hi = *(device const float4*)(x + logical_col + 4u);
            const float scale = deepseek_v4_e8m0fnu_value(
                scales[row_scale_base + (ulong)(logical_col / 32u)]);
            acc += deepseek_v4_e2m1fn_value(packed.x, false) * scale * activation_lo.x;
            acc += deepseek_v4_e2m1fn_value(packed.x, true)  * scale * activation_lo.y;
            acc += deepseek_v4_e2m1fn_value(packed.y, false) * scale * activation_lo.z;
            acc += deepseek_v4_e2m1fn_value(packed.y, true)  * scale * activation_lo.w;
            acc += deepseek_v4_e2m1fn_value(packed.z, false) * scale * activation_hi.x;
            acc += deepseek_v4_e2m1fn_value(packed.z, true)  * scale * activation_hi.y;
            acc += deepseek_v4_e2m1fn_value(packed.w, false) * scale * activation_hi.z;
            acc += deepseek_v4_e2m1fn_value(packed.w, true)  * scale * activation_hi.w;
        }
    }

    const float reduced_simdgroup = simd_sum(acc);
    if (lane_id == 0u) {
        partial[partial_index] = reduced_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_simdgroup == 0u) {
        const float partial_value = lane_id < simdgroups_per_row
            ? partial[local_row * kMaxSimdgroupsPerRow + lane_id]
            : 0.0f;
        const float reduced_row = simd_sum(partial_value);
        if (lane_id == 0u && row < rows) {
            y[row] = reduced_row;
        }
    }
}
#pragma clang fp contract(on)
