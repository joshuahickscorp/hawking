// common.metal — shared primitives across attention, MoE, and the
// per-token residual path.
//
// Kernels:
//   rmsnorm               — RMS normalization. fp32 reduction, fp16 mul.
//                           [Phase 0]
//   rmsnorm_f32           — RMS normalization, full fp32 I/O. Used by the
//                           Wedge B TCB path (f32 residual stream).
//                           [v1.0.0-B]
//   silu_mul              — SwiGLU activation (silu(a) * b) for the
//                           gate-up projection.
//                           [Phase 0]
//   rope_inplace          — Rotary position embedding, applied in-place
//                           to Q and K projections.
//                           [Phase 0]
//   embed_lookup          — input-token embedding lookup with optional
//                           tied LM head.
//                           [Phase 0]
//   add_inplace           — element-wise residual add: a[i] += b[i].
//                           [Phase 4 Wedge 4a]

#include <metal_stdlib>
using namespace metal;

// ── Phase 3 argbuf structs ────────────────────────────────────────────────────
// One packed struct per distinct scalar-arg pattern.  Kernels refactored to the
// argbuf pattern declare `constant ArgbufXxx& args [[buffer(N)]]` at the index
// that previously held the first `set_bytes` arg.

/// (rows: u32, cols: u32) — used by GEMV kernels with weight+activation buffers.
struct ArgbufRowsCols { uint rows; uint cols; };

/// (hidden: u32, eps: f32) — used by rmsnorm_f32 TCB path.
struct ArgbufRmsnorm { uint hidden; float eps; };

/// (n_experts: u32, top_k: u32) — used by moe_topk_gate.
struct ArgbufTopkGate { uint n_experts; uint top_k; };

/// (n: u32) — used by silu_mul / moe_batched_silu_mul.
struct ArgbufN { uint n; };

/// (seq_slot: u32, kv_lora_rank: u32, qk_rope_head_dim: u32) — used by kv_append_f32.
struct ArgbufKvAppend { uint seq_slot; uint kv_lora_rank; uint qk_rope_head_dim; };

/// (hidden: u32, routes: u32, has_shared: u32) — used by moe_route_accumulate.
struct ArgbufRouteAcc { uint hidden; uint routes; uint has_shared; };

/// (n_heads: u32, q_head_dim: u32, qk_nope_dim: u32, qk_rope_dim: u32, pos: u32, base: float)
/// — used by rope_q_f32_inplace.
struct ArgbufRopeQ {
    uint n_heads;
    uint q_head_dim;
    uint qk_nope_dim;
    uint qk_rope_dim;
    uint pos;
    float base;
};

// ─────────────────────────────────────────────────────────────────────────────

// One workgroup normalizes one (hidden,) row.
kernel void rmsnorm(
    device const half*  x        [[buffer(0)]],
    device const half*  weight   [[buffer(1)]],
    device       half*  out      [[buffer(2)]],
    constant     uint&  hidden   [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    threadgroup  float* shmem    [[threadgroup(0)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                tg_size  [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = (float)x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / float(hidden) + eps);
    float inv = 1.0f / rms;
    for (uint i = tid; i < hidden; i += tg_size) {
        out[i] = half((float)x[i] * inv * (float)weight[i]);
    }
}

// v1.0.0-B Wedge B — full fp32 rmsnorm for the f32 residual stream TCB path.
// Same math as rmsnorm above; operates on f32 x, f32 weight, f32 out.
// Threadgroup reduction accumulates variance in f32 (no precision loss).
kernel void rmsnorm_f32(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant ArgbufRmsnorm& args [[buffer(3)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        float v = x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / (float)args.hidden + args.eps);
    float inv = 1.0f / rms;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        out[i] = x[i] * inv * weight[i];
    }
}

// Session F (sketch) — fused add_inplace + rmsnorm_f32.
//
// Replaces two back-to-back dispatches:
//   add_inplace(x, attn_out, n)          // x[i] += attn_out[i]
//   rmsnorm_f32(x, weight, eps -> x_norm)
//
// Combined into a single TG that:
//   1. Loads attn_out[i], adds to x[i], stores x[i] back, accumulates v*v.
//   2. Reduces partial → inv_rms.
//   3. Re-reads x[i] (already in cache), writes x_norm[i] = x[i] * inv * weight[i].
//
// Eliminates one full pass over x (DRAM) and one dispatch's launch overhead.
// Single TG of 256 threads strides over `hidden` (≤ 2048 on V2-Lite, so one TG
// is sufficient — same shape contract as `rmsnorm_f32`).
//
// Bindings:
//   0  x         (hidden,) f32   IN/OUT — residual stream, gets += attn_out
//   1  attn_out  (hidden,) f32   read-only
//   2  weight    (hidden,) f32   rmsnorm learnable scale
//   3  x_norm    (hidden,) f32   OUT — normalized x
//   4  args      ArgbufRmsnorm   { hidden, eps }
//   threadgroup(0): shmem (tg_size × f32) — variance reduction
//
// Grid: (TG_SIZE, 1, 1) — single TG per dispatch. Cf. rmsnorm_f32 caller.
kernel void add_rmsnorm_fused(
    device       float* x        [[buffer(0)]],
    device const float* attn_out [[buffer(1)]],
    device const float* weight   [[buffer(2)]],
    device       float* x_norm   [[buffer(3)]],
    constant ArgbufRmsnorm& args [[buffer(4)]],
    threadgroup  float* shmem    [[threadgroup(0)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                tg_size  [[threads_per_threadgroup]])
{
    // Phase 1: add residual + accumulate variance in one pass.
    float partial = 0.0f;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        float v = x[i] + attn_out[i];
        x[i] = v;
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / (float)args.hidden + args.eps);
    float inv = 1.0f / rms;

    // Phase 2: normalize and write x_norm. x[i] is hot in cache after phase 1.
    for (uint i = tid; i < args.hidden; i += tg_size) {
        x_norm[i] = x[i] * inv * weight[i];
    }
}

// W4A8 fusion (2026-05-24): same add+rmsnorm as add_rmsnorm_fused, plus per-256-
// block int8 quantize of the normalized output. Saves one dispatch + one DRAM
// round-trip on x_norm vs back-to-back add_rmsnorm_fused + quantize_f32_to_int8_per_block.
//
// Phase 3 reads x_norm back from DRAM (rather than a register held over from
// phase 2) so the quantization math is byte-for-byte identical to the standalone
// quantize kernel, which is the bit-identical fusion gate (parity test).
//
// Requires hidden % 256 == 0; one TG (TG_SIZE=256 threads), single dispatch.
kernel void add_rmsnorm_fused_q8(
    device       float*       x             [[buffer(0)]],
    device const float*       attn_out      [[buffer(1)]],
    device const float*       weight        [[buffer(2)]],
    device       float*       x_norm        [[buffer(3)]],
    device       signed char* x_norm_int8   [[buffer(4)]],
    device       float*       x_norm_scales [[buffer(5)]],
    constant ArgbufRmsnorm&   args          [[buffer(6)]],
    threadgroup  float*       shmem         [[threadgroup(0)]],
    uint                      tid           [[thread_position_in_threadgroup]],
    uint                      tg_size       [[threads_per_threadgroup]])
{
    // Phase 1: residual add + accumulate variance (unchanged from add_rmsnorm_fused).
    float partial = 0.0f;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        float v = x[i] + attn_out[i];
        x[i] = v;
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / (float)args.hidden + args.eps);
    float inv = 1.0f / rms;

    // Phase 2: normalize and write x_norm (unchanged).
    for (uint i = tid; i < args.hidden; i += tg_size) {
        x_norm[i] = x[i] * inv * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: per-256-element block int8 quantize, **parallel across blocks**.
    // Each simdgroup (32 lanes) owns one 256-element block. Each lane handles
    // 8 elements via a strided pattern. simd_max gives the per-block reduction
    // in one warp-shuffle (no barriers, no shmem). All `hidden/256` blocks
    // execute concurrently within the single TG — matches the parallelism of
    // the standalone quantize kernel (which dispatched one TG per block) while
    // saving the inter-dispatch encoding overhead and the x_norm DRAM round-trip.
    //
    // Bit-identical to standalone: scales are max(|x|) of the same 256-elem
    // set (order-independent for max-of-floats); int8 conversion uses identical
    // precise::divide/round/clamp math.
    //
    // Requires tg_size == 256 (8 simdgroups × 32 lanes).
    uint blocks = args.hidden / 256u;
    uint simd_id   = tid / 32u;
    uint simd_lane = tid % 32u;
    if (simd_id < blocks) {
        uint block_off = simd_id * 256u;
        // Each lane reads 8 elements via stride-32 (lane, lane+32, ..., lane+224)
        // to coalesce DRAM loads across the simdgroup.
        float vals[8];
        float my_abs_max = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            float v = x_norm[block_off + simd_lane + k * 32u];
            vals[k] = v;
            my_abs_max = max(my_abs_max, fabs(v));
        }
        float max_abs = simd_max(my_abs_max);
        float scale = (max_abs > 0.0f)
                    ? metal::precise::divide(max_abs, 127.0f)
                    : 1.0f;
        if (simd_lane == 0u) x_norm_scales[simd_id] = scale;
        float inv_s = metal::precise::divide(1.0f, scale);
        for (uint k = 0u; k < 8u; ++k) {
            float q = round(vals[k] * inv_s);
            q = clamp(q, -127.0f, 127.0f);
            x_norm_int8[block_off + simd_lane + k * 32u] = (signed char)q;
        }
    }
}

// P3 — Batched add_rmsnorm_fused: B rows in one dispatch.
// Grid: (TG_SIZE * B, 1, 1) — B TGs, each TG handles one row.
// All buffers laid out (B, hidden) contiguous.
kernel void add_rmsnorm_fused_batched(
    device       float* x        [[buffer(0)]],
    device const float* attn_out [[buffer(1)]],
    device const float* weight   [[buffer(2)]],
    device       float* x_norm   [[buffer(3)]],
    constant ArgbufRmsnorm& args [[buffer(4)]],
    threadgroup  float* shmem    [[threadgroup(0)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                tg_id    [[threadgroup_position_in_grid]],
    uint                tg_size  [[threads_per_threadgroup]])
{
    uint row_off = tg_id * args.hidden;
    device       float* x_row        = x        + row_off;
    device const float* attn_out_row = attn_out + row_off;
    device       float* x_norm_row   = x_norm   + row_off;

    float partial = 0.0f;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        float v = x_row[i] + attn_out_row[i];
        x_row[i] = v;
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / (float)args.hidden + args.eps);
    float inv = 1.0f / rms;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        x_norm_row[i] = x_row[i] * inv * weight[i];
    }
}

// P3 — Batched add-bias: broadcasts a single (dim) bias vector across
// B output rows in one dispatch. Grid: (B*dim,). Replaces B sequential
// add_inplace calls for Qwen2 q/k/v biases in the batched prefill path.
struct ArgbufAddBroadcast {
    uint n;     // total elems = B * dim
    uint dim;   // bias length
};
kernel void add_inplace_broadcast(
    device       float* a       [[buffer(0)]],
    device const float* bias    [[buffer(1)]],
    constant ArgbufAddBroadcast& args [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= args.n) return;
    a[id] = a[id] + bias[id % args.dim];
}

kernel void silu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device       half* out  [[buffer(2)]],
    constant     uint& n    [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    if (id >= n) return;
    float g = (float)gate[id];
    float s = g / (1.0f + exp(-g));
    out[id] = half(s * (float)up[id]);
}

// P1f: generic GPU memcpy of `n` f32 elements from src+src_off to dst+dst_off.
// Offsets are in element units (not bytes). Used by GQA KV-cache append:
// copy k_token / v_token into the per-layer K/V cache slice at the
// current `seq_slot * kv_dim` offset.
struct ArgbufMemcpyF32 {
    uint n;
    uint src_off;
    uint dst_off;
};

kernel void memcpy_f32_off(
    device const float*           src  [[buffer(0)]],
    device       float*           dst  [[buffer(1)]],
    constant ArgbufMemcpyF32&     args [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= args.n) return;
    dst[args.dst_off + id] = src[args.src_off + id];
}

kernel void rope_inplace(
    device       half* x        [[buffer(0)]],
    constant     uint& head_dim [[buffer(1)]],
    constant     uint& pos      [[buffer(2)]],
    constant     float& base    [[buffer(3)]],
    uint id                      [[thread_position_in_grid]])
{
    uint half_dim = head_dim / 2;
    if (id >= half_dim) return;
    float theta = (float)pos / pow(base, 2.0f * float(id) / float(head_dim));
    float c = cos(theta);
    float s = sin(theta);
    float x0 = (float)x[2 * id];
    float x1 = (float)x[2 * id + 1];
    x[2 * id]     = half(x0 * c - x1 * s);
    x[2 * id + 1] = half(x0 * s + x1 * c);
}

kernel void rope_q_f32_inplace(
    device       float* q              [[buffer(0)]],
    constant ArgbufRopeQ& args         [[buffer(1)]],
    uint id                            [[thread_position_in_grid]])
{
    uint pairs_per_head = args.qk_rope_dim / 2u;
    uint total_pairs = args.n_heads * pairs_per_head;
    if (id >= total_pairs) return;

    uint head = id / pairs_per_head;
    uint pair = id - head * pairs_per_head;
    uint off = head * args.q_head_dim + args.qk_nope_dim + 2u * pair;

    float theta = (float)args.pos / pow(args.base, 2.0f * float(pair) / float(args.qk_rope_dim));
    float c = cos(theta);
    float s = sin(theta);
    float x0 = q[off];
    float x1 = q[off + 1u];
    q[off]      = x0 * c - x1 * s;
    q[off + 1u] = x0 * s + x1 * c;
}

kernel void rope_slice_f32_inplace(
    device       float* x        [[buffer(0)]],
    constant     uint&  offset   [[buffer(1)]],
    constant     uint&  head_dim [[buffer(2)]],
    constant     uint&  pos      [[buffer(3)]],
    constant     float& base     [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    uint half_dim = head_dim / 2u;
    if (id >= half_dim) return;
    uint off = offset + 2u * id;

    float theta = (float)pos / pow(base, 2.0f * float(id) / float(head_dim));
    float c = cos(theta);
    float s = sin(theta);
    float x0 = x[off];
    float x1 = x[off + 1u];
    x[off]      = x0 * c - x1 * s;
    x[off + 1u] = x0 * s + x1 * c;
}

// v1.0.0-D — embed lookup writing f32 residual stream.
// Reads f16 embed table, writes f32 x_buf directly (no CPU round-trip).
kernel void embed_lookup_f32(
    device const half*  embed  [[buffer(0)]],
    device       float* out    [[buffer(1)]],
    constant     uint&  hidden [[buffer(2)]],
    constant     uint&  token  [[buffer(3)]],
    uint id                     [[thread_position_in_grid]])
{
    if (id >= hidden) return;
    out[id] = (float)embed[token * hidden + id];
}

// G1.2 — fp16-weight × fp32-vec → fp32 GEMV (LM-head shape).
//
// One workgroup per output row, tg_size threads per group, threadgroup
// reduction across the inner-product accumulator (same pattern as
// rmsnorm above). Output is fp32 because rows like the LM head reach
// magnitudes ~√cols, where fp16 precision (~1 part in 1024) is too
// coarse for the parity tolerance.
kernel void gemv_f16(
    device const half*  w      [[buffer(0)]],   // (rows, cols) row-major fp16
    device const float* x      [[buffer(1)]],   // (cols,)
    device       float* y      [[buffer(2)]],   // (rows,)
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint                tid        [[thread_position_in_threadgroup]],
    uint                gid        [[threadgroup_position_in_grid]],
    uint                tg_size    [[threads_per_threadgroup]])
{
    if (gid >= rows) return;
    device const half* row = w + (uint64_t)gid * (uint64_t)cols;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += (float)row[c] * x[c];
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) y[gid] = shmem[0];
}

// Phase 4 Wedge 4a — element-wise residual add.
// Computes a[i] += b[i] for i in [0, n).
// One thread per element; grid (n, 1, 1), threadgroup (TG_SIZE, 1, 1).
kernel void add_inplace(
    device       float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    constant     uint&  n    [[buffer(2)]],
    uint                gid  [[thread_position_in_grid]])
{
    if (gid >= n) return;
    a[gid] += b[gid];
}

// Phase 7 Wedge 7b — fp16 rmsnorm.
// Reads f16 input, computes variance in f32 (sensitive to overflow at
// large activations), writes f16 output. Weight is f32.
//
// Threadgroup size 256 (parallel reduction; must be power of two ≤ 1024).
kernel void rmsnorm_f16(
    device const half*  x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    constant     float& eps     [[buffer(2)]],
    constant     uint&  hidden  [[buffer(3)]],
    device       half*  out     [[buffer(4)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = (float)x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shmem[0] / (float)hidden;
    float scale = rsqrt(mean + eps);

    for (uint i = tid; i < hidden; i += tg_size) {
        float v = (float)x[i];
        out[i] = (half)(v * scale * weight[i]);
    }
}

// Phase 7 Wedge 7d-prep — fp16 silu_mul.
// Computes out[i] = silu(gate[i]) * up[i] reading f16, writing f16.
// Internal sigmoid + multiply in f32 (silu's exp is sensitive).
kernel void silu_mul_f16(
    device const half*  gate   [[buffer(0)]],
    device const half*  up     [[buffer(1)]],
    device       half*  out    [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = (float)gate[gid];
    float u = (float)up[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = (half)(silu_g * u);
}

// v0.5.9-E — standalone f16 softmax.
// Single-threadgroup kernel: reads n f16 logits, writes n f16 probabilities.
// Max and exp-sum computed in f32. Grid: (TG_SIZE, 1, 1), TG: (TG_SIZE, 1, 1).
kernel void softmax_f16(
    device const half*  x     [[buffer(0)]],
    device       half*  out   [[buffer(1)]],
    constant     uint&  n     [[buffer(2)]],
    threadgroup  float* shmem [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    // Phase 1: parallel max reduction.
    float local_max = -INFINITY;
    for (uint i = tid; i < n; i += tg_size) {
        float v = (float)x[i];
        if (v > local_max) local_max = v;
    }
    shmem[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shmem[0];

    // Phase 2: parallel sum of exp(x - max).
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local_sum += exp((float)x[i] - max_val);
    }
    shmem[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shmem[0];

    // Phase 3: write normalized probabilities.
    for (uint i = tid; i < n; i += tg_size) {
        out[i] = (half)(exp((float)x[i] - max_val) * inv_sum);
    }
}

// v0.5.9-F — f16 layer normalization (mean-centering + variance + bias).
// Like rmsnorm_f16 but subtracts mean first and adds a bias term.
// Single-threadgroup kernel. Grid: (TG_SIZE, 1, 1), TG: (TG_SIZE, 1, 1).
kernel void layer_norm_f16(
    device const half*  x       [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const half*  bias    [[buffer(2)]],
    constant     float& eps     [[buffer(3)]],
    constant     uint&  n       [[buffer(4)]],
    device       half*  out     [[buffer(5)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    // Phase 1: mean.
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local_sum += (float)x[i];
    }
    shmem[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shmem[0] / float(n);

    // Phase 2: variance.
    float local_var = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float v = (float)x[i] - mean;
        local_var += v * v;
    }
    shmem[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shmem[0] / float(n) + eps);

    // Phase 3: normalize, scale, bias.
    for (uint i = tid; i < n; i += tg_size) {
        float v = ((float)x[i] - mean) * inv_std;
        out[i] = (half)(v * (float)weight[i] + (float)bias[i]);
    }
}

// Phase 5C.2 — f32 residual → f16 normed activation.
// Keeps the canonical residual stream as f32 between layers. Only the
// per-layer normed activation buffer is f16, halving downstream GEMV
// activation read bandwidth (e.g. LM head reads hidden×2 bytes vs hidden×4).
//
// Variance accumulator stays f32 (non-negotiable for stability).
// Binding layout matches rmsnorm_f32 so the Rust dispatcher can share the
// ArgbufRmsnorm struct. Buffer(3) is half* out instead of float* out.
// Grid: (TG_SIZE, 1, 1), TG: (TG_SIZE, 1, 1).
kernel void rmsnorm_f32_to_f16(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    constant ArgbufRmsnorm& args [[buffer(2)]],
    device       half*  out     [[buffer(3)]],
    threadgroup  float* shmem   [[threadgroup(0)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        float v = x[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shmem[0] / (float)args.hidden + args.eps);
    float inv = 1.0f / rms;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        out[i] = (half)(x[i] * inv * weight[i]);
    }
}

// Phase 5C.2 — f16-weight × f16-activation GEMV → f32 output.
// Used when x_norm_dtype="f16": LM head reads f16 activation (x_norm_f16_buf)
// instead of f32, halving the activation read bandwidth for the vocab GEMV.
// Weight is still f16 (same as gemv_f16). Output is f32 (logits need f32 range).
// MAC accumulates in f32. Binding layout identical to gemv_f16 except
// buffer(1) is half* instead of float*.
// Grid: (rows * TG_SIZE, 1, 1), TG: (TG_SIZE, 1, 1).
kernel void gemv_f16_f16in(
    device const half*  w      [[buffer(0)]],   // (rows, cols) row-major f16
    device const half*  x      [[buffer(1)]],   // (cols,) f16 activation
    device       float* y      [[buffer(2)]],   // (rows,) f32 output
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    threadgroup  float* shmem  [[threadgroup(0)]],
    uint                tid        [[thread_position_in_threadgroup]],
    uint                gid        [[threadgroup_position_in_grid]],
    uint                tg_size    [[threads_per_threadgroup]])
{
    if (gid >= rows) return;
    device const half* row = w + (uint64_t)gid * (uint64_t)cols;
    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += (float)row[c] * (float)x[c];
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) y[gid] = shmem[0];
}
