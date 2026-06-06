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

// AWQ Option B fused variant: residual-add + rmsnorm + per-block int8
// quantization with a per-channel smoothing-divide folded into phase 3.
//
// Same five-buffer plus argbuf shape as `add_rmsnorm_fused_q8`, with an
// added `s` buffer holding the per-channel AWQ smoothing factors. The
// int8 quantize sees `x_norm[i] / s[i]` rather than `x_norm[i]`, but the
// stored `x_norm` is unchanged — downstream f32 consumers (`o_proj`-
// style fallback paths when W4A8 is force-disabled per-projection)
// still see the canonical normalized activation.
//
// Phases 1 (add + variance) and 2 (write x_norm) are byte-identical to
// `add_rmsnorm_fused_q8`. Phase 3 reads `s[block_off + ...]` alongside
// `x_norm[...]` and divides before fabs/simd_max.
kernel void add_rmsnorm_fused_q8_scaled(
    device       float*       x             [[buffer(0)]],
    device const float*       attn_out      [[buffer(1)]],
    device const float*       weight        [[buffer(2)]],
    device       float*       x_norm        [[buffer(3)]],
    device       signed char* x_norm_int8   [[buffer(4)]],
    device       float*       x_norm_scales [[buffer(5)]],
    device const float*       s             [[buffer(6)]],
    constant ArgbufRmsnorm&   args          [[buffer(7)]],
    threadgroup  float*       shmem         [[threadgroup(0)]],
    uint                      tid           [[thread_position_in_threadgroup]],
    uint                      tg_size       [[threads_per_threadgroup]])
{
    // Phase 1: residual add + accumulate variance.
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

    // Phase 2: normalize and write x_norm (unscaled).
    for (uint i = tid; i < args.hidden; i += tg_size) {
        x_norm[i] = x[i] * inv * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: per-256-element block int8 quantize with smoothing-divide.
    uint blocks = args.hidden / 256u;
    uint simd_id   = tid / 32u;
    uint simd_lane = tid % 32u;
    if (simd_id < blocks) {
        uint block_off = simd_id * 256u;
        float vals[8];
        float my_abs_max = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            uint idx = block_off + simd_lane + k * 32u;
            float sv = s[idx];
            float inv_s = (sv > 1e-12f) ? metal::precise::divide(1.0f, sv) : 0.0f;
            float v = x_norm[idx] * inv_s;
            vals[k] = v;
            my_abs_max = max(my_abs_max, fabs(v));
        }
        float max_abs = simd_max(my_abs_max);
        float scale = (max_abs > 0.0f)
                    ? metal::precise::divide(max_abs, 127.0f)
                    : 1.0f;
        if (simd_lane == 0u) x_norm_scales[simd_id] = scale;
        float inv_sc = metal::precise::divide(1.0f, scale);
        for (uint k = 0u; k < 8u; ++k) {
            float q = round(vals[k] * inv_sc);
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

// f32->f16 KV-append (Phase 2.1-a). Clone of memcpy_f32_off that writes the
// per-token f32 K/V slice into a `half` cache at `dst_off` (ELEMENT units —
// dst_off indexes half elements, identical convention to memcpy_f32_off).
// Reuses ArgbufMemcpyF32 (declared above). half(x) round-to-nearest-even
// matches Rust half::f16::from_f32 (the parity test asserts bit-equality).
// Default-off lever (reached only when DISMANTLE_QWEN_F16_KV=1).
kernel void memcpy_f32_to_f16_off(
    device const float*           src  [[buffer(0)]],
    device       half*            dst  [[buffer(1)]],
    constant ArgbufMemcpyF32&     args [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= args.n) return;
    dst[args.dst_off + id] = half(src[args.src_off + id]);
}

// R3 — batched KV scatter-append over B multi-seq decode slots. Each slot writes
// its kv_dim K and V elements into its OWN stable region (regions[bi]) at its OWN
// position (positions[bi]) within layer `layer_off`, in ONE dispatch (K+V) instead
// of 2B memcpys. Byte-identical to the per-slot memcpy_f32_off loop (pure copy):
//   dst_elem = layer_off + regions[bi]*slot_stride + positions[bi]*kv_dim + i
struct ArgbufKvScatter {
    uint kv_dim;
    uint b;
    uint slot_stride;  // max_seq_per_slot * kv_dim (one slot's per-layer region)
    uint layer_off;    // li * (max_batch * max_seq_per_slot * kv_dim)
};

kernel void kv_scatter_append_multiseq(
    device const float* src_k      [[buffer(0)]],
    device const float* src_v      [[buffer(1)]],
    device       float* k_cache    [[buffer(2)]],
    device       float* v_cache    [[buffer(3)]],
    constant ArgbufKvScatter& args [[buffer(4)]],
    device const uint*  regions    [[buffer(5)]],
    device const uint*  positions  [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = args.b * args.kv_dim;
    if (id >= total) return;
    uint bi = id / args.kv_dim;
    uint i  = id - bi * args.kv_dim;
    uint dst = args.layer_off + regions[bi] * args.slot_stride + positions[bi] * args.kv_dim + i;
    uint src = bi * args.kv_dim + i;
    k_cache[dst] = src_k[src];
    v_cache[dst] = src_v[src];
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

// R2 — batched RoPE over B multi-seq decode slots, each at its OWN position
// (positions[bi]). Bit-identical to running rope_q_f32_inplace per slot: the
// rotation is elementwise (no reduction), so batching over B changes no element.
// `x` is laid out [B, n_heads*head_dim] with `slot_stride` elements per slot.
// Called once for Q (n_heads, slot_stride=q_dim) and once for K (n_kv_heads,
// slot_stride=kv_dim). qk_nope_dim is 0 here (full-head rope, the dense path).
struct ArgbufRopeMultiseq {
    uint  n_heads;
    uint  head_dim;     // full head (= qk_rope_dim); qk_nope_dim is 0
    uint  slot_stride;  // elements per slot (q_dim for Q, kv_dim for K)
    uint  b;
    float base;
};

kernel void rope_f32_batched_multiseq(
    device       float* x             [[buffer(0)]],
    constant ArgbufRopeMultiseq& args [[buffer(1)]],
    device const uint*  positions     [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint pairs_per_head = args.head_dim / 2u;
    uint per_slot = args.n_heads * pairs_per_head;
    uint total = args.b * per_slot;
    if (id >= total) return;

    uint bi   = id / per_slot;
    uint rem  = id - bi * per_slot;
    uint head = rem / pairs_per_head;
    uint pair = rem - head * pairs_per_head;
    uint off  = bi * args.slot_stride + head * args.head_dim + 2u * pair;

    float theta = (float)positions[bi] / pow(args.base, 2.0f * float(pair) / float(args.head_dim));
    float c = cos(theta);
    float s = sin(theta);
    float x0 = x[off];
    float x1 = x[off + 1u];
    x[off]      = x0 * c - x1 * s;
    x[off + 1u] = x0 * s + x1 * c;
}

// Fused Q+K RoPE for multiseq batched decode (Track 3.4).
// Replaces two separate rope_f32_batched_multiseq dispatches per layer with one,
// saving 1 dispatch/layer × n_layers = 28 dispatches on Qwen-3B.
//
// Grid: (b * (n_q_heads + n_k_heads) * head_dim/2, 1, 1).
// Threads with id < b*n_q_heads*(head_dim/2) process Q; the rest process K.
// Q and K have different slot strides (q_dim vs kv_dim for GQA) but share
// the same rope base and positions buffer.
struct ArgbufRopeQKMultiseq {
    uint  n_q_heads;     // Q heads (n_heads)
    uint  n_k_heads;     // K heads (n_kv_heads; may differ for GQA)
    uint  head_dim;
    uint  q_slot_stride; // elements per Q slot = n_q_heads * head_dim
    uint  k_slot_stride; // elements per K slot = n_k_heads * head_dim
    uint  b;
    float base;
};
kernel void rope_qk_f32_batched_multiseq(
    device       float* q             [[buffer(0)]],  // (B, q_slot_stride)
    device       float* k             [[buffer(1)]],  // (B, k_slot_stride)
    constant ArgbufRopeQKMultiseq& args [[buffer(2)]],
    device const uint*  positions     [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint pairs_per_head = args.head_dim / 2u;
    uint q_per_slot = args.n_q_heads * pairs_per_head;
    uint k_per_slot = args.n_k_heads * pairs_per_head;
    uint q_total    = args.b * q_per_slot;
    uint total      = q_total + args.b * k_per_slot;
    if (id >= total) return;

    uint pair, off;
    float theta_denom;
    device float* buf;

    if (id < q_total) {
        uint bi   = id / q_per_slot;
        uint rem  = id - bi * q_per_slot;
        uint head = rem / pairs_per_head;
        pair      = rem - head * pairs_per_head;
        off       = bi * args.q_slot_stride + head * args.head_dim + 2u * pair;
        buf       = q;
        theta_denom = pow(args.base, 2.0f * float(pair) / float(args.head_dim));
        float theta = (float)positions[bi] / theta_denom;
        float c = cos(theta); float s = sin(theta);
        float x0 = buf[off]; float x1 = buf[off + 1u];
        buf[off]      = x0 * c - x1 * s;
        buf[off + 1u] = x0 * s + x1 * c;
    } else {
        uint kid  = id - q_total;
        uint bi   = kid / k_per_slot;
        uint rem  = kid - bi * k_per_slot;
        uint head = rem / pairs_per_head;
        pair      = rem - head * pairs_per_head;
        off       = bi * args.k_slot_stride + head * args.head_dim + 2u * pair;
        theta_denom = pow(args.base, 2.0f * float(pair) / float(args.head_dim));
        float theta = (float)positions[bi] / theta_denom;
        float c = cos(theta); float s = sin(theta);
        float x0 = k[off]; float x1 = k[off + 1u];
        k[off]      = x0 * c - x1 * s;
        k[off + 1u] = x0 * s + x1 * c;
    }
}

// Track 3.2 — Batched embed lookup: B tokens in one dispatch.
// Grid: (B * hidden, 1, 1). Saves B-1 dispatches vs the per-slot loop.
// tokens: GPU buffer of B u32 token ids (packed, no stride).
// out: (B, hidden) f32, row-major, slot-contiguous.
kernel void embed_lookup_f32_batched(
    device const half*  embed   [[buffer(0)]],  // (vocab, hidden) f16
    device const uint*  tokens  [[buffer(1)]],  // (B,) token ids
    device       float* out     [[buffer(2)]],  // (B, hidden) f32
    constant     uint&  hidden  [[buffer(3)]],
    constant     uint&  b       [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint total = b * hidden;
    if (id >= total) return;
    uint slot = id / hidden;
    uint elem = id - slot * hidden;
    uint tok  = tokens[slot];
    out[slot * hidden + elem] = (float)embed[(uint64_t)tok * (uint64_t)hidden + elem];
}

// Track 3.2 — Batched cold rmsnorm: B rows in one dispatch.
// Grid: (TG_SIZE * B, 1, 1). One TG per slot.
// Unlike add_rmsnorm_fused_batched, does NOT add a delta to x.
// Used for the layer-0 pre-norm where x is the embed output (no residual).
kernel void rmsnorm_f32_batched(
    device const float*     x      [[buffer(0)]],  // (B, hidden) input
    device const float*     weight [[buffer(1)]],  // (hidden,) scale
    device       float*     out    [[buffer(2)]],  // (B, hidden) output
    constant ArgbufRmsnorm& args   [[buffer(3)]],
    threadgroup  float*     shmem  [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row_off              = tg_id * args.hidden;
    device const float* x_row = x   + row_off;
    device       float* o_row = out + row_off;

    float partial = 0.0f;
    for (uint i = tid; i < args.hidden; i += tg_size) {
        float v = x_row[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shmem[tid] += shmem[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(shmem[0] / (float)args.hidden + args.eps);
    for (uint i = tid; i < args.hidden; i += tg_size) {
        o_row[i] = x_row[i] * inv * weight[i];
    }
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

// ── use_resource + gpu_address POC ──────────────────────────────────────────
//
// Canonical art for the megakernel day-3 dispatch harness. Demonstrates
// reading device buffers through pointer-in-argbuf (gpu_address)
// rather than per-dispatch set_buffer calls. The Metal driver requires
// either (a) set_buffer for residency tracking, or (b) explicit
// use_resource declaration when the buffer is referenced via raw
// pointer/gpu_address.
//
// Pattern used by the megakernel (each layer has 13 weight pointers
// in a single argbuf; calling set_buffer 13× per dispatch hits the
// 31-binding ceiling — see commit 7e4dc2c that compressed bindings
// 32→8). This kernel exists to validate the use_resource pattern in
// isolation before the megakernel relies on it.

struct PointerArgs {
    device const float* a;   // gpu_address of buffer A
    device const float* b;   // gpu_address of buffer B
    uint n;
};

kernel void use_resource_poc_add(
    constant PointerArgs& args [[buffer(0)]],
    device float* out          [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    if (tid >= args.n) return;
    out[tid] = args.a[tid] + args.b[tid];
}

// ── kv_append_vbias_f32 ───────────────────────────────────────────────────────
// Track 3.7 — Fused KV-cache append with V-bias addition.
//
// Replaces THREE dispatches per layer in forward_token_greedy_tcb:
//   add_inplace(v_token_buf, v_bias)         ← V bias (no rope)
//   memcpy_f32_off(k_token_buf → k_cache)   ← K cache append
//   memcpy_f32_off(v_token_buf → v_cache)   ← V cache append
// with ONE dispatch, saving 2 dispatches/layer × n_layers = 56 on Qwen-3B.
//
// K-token is written verbatim (K bias was already applied in rope_qk_b1_bias).
// V-token has v_bias optionally added before writing to v_cache.
//
// Buffer layout:
//   0: k_tok     (float*, kv_dim, input — K token vector, bias already applied)
//   1: v_tok     (float*, kv_dim, input — V token vector, NO bias yet)
//   2: v_bias    (float*, kv_dim, optional V bias — only read when has_v_bias=1)
//   3: k_cache   (float*, large KV cache, output)
//   4: v_cache   (float*, large KV cache, output)
//   5: args      (ArgbufKvAppendVbias)
//
// Grid: (kv_dim, 1, 1)
struct ArgbufKvAppendVbias {
    uint kv_dim;
    uint kv_off;      // element offset into k_cache and v_cache
    uint has_v_bias;  // 1 if v_bias is valid, 0 otherwise
};

kernel void kv_append_vbias_f32(
    device const float* k_tok   [[buffer(0)]],
    device const float* v_tok   [[buffer(1)]],
    device const float* v_bias  [[buffer(2)]],
    device       float* k_cache [[buffer(3)]],
    device       float* v_cache [[buffer(4)]],
    constant ArgbufKvAppendVbias& args [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= args.kv_dim) return;
    k_cache[args.kv_off + id] = k_tok[id];
    v_cache[args.kv_off + id] = v_tok[id] + (args.has_v_bias ? v_bias[id] : 0.0f);
}

// ── rope_qk_f32_b1_bias ───────────────────────────────────────────────────────
// Track 3.6 — B=1 fused Q+K RoPE with in-place bias addition.
//
// Replaces FOUR dispatches per layer in forward_token_greedy_tcb:
//   add_inplace(q_buf, q_bias)   ← Q bias
//   rope_q_f32_inplace(q_buf)    ← Q rope
//   add_inplace(k_buf, k_bias)   ← K bias
//   rope_q_f32_inplace(k_buf)    ← K rope
// with ONE dispatch, saving 3 dispatches/layer × n_layers = 84 on Qwen-3B.
//
// The bias is added BEFORE rotation (as in the original sequence). Each thread
// handles ONE complex pair (2 float elements) for one head of Q or K.
//
// Buffer layout:
//   0: q      (float*, n_q_heads * head_dim, in-place)
//   1: k      (float*, n_k_heads * head_dim, in-place)
//   2: q_bias (float*, n_q_heads * head_dim, or NULL-equivalent if no bias)
//   3: k_bias (float*, n_k_heads * head_dim, or NULL-equivalent if no bias)
//   4: args   (ArgbufRopeQKB1Bias)
//
// Grid: (n_q_heads * head_dim/2 + n_k_heads * head_dim/2, 1, 1)
// Threads id < n_q_heads * (head_dim/2) handle Q; the rest handle K.
struct ArgbufRopeQKB1Bias {
    uint  n_q_heads;
    uint  n_k_heads;
    uint  head_dim;
    uint  pos;
    float base;
    uint  has_q_bias; // 1 if q_bias is valid, 0 otherwise
    uint  has_k_bias; // 1 if k_bias is valid, 0 otherwise
};

kernel void rope_qk_f32_b1_bias(
    device       float* q    [[buffer(0)]],
    device       float* k    [[buffer(1)]],
    device const float* qb   [[buffer(2)]],
    device const float* kb   [[buffer(3)]],
    constant ArgbufRopeQKB1Bias& args [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint pairs_per_head = args.head_dim / 2u;
    uint q_total = args.n_q_heads * pairs_per_head;
    uint total   = q_total + args.n_k_heads * pairs_per_head;
    if (id >= total) return;

    device float* buf;
    device const float* bias;
    uint head, pair, off;
    bool use_bias;

    if (id < q_total) {
        head     = id / pairs_per_head;
        pair     = id - head * pairs_per_head;
        off      = head * args.head_dim + 2u * pair;
        buf      = q + off;
        bias     = qb + off;
        use_bias = args.has_q_bias != 0u;
    } else {
        uint kid = id - q_total;
        head     = kid / pairs_per_head;
        pair     = kid - head * pairs_per_head;
        off      = head * args.head_dim + 2u * pair;
        buf      = k + off;
        bias     = kb + off;
        use_bias = args.has_k_bias != 0u;
    }

    float x0 = buf[0] + (use_bias ? bias[0] : 0.0f);
    float x1 = buf[1] + (use_bias ? bias[1] : 0.0f);
    float theta = (float)args.pos / pow(args.base, 2.0f * float(pair) / float(args.head_dim));
    float c = cos(theta), s = sin(theta);
    buf[0] = x0 * c - x1 * s;
    buf[1] = x0 * s + x1 * c;
}

// ── rope_qk_kv_append_vbias_f32 ───────────────────────────────────────────────
// Track B6 — Fuses rope_qk_f32_b1_bias + kv_append_vbias_f32 into ONE dispatch.
//
// Saves 1 dispatch/layer × n_layers = 36 on Qwen-3B vs the two-dispatch path.
//
// Thread partition (id ∈ [0, total)):
//   [0,               q_pairs):           Q: bias + RoPE, write q_buf in-place
//   [q_pairs,         q_pairs+k_pairs):   K: bias + RoPE k_tok, write k_cache
//   [q_pairs+k_pairs, q_pairs+k_pairs+kv_dim): V: add v_bias, write v_cache
//
// Note: k_tok is READ-ONLY in this kernel (the rotated K is written directly to
// k_cache; k_token_buf is left in pre-rope state, which is fine since nothing
// reads it after this dispatch).
//
// Buffer layout:
//   0: q_buf    (float*, n_q_heads*head_dim, in-place Q rope destination)
//   1: k_tok    (float*, kv_dim, read-only; pre-rope K token)
//   2: v_tok    (float*, kv_dim, read-only)
//   3: q_bias   (float*, n_q_heads*head_dim, or unused when has_q_bias=0)
//   4: k_bias   (float*, kv_dim, or unused when has_k_bias=0)
//   5: v_bias   (float*, kv_dim, or unused when has_v_bias=0)
//   6: k_cache  (float*, large KV cache, write only)
//   7: v_cache  (float*, large KV cache, write only)
//   8: args     (ArgbufRopeQKKVAppend)
//
// Grid: (n_q_heads*head_dim/2 + n_k_heads*head_dim/2 + kv_dim, 1, 1)
struct ArgbufRopeQKKVAppend {
    uint  n_q_heads;
    uint  n_k_heads;
    uint  head_dim;
    uint  pos;
    float base;
    uint  has_q_bias;
    uint  has_k_bias;
    uint  has_v_bias;
    uint  kv_dim;    // n_k_heads * head_dim
    uint  kv_off;    // element offset into k_cache / v_cache
};

kernel void rope_qk_kv_append_vbias_f32(
    device       float* q_buf   [[buffer(0)]],
    device const float* k_tok   [[buffer(1)]],
    device const float* v_tok   [[buffer(2)]],
    device const float* q_bias  [[buffer(3)]],
    device const float* k_bias  [[buffer(4)]],
    device const float* v_bias  [[buffer(5)]],
    device       float* k_cache [[buffer(6)]],
    device       float* v_cache [[buffer(7)]],
    constant ArgbufRopeQKKVAppend& args [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    uint pairs_per_head = args.head_dim / 2u;
    uint q_pairs  = args.n_q_heads * pairs_per_head;
    uint k_pairs  = args.n_k_heads * pairs_per_head;
    uint q_end    = q_pairs;
    uint k_end    = q_pairs + k_pairs;
    uint v_end    = k_end + args.kv_dim;
    if (id >= v_end) return;

    if (id < q_end) {
        // ── Q: bias + RoPE, in-place write to q_buf ─────────────────────────
        uint head = id / pairs_per_head;
        uint pair = id - head * pairs_per_head;
        uint off  = head * args.head_dim + 2u * pair;
        float x0 = q_buf[off]   + (args.has_q_bias ? q_bias[off]   : 0.0f);
        float x1 = q_buf[off+1] + (args.has_q_bias ? q_bias[off+1] : 0.0f);
        float theta = (float)args.pos / pow(args.base, 2.0f * float(pair) / float(args.head_dim));
        float c = cos(theta), s = sin(theta);
        q_buf[off]   = x0 * c - x1 * s;
        q_buf[off+1] = x0 * s + x1 * c;
    } else if (id < k_end) {
        // ── K: bias + RoPE k_tok, write directly to k_cache ─────────────────
        uint kid  = id - q_end;
        uint head = kid / pairs_per_head;
        uint pair = kid - head * pairs_per_head;
        uint off  = head * args.head_dim + 2u * pair;
        float x0 = k_tok[off]   + (args.has_k_bias ? k_bias[off]   : 0.0f);
        float x1 = k_tok[off+1] + (args.has_k_bias ? k_bias[off+1] : 0.0f);
        float theta = (float)args.pos / pow(args.base, 2.0f * float(pair) / float(args.head_dim));
        float c = cos(theta), s = sin(theta);
        k_cache[args.kv_off + off]   = x0 * c - x1 * s;
        k_cache[args.kv_off + off+1] = x0 * s + x1 * c;
    } else {
        // ── V: add optional bias, write to v_cache ───────────────────────────
        uint vid = id - k_end;
        v_cache[args.kv_off + vid] = v_tok[vid] + (args.has_v_bias ? v_bias[vid] : 0.0f);
    }
}
