#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// RWKV-7 ("Goose") WKV-7 single-step DECODE recurrence.
//
// This is the novel, tps-critical kernel of the RWKV-7 GPU decode path. RWKV-7
// is a state-space model: instead of a growing KV cache it carries a fixed
// per-layer recurrent state `S` (one `head_size x head_size` matrix per head),
// so decode is O(1) in context — the whole point of this slice.
//
// One threadgroup processes one head; `head_size` threads per group, thread `i`
// owns row `i` of that head's state. The state lives in a persistent GPU buffer
// (the RwkvDecodeArena), advanced in place each step — never reallocated, never
// grown. This file is self-contained (its own helpers, no shared symbols) and
// is appended LAST to the runtime library so it cannot perturb any existing
// kernel's codegen, exactly like the strand_bitslice family.
//
// Math (per head, matching the CPU reference `rwkv7.rs::time_mix` EXACTLY so the
// GPU path is bit-for-bit within f32 tolerance against the validated oracle).
// With `a_op = -kk`, `b_op = kk * iclr`:
//
//   sa[i]   = sum_j a_op[j] * S_prev[i][j]
//   S[i][j] = S_prev[i][j]*w[j] + v[i]*k[j] + sa[i]*b_op[j]
//   out[i]  = sum_j S[i][j] * r[j]
//
// `S[i][j]` is stored row-major at `head*hs*hs + i*hs + j` (row i, col j),
// matching ggml's `state_prev[i*h_stride + j]` and the CPU reference layout.
//
// The kernel then folds the per-head tail that the CPU `time_mix` applies right
// after the recurrence — group-norm, the r*k*r_k bonus, and the gate — into the
// SAME dispatch (all of them are per-head ops), so the whole WKV stage is one
// dispatch per layer and the f32 op-order matches the reference step for step.
//
// Bindings:
//   0  state    (head_count*hs*hs,) f32   persistent S, advanced in place
//   1  r        (n_embd,)           f32   receptance  (Wr @ xr)
//   2  w        (n_embd,)           f32   decay       (exp(-0.606531*sigmoid(..)))
//   3  k        (n_embd,)           f32   key         (post k_a mix)
//   4  v        (n_embd,)           f32   value       (post value-residual mix)
//   5  a_op     (n_embd,)           f32   = -kk
//   6  b_op     (n_embd,)           f32   = kk * iclr
//   7  r_k      (n_embd,)           f32   per-channel r_k vector (bonus)
//   8  ln_w     (n_embd,)           f32   group-norm weight
//   9  ln_b     (n_embd,)           f32   group-norm bias
//  10  gate     (n_embd,)           f32   gate (g = G2 @ sigmoid(G1 @ xg))
//  11  out      (n_embd,)           f32   WKV-7 output for this token
//  12  args     ArgbufRwkv7Wkv { head_size; gn_eps; has_gate }
//
//  threadgroup(0): red  (hs floats) — head-local reduction scratch (gn + bonus)
//
// Grid:  (head_count * hs, 1, 1) threads, threadgroups of (hs, 1, 1).
// ─────────────────────────────────────────────────────────────────────────────

struct ArgbufRwkv7Wkv {
    uint  head_size;
    float gn_eps;     // group-norm epsilon (64e-5 in the reference)
    uint  has_gate;   // 1 → apply the gate multiply, 0 → skip
};

kernel void rwkv7_wkv_decode(
    device       float* state [[buffer(0)]],
    device const float* r     [[buffer(1)]],
    device const float* w     [[buffer(2)]],
    device const float* k     [[buffer(3)]],
    device const float* v     [[buffer(4)]],
    device const float* a_op  [[buffer(5)]],
    device const float* b_op  [[buffer(6)]],
    device const float* r_k   [[buffer(7)]],
    device const float* ln_w  [[buffer(8)]],
    device const float* ln_b  [[buffer(9)]],
    device const float* gate  [[buffer(10)]],
    device       float* out   [[buffer(11)]],
    constant ArgbufRwkv7Wkv& args [[buffer(12)]],
    threadgroup  float* red   [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],   // row index i within the head
    uint head    [[threadgroup_position_in_grid]],     // head index
    uint tg_size [[threads_per_threadgroup]])           // == head_size
{
    const uint hs = args.head_size;
    if (tid >= hs) return;

    const uint ho = head * hs;        // offset into the n-dim vectors for this head
    const uint so = head * hs * hs;   // offset into the state for this head
    const uint row = so + tid * hs;   // start of this thread's state row (row i)

    const float v_i = v[ho + tid];

    // sa[i] = sum_j a_op[j] * S_prev[i][j]   (over this thread's row).
    float sa = 0.0f;
    for (uint j = 0; j < hs; ++j) {
        sa += a_op[ho + j] * state[row + j];
    }

    // S[i][j] = S_prev[i][j]*w[j] + v[i]*k[j] + sa*b_op[j]   (in place)
    // out_raw[i] = sum_j S[i][j] * r[j]
    float result = 0.0f;
    for (uint j = 0; j < hs; ++j) {
        float s_new = state[row + j] * w[ho + j]
                    + v_i * k[ho + j]
                    + sa * b_op[ho + j];
        state[row + j] = s_new;
        result += s_new * r[ho + j];
    }

    // ── per-head group-norm over out_raw (population variance, eps=gn_eps) ──
    // mean
    red[tid] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / (float)hs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // variance
    float d = result - mean;
    red[tid] = d * d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = red[0] / (float)hs;
    float inv = 1.0f / sqrt(var + args.gn_eps);
    // out[i] = (out_raw[i]-mean)*inv*ln_w[i] + ln_b[i]
    float o = (result - mean) * inv * ln_w[ho + tid] + ln_b[ho + tid];

    // ── bonus: out += v * rowsum_per_head(k * r * r_k) ──
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[tid] = k[ho + tid] * r[ho + tid] * r_k[ho + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rk = red[0];
    o += v_i * rk;

    // ── gate ──
    if (args.has_gate != 0u) {
        o *= gate[ho + tid];
    }

    out[ho + tid] = o;
}

// ─────────────────────────────────────────────────────────────────────────────
// RWKV-7 elementwise "glue" kernels.
//
// These realize the RWKV-7-specific, non-GEMM steps of `time_mix` / `channel_mix`
// on the GPU so the whole decode forward stays inside one TokenCommandBuffer
// (no per-op CPU round-trip, which would serialize the GPU and destroy the flat
// decode curve). Each mirrors the CPU reference EXACTLY for f32 parity. All are
// self-contained — no shared symbols with the rest of the library.
// ─────────────────────────────────────────────────────────────────────────────

// Token-shift + per-slot lerp for the time-mix branch.
//   sx[i] = fresh ? -att_in[i] : (x_prev[i] - att_in[i])
//   for slot s in 0..n_slots:  x_s[s*n + i] = att_in[i] + sx[i] * lerp[s*n + i]
// `lerp` is the fused [n_slots * n] slot-major coefficient block; `xs` is the
// [n_slots * n] output (slot-major), so slot s lands at xs + s*n. The reference
// packs slots as r,w,k,v,a,g. n_slots is 5 (no gate) or 6 (gate).
// Grid: (n, 1, 1).
struct ArgbufRwkv7Lerp {
    uint n;
    uint n_slots;
    uint fresh;   // 1 → x_prev treated as 0 (first token)
};
kernel void rwkv7_token_shift_lerp(
    device const float* att_in [[buffer(0)]],
    device const float* x_prev [[buffer(1)]],
    device const float* lerp   [[buffer(2)]],
    device       float* xs     [[buffer(3)]],
    constant ArgbufRwkv7Lerp& args [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= args.n) return;
    float a = att_in[gid];
    float sx = (args.fresh != 0u) ? (-a) : (x_prev[gid] - a);
    for (uint s = 0; s < args.n_slots; ++s) {
        uint o = s * args.n + gid;
        xs[o] = a + sx * lerp[o];
    }
}

// Channel-mix token-shift + single lerp.
//   xk[i] = ffn_in[i] + (fresh ? -ffn_in[i] : (x_prev[i]-ffn_in[i])) * lerp_k[i]
// Grid: (n, 1, 1).
kernel void rwkv7_channel_mix_shift(
    device const float* ffn_in [[buffer(0)]],
    device const float* x_prev [[buffer(1)]],
    device const float* lerp_k [[buffer(2)]],
    device       float* xk     [[buffer(3)]],
    constant ArgbufRwkv7Lerp& args [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= args.n) return;
    float f = ffn_in[gid];
    float d = (args.fresh != 0u) ? (-f) : (x_prev[gid] - f);
    xk[gid] = f + d * lerp_k[gid];
}

// tanh in place over a vector (decay LoRA low projection). Grid: (n, 1, 1).
kernel void rwkv7_tanh_inplace(
    device float* x   [[buffer(0)]],
    constant uint& n  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = tanh(x[gid]);
}

// Decay activation: w[i] = exp(-0.606531 * sigmoid(w_raw[i] + w0[i])).
// `w_raw` is W2 @ tanh(W1 @ xw); `w0` is the per-channel bias. Grid: (n, 1, 1).
kernel void rwkv7_decay_act(
    device const float* w_raw [[buffer(0)]],
    device const float* w0    [[buffer(1)]],
    device       float* w     [[buffer(2)]],
    constant     uint&  n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float s = 1.0f / (1.0f + exp(-(w_raw[gid] + w0[gid])));
    w[gid] = exp(-0.606531f * s);
}

// Sigmoid-with-bias in place: x[i] = sigmoid(x[i] + bias[i]).
// Reused for the iclr activation a = sigmoid(a_raw + a0). Grid: (n, 1, 1).
kernel void rwkv7_sigmoid_bias(
    device       float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant     uint&  n    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = 1.0f / (1.0f + exp(-(x[gid] + bias[gid])));
}

// Plain sigmoid in place: x[i] = sigmoid(x[i]). Used for the gate's inner
// G1@xg sigmoid (no bias). Grid: (n, 1, 1).
kernel void rwkv7_sigmoid_inplace(
    device       float* x [[buffer(0)]],
    constant     uint&  n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] = 1.0f / (1.0f + exp(-x[gid]));
}

// Value-residual mix:
//   v[i] += (v_first[i] - v[i]) * sigmoid(v_mix[i] + v0[i])
// where v_mix = V2 @ (V1 @ xv). Skipped on layer 0 by the host. Grid: (n, 1, 1).
kernel void rwkv7_value_residual_mix(
    device       float* v       [[buffer(0)]],
    device const float* v_first [[buffer(1)]],
    device const float* v_mix   [[buffer(2)]],
    device const float* v0      [[buffer(3)]],
    constant     uint&  n       [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = 1.0f / (1.0f + exp(-(v_mix[gid] + v0[gid])));
    float vi = v[gid];
    v[gid] = vi + (v_first[gid] - vi) * g;
}

// kk = l2norm_per_head(k * k_k); then k += (a-1)*(k*k_a); a_op = -kk; b_op = kk*a.
// One threadgroup per head (hs threads), matching the WKV kernel's mapping.
// `k` is updated in place. Mirrors the reference order exactly. Grid:
// (head_count*hs, 1, 1), threadgroups (hs, 1, 1).
struct ArgbufRwkv7Kk { uint head_size; };
kernel void rwkv7_kk_kmix(
    device       float* k    [[buffer(0)]],
    device const float* k_k  [[buffer(1)]],
    device const float* k_a  [[buffer(2)]],
    device const float* a    [[buffer(3)]],
    device       float* a_op [[buffer(4)]],
    device       float* b_op [[buffer(5)]],
    constant ArgbufRwkv7Kk& args [[buffer(6)]],
    threadgroup  float* red  [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint head    [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint hs = args.head_size;
    if (tid >= hs) return;
    const uint idx = head * hs + tid;

    // kk_raw = k * k_k for this lane.
    float kk = k[idx] * k_k[idx];

    // l2 norm over the head: scale = 1 / max(sqrt(sum(kk^2)), 1e-12).
    red[tid] = kk * kk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / max(sqrt(red[0]), 1e-12f);
    kk *= scale;

    // k = k + (a-1)*(k*k_a)   [== k + a*ka - ka]
    float ka = k[idx] * k_a[idx];
    float av = a[idx];
    k[idx] = k[idx] + av * ka - ka;

    // a_op = -kk ; b_op = kk * a
    a_op[idx] = -kk;
    b_op[idx] = kk * av;
}

// Channel-mix activation in place: k[i] = relu(k[i])^2. Grid: (n_ff, 1, 1).
kernel void rwkv7_relu_sq_inplace(
    device float* k   [[buffer(0)]],
    constant uint& n  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = max(k[gid], 0.0f);
    k[gid] = v * v;
}

// add two vectors into a fresh destination: out[i] = a[i] + b[i].
// Used for the residual adds (ffn_inp = cur + x; x = cmix + ffn_inp) without
// clobbering an operand that is still needed. Grid: (n, 1, 1).
kernel void rwkv7_add_into(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant     uint&  n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

// LayerNorm with weight+bias over the whole vector (population variance),
// matching the CPU `layernorm` (ggml_norm semantics): subtract mean, divide by
// sqrt(var+eps), then *w + b. Single threadgroup (TG_SIZE threads), grid-strided
// over `hidden`. Grid: (TG_SIZE, 1, 1).
struct ArgbufRwkv7Ln { uint hidden; float eps; };
kernel void rwkv7_layernorm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device       float* out    [[buffer(3)]],
    constant ArgbufRwkv7Ln& args [[buffer(4)]],
    threadgroup  float* red    [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint n = args.hidden;
    // mean
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg_size) partial += x[i];
    red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / (float)n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // variance
    partial = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = x[i] - mean;
        partial += d * d;
    }
    red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(red[0] / (float)n + args.eps);
    for (uint i = tid; i < n; i += tg_size) {
        out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

// Copy a vector: dst[i] = src[i]. Used to snapshot att_in/ffn_in into the
// persistent token-shift state, and to seed v_first on layer 0. Grid: (n,1,1).
kernel void rwkv7_copy(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant     uint&  n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = src[gid];
}

// Gate multiply in place: out[i] *= gate[i]. (The WKV kernel already folds the
// gate when present; this standalone form is unused by the default path but kept
// symmetric for clarity / future fusion. Grid: (n,1,1).)
kernel void rwkv7_mul_inplace(
    device       float* out  [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant     uint&  n    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] *= gate[gid];
}
