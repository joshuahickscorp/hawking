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

// ═════════════════════════════════════════════════════════════════════════════
// RWKV-7 CONTINUOUS-BATCH (multi-seq) DECODE kernels.
//
// B INDEPENDENT streams advanced in ONE pass. Every projection/LM-head weight is
// read ONCE across the B activation columns by the existing batched Q4_K GEMV
// (gemm_q4_k_m_batched_v3w_predec, x_batch[b*cols+off] / y_batch[b*rows+row]) —
// that is the bandwidth win. These kernels cover the RWKV-specific,
// non-GEMM steps for B streams. Each is byte-for-byte the single-stream kernel
// above with a stream index `b` added, so stream b is bit-for-bit its own
// single-stream decode (the multiseq parity oracle).
//
// LAYOUT CONTRACT (matches RwkvDecodeArena::new_with_batch):
//   - per-token activation buffers (att_in, k, v, a, a_op, b_op, r, w, gate,
//     out, ffn_in, xk, ...) are (B, n) ROW-major:  buf[b*n + i].
//   - the WKV state plane is STREAM-major; the host passes a per-(stream,layer)
//     byte offset so `state` already points at stream b's layer window... EXCEPT
//     the multiseq WKV kernel below indexes the stream itself (one dispatch for
//     all B), so it takes the per-LAYER stride and the slot base offset.
//   - xs (lerp output) is (slot, B, n); slot s lands at xs + s*B*n.
//   - per-channel vectors (lerp coeffs, w0/a0/v0, k_k/k_a/r_k, ln_w/ln_b,
//     v_first) are SHARED across streams → indexed by the channel i = gid % n.
// ═════════════════════════════════════════════════════════════════════════════

// ── WKV-7 recurrence + per-head group-norm + bonus + gate, for B streams ──
// One threadgroup processes ONE (stream, head); `head_size` threads, thread i
// owns row i of that (stream, head) state. Grid: (B*head_count*hs, 1, 1),
// threadgroups (hs,1,1); threadgroup index = b*head_count + head.
// `state` is the stream-major plane; `state_layer_stride` is the per-stream
// per-layer element stride (head_count*hs*hs) and `state_base` the element
// offset of THIS layer within each stream's window — so stream b's state for
// this layer starts at b*n_layer*state_layer_stride*... handled via base+stride.
struct ArgbufRwkv7WkvMs {
    uint  head_size;
    uint  head_count;
    uint  n;                 // n_embd (per-stream activation width)
    uint  batch;             // B
    uint  state_stream_stride; // elems per stream in the wkv plane (= n_layer*head_count*hs*hs)
    uint  state_layer_base;    // elems offset of this layer within a stream window (= layer*head_count*hs*hs)
    float gn_eps;
    uint  has_gate;
};
kernel void rwkv7_wkv_decode_multiseq(
    device       float* state [[buffer(0)]],   // stream-major S plane (whole buffer)
    device const float* r     [[buffer(1)]],   // (B, n)
    device const float* w     [[buffer(2)]],   // (B, n)
    device const float* k     [[buffer(3)]],   // (B, n)
    device const float* v     [[buffer(4)]],   // (B, n)
    device const float* a_op  [[buffer(5)]],   // (B, n)
    device const float* b_op  [[buffer(6)]],   // (B, n)
    device const float* r_k   [[buffer(7)]],   // (n,) shared
    device const float* ln_w  [[buffer(8)]],   // (n,) shared
    device const float* ln_b  [[buffer(9)]],   // (n,) shared
    device const float* gate  [[buffer(10)]],  // (B, n)
    device       float* out   [[buffer(11)]],  // (B, n)
    constant ArgbufRwkv7WkvMs& args [[buffer(12)]],
    threadgroup  float* red   [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],   // row i within the head
    uint tg      [[threadgroup_position_in_grid]],     // b*head_count + head
    uint tg_size [[threads_per_threadgroup]])           // == head_size
{
    const uint hs = args.head_size;
    if (tid >= hs) return;

    const uint b    = tg / args.head_count;            // stream index
    const uint head = tg - b * args.head_count;        // head within the stream
    const uint ho   = b * args.n + head * hs;          // (B,n) offset for this (stream,head)

    // State row for (stream b, this layer, head, row tid).
    const uint s_base = b * args.state_stream_stride + args.state_layer_base
                      + head * hs * hs;
    const uint row    = s_base + tid * hs;

    const float v_i = v[ho + tid];

    // sa[i] = sum_j a_op[j] * S_prev[i][j]
    float sa = 0.0f;
    for (uint j = 0; j < hs; ++j) {
        sa += a_op[ho + j] * state[row + j];
    }

    // S[i][j] = S_prev[i][j]*w[j] + v[i]*k[j] + sa*b_op[j]  (in place)
    // out_raw[i] = sum_j S[i][j] * r[j]
    float result = 0.0f;
    for (uint j = 0; j < hs; ++j) {
        float s_new = state[row + j] * w[ho + j]
                    + v_i * k[ho + j]
                    + sa * b_op[ho + j];
        state[row + j] = s_new;
        result += s_new * r[ho + j];
    }

    // per-head group-norm (population variance, eps=gn_eps)
    red[tid] = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / (float)hs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float d = result - mean;
    red[tid] = d * d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = red[0] / (float)hs;
    float inv = 1.0f / sqrt(var + args.gn_eps);
    // ln_w / ln_b are SHARED (per-channel): index by the within-stream channel.
    const uint ch = head * hs + tid;
    float o = (result - mean) * inv * ln_w[ch] + ln_b[ch];

    // bonus: out += v * rowsum_per_head(k * r * r_k)   (r_k shared per-channel)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[tid] = k[ho + tid] * r[ho + tid] * r_k[ch];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rk = red[0];
    o += v_i * rk;

    if (args.has_gate != 0u) {
        o *= gate[ho + tid];
    }
    out[ho + tid] = o;
}

// ── kk / k-mix for B streams. One threadgroup per (stream, head). ──
// k/a/a_op/b_op are (B, n); k_k/k_a are (n,) shared. Grid: (B*head_count*hs).
struct ArgbufRwkv7KkMs { uint head_size; uint head_count; uint n; };
kernel void rwkv7_kk_kmix_multiseq(
    device       float* k    [[buffer(0)]],   // (B, n)
    device const float* k_k  [[buffer(1)]],   // (n,) shared
    device const float* k_a  [[buffer(2)]],   // (n,) shared
    device const float* a    [[buffer(3)]],   // (B, n)
    device       float* a_op [[buffer(4)]],   // (B, n)
    device       float* b_op [[buffer(5)]],   // (B, n)
    constant ArgbufRwkv7KkMs& args [[buffer(6)]],
    threadgroup  float* red  [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg      [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint hs = args.head_size;
    if (tid >= hs) return;
    const uint b    = tg / args.head_count;
    const uint head = tg - b * args.head_count;
    const uint ch   = head * hs + tid;           // within-stream channel (shared vecs)
    const uint idx  = b * args.n + ch;           // (B,n) index

    float kk = k[idx] * k_k[ch];

    red[tid] = kk * kk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / max(sqrt(red[0]), 1e-12f);
    kk *= scale;

    float ka = k[idx] * k_a[ch];
    float av = a[idx];
    k[idx] = k[idx] + av * ka - ka;

    a_op[idx] = -kk;
    b_op[idx] = kk * av;
}

// ── token-shift + per-slot lerp for B streams. ──
// att_in is (B, n) ROW-major (att_in[b*n + i]). x_prev is the token-shift STATE
// plane, which is STREAM-major [stream][layer][n] (see RwkvDecodeArena): the host
// passes it offset to (stream 0, layer li), and the kernel reaches stream b's
// window with the per-stream element stride `x_prev_stride` (= n_layer*n). lerp
// is (n_slots, n) shared. xs is (n_slots, B, n): slot s, stream b, channel i
// lands at (s*B + b)*n + i. Grid: (B*n, 1, 1).
struct ArgbufRwkv7LerpMs { uint n; uint n_slots; uint batch; uint fresh; uint x_prev_stride; };
kernel void rwkv7_token_shift_lerp_multiseq(
    device const float* att_in [[buffer(0)]],   // (B, n) row-major
    device const float* x_prev [[buffer(1)]],   // stream-major plane @ (s0,li); stride x_prev_stride
    device const float* lerp   [[buffer(2)]],   // (n_slots, n) shared
    device       float* xs      [[buffer(3)]],  // (n_slots, B, n)
    constant ArgbufRwkv7LerpMs& args [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = args.batch * args.n;
    if (gid >= total) return;
    const uint b = gid / args.n;
    const uint i = gid - b * args.n;
    float av = att_in[gid];
    // x_prev is stream-major: stream b's layer window starts at b*x_prev_stride.
    float xp = x_prev[b * args.x_prev_stride + i];
    float sx = (args.fresh != 0u) ? (-av) : (xp - av);
    for (uint s = 0; s < args.n_slots; ++s) {
        // slot coeff is shared per-channel; output is (slot, B, n).
        xs[(s * args.batch + b) * args.n + i] = av + sx * lerp[s * args.n + i];
    }
}

// ── channel-mix token-shift + single lerp for B streams. ──
// ffn_in (B,n) row-major; x_prev the STREAM-major ffn token-shift plane @ (s0,li)
// with per-stream stride x_prev_stride; lerp_k (n,) shared; xk (B,n). Grid: (B*n).
kernel void rwkv7_channel_mix_shift_multiseq(
    device const float* ffn_in [[buffer(0)]],   // (B, n) row-major
    device const float* x_prev [[buffer(1)]],   // stream-major plane @ (s0,li); stride x_prev_stride
    device const float* lerp_k [[buffer(2)]],   // (n,) shared
    device       float* xk      [[buffer(3)]],  // (B, n)
    constant ArgbufRwkv7LerpMs& args [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = args.batch * args.n;
    if (gid >= total) return;
    const uint b = gid / args.n;
    const uint i = gid - b * args.n;
    float f = ffn_in[gid];
    float xp = x_prev[b * args.x_prev_stride + i];
    float dd = (args.fresh != 0u) ? (-f) : (xp - f);
    xk[gid] = f + dd * lerp_k[i];
}

// ── write a (B, n) row-major plane back into the STREAM-major token-shift state
// plane for layer `li`. src is att_in/ffn_in (B,n); dst is the whole state plane
// (host passes it offset to (stream 0, layer li)); stream b lands at
// b*dst_stride + i (dst_stride = n_layer*n). This is the B-stream analogue of the
// single-stream `rwkv7_copy` into a per-layer window. Grid: (B*n, 1, 1).
struct ArgbufRwkv7ShiftWb { uint n; uint batch; uint dst_stride; };
kernel void rwkv7_shift_writeback_multiseq(
    device const float* src [[buffer(0)]],   // (B, n) row-major
    device       float* dst [[buffer(1)]],   // stream-major plane @ (s0,li)
    constant ArgbufRwkv7ShiftWb& args [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = args.batch * args.n;
    if (gid >= total) return;
    const uint b = gid / args.n;
    const uint i = gid - b * args.n;
    dst[b * args.dst_stride + i] = src[gid];
}

// ── B independent LayerNorms. One threadgroup per stream; grid-strided over n. ──
// x / out are (B, n); weight / bias are (n,) shared. Grid: (B*TG, 1, 1),
// threadgroups (TG, 1, 1) → threadgroup index = stream.
struct ArgbufRwkv7LnMs { uint hidden; uint batch; float eps; };
kernel void rwkv7_layernorm_multiseq(
    device const float* x      [[buffer(0)]],   // (B, n)
    device const float* weight [[buffer(1)]],   // (n,) shared
    device const float* bias   [[buffer(2)]],   // (n,) shared
    device       float* out    [[buffer(3)]],   // (B, n)
    constant ArgbufRwkv7LnMs& args [[buffer(4)]],
    threadgroup  float* red    [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint b       [[threadgroup_position_in_grid]],   // stream index
    uint tg_size [[threads_per_threadgroup]])
{
    const uint n = args.hidden;
    if (b >= args.batch) return;
    device const float* xb = x   + (uint64_t)b * n;
    device       float* ob = out + (uint64_t)b * n;
    float partial = 0.0f;
    for (uint i = tid; i < n; i += tg_size) partial += xb[i];
    red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / (float)n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float d = xb[i] - mean;
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
        ob[i] = (xb[i] - mean) * inv * weight[i] + bias[i];
    }
}

// ── per-channel-bias elementwise kernels for B streams ──
// These mirror rwkv7_decay_act / rwkv7_sigmoid_bias / rwkv7_value_residual_mix
// but the bias/residual vectors (w0, a0, v0, v_first) are SHARED per-channel, so
// index them by i = gid % n. Grid: (B*n, 1, 1).
struct ArgbufRwkv7ElemMs { uint n; uint batch; };

// w[i] = exp(-0.606531 * sigmoid(w_raw[i] + w0[i%n])).  (B,n) over w_raw/w.
kernel void rwkv7_decay_act_multiseq(
    device const float* w_raw [[buffer(0)]],   // (B, n)
    device const float* w0    [[buffer(1)]],   // (n,) shared
    device       float* w     [[buffer(2)]],   // (B, n)
    constant ArgbufRwkv7ElemMs& args [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = args.batch * args.n;
    if (gid >= total) return;
    const uint i = gid % args.n;
    float s = 1.0f / (1.0f + exp(-(w_raw[gid] + w0[i])));
    w[gid] = exp(-0.606531f * s);
}

// x[i] = sigmoid(x[i] + bias[i%n]).  (B,n) over x.
kernel void rwkv7_sigmoid_bias_multiseq(
    device       float* x    [[buffer(0)]],   // (B, n)
    device const float* bias [[buffer(1)]],   // (n,) shared
    constant ArgbufRwkv7ElemMs& args [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = args.batch * args.n;
    if (gid >= total) return;
    const uint i = gid % args.n;
    x[gid] = 1.0f / (1.0f + exp(-(x[gid] + bias[i])));
}

// v[i] += (v_first[i%n] - v[i]) * sigmoid(v_mix[i] + v0[i%n]).  (B,n) over v/v_mix.
// NOTE: v_first is SHARED per-channel here — it is layer 0's value projection,
// which on the multiseq path is established PER STREAM. To keep v_first per
// stream the host stages a (B,n) v_first; this kernel then needs v_first indexed
// by gid (not i). We expose the per-stream form (v_first is (B,n)).
kernel void rwkv7_value_residual_mix_multiseq(
    device       float* v       [[buffer(0)]],   // (B, n)
    device const float* v_first [[buffer(1)]],   // (B, n) per-stream
    device const float* v_mix   [[buffer(2)]],   // (B, n)
    device const float* v0      [[buffer(3)]],   // (n,) shared
    constant ArgbufRwkv7ElemMs& args [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = args.batch * args.n;
    if (gid >= total) return;
    const uint i = gid % args.n;
    float g = 1.0f / (1.0f + exp(-(v_mix[gid] + v0[i])));
    float vi = v[gid];
    v[gid] = vi + (v_first[gid] - vi) * g;
}

// out[i] = a[i] + b[i] over a flat (B*n) range — residual adds for B streams.
// (a/b/out all (B,n); the op is purely elementwise so no per-stream indexing.)
kernel void rwkv7_add_into_flat(
    device const float* a   [[buffer(0)]],
    device const float* bb  [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant     uint&  total [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    out[gid] = a[gid] + bb[gid];
}

// ─────────────────────────────────────────────────────────────────────────────
// RWKV-7 LoRA fusion: grouped GEMV + fused mid-activation.
//
// The time-mix has four low-rank (LoRA) paths — w (decay), a (iclr), v (value-
// residual), g (gate) — each `x @ W1 → act → @ W2`. Run as eight tiny separate
// GEMV dispatches per layer, they dominated the GPU-time profile (≈190 tiny
// dispatches/step at ≈39% of GPU time), dispatch-overhead-bound not bandwidth-
// bound. These two kernels collapse the eight GEMVs + two inter-activations into
// THREE dispatches/layer: one grouped down-GEMV, one fused mid-act, one grouped
// up-GEMV.
//
// PARITY: each output row's arithmetic is bit-identical to the standalone
// `gemv_f32_attn` / `rwkv7_gemv_f32_off` it replaces — same f32 `row[c]*x[c]`
// MAC, same tree reduction over the same `cols`. Only the *dispatch* that carries
// the row changes, never the math, so the GPU↔CPU parity gate is preserved.
//
// A "group" is one LoRA's W1 (or W2). The host stacks the per-LoRA weight rows
// into one contiguous buffer and passes a small group table (≤4 entries). Each
// threadgroup owns one GLOBAL output row across all stacked groups; it scans the
// (tiny, ≤4) table to find its group, recovers the local row, then runs the
// reduction GEMV reading `x` at the group's input offset and writing `out` at
// the group's output offset.
//
// One threadgroup per output row; `tg_size` threads stride the reduction.
// Grid: (total_rows * tg_size, 1, 1), threadgroups (tg_size, 1, 1).
//
// Bindings:
//   0  w        stacked W rows (all groups), row-major f32
//   1  x        activation source (slot-major xs for down; stacked lo for up)
//   2  out      stacked outputs (all groups), f32
//   3  table    RwkvLoraGroup[ngroups]  (row_start, w_off, x_off, out_off, cols)
//   4  args     { ngroups; total_rows }
//   threadgroup(0): red (tg_size floats) reduction scratch
struct RwkvLoraGroup {
    uint row_start;  // first global row index of this group
    uint w_off;      // element offset of this group's W block in `w`
    uint x_off;      // element offset of this group's input in `x`
    uint out_off;    // element offset of this group's output in `out`
    uint cols;       // reduction length (== n for down, == rank for up)
};
struct ArgbufRwkv7Lora { uint ngroups; uint total_rows; };

kernel void rwkv7_lora_grouped_gemv(
    device const float*          w     [[buffer(0)]],
    device const float*          x     [[buffer(1)]],
    device       float*          out   [[buffer(2)]],
    device const RwkvLoraGroup*  table [[buffer(3)]],
    constant ArgbufRwkv7Lora&    args  [[buffer(4)]],
    threadgroup  float*          red   [[threadgroup(0)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint gid     [[threadgroup_position_in_grid]],   // global output row
    uint tg_size [[threads_per_threadgroup]])
{
    if (gid >= args.total_rows) return;

    // Locate the group owning this global row (≤4 groups → linear scan). Pick
    // the last group whose row_start <= gid.
    uint g = 0;
    for (uint i = 1; i < args.ngroups; ++i) {
        if (table[i].row_start <= gid) g = i; else break;
    }
    const uint local_row = gid - table[g].row_start;
    const uint cols      = table[g].cols;
    device const float* row  = w + (uint64_t)table[g].w_off + (uint64_t)local_row * (uint64_t)cols;
    device const float* xin  = x + table[g].x_off;

    float partial = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        partial += row[c] * xin[c];
    }
    red[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[table[g].out_off + local_row] = red[0];
}

// Fused LoRA mid-activation over the stacked low-rank buffer
// `lo = [w_lo | a_lo | v_lo | g_lo]`:
//   * tanh    on the w segment  [0, decay)
//   * sigmoid on the g segment  [decay+iclr+vres, decay+iclr+vres+gate)
//   * identity on the a and v segments
// Mirrors the CPU reference exactly (w: tanh before W2; g: sigmoid before G2;
// a/v: no inter-activation). Replaces the separate tanh_inplace + sigmoid_inplace
// dispatches. Grid: (decay+iclr+vres+gate, 1, 1).
struct ArgbufRwkv7LoraAct { uint w_end; uint g_begin; uint total; };
kernel void rwkv7_lora_mid_act(
    device float* lo [[buffer(0)]],
    constant ArgbufRwkv7LoraAct& args [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= args.total) return;
    float v = lo[gid];
    if (gid < args.w_end) {
        lo[gid] = tanh(v);
    } else if (gid >= args.g_begin) {
        lo[gid] = 1.0f / (1.0f + exp(-v));
    }
    // a and v segments: identity (leave as-is).
}
