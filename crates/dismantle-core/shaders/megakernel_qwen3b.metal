// Megakernel POC for Qwen-3B-Q4_K (2026-05-25, build/megakernel day 3+).
//
// SKELETON. Stage bodies (A..L per layer) are TODOs — see
// `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/build_megakernel_design_2026_05_25.md`
// and the day-2 / day-3 closeouts. This file ships the threadgroup
// memory layout, the argument-buffer-based binding scheme that keeps
// the buffer-binding count under Metal's 30-slot limit, and the
// stage scaffolding the production port will fill with real GEMV
// bodies, RoPE, MHA, and KV-write.
//
// Shape constants are HARDCODED for Qwen-3B (hidden=2048, n_heads=16,
// n_kv_heads=2, head_dim=128, intermediate=11008). Do not reuse for
// other models without regenerating these.
//
// Weight format: pre-dequantized f16 (POC choice). Full Q4_K-inline
// is followup work. See memo § "Q4_K inline decode".

#include <metal_stdlib>
using namespace metal;

// Qwen-3B compile-time shape (see memo § "Qwen-3B shape constants").
constant constexpr uint MK_HIDDEN       = 2048u;
constant constexpr uint MK_N_HEADS      = 16u;
constant constexpr uint MK_N_KV_HEADS   = 2u;
constant constexpr uint MK_HEAD_DIM     = 128u;
constant constexpr uint MK_Q_DIM        = MK_N_HEADS * MK_HEAD_DIM;    // 2048
constant constexpr uint MK_KV_DIM       = MK_N_KV_HEADS * MK_HEAD_DIM; // 256
constant constexpr uint MK_INTERMEDIATE = 11008u;
constant constexpr uint MK_GROUP_SIZE   = MK_N_HEADS / MK_N_KV_HEADS;  // 8
constant constexpr uint MK_MAX_SEQ      = 256u;                        // POC cap

// Threadgroup layout (see memo § "Threadgroup memory layout").
//   shmem layout, all f16:
//     [0 .. 2048)        residual
//     [2048 .. 4096)     x_norm  (also reused for o_proj, ffn_down output)
//     [4096 .. 6144)     q
//     [6144 .. 6400)     k_token
//     [6400 .. 6656)     v_token
//     [6656 .. 6912)     scores (cap MK_MAX_SEQ)
//     [6912 .. 8960)     attn_out
//   Total: 8960 × 2B = 17920 bytes (~17.5 KB), under M3 Pro's 32 KB.
constant constexpr uint SH_RESIDUAL = 0u;
constant constexpr uint SH_XNORM    = MK_HIDDEN;
constant constexpr uint SH_Q        = SH_XNORM + MK_HIDDEN;
constant constexpr uint SH_K        = SH_Q + MK_Q_DIM;
constant constexpr uint SH_V        = SH_K + MK_KV_DIM;
constant constexpr uint SH_SCORES   = SH_V + MK_KV_DIM;
constant constexpr uint SH_ATTNOUT  = SH_SCORES + MK_MAX_SEQ;
constant constexpr uint SH_TOTAL    = SH_ATTNOUT + MK_HIDDEN;  // = 8960 half

// Per-layer argument buffer. Pointer fields hold Metal 3 GPU virtual
// addresses (populated host-side via `Buffer::gpuAddress()` then
// written into the argbuf as u64 → device pointer). The host must
// also call `useResource:usage:` on every referenced buffer before
// dispatching so the driver keeps them resident across the kernel.
//
// Layout MUST match the `#[repr(C)] struct MkLayerArgs` in
// `crates/dismantle-core/src/kernels/megakernel.rs` byte-for-byte.
// Total: 12 × 8B pointers + 2 × 4B floats + 4 × 4B uints = 120 B.
struct MkLayerArgs {
    // f16 weight pointers (rows × cols, row-major):
    device const half*  qw;    // q_proj   (q_dim × hidden)
    device const half*  kw;    // k_proj   (kv_dim × hidden)
    device const half*  vw;    // v_proj   (kv_dim × hidden)
    device const half*  ow;    // o_proj   (hidden × q_dim)
    device const half*  gw;    // ffn_gate (intermediate × hidden)
    device const half*  uw;    // ffn_up   (intermediate × hidden)
    device const half*  dw;    // ffn_down (hidden × intermediate)
    // f32 norm + bias pointers:
    device const float* attn_norm; // (hidden,)
    device const float* ffn_norm;  // (hidden,)
    device const float* qb;        // (q_dim,)  — null when has_qbias==0
    device const float* kb;        // (kv_dim,) — null when has_kbias==0
    device const float* vb;        // (kv_dim,) — null when has_vbias==0
    // Scalars (same shape as previous MkLayerWeights):
    float rms_eps;
    float rope_theta;
    uint  has_qbias;
    uint  has_kbias;
    uint  has_vbias;
    uint  _padding;
};

// Megakernel argbuf. Position + scratch cache strides.
struct MkArgs {
    uint pos;        // current decode position (RoPE phase)
    uint seq_len;    // pos + 1 (length of attended KV slice)
    uint max_seq;    // K/V cache stride per layer
    uint _padding;   // align to 16
};

// 2-layer Qwen-3B megakernel POC. Grid = (1, 1, 1). TG size = 256.
//
// Buffer bindings (8 total — well under Metal's 30 limit):
//   0  args        MkArgs (single, constant)
//   1  x_in        device const half* (residual input, hidden)
//   2  x_out       device half*       (residual output, hidden)
//   3  k_cache     device half*       (n_layers × max_seq × kv_dim)
//   4  v_cache     device half*       (n_layers × max_seq × kv_dim)
//   5  ffn_scratch device half*       (intermediate, used for ffn_act)
//   6  l0          device const MkLayerArgs* (layer-0 argbuf)
//   7  l1          device const MkLayerArgs* (layer-1 argbuf)
//
// TODO(megakernel-day3+): implement stages. Currently a no-op that
// passes x_in through to x_out so the harness can wire it.
//
// Stage outline (see memo § "Synchronization points"):
//   for each layer:
//     A) rmsnorm residual → x_norm                     [tg_barrier]
//     B) q/k/v GEMV  shmem activation × DRAM weights   [tg_barrier]
//     C) +q/k/v biases                                 [tg_barrier]
//     D) RoPE q (16 heads × 128) + RoPE k              [tg_barrier]
//     E) write k_token, v_token → DRAM kv_cache        [tg_barrier(threadgroup|device)]
//     F) MHA decode: scores[t] = q·K[t,kv]; softmax;
//        out[i] = Σ scores[t] V[t,kv,i]                [tg_barrier]
//     G) o_proj into shmem[attn_out]                   [tg_barrier]
//     H) fused add+rmsnorm:
//        residual += attn_out; x_norm = norm(residual) [tg_barrier]
//     I) ffn_gate/up into DRAM ffn_scratch (spill)     [tg_barrier(threadgroup|device)]
//     J) silu_mul in-place on ffn_scratch              [tg_barrier(threadgroup|device)]
//     K) ffn_down: DRAM ffn_scratch → shmem[xnorm]     [tg_barrier]
//     L) fused add+(next_norm or final): residual+=xnorm [tg_barrier]
//   write shmem[residual] → DRAM x_out
//
// Each `device` barrier point is also where the next layer's KV
// read sees the just-written KV slot.

kernel void qwen3b_megakernel_2layer(
    constant MkArgs&                  args        [[buffer(0)]],
    device const half*                x_in        [[buffer(1)]],
    device       half*                x_out       [[buffer(2)]],
    device       half*                k_cache     [[buffer(3)]],
    device       half*                v_cache     [[buffer(4)]],
    device       half*                ffn_scratch [[buffer(5)]],
    device const MkLayerArgs&         l0          [[buffer(6)]],
    device const MkLayerArgs&         l1          [[buffer(7)]],
    threadgroup half*                 shmem       [[threadgroup(0)]],
    uint tid                                      [[thread_position_in_threadgroup]],
    uint tg_size                                  [[threads_per_threadgroup]])
{
    // Stage 0: load residual from DRAM into shmem.
    threadgroup half* residual = shmem + SH_RESIDUAL;
    threadgroup half* xnorm    = shmem + SH_XNORM;
    threadgroup half* qbuf     = shmem + SH_Q;
    threadgroup half* kbuf     = shmem + SH_K;
    threadgroup half* vbuf     = shmem + SH_V;
    threadgroup half* scores   = shmem + SH_SCORES;
    threadgroup half* attnout  = shmem + SH_ATTNOUT;
    (void)qbuf; (void)kbuf; (void)vbuf; (void)scores; (void)attnout; (void)xnorm;
    (void)args; (void)k_cache; (void)v_cache; (void)ffn_scratch;
    (void)l0; (void)l1;

    for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
        residual[i] = x_in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TODO(megakernel-day3+): stages A..L for layer 0 (use l0.qw, l0.kw, ...)
    // TODO(megakernel-day3+): stages A..L for layer 1 (use l1.qw, l1.kw, ...)

    // Stage 13: write residual → DRAM x_out (POC: pass-through).
    for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
        x_out[i] = residual[i];
    }
}

// ── gpu_address probe (day-3 microbench) ─────────────────────────────
// Smallest test of the Metal 3 `gpuAddress` + `useResource:` pattern
// that the megakernel dispatcher will scale up to ~24 buffers/layer.
// One buffer in, one buffer out, both referenced via raw device pointers
// passed through a constant argbuf — no set_buffer for the data
// buffers; only the argbuf itself is bound. The dispatcher MUST call
// `useResource(in, Read)` and `useResource(out, Write)` on the
// encoder before dispatch, otherwise the driver page-faults on first
// dereference.
struct GpuAddrProbeArgs {
    device const float* in_ptr;
    device float*       out_ptr;
    uint n;
    uint _pad;
};

kernel void gpu_address_probe(
    constant GpuAddrProbeArgs& args [[buffer(0)]],
    uint tid                        [[thread_position_in_grid]])
{
    if (tid < args.n) {
        args.out_ptr[tid] = args.in_ptr[tid] * 2.0f;
    }
}
