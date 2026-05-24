// Megakernel POC for Qwen-3B-Q4_K (2026-05-25, build/megakernel).
//
// SKELETON. Not yet functional. See
// `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/build_megakernel_design_2026_05_25.md`
// for the full design memo and the TODO checklist for filling this
// out. This file ships the threadgroup-memory layout, stage
// scaffolding, and barrier topology that the production port will
// fill with real GEMV bodies, RoPE, MHA, and KV-write.
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

// Per-layer weight pointer set. POC uses pre-dequantized f16 weights
// for ease of inline GEMV — see memo. Production port replaces these
// with Q4_K block pointers + inline decode.
struct MkLayerWeights {
    // f16 weights, contiguous in DRAM.
    // q_proj: (q_dim × hidden) row-major = 2048 × 2048
    // k_proj: (kv_dim × hidden) = 256 × 2048
    // v_proj: (kv_dim × hidden) = 256 × 2048
    // o_proj: (hidden × q_dim) = 2048 × 2048
    // ffn_gate: (intermediate × hidden) = 11008 × 2048
    // ffn_up:   (intermediate × hidden) = 11008 × 2048
    // ffn_down: (hidden × intermediate) = 2048 × 11008
    // attn_norm: (hidden,) f32
    // ffn_norm:  (hidden,) f32
    // q_bias / k_bias / v_bias: (q_dim or kv_dim,) f32 — optional
    // All passed via separate buffer slots; this struct only carries
    // scalars and offsets if needed.
    float rms_eps;
    float rope_theta;
    uint  has_qbias;
    uint  has_kbias;
    uint  has_vbias;
};

// Megakernel argbuf. Position and scratch addresses.
struct MkArgs {
    uint pos;        // current decode position
    uint seq_len;    // pos + 1
    uint max_seq;    // K/V cache stride (per layer)
};

// 2-layer Qwen-3B megakernel POC. Grid = (1, 1, 1). TG size = 256.
//
// Buffer layout (best-effort enumeration; Rust dispatcher binds in this order):
//   0  args        MkArgs (single)
//   1  x_in        device const half* (residual input, hidden)
//   2  x_out       device half*       (residual output, hidden)
//   3  k_cache     device half*       (n_layers × max_seq × kv_dim)
//   4  v_cache     device half*       (n_layers × max_seq × kv_dim)
//   5  ffn_scratch device half*       (intermediate, used for ffn_act)
//   6  layer0_wts  MkLayerWeights (single)
//   7  l0_qw       device const half* (q_dim × hidden)
//   8  l0_kw       device const half* (kv_dim × hidden)
//   9  l0_vw       device const half* (kv_dim × hidden)
//  10  l0_ow       device const half* (hidden × q_dim)
//  11  l0_gw       device const half* (intermediate × hidden)
//  12  l0_uw       device const half* (intermediate × hidden)
//  13  l0_dw       device const half* (hidden × intermediate)
//  14  l0_an       device const float* (hidden,)         // attn_norm
//  15  l0_fn       device const float* (hidden,)         // ffn_norm
//  16  l0_qb       device const float* (q_dim,) or null  // q_bias
//  17  l0_kb       device const float* (kv_dim,)         // k_bias
//  18  l0_vb       device const float* (kv_dim,)         // v_bias
//  19..31 layer 1, same shape
//
// TODO(megakernel-poc): implement stages. Currently a no-op that
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
    constant MkArgs&            args        [[buffer(0)]],
    device const half*          x_in        [[buffer(1)]],
    device       half*          x_out       [[buffer(2)]],
    device       half*          k_cache     [[buffer(3)]],
    device       half*          v_cache     [[buffer(4)]],
    device       half*          ffn_scratch [[buffer(5)]],
    // Layer 0 + weights:
    constant MkLayerWeights&    l0          [[buffer(6)]],
    device const half*          l0_qw       [[buffer(7)]],
    device const half*          l0_kw       [[buffer(8)]],
    device const half*          l0_vw       [[buffer(9)]],
    device const half*          l0_ow       [[buffer(10)]],
    device const half*          l0_gw       [[buffer(11)]],
    device const half*          l0_uw       [[buffer(12)]],
    device const half*          l0_dw       [[buffer(13)]],
    device const float*         l0_an       [[buffer(14)]],
    device const float*         l0_fn       [[buffer(15)]],
    device const float*         l0_qb       [[buffer(16)]],
    device const float*         l0_kb       [[buffer(17)]],
    device const float*         l0_vb       [[buffer(18)]],
    // Layer 1 + weights:
    constant MkLayerWeights&    l1          [[buffer(19)]],
    device const half*          l1_qw       [[buffer(20)]],
    device const half*          l1_kw       [[buffer(21)]],
    device const half*          l1_vw       [[buffer(22)]],
    device const half*          l1_ow       [[buffer(23)]],
    device const half*          l1_gw       [[buffer(24)]],
    device const half*          l1_uw       [[buffer(25)]],
    device const half*          l1_dw       [[buffer(26)]],
    device const float*         l1_an       [[buffer(27)]],
    device const float*         l1_fn       [[buffer(28)]],
    device const float*         l1_qb       [[buffer(29)]],
    device const float*         l1_kb       [[buffer(30)]],
    device const float*         l1_vb       [[buffer(31)]],
    threadgroup half*           shmem       [[threadgroup(0)]],
    uint tid                                [[thread_position_in_threadgroup]],
    uint tg_size                            [[threads_per_threadgroup]])
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
    (void)l0; (void)l0_qw; (void)l0_kw; (void)l0_vw; (void)l0_ow;
    (void)l0_gw; (void)l0_uw; (void)l0_dw; (void)l0_an; (void)l0_fn;
    (void)l0_qb; (void)l0_kb; (void)l0_vb;
    (void)l1; (void)l1_qw; (void)l1_kw; (void)l1_vw; (void)l1_ow;
    (void)l1_gw; (void)l1_uw; (void)l1_dw; (void)l1_an; (void)l1_fn;
    (void)l1_qb; (void)l1_kb; (void)l1_vb;

    for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
        residual[i] = x_in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TODO(megakernel-poc): stages A..L for layer 0
    // TODO(megakernel-poc): stages A..L for layer 1

    // Stage 13: write residual → DRAM x_out (POC: pass-through).
    for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
        x_out[i] = residual[i];
    }
}
