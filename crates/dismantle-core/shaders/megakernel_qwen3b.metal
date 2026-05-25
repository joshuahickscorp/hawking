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
    uint pos;          // current decode position (RoPE phase)
    uint seq_len;      // pos + 1 (length of attended KV slice)
    uint max_seq;      // K/V cache stride per layer
    uint probe_stage;  // dev-only: which intermediate is written to x_out
};

// Probe-stage IDs (must match `MK_PROBE_*` in
// `crates/dismantle-core/src/kernels/megakernel.rs`). Dev-only escape
// hatch — pinned to a single integer-compare in the terminal write
// stage so the shader can be parity-tested per stage without rewiring
// each commit. Eliminated when 2-layer parity ships (probe_stage 6
// becomes the only terminal write).
constant constexpr uint MK_PROBE_XNORM_A   = 0u;  // layer-0 stage A
constant constexpr uint MK_PROBE_Q_ROT     = 1u;  // layer-0 stage D (post-RoPE Q)
constant constexpr uint MK_PROBE_ATTN_OUT  = 2u;  // layer-0 stage F (MHA out)
constant constexpr uint MK_PROBE_O_PROJ    = 3u;  // layer-0 stage G (o_proj out)
constant constexpr uint MK_PROBE_XNORM_FFN = 4u;  // layer-0 stage H (post-attn rmsnorm)
constant constexpr uint MK_PROBE_FFN_DOWN  = 5u;  // layer-0 stage K (ffn_down out)
constant constexpr uint MK_PROBE_RESIDUAL  = 6u;  // final 2-layer residual

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
    // Voids for slots not yet referenced by stages B..F (day-5).
    (void)ffn_scratch;
    (void)l1;

    for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
        residual[i] = x_in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage A (day-3 step 4): layer-0 pre-attention rmsnorm.
    //   xnorm[i] = (residual[i] / sqrt(mean(residual^2) + eps))
    //                * l0.attn_norm[i]
    // Reduction strategy: per-thread partial sum of squares → simdgroup
    // reduce via simd_sum → 8 partials in shmem (TG=256 → 8 simdgroups)
    // → final 8-way sum is computed locally by every thread (the
    // barrier upstream makes all simd partials visible). f32
    // accumulation, f16 store to xnorm.
    {
        threadgroup float simd_partials[8];

        float local = 0.0f;
        for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
            float v = (float)residual[i];
            local += v * v;
        }
        float simd_red = simd_sum(local);
        uint simd_lane  = tid & 31u;
        uint simd_group = tid >> 5u;
        if (simd_lane == 0u) {
            simd_partials[simd_group] = simd_red;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            total += simd_partials[i];
        }
        float rnorm = rsqrt(total / (float)MK_HIDDEN + l0.rms_eps);

        for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
            float v = (float)residual[i];
            xnorm[i] = (half)(v * rnorm * l0.attn_norm[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage B (day-4): Q / K / V f16 GEMVs from shmem `xnorm`.
    //   qbuf[r] = Σ_c qw[r, c] * xnorm[c]   for r ∈ [0, Q_DIM)
    //   kbuf[r] = Σ_c kw[r, c] * xnorm[c]   for r ∈ [0, KV_DIM)
    //   vbuf[r] = Σ_c vw[r, c] * xnorm[c]   for r ∈ [0, KV_DIM)
    // Row-major weights. Per-row f32 accumulator, f16 store. Each
    // thread strides by tg_size across the output rows. TG=256 ⇒ Q has
    // 8 rows/thread, K/V have one row per thread for tid<256 (with
    // only 256 rows total → first 256 threads do work).
    for (uint r = tid; r < MK_Q_DIM; r += tg_size) {
        float acc = 0.0f;
        device const half* row = l0.qw + (uint64_t)r * MK_HIDDEN;
        for (uint c = 0; c < MK_HIDDEN; ++c) {
            acc += (float)row[c] * (float)xnorm[c];
        }
        qbuf[r] = (half)acc;
    }
    for (uint r = tid; r < MK_KV_DIM; r += tg_size) {
        float acc_k = 0.0f;
        float acc_v = 0.0f;
        device const half* row_k = l0.kw + (uint64_t)r * MK_HIDDEN;
        device const half* row_v = l0.vw + (uint64_t)r * MK_HIDDEN;
        for (uint c = 0; c < MK_HIDDEN; ++c) {
            acc_k += (float)row_k[c] * (float)xnorm[c];
            acc_v += (float)row_v[c] * (float)xnorm[c];
        }
        kbuf[r] = (half)acc_k;
        vbuf[r] = (half)acc_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage C (day-4): add Q/K/V biases (Qwen2 always has them).
    if (l0.has_qbias != 0u) {
        for (uint r = tid; r < MK_Q_DIM; r += tg_size) {
            qbuf[r] = (half)((float)qbuf[r] + l0.qb[r]);
        }
    }
    if (l0.has_kbias != 0u) {
        for (uint r = tid; r < MK_KV_DIM; r += tg_size) {
            kbuf[r] = (half)((float)kbuf[r] + l0.kb[r]);
        }
    }
    if (l0.has_vbias != 0u) {
        for (uint r = tid; r < MK_KV_DIM; r += tg_size) {
            vbuf[r] = (half)((float)vbuf[r] + l0.vb[r]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage D (day-4): RoPE on Q and K (V unrotated).
    //   For each (head, pair j ∈ 0..head_dim/2):
    //     θ = pos / base^(2j/head_dim)
    //     (x[2j], x[2j+1]) ← (x0·cosθ − x1·sinθ, x0·sinθ + x1·cosθ)
    // Q: n_heads × (head_dim/2) = 16 × 64 = 1024 pairs.
    // K: n_kv_heads × (head_dim/2) = 2 × 64 = 128 pairs.
    {
        const uint half_dim = MK_HEAD_DIM / 2u;
        const float pos_f = (float)args.pos;
        const float base = l0.rope_theta;
        const float inv_hd = 1.0f / (float)MK_HEAD_DIM;
        // Q (Q_DIM/2 = 1024 pair-rotations).
        const uint q_pairs = MK_Q_DIM / 2u;
        for (uint idx = tid; idx < q_pairs; idx += tg_size) {
            uint h = idx / half_dim;
            uint j = idx % half_dim;
            float theta = pos_f / pow(base, 2.0f * (float)j * inv_hd);
            float c = cos(theta);
            float s = sin(theta);
            uint off = h * MK_HEAD_DIM + 2u * j;
            float x0 = (float)qbuf[off];
            float x1 = (float)qbuf[off + 1u];
            qbuf[off]      = (half)(x0 * c - x1 * s);
            qbuf[off + 1u] = (half)(x0 * s + x1 * c);
        }
        // K (KV_DIM/2 = 128 pair-rotations).
        const uint k_pairs = MK_KV_DIM / 2u;
        for (uint idx = tid; idx < k_pairs; idx += tg_size) {
            uint h = idx / half_dim;
            uint j = idx % half_dim;
            float theta = pos_f / pow(base, 2.0f * (float)j * inv_hd);
            float c = cos(theta);
            float s = sin(theta);
            uint off = h * MK_HEAD_DIM + 2u * j;
            float x0 = (float)kbuf[off];
            float x1 = (float)kbuf[off + 1u];
            kbuf[off]      = (half)(x0 * c - x1 * s);
            kbuf[off + 1u] = (half)(x0 * s + x1 * c);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage E (day-5): write rotated K and V into DRAM kv_cache at
    // (li=0, slot=args.pos). Layout per layer: row-major
    //   k_cache[li * max_seq * KV_DIM + pos * KV_DIM + r]
    // For layer 0 the layer-offset is 0.
    {
        const uint64_t slot_off = (uint64_t)args.pos * MK_KV_DIM;
        device half* k_slot = k_cache + slot_off;
        device half* v_slot = v_cache + slot_off;
        for (uint r = tid; r < MK_KV_DIM; r += tg_size) {
            k_slot[r] = kbuf[r];
            v_slot[r] = vbuf[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Stage F (day-5): MHA decode for layer 0.
    //   per head h: scores[t] = (q_h · K[t, kv_h]) / √head_dim, t<seq_len
    //               softmax(scores)
    //               attn_h[d] = Σ_t scores[t] · V[t, kv_h, d]
    // group_size = n_heads / n_kv_heads = 8.
    //
    // Per-head serial schedule: all 256 threads cooperate on one head at
    // a time. Cheap parallelism waste (B=1 decode is gap-bound, not
    // compute-bound — see decode_gap_anatomy_2026_05_24.md).
    {
        const uint group_size = MK_N_HEADS / MK_N_KV_HEADS; // 8
        const float scale = 1.0f / sqrt((float)MK_HEAD_DIM);
        const uint seq_len = args.seq_len;
        threadgroup float reduce_partials[8];

        for (uint hh = 0u; hh < MK_N_HEADS; ++hh) {
            uint kv_h = hh / group_size;
            threadgroup half* q_head = qbuf + hh * MK_HEAD_DIM;
            threadgroup half* out_head = attnout + hh * MK_HEAD_DIM;

            // Scores: one thread per position, full-dim dot product per
            // position. Tail threads (tid ≥ seq_len) idle.
            for (uint t = tid; t < seq_len; t += tg_size) {
                device const half* k_t =
                    k_cache + (uint64_t)t * MK_KV_DIM + kv_h * MK_HEAD_DIM;
                float s = 0.0f;
                for (uint i = 0u; i < MK_HEAD_DIM; ++i) {
                    s += (float)q_head[i] * (float)k_t[i];
                }
                scores[t] = (half)(s * scale);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Stable softmax — max over seq_len, then exp/sum, then div.
            float local_max = -INFINITY;
            for (uint t = tid; t < seq_len; t += tg_size) {
                float v = (float)scores[t];
                if (v > local_max) local_max = v;
            }
            float simd_m = simd_max(local_max);
            uint simd_lane  = tid & 31u;
            uint simd_group = tid >> 5u;
            if (simd_lane == 0u) reduce_partials[simd_group] = simd_m;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float gmax = reduce_partials[0];
            for (uint k = 1u; k < 8u; ++k) {
                gmax = max(gmax, reduce_partials[k]);
            }

            float local_sum = 0.0f;
            for (uint t = tid; t < seq_len; t += tg_size) {
                float e = exp((float)scores[t] - gmax);
                scores[t] = (half)e;
                local_sum += e;
            }
            float simd_s = simd_sum(local_sum);
            if (simd_lane == 0u) reduce_partials[simd_group] = simd_s;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float gsum = 0.0f;
            for (uint k = 0u; k < 8u; ++k) gsum += reduce_partials[k];
            float inv_sum = 1.0f / gsum;
            for (uint t = tid; t < seq_len; t += tg_size) {
                scores[t] = (half)((float)scores[t] * inv_sum);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Weighted V sum into shmem attnout for this head.
            for (uint d = tid; d < MK_HEAD_DIM; d += tg_size) {
                float a = 0.0f;
                for (uint t = 0u; t < seq_len; ++t) {
                    device const half* v_t =
                        v_cache + (uint64_t)t * MK_KV_DIM + kv_h * MK_HEAD_DIM;
                    a += (float)scores[t] * (float)v_t[d];
                }
                out_head[d] = (half)a;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // TODO(megakernel-day6+): stages G..L for layer 0, then A..L for layer 1.

    // Terminal probe-write. `args.probe_stage` selects which intermediate
    // is emitted to x_out for parity testing. Dev-only — collapses to a
    // single residual write when full 2-layer parity ships.
    if (args.probe_stage == MK_PROBE_XNORM_A) {
        for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
            x_out[i] = xnorm[i];
        }
    } else if (args.probe_stage == MK_PROBE_Q_ROT) {
        for (uint i = tid; i < MK_Q_DIM; i += tg_size) {
            x_out[i] = qbuf[i];
        }
    } else if (args.probe_stage == MK_PROBE_ATTN_OUT) {
        for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
            x_out[i] = attnout[i];
        }
    } else {
        // Stages F..residual not yet implemented — return zeros so the
        // test fails loud (rather than passing on uninitialised memory).
        for (uint i = tid; i < MK_HIDDEN; i += tg_size) {
            x_out[i] = (half)0.0f;
        }
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
