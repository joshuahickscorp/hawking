// Decode-step multi-head attention for Qwen-class GQA models.
//
// Computes attention for ONE new token across all heads in a single
// dispatch. Each threadgroup handles one query head; KV head is derived
// via `h / group_size` where `group_size = n_heads / n_kv_heads`.
//
// Layout:
//   - grid: (n_heads, 1, 1)  -- one TG per head
//   - tg size: 64 (power-of-2; required by tree reductions below)
//
// Buffers:
//   0 args  : ArgbufMhaDecode
//   1 q     : (n_heads, head_dim)                f32
//   2 k     : (seq_len, n_kv_heads, head_dim)    f32
//   3 v     : (seq_len, n_kv_heads, head_dim)    f32
//   4 out   : (n_heads, head_dim)                f32
//
// Threadgroup memory layout:
//   shmem[0 .. seq_len]              -- scores buffer (reused for exp(score))
//   shmem[seq_len .. seq_len+tg]     -- reduction scratch (max, then sum)
//
// Capacity: seq_len * 4B + tg_size * 4B threadgroup memory.
// At tg_size=64 and 32 KB shared-mem ceiling, scores cap ~8000 tokens.
// Bench targets (64-256 token generation) fit comfortably.

struct ArgbufMhaDecode {
    uint seq_len;
    uint head_dim;
    uint n_kv_heads;
    uint group_size;     // n_heads / n_kv_heads
    float scale;         // 1 / sqrt(head_dim)
};

kernel void mha_decode_f32(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const float*       k_cache[[buffer(2)]],
    device const float*       v_cache[[buffer(3)]],
    device       float*       out    [[buffer(4)]],
    threadgroup  float*       shmem  [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint h      = tg_id;
    const uint H_DIM  = args.head_dim;
    const uint SEQ    = args.seq_len;
    const uint NKV    = args.n_kv_heads;
    const uint GROUP  = args.group_size;
    const uint kv_h   = h / GROUP;
    const float scale = args.scale;

    threadgroup float* scores = shmem;
    threadgroup float* red    = shmem + SEQ;

    device const float* q_h = q + h * H_DIM;

    // Phase 1: scores[t] = dot(q_h, K[t, kv_h]) * scale
    for (uint t = tid; t < SEQ; t += tg_size) {
        device const float* kt = k_cache + (t * NKV + kv_h) * H_DIM;
        float acc = 0.0f;
        for (uint i = 0; i < H_DIM; ++i) {
            acc += q_h[i] * kt[i];
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce max(scores[0..SEQ])
    float local_max = -INFINITY;
    for (uint t = tid; t < SEQ; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            red[tid] = max(red[tid], red[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: scores[t] = exp(scores[t] - max); reduce sum
    float local_sum = 0.0f;
    for (uint t = tid; t < SEQ; t += tg_size) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out[h, i] = inv_sum * sum_t(scores[t] * V[t, kv_h, i])
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < SEQ; ++t) {
            device const float* vt = v_cache + (t * NKV + kv_h) * H_DIM;
            acc += scores[t] * vt[i];
        }
        out_h[i] = acc * inv_sum;
    }
}

// ── mha_decode_flash_f32 ──────────────────────────────────────────────────
// Phase 2.3 — GQA flash-style decode attention (online softmax).
//
// Drop-in numerical equivalent of mha_decode_f32 (:34) that does NOT
// materialize scores[seq_len] in threadgroup memory. Instead it streams
// K/V once in tiles of FLASH_TG tokens, carrying a running max (state[0])
// and running sum (state[1]) and rescaling the value accumulator by
// exp(m_old - m_new) per tile (Flash-Attention-v2 online softmax for a
// single-token decode). Threadgroup memory is CONSTANT and
// context-independent — (head_dim + FLASH_TG + 8) floats — so it removes
// the ~7800-token shmem cap that mha_decode_f32 hits at the 32 KB ceiling
// (see :23). This is a GQA re-skin of flash_attn_decode_kernel
// (attn.metal:536) with the MLA latent projection stripped: the score is
// the raw dot(q_h, K[t, kv_h]) and the accumulator is V-weighted over
// head_dim. The online-softmax state[0..7] block is byte-for-byte the MLA
// reference's.
//
// Grid: (n_heads * FLASH_TG, 1, 1)   TG: (FLASH_TG, 1, 1)   one TG / head.
// FLASH_TG = 128 = 4 simdgroups × 32 threads; head_dim=128 (Qwen2.5-3B)
// gives full Phase-4 occupancy AND exactly 4 simdgroups for the state[4..7]
// reduction. GQA kv head: kv_h = h / group_size (as in mha_decode_f32:50).
//
// Buffers: identical ArgbufMhaDecode + (q, k_cache, v_cache, out) surface
// as mha_decode_f32 — only the threadgroup-memory layout differs.
//
// Threadgroup memory slots (host sets sizes at dispatch):
//   slot 0 — acc:         head_dim floats   (running value accumulator)
//   slot 1 — scores_tile: FLASH_TG floats   (current tile's scaled scores)
//   slot 2 — state[8]:    {m_run, l_run, corr, m_bc, simd[0..3]_max}

#ifndef MHA_FLASH_TG
#define MHA_FLASH_TG  128u
#endif
#ifndef MHA_FLASH_NSG
#define MHA_FLASH_NSG 4u
#endif

kernel void mha_decode_flash_f32(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const float*       k_cache[[buffer(2)]],
    device const float*       v_cache[[buffer(3)]],
    device       float*       out    [[buffer(4)]],
    threadgroup  float*       acc         [[threadgroup(0)]],
    threadgroup  float*       scores_tile [[threadgroup(1)]],
    threadgroup  float*       state       [[threadgroup(2)]],
    uint tid       [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    const uint h      = tg_id;
    const uint H_DIM  = args.head_dim;
    const uint SEQ    = args.seq_len;
    const uint NKV    = args.n_kv_heads;
    const uint GROUP  = args.group_size;
    const uint kv_h   = h / GROUP;
    const float scale = args.scale;

    device const float* q_h = q + h * H_DIM;

    // Init: acc = 0, running max = -inf, running sum = 0.
    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) acc[i] = 0.0f;
    if (tid == 0u) { state[0] = -INFINITY; state[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_tiles = (SEQ + MHA_FLASH_TG - 1u) / MHA_FLASH_TG;

    for (uint tile = 0u; tile < n_tiles; ++tile) {
        const uint t_base = tile * MHA_FLASH_TG;
        const uint t_len  = min(MHA_FLASH_TG, SEQ - t_base);
        const uint t      = t_base + tid;

        // 1. Score for this thread's token (one token per thread).
        //    Padding lanes (tid >= t_len) keep -INFINITY so simd_max over a
        //    full 32-lane simdgroup is correct and they contribute no mass.
        float s_local = -INFINITY;
        if (tid < t_len) {
            device const float* kt = k_cache + (t * NKV + kv_h) * H_DIM;
            float s = 0.0f;
            for (uint i = 0u; i < H_DIM; ++i) s += q_h[i] * kt[i];
            s_local = s * scale;
        }
        scores_tile[tid] = s_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Parallel tile max via simdgroup reduction (4 simdgroups).
        float simd_mx = simd_max(s_local);
        if (simd_lane == 0u) state[4u + simd_id] = simd_mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Thread 0: online-softmax state update (byte-for-byte the MLA
        //    reference, attn.metal:640-652).
        if (tid == 0u) {
            float tile_max = max(max(state[4], state[5]), max(state[6], state[7]));
            float m_old    = state[0];
            float m_new    = max(m_old, tile_max);
            float corr     = exp(m_old - m_new);
            float tile_sum = 0.0f;
            for (uint ti = 0u; ti < t_len; ++ti)
                tile_sum += exp(scores_tile[ti] - m_new);
            state[0] = m_new;
            state[1] = state[1] * corr + tile_sum;
            state[2] = corr;
            state[3] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Rescale acc by the correction, then accumulate this tile's
        //    V-weighted contributions. Each thread owns a head_dim slice.
        const float corr_bc = state[2];
        const float m_bc    = state[3];
        for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
            float a = acc[i] * corr_bc;
            for (uint ti = 0u; ti < t_len; ++ti) {
                float w = exp(scores_tile[ti] - m_bc);
                device const float* vt =
                    v_cache + ((t_base + ti) * NKV + kv_h) * H_DIM;
                a += w * vt[i];
            }
            acc[i] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by the running sum and write out.
    const float inv_l = 1.0f / state[1];
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
        out_h[i] = acc[i] * inv_l;
    }
}

// P3 — Batched MHA decode: B query tokens at consecutive positions
// [p0..p0+B) share the K/V cache. Each TG handles one (head, batch_elem)
// pair via a 2D grid (n_heads, B). Per-batch seq_len is computed as
// p0 + batch_id + 1 so each batch element sees its causal prefix.
//
// Buffer layout:
//   q   : (B, n_heads, head_dim) f32 — B tokens' Q rows contiguous
//   out : (B, n_heads, head_dim) f32 — B output rows contiguous
//   k/v_cache: same as the unbatched kernel (single per-layer window).
//
// Replaces B sequential mha_decode_f32 dispatches; saves (B-1)*n_heads
// TG launches per layer plus encode overhead.

struct ArgbufMhaDecodeBatched {
    uint p0;           // base position; batch_id b sees seq_len = p0 + b + 1
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint group_size;
    float scale;
};

kernel void mha_decode_f32_batched(
    constant ArgbufMhaDecodeBatched& args [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const float*       k_cache[[buffer(2)]],
    device const float*       v_cache[[buffer(3)]],
    device       float*       out    [[buffer(4)]],
    threadgroup  float*       shmem  [[threadgroup(0)]],
    uint3 tg_id     [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_dim    [[threads_per_threadgroup]])
{
    const uint tid       = tid_in_tg.x;
    const uint tg_size   = tg_dim.x;
    const uint h         = tg_id.x;
    const uint batch_id  = tg_id.y;
    const uint H_DIM     = args.head_dim;
    const uint SEQ       = args.p0 + batch_id + 1u;
    const uint NKV       = args.n_kv_heads;
    const uint GROUP     = args.group_size;
    const uint NHEADS    = args.n_heads;
    const uint kv_h      = h / GROUP;
    const float scale    = args.scale;

    threadgroup float* scores = shmem;
    threadgroup float* red    = shmem + SEQ;

    // Q/OUT row strides by (batch, head).
    device const float* q_h = q + (batch_id * NHEADS + h) * H_DIM;

    // Phase 1: scores[t] = dot(q_h, K[t, kv_h]) * scale
    for (uint t = tid; t < SEQ; t += tg_size) {
        device const float* kt = k_cache + (t * NKV + kv_h) * H_DIM;
        float acc = 0.0f;
        for (uint i = 0; i < H_DIM; ++i) {
            acc += q_h[i] * kt[i];
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce max.
    float local_max = -INFINITY;
    for (uint t = tid; t < SEQ; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) red[tid] = max(red[tid], red[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: exp + reduce sum.
    float local_sum = 0.0f;
    for (uint t = tid; t < SEQ; t += tg_size) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out[batch, h, i] = inv_sum * sum_t(scores[t] * V[t, kv_h, i])
    device float* out_h = out + (batch_id * NHEADS + h) * H_DIM;
    for (uint i = tid; i < H_DIM; i += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < SEQ; ++t) {
            device const float* vt = v_cache + (t * NKV + kv_h) * H_DIM;
            acc += scores[t] * vt[i];
        }
        out_h[i] = acc * inv_sum;
    }
}

// ── mha_decode_f32_batched_multiseq (continuous-batching decode) ────────────
// B INDEPENDENT sequences in ONE dispatch. Unlike mha_decode_f32_batched (which
// is B tokens of ONE sequence sharing a single growing K/V window), here each
// batch element `bi` is its own sequence with:
//   - its own position positions[bi]  => SEQ_bi = positions[bi] + 1
//   - its own slot-strided K/V region at element offset bi * kv_slot_stride
// Grid (n_heads*TG, B, 1); q/out laid out (B, n_heads, head_dim). shmem is sized
// to the LARGEST SEQ in the batch; each TG uses only its own SEQ_bi. The four
// softmax phases are byte-identical to mha_decode_f32_batched — only the K/V
// base (per-slot) and SEQ (per-slot position) differ.
struct ArgbufMhaDecodeMultiseq {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint group_size;
    uint kv_slot_stride;   // elements between consecutive slots' K (and V) regions
    float scale;
};

kernel void mha_decode_f32_batched_multiseq(
    constant ArgbufMhaDecodeMultiseq& args [[buffer(0)]],
    device const float*       q         [[buffer(1)]],
    device const float*       k_cache   [[buffer(2)]],
    device const float*       v_cache   [[buffer(3)]],
    device       float*       out       [[buffer(4)]],
    device const uint*        positions [[buffer(5)]],
    device const uint*        regions   [[buffer(6)]],
    threadgroup  float*       shmem     [[threadgroup(0)]],
    uint3 tg_id     [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_dim    [[threads_per_threadgroup]])
{
    const uint tid       = tid_in_tg.x;
    const uint tg_size   = tg_dim.x;
    const uint h         = tg_id.x;
    const uint batch_id  = tg_id.y;
    const uint H_DIM     = args.head_dim;
    const uint SEQ       = positions[batch_id] + 1u;
    const uint NKV       = args.n_kv_heads;
    const uint GROUP     = args.group_size;
    const uint NHEADS    = args.n_heads;
    const uint kv_h      = h / GROUP;
    const float scale    = args.scale;

    // Per-slot K/V base: each sequence's cache lives at its STABLE region
    // (regions[batch_id]), NOT the compacted dispatch index — so a slot keeps
    // its KV history even as the active/ready set shrinks/grows between steps.
    const uint region = regions[batch_id];
    device const float* k_slot = k_cache + region * args.kv_slot_stride;
    device const float* v_slot = v_cache + region * args.kv_slot_stride;

    threadgroup float* scores = shmem;
    threadgroup float* red    = shmem + SEQ;

    device const float* q_h = q + (batch_id * NHEADS + h) * H_DIM;

    // Phase 1: scores[t] = dot(q_h, K_slot[t, kv_h]) * scale
    for (uint t = tid; t < SEQ; t += tg_size) {
        device const float* kt = k_slot + (t * NKV + kv_h) * H_DIM;
        float acc = 0.0f;
        for (uint i = 0; i < H_DIM; ++i) {
            acc += q_h[i] * kt[i];
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce max.
    float local_max = -INFINITY;
    for (uint t = tid; t < SEQ; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) red[tid] = max(red[tid], red[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: exp + reduce sum.
    float local_sum = 0.0f;
    for (uint t = tid; t < SEQ; t += tg_size) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out[batch, h, i] = inv_sum * sum_t(scores[t] * V_slot[t, kv_h, i])
    device float* out_h = out + (batch_id * NHEADS + h) * H_DIM;
    for (uint i = tid; i < H_DIM; i += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < SEQ; ++t) {
            device const float* vt = v_slot + (t * NKV + kv_h) * H_DIM;
            acc += scores[t] * vt[i];
        }
        out_h[i] = acc * inv_sum;
    }
}

// ── f16-KV decode (Phase 2.1-a) ─────────────────────────────────────────────
// Byte-for-byte clone of mha_decode_f32 EXCEPT k_cache/v_cache are `half*` and
// each cached element is widened to float inside the dot loops. Q stays f32
// (tiny: one row), out/scores/reductions stay f32 (the residual is never f16 —
// that is a recorded Type-1 kill). Halves KV traffic + KV footprint at long
// context. Reuses ArgbufMhaDecode (declared above). Default-off lever, reached
// only when DISMANTLE_QWEN_F16_KV=1 routes the dispatch here.
kernel void mha_decode_f16kv(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const half*        k_cache[[buffer(2)]],
    device const half*        v_cache[[buffer(3)]],
    device       float*       out    [[buffer(4)]],
    threadgroup  float*       shmem  [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint h      = tg_id;
    const uint H_DIM  = args.head_dim;
    const uint SEQ    = args.seq_len;
    const uint NKV    = args.n_kv_heads;
    const uint GROUP  = args.group_size;
    const uint kv_h   = h / GROUP;
    const float scale = args.scale;

    threadgroup float* scores = shmem;
    threadgroup float* red    = shmem + SEQ;

    device const float* q_h = q + h * H_DIM;

    // Phase 1: scores[t] = dot(q_h, K[t, kv_h]) * scale  (K widened from half)
    for (uint t = tid; t < SEQ; t += tg_size) {
        device const half* kt = k_cache + (t * NKV + kv_h) * H_DIM;
        float acc = 0.0f;
        for (uint i = 0; i < H_DIM; ++i) {
            acc += q_h[i] * (float)kt[i];
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce max(scores[0..SEQ])
    float local_max = -INFINITY;
    for (uint t = tid; t < SEQ; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            red[tid] = max(red[tid], red[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: scores[t] = exp(scores[t] - max); reduce sum
    float local_sum = 0.0f;
    for (uint t = tid; t < SEQ; t += tg_size) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out[h, i] = inv_sum * sum_t(scores[t] * V[t, kv_h, i])  (V widened)
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < SEQ; ++t) {
            device const half* vt = v_cache + (t * NKV + kv_h) * H_DIM;
            acc += scores[t] * (float)vt[i];
        }
        out_h[i] = acc * inv_sum;
    }
}

// ── f16-KV batched decode (Phase 2.1-a) ─────────────────────────────────────
// Clone of mha_decode_f32_batched with half k/v_cache + (float) widen. Needed
// because the batched-prefill path (forward_tokens_batch_tcb) is the PRODUCER
// of the KV the single-token decode CONSUMES — when DISMANTLE_QWEN_F16_KV=1 it
// must write+read the SAME half cache, else the decode reads f32 garbage from
// f16-prefilled slots. Reuses ArgbufMhaDecodeBatched (declared above). Q/out
// stay f32. Default-off lever.
kernel void mha_decode_f16kv_batched(
    constant ArgbufMhaDecodeBatched& args [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const half*        k_cache[[buffer(2)]],
    device const half*        v_cache[[buffer(3)]],
    device       float*       out    [[buffer(4)]],
    threadgroup  float*       shmem  [[threadgroup(0)]],
    uint3 tg_id     [[threadgroup_position_in_grid]],
    uint3 tid_in_tg [[thread_position_in_threadgroup]],
    uint3 tg_dim    [[threads_per_threadgroup]])
{
    const uint tid       = tid_in_tg.x;
    const uint tg_size   = tg_dim.x;
    const uint h         = tg_id.x;
    const uint batch_id  = tg_id.y;
    const uint H_DIM     = args.head_dim;
    const uint SEQ       = args.p0 + batch_id + 1u;
    const uint NKV       = args.n_kv_heads;
    const uint GROUP     = args.group_size;
    const uint NHEADS    = args.n_heads;
    const uint kv_h      = h / GROUP;
    const float scale    = args.scale;

    threadgroup float* scores = shmem;
    threadgroup float* red    = shmem + SEQ;

    device const float* q_h = q + (batch_id * NHEADS + h) * H_DIM;

    // Phase 1: scores[t] = dot(q_h, K[t, kv_h]) * scale  (K widened from half)
    for (uint t = tid; t < SEQ; t += tg_size) {
        device const half* kt = k_cache + (t * NKV + kv_h) * H_DIM;
        float acc = 0.0f;
        for (uint i = 0; i < H_DIM; ++i) {
            acc += q_h[i] * (float)kt[i];
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce max.
    float local_max = -INFINITY;
    for (uint t = tid; t < SEQ; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) red[tid] = max(red[tid], red[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: exp + reduce sum.
    float local_sum = 0.0f;
    for (uint t = tid; t < SEQ; t += tg_size) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out[batch, h, i] = inv_sum * sum_t(scores[t] * V[t, kv_h, i])  (V widened)
    device float* out_h = out + (batch_id * NHEADS + h) * H_DIM;
    for (uint i = tid; i < H_DIM; i += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < SEQ; ++t) {
            device const half* vt = v_cache + (t * NKV + kv_h) * H_DIM;
            acc += scores[t] * (float)vt[i];
        }
        out_h[i] = acc * inv_sum;
    }
}
