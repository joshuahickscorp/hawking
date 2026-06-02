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
