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

// ── mha_decode_flash_f16kv (long-context decode-at-depth) ────────────────────
// Wave-R6: byte-for-byte clone of mha_decode_flash_f32 with the K/V cache typed
// `half*` (widened to float at read), exactly as mha_decode_f16kv does. This is
// the ONLY attention kernel that BOTH (a) runs at 32K — flash online-softmax uses
// CONSTANT threadgroup memory, whereas standalone mha_decode_f16kv allocates
// O(seq) scores shmem and exceeds 32 KB past ~7800 tokens — AND (b) halves the
// dominant per-token KV byte stream at depth. q/out/acc/scores/state stay f32
// (the residual is never f16 — recorded Type-1 kill). Numerically equals
// mha_decode_f16kv on the same f16 cache up to the online-softmax reorder
// (gate atol 1e-3 + rtol 1e-4, NOT the strict 1e-4). Opt-in DISMANTLE_QWEN_FLASH_F16KV
// (rides the F16_KV cache machinery; only the decode kernel changes).
kernel void mha_decode_flash_f16kv(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const half*        k_cache[[buffer(2)]],
    device const half*        v_cache[[buffer(3)]],
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

    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) acc[i] = 0.0f;
    if (tid == 0u) { state[0] = -INFINITY; state[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_tiles = (SEQ + MHA_FLASH_TG - 1u) / MHA_FLASH_TG;

    for (uint tile = 0u; tile < n_tiles; ++tile) {
        const uint t_base = tile * MHA_FLASH_TG;
        const uint t_len  = min(MHA_FLASH_TG, SEQ - t_base);
        const uint t      = t_base + tid;

        float s_local = -INFINITY;
        if (tid < t_len) {
            device const half* kt = k_cache + (t * NKV + kv_h) * H_DIM;
            float s = 0.0f;
            for (uint i = 0u; i < H_DIM; ++i) s += q_h[i] * (float)kt[i];
            s_local = s * scale;
        }
        scores_tile[tid] = s_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float simd_mx = simd_max(s_local);
        if (simd_lane == 0u) state[4u + simd_id] = simd_mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        const float corr_bc = state[2];
        const float m_bc    = state[3];
        for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
            float a = acc[i] * corr_bc;
            for (uint ti = 0u; ti < t_len; ++ti) {
                float w = exp(scores_tile[ti] - m_bc);
                device const half* vt =
                    v_cache + ((t_base + ti) * NKV + kv_h) * H_DIM;
                a += w * (float)vt[i];
            }
            acc[i] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = 1.0f / state[1];
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
        out_h[i] = acc[i] * inv_l;
    }
}

// ── int4 (per-row symmetric) KV cache — Track 5.3 / silicon #15 ──────────────
// Footprint: 64 packed bytes + 1 f16 scale per (token,kv_head) row of head_dim=128
// (~4× vs f32, 2× vs f16-KV); long-context enabler. NOT bit-identical — symmetric
// int4 [-7,7], one f16 scale/row. Quant math (BOTH kernels must agree exactly):
//   scale = max|row| / 7  (1.0 if max==0);  q = clamp(rint(x/scale), -7, 7);
//   nibble u = q & 0xF (two's-complement); byte j = u_{2j} | (u_{2j+1} << 4);
//   dequant: s = sign_extend4(u);  x ≈ s * scale.
struct ArgbufKvQuantInt4 {
    uint kv_dim;        // n_kv_heads * head_dim (elements in the per-token K/V slice)
    uint head_dim;      // per-row width (128)
    uint dst_row_base;  // first ROW index for this token: (layer*max_seq + seq_slot)*n_kv_heads
};

// One threadgroup per (row, K|V). grid = (n_kv_heads, 2, 1); tg = head_dim threads.
// Each thread owns one element; tree-reduce row max|x|; thread 0 writes the f16
// scale; threads cooperatively pack 2 nibbles/byte.
kernel void kv_quant_int4_append(
    device const float* src_k        [[buffer(0)]],
    device const float* src_v        [[buffer(1)]],
    device       uchar* k_packed     [[buffer(2)]],
    device       half*  k_scales     [[buffer(3)]],
    device       uchar* v_packed     [[buffer(4)]],
    device       half*  v_scales     [[buffer(5)]],
    constant ArgbufKvQuantInt4& args [[buffer(6)]],
    threadgroup float* red           [[threadgroup(0)]],   // head_dim floats
    uint3 tg_id  [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tsz3   [[threads_per_threadgroup]])
{
    // Metal requires position attributes to be all-vector or all-scalar; the 2D
    // grid (rows, K|V) needs tg_id.y, so all three are uint3 (index .x for the 1D
    // thread axis).
    const uint tid  = tid3.x;
    const uint tsz  = tsz3.x;
    const uint HD   = args.head_dim;
    const uint kvh  = tg_id.x;            // which kv-head row within this token
    const bool isV  = (tg_id.y == 1u);
    device const float* src = (isV ? src_v : src_k) + kvh * HD;
    device uchar* packed    = (isV ? v_packed : k_packed);
    device half*  scales    = (isV ? v_scales : k_scales);
    const uint row = args.dst_row_base + kvh;

    // 1. load element, reduce row max|x|.
    float x = (tid < HD) ? src[tid] : 0.0f;
    red[tid] = fabs(x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tsz / 2u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxabs = red[0];
    float scale  = (maxabs > 0.0f) ? (maxabs / 7.0f) : 1.0f;
    if (tid == 0u) scales[row] = (half)scale;

    // 2. quantize this thread's element → 4-bit two's-complement nibble.
    int q = (int)rint(x / scale);
    q = clamp(q, -7, 7);
    uint u = (uint)(q & 0xF);

    // 3. pack 2 nibbles/byte. Stash nibble in `red` (exact for 0..15), then the
    //    even thread of each pair writes the byte (reads its odd partner).
    threadgroup_barrier(mem_flags::mem_threadgroup);   // all done reading red[0]
    red[tid] = (float)u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 1u) == 0u && tid < HD) {
        uint lo = (uint)red[tid];
        uint hi = (uint)red[tid + 1u];
        packed[row * (HD / 2u) + (tid >> 1)] = (uchar)((lo & 0xF) | ((hi & 0xF) << 4));
    }
}

// mha_decode_flash_int4kv — flash online-softmax decode reading an int4 K/V cache.
// Identical constant-shmem structure to mha_decode_flash_f16kv (runs at 32K), but
// each K/V row is 64 packed bytes + one f16 scale, dequantized in-register.
// q/out/acc/scores/state stay f32. Reuses ArgbufMhaDecode.
kernel void mha_decode_flash_int4kv(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*  q           [[buffer(1)]],
    device const uchar*  k_packed    [[buffer(2)]],
    device const half*   k_scales    [[buffer(3)]],
    device const uchar*  v_packed    [[buffer(4)]],
    device const half*   v_scales    [[buffer(5)]],
    device       float*  out         [[buffer(6)]],
    threadgroup  float*  acc         [[threadgroup(0)]],
    threadgroup  float*  scores_tile [[threadgroup(1)]],
    threadgroup  float*  state       [[threadgroup(2)]],
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
    const uint HALF   = H_DIM / 2u;             // packed bytes per row

    device const float* q_h = q + h * H_DIM;

    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) acc[i] = 0.0f;
    if (tid == 0u) { state[0] = -INFINITY; state[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_tiles = (SEQ + MHA_FLASH_TG - 1u) / MHA_FLASH_TG;
    for (uint tile = 0u; tile < n_tiles; ++tile) {
        const uint t_base = tile * MHA_FLASH_TG;
        const uint t_len  = min(MHA_FLASH_TG, SEQ - t_base);
        const uint t      = t_base + tid;

        float s_local = -INFINITY;
        if (tid < t_len) {
            const uint row = t * NKV + kv_h;
            device const uchar* kp = k_packed + (uint64_t)row * HALF;
            float ks = (float)k_scales[row];
            float s = 0.0f;
            for (uint b = 0u; b < HALF; ++b) {
                uchar byte = kp[b];
                int lo = ((int)((byte & 0x0F) << 28)) >> 28;   // sign-extend nibble
                int hi = ((int)((byte & 0xF0) << 24)) >> 28;
                s += q_h[2u * b] * ((float)lo * ks) + q_h[2u * b + 1u] * ((float)hi * ks);
            }
            s_local = s * scale;
        }
        scores_tile[tid] = s_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float simd_mx = simd_max(s_local);
        if (simd_lane == 0u) state[4u + simd_id] = simd_mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        const float corr_bc = state[2];
        const float m_bc    = state[3];
        for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
            float a = acc[i] * corr_bc;
            const uint byte_i = i >> 1;
            const bool hi_nib = (i & 1u) != 0u;
            for (uint ti = 0u; ti < t_len; ++ti) {
                float w = exp(scores_tile[ti] - m_bc);
                const uint row = (t_base + ti) * NKV + kv_h;
                uchar byte = v_packed[(uint64_t)row * HALF + byte_i];
                int nib = hi_nib ? (((int)((byte & 0xF0) << 24)) >> 28)
                                 : (((int)((byte & 0x0F) << 28)) >> 28);
                a += w * ((float)nib * (float)v_scales[row]);
            }
            acc[i] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = 1.0f / state[1];
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
        out_h[i] = acc[i] * inv_l;
    }
}

// ── int4 PER-CHANNEL KV cache (#15 redesign) ────────────────────────────────
// The per-ROW scheme above collapses on real K/V (a few post-RoPE channels
// dominate max|row| → rest round to ~0). PER-CHANNEL gives each head_dim channel
// c its OWN fixed scale s[layer,kvh,c] = max_t|x[t,c]| / 7, calibrated over the
// prompt (running max via kv_int4_calib_max as prefill streams tokens; host
// finalizes /7 at the prefill→decode boundary). Same packing/sign-extend/flash
// structure as the per-row kernels — only the scale index changes (per-channel,
// not per-row). dead_levers #15 measured per-channel int4 at cosine 0.998.
struct ArgbufKvInt4Calib {
    uint head_dim;       // channels per (kvh) row (128)
    uint scale_row_base; // first CHANNEL index for this layer: layer*n_kv_heads*head_dim
};
// Running-max fold of ONE token into the per-channel scale table. Each thread c
// touches a DISTINCT slot (no race); host runs it once per prefill token.
// grid=(n_kv_heads*head_dim,2,1)  tg=(head_dim,1,1). Table must be ZEROED first
// (new_buffer is not zero-initialized).
kernel void kv_int4_calib_max(
    device const float* src_k         [[buffer(0)]],
    device const float* src_v         [[buffer(1)]],
    device       half*  k_chan_scales [[buffer(2)]],
    device       half*  v_chan_scales [[buffer(3)]],
    constant ArgbufKvInt4Calib& args  [[buffer(4)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]])
{
    const uint c   = tid3.x;
    const uint HD  = args.head_dim;
    if (c >= HD) return;
    const uint kvh = tg_id.x;
    const bool isV = (tg_id.y == 1u);
    device const float* src = (isV ? src_v : src_k) + kvh * HD;
    device half*  scales    = (isV ? v_chan_scales : k_chan_scales);
    const uint slot = args.scale_row_base + kvh * HD + c;
    float a   = fabs(src[c]);
    float cur = (float)scales[slot];
    scales[slot] = (half)max(cur, a);
}

// Per-channel int4 append: identical packing to kv_quant_int4_append, but reads
// the FIXED per-channel scale s_c (no row reduction). grid=(n_kv_heads*head_dim,2,1)
// tg=(head_dim,1,1); threadgroup `red` (head_dim floats) carries nibbles for pack.
struct ArgbufKvQuantInt4PC {
    uint head_dim;       // 128
    uint dst_row_base;   // first ROW for this token: (layer*max_seq+slot)*n_kv_heads
    uint scale_row_base; // first CHANNEL for this layer: layer*n_kv_heads*head_dim
};
kernel void kv_quant_int4_append_pc(
    device const float* src_k         [[buffer(0)]],
    device const float* src_v         [[buffer(1)]],
    device       uchar* k_packed      [[buffer(2)]],
    device const half*  k_chan_scales [[buffer(3)]],
    device       uchar* v_packed      [[buffer(4)]],
    device const half*  v_chan_scales [[buffer(5)]],
    constant ArgbufKvQuantInt4PC& args[[buffer(6)]],
    threadgroup float* red            [[threadgroup(0)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]])
{
    const uint c   = tid3.x;
    const uint HD  = args.head_dim;
    const uint kvh = tg_id.x;
    const bool isV = (tg_id.y == 1u);
    device const float* src    = (isV ? src_v : src_k) + kvh * HD;
    device uchar*      packed   = (isV ? v_packed : k_packed);
    device const half* scales   = (isV ? v_chan_scales : k_chan_scales);
    const uint row  = args.dst_row_base + kvh;
    const uint cbas = args.scale_row_base + kvh * HD;

    float x  = (c < HD) ? src[c] : 0.0f;
    float sc = (c < HD) ? (float)scales[cbas + c] : 1.0f;
    sc = (sc > 0.0f) ? sc : 1.0f;
    int q = (int)rint(x / sc);
    q = clamp(q, -7, 7);
    uint u = (uint)(q & 0xF);

    red[c] = (float)u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((c & 1u) == 0u && c < HD) {
        uint lo = (uint)red[c];
        uint hi = (uint)red[c + 1u];
        packed[row * (HD / 2u) + (c >> 1)] = (uchar)((lo & 0xF) | ((hi & 0xF) << 4));
    }
}

// Per-channel int4 flash decode: clone of mha_decode_flash_int4kv, but each
// nibble is multiplied by its CHANNEL's fixed scale (k/v_chan_scales[cbas+chan])
// instead of one row scale. scale_row_base passed as a scalar buffer(7).
kernel void mha_decode_flash_int4kv_pc(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*  q           [[buffer(1)]],
    device const uchar*  k_packed    [[buffer(2)]],
    device const half*   k_chan_scales [[buffer(3)]],
    device const uchar*  v_packed    [[buffer(4)]],
    device const half*   v_chan_scales [[buffer(5)]],
    device       float*  out         [[buffer(6)]],
    constant uint&       scale_row_base [[buffer(7)]],
    threadgroup  float*  acc         [[threadgroup(0)]],
    threadgroup  float*  scores_tile [[threadgroup(1)]],
    threadgroup  float*  state       [[threadgroup(2)]],
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
    const uint HALF   = H_DIM / 2u;
    const uint cbas   = scale_row_base + kv_h * H_DIM;

    device const float* q_h = q + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) acc[i] = 0.0f;
    if (tid == 0u) { state[0] = -INFINITY; state[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_tiles = (SEQ + MHA_FLASH_TG - 1u) / MHA_FLASH_TG;
    for (uint tile = 0u; tile < n_tiles; ++tile) {
        const uint t_base = tile * MHA_FLASH_TG;
        const uint t_len  = min(MHA_FLASH_TG, SEQ - t_base);
        const uint t      = t_base + tid;

        float s_local = -INFINITY;
        if (tid < t_len) {
            const uint row = t * NKV + kv_h;
            device const uchar* kp = k_packed + (uint64_t)row * HALF;
            float s = 0.0f;
            for (uint b = 0u; b < HALF; ++b) {
                uchar byte = kp[b];
                int lo = ((int)((byte & 0x0F) << 28)) >> 28;
                int hi = ((int)((byte & 0xF0) << 24)) >> 28;
                float slo = (float)k_chan_scales[cbas + 2u * b];
                float shi = (float)k_chan_scales[cbas + 2u * b + 1u];
                s += q_h[2u * b] * ((float)lo * slo) + q_h[2u * b + 1u] * ((float)hi * shi);
            }
            s_local = s * scale;
        }
        scores_tile[tid] = s_local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float simd_mx = simd_max(s_local);
        if (simd_lane == 0u) state[4u + simd_id] = simd_mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tile_max = max(max(state[4], state[5]), max(state[6], state[7]));
            float m_old = state[0];
            float m_new = max(m_old, tile_max);
            float corr  = exp(m_old - m_new);
            float tile_sum = 0.0f;
            for (uint ti = 0u; ti < t_len; ++ti) tile_sum += exp(scores_tile[ti] - m_new);
            state[0] = m_new; state[1] = state[1] * corr + tile_sum;
            state[2] = corr;  state[3] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float corr_bc = state[2];
        const float m_bc    = state[3];
        for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) {
            float a = acc[i] * corr_bc;
            const uint byte_i = i >> 1;
            const bool hi_nib = (i & 1u) != 0u;
            float vsc = (float)v_chan_scales[cbas + i];
            for (uint ti = 0u; ti < t_len; ++ti) {
                float w = exp(scores_tile[ti] - m_bc);
                const uint row = (t_base + ti) * NKV + kv_h;
                uchar byte = v_packed[(uint64_t)row * HALF + byte_i];
                int nib = hi_nib ? (((int)((byte & 0xF0) << 24)) >> 28)
                                 : (((int)((byte & 0x0F) << 28)) >> 28);
                a += w * ((float)nib * vsc);
            }
            acc[i] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_l = 1.0f / state[1];
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += MHA_FLASH_TG) out_h[i] = acc[i] * inv_l;
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
