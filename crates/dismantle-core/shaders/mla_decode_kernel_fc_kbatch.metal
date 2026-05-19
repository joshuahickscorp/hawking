// Path B Stage 2.4 — K-batched MLA decode (compressed KV cache).
//
// Mirrors mla_decode_kernel (attn.metal:48) but processes K queries
// against the SAME (c_kv, k_pe) KV cache in a single dispatch, with
// weight reads (kv_b_proj) and KV-cache reads amortized across the K
// queries inside one threadgroup.
//
// At K=4 the naive extension of the K=1 kernel busts the M3 Pro 32 KB /
// core threadgroup-memory budget because `scores[seq_len]` × K = 64 KB
// at seq_len=4096. To stay within budget we move scores to a device-
// scratch buffer (n_heads × K × seq_len f32), passed in by the
// dispatcher. TG memory stays at:
//   - q_nope_proj_k  : K × kv_lora_rank f32  (K × 2 KB)
//   - c_kv_wt_k      : K × kv_lora_rank f32  (K × 2 KB)
// = 16 KB at K=4 (k_lora_rank=512, V2-Lite), within budget at K ≤ 8.
//
// Flash-style online softmax is a future optimization; for correctness-
// first the explicit-scores approach matches the K=1 kernel's math
// exactly (just K-fold).
//
// At K=1 the kernel is bit-equivalent to mla_decode_kernel by
// construction (the K-fold loops collapse to single iterations).
//
// Requires k_batch ∈ [1, 8]; seq_len ≥ 1.

kernel void mla_decode_kernel_fc_kbatch(
    device const float* q_kbatch         [[buffer(0)]],   // (K, n_heads, qk_nope+qk_rope)
    device const float* c_kv             [[buffer(1)]],   // (seq_len, kv_lora_rank)
    device const float* k_pe             [[buffer(2)]],   // (seq_len, qk_rope_head_dim)
    device const float* kv_b_proj        [[buffer(3)]],   // (n_heads, (qk_nope+v_head)·kv_lora_rank)
    device       float* out_kbatch       [[buffer(4)]],   // (K, n_heads · v_head_dim)
    device       float* scores_scratch   [[buffer(5)]],   // (n_heads, K, seq_len) device scratch
    constant     uint&  n_heads             [[buffer(6)]],
    constant     uint&  qk_nope_head_dim    [[buffer(7)]],
    constant     uint&  qk_rope_head_dim    [[buffer(8)]],
    constant     uint&  v_head_dim          [[buffer(9)]],
    constant     uint&  kv_lora_rank        [[buffer(10)]],
    constant     uint&  seq_len             [[buffer(11)]],
    constant     float& scale               [[buffer(12)]],
    constant     uint&  k_batch             [[buffer(13)]],
    threadgroup  float* q_nope_proj_k     [[threadgroup(0)]],  // K × kv_lora_rank f32
    threadgroup  float* c_kv_wt_k         [[threadgroup(1)]],  // K × kv_lora_rank f32
    uint                tid     [[thread_position_in_threadgroup]],
    uint                gid     [[threadgroup_position_in_grid]],
    uint                tg_size [[threads_per_threadgroup]])
{
    if (gid >= n_heads) return;

    const uint head        = gid;
    const uint q_head_dim  = qk_nope_head_dim + qk_rope_head_dim;

    const uint kv_b_per_head = (qk_nope_head_dim + v_head_dim) * kv_lora_rank;
    device const float* w_uk = kv_b_proj + (uint64_t)head * (uint64_t)kv_b_per_head;
    device const float* w_uv = w_uk + (uint64_t)qk_nope_head_dim * (uint64_t)kv_lora_rank;

    // Scratch slice for this head: (k_batch × seq_len) f32.
    device float* scores_head = scores_scratch
        + (uint64_t)head * (uint64_t)k_batch * (uint64_t)seq_len;

    // ── Phase 0: q_nope_proj_k[kk, r] = Σ_i w_uk[i, r] × q_nope_kbatch[kk, head, i]
    for (uint kk = 0; kk < k_batch; ++kk) {
        device const float* q_kk = q_kbatch
            + (uint64_t)kk * (uint64_t)n_heads * (uint64_t)q_head_dim
            + (uint64_t)head * (uint64_t)q_head_dim;
        device const float* q_nope = q_kk;
        threadgroup float* qp_kk = q_nope_proj_k + kk * kv_lora_rank;

        for (uint r = tid; r < kv_lora_rank; r += tg_size) {
            float acc = 0.0f;
            for (uint i = 0; i < qk_nope_head_dim; ++i) {
                acc += w_uk[i * kv_lora_rank + r] * q_nope[i];
            }
            qp_kk[r] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 1: scores_head[kk, t] = (qp_kk · c_kv[t] + q_rope_kk · k_pe[t]) × scale
    for (uint kk = 0; kk < k_batch; ++kk) {
        device const float* q_kk = q_kbatch
            + (uint64_t)kk * (uint64_t)n_heads * (uint64_t)q_head_dim
            + (uint64_t)head * (uint64_t)q_head_dim;
        device const float* q_rope = q_kk + qk_nope_head_dim;
        threadgroup const float* qp_kk = q_nope_proj_k + kk * kv_lora_rank;
        device float* s_kk = scores_head + (uint64_t)kk * (uint64_t)seq_len;

        for (uint t = tid; t < seq_len; t += tg_size) {
            device const float* c_kv_t = c_kv + (uint64_t)t * (uint64_t)kv_lora_rank;
            device const float* k_pe_t = k_pe + (uint64_t)t * (uint64_t)qk_rope_head_dim;
            float s = 0.0f;
            for (uint r = 0; r < kv_lora_rank; ++r)     s += qp_kk[r]    * c_kv_t[r];
            for (uint r = 0; r < qk_rope_head_dim; ++r) s += q_rope[r]   * k_pe_t[r];
            s_kk[t] = s * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: softmax per kk (serial in thread 0 per kk; matches K=1 path).
    if (tid == 0) {
        for (uint kk = 0; kk < k_batch; ++kk) {
            device float* s_kk = scores_head + (uint64_t)kk * (uint64_t)seq_len;
            float mx = -INFINITY;
            for (uint t = 0; t < seq_len; ++t) if (s_kk[t] > mx) mx = s_kk[t];
            float sum = 0.0f;
            for (uint t = 0; t < seq_len; ++t) {
                s_kk[t] = exp(s_kk[t] - mx);
                sum += s_kk[t];
            }
            float inv = 1.0f / sum;
            for (uint t = 0; t < seq_len; ++t) s_kk[t] *= inv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: c_kv_wt_k[kk, r] = Σ_t s_kk[t] × c_kv[t, r]
    for (uint kk = 0; kk < k_batch; ++kk) {
        device const float* s_kk = scores_head + (uint64_t)kk * (uint64_t)seq_len;
        threadgroup float* cwt_kk = c_kv_wt_k + kk * kv_lora_rank;
        for (uint r = tid; r < kv_lora_rank; r += tg_size) {
            float acc = 0.0f;
            for (uint t = 0; t < seq_len; ++t) {
                acc += s_kk[t] * c_kv[(uint64_t)t * (uint64_t)kv_lora_rank + r];
            }
            cwt_kk[r] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: out_kbatch[kk, head, vi] = Σ_r w_uv[vi, r] × c_kv_wt_k[kk, r]
    for (uint kk = 0; kk < k_batch; ++kk) {
        threadgroup const float* cwt_kk = c_kv_wt_k + kk * kv_lora_rank;
        device float* out_kk = out_kbatch
            + (uint64_t)kk * (uint64_t)n_heads * (uint64_t)v_head_dim
            + (uint64_t)head * (uint64_t)v_head_dim;
        for (uint vi = tid; vi < v_head_dim; vi += tg_size) {
            device const float* w_uv_row = w_uv + (uint64_t)vi * (uint64_t)kv_lora_rank;
            float acc = 0.0f;
            for (uint r = 0; r < kv_lora_rank; ++r) acc += w_uv_row[r] * cwt_kk[r];
            out_kk[vi] = acc;
        }
    }
}
