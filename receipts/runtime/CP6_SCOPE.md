# CP6 — what integration actually costs, counted rather than estimated

CP3 through CP5c answered every physics question a harness can answer. What remains is an
integration, and this counts it instead of calling it "large".

## The obstacle is the workspace, not the kernels

`step()` builds one `TokenCommandBuffer` and calls `encode_full_token` = embed → layers →
terminal. Everything downstream reads and writes `self.workspace.*` buffers sized for ONE
position — `normalized`, `act`, `gate`, `up`, `down`, `hidden`, `first_residual`, `repeated_q`,
`repeated_k`, `conv_v`, `ba`, `rec_out`, `mixer`, `logits`, `sampled`.

A chunked path needs every one of those K-wide, and every organ encoder taught to address them
per position. That is a parallel encode graph, not a patch to the existing one — which is why
the sequential path must be retained rather than converted, exactly as the directive requires.

## Every kernel a real step dispatches, classified

From the runtime's own record (`_ORGAN_BANDWIDTH_raw.json`, `dispatched_kernels_rep0`), 16
distinct kernels:

**Have a multi-position path already — 4**

| kernel | path |
|---|---|
| `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` | `matmul_r4k4`, CP3, 2.474x |
| `qwen_affine_q2_group64_matvec_gate_up_swiglu_...` | affine gate_up `r4k4`, CP4, 2.49x (the non-swiglu pair variant; the swiglu fusion needs its own RxK) |
| `qwen_affine_q2_group32_matvec_geo_tpr64_tg128` | affine `r4k4`, CP4b, 2.27x |
| `qwen38_gated_delta_decode_vi_simd` | serial by nature, one dispatch per position — CP5b proved that is fine and marginally cheaper inside a shared CB |

**Trivially K-parallel — 7.** Elementwise or per-position normalisations. Launching K× the
threads over a K-wide buffer is the whole change:

> **SUPERSEDED 2026-09-04 by `CP6E_RESULT.md`.** That claim is WRONG. The
> multi-position matmuls require INTERLEAVED activations `[col*K+k]` for
> coalescing, so a per-position kernel cannot be rebound at a byte offset --
> offsets address a BLOCKED layout. Each of these needs its own strided
> variant. Stage 1 is 7 kernels, not 7 launch changes. Two are now built and
> bit-identical; with them the MLP half needs no further new kernels.
 `qwen80_add_residual_rmsnorm_tg`,
`qwen80_residual_rmsnorm_tg`, `qwen80_deltanet_gated_rmsnorm_tg`, `qwen80_ba_to_decay_beta_f32`,
`qwen38_attention_apply_sigmoid_gate`, `qwen_uniform_q4_embedding_lookup`, `sample_argmax_f32`
(prefill needs the last only for the final position).

**Need real work — 5**

| kernel | why |
|---|---|
| `mha_decode_f32` | attention over a growing KV; a chunk attends over prior positions AND causally within itself. This is standard chunked attention but it is new code. |
| `qwen38_gqa_qk_norm_rope_cache_tg` | RoPE is position-dependent; K positions need K rotations and K cache appends |
| `qwen38_qkvz_rearrange_conv_l2_f32` | causal conv, kernel 4 — a sliding window that crosses the chunk boundary and needs the previous chunk's tail |
| `qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128` | fused 3-tensor variant; needs its own RxK rather than the generic one |
| `qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128` | same, fused pair |

So: **4 done, 7 mechanical, 5 genuinely new** — of which two (`qkv`, `pair_concat`) are ports of
a kernel family that already exists and three are new algorithms.

## The order that keeps a measurement at every step

1. K-wide workspace plus the 7 mechanical kernels, DeltaNet layers only, attention layers still
   stepping one position at a time. Measures the 48 DeltaNet layers against the retained
   sequential baseline.
2. The two fused q4 RxK ports, which lift DeltaNet `in_proj` off the generic kernel.
3. Chunked attention — RoPE, KV append, `mha`. The 16 GQA layers, 6.5% of the step.
4. The causal conv's cross-chunk tail.

Step 1 alone covers the organs that CP3/CP4/CP4b measured and is the first point at which a
PROMPT WALL exists to compare. Everything before it is harness.

## What this does not change

Nothing here is a measurement. The projections still stand or fall on step 1: the arithmetic in
`CP5_DECOMPOSITION.md` (1.79-1.89x on step GPU ns from the batchable 78.7%) is organ arithmetic,
and this mission has retracted one projection of that shape already. All three kernel families
still have zero call sites in `qwen38_hybrid_decode.rs`.
