# G16 — runtime dispatch trace vs NX declared kernel set

Date: 2026-08-18. Worktree `g16-dispatch-trace-20260818-155751` at `3a526bc8e`.
Every set below is **MEASURED** unless tagged SOURCE.

Vehicle: Qwen3.8-27B hybrid decode (`Qwen38HybridDecodeSession::step` →
`generate_greedy`). Probe:
`crates/hawking-core/examples/qwen38_dispatch_trace.rs`.

## Verdict

**declared ≠ dispatched-union.**

The NX genome's 38-name `kernel_binding.dispatched` is a source-literal
intersection, not the set the runtime actually launches. Against a real
token on the two required artifacts:

| claim | result |
|---|---|
| uniform-q4-v1 ⊆ declared 38 | **FAIL** (4 extras) |
| mixed-q3mlp-q3attn-v1 ⊆ declared 38 | **FAIL** (same 4 extras) |
| union(artifacts) == declared 38 | **FAIL** (18 vs 38) |
| union(artifacts) == codec-relevant declared | **FAIL** (18 vs 14) |

The 4 extras are real Metal kernels dispatched on every token of both
artifacts. They are invisible to `./tools/nx_genome.py --seal` because
their names are not string literals in `qwen38_hybrid_decode.rs`.

24 of the 38 declared names never fired on these two artifacts under
default env. They are named dead-declared below.

## How the runtime set was derived

`MetalContext::new_with_trace` + `drain_trace` already exist. Two facts
blocked using `drain_trace` as the sole witness:

1. **Clone sharing is already correct (SOURCE).** `MetalContext` is
   `#[derive(Clone)]` with `trace: Arc<DispatchTrace>`. Session attach
   does `context: weights.context.clone()` and now asserts
   `Arc::ptr_eq` on that buffer. The probe completed, so the assert
   held. `crates/hawking-core/src/metal/mod.rs` was not edited.
2. **TCB Off does not write kernel names into that buffer (SOURCE).**
   `TokenCommandBuffer::dispatch_threads` only records
   `DispatchSample`s when `HAWKING_TCB_TRACE` is cpu/gpu/gpu_prod.
   Default decode (`HAWKING_TCB_TRACE` unset/`0`) commits without
   flushing names. Enabling `HAWKING_TCB_TRACE=cpu` would populate
   `drain_trace`, but `static_kernel_name` remaps unknown labels to
   `"other"`. Nine of the 38 declared names are missing from that
   match (including every q3 / hgravu-q4 / `*_tg` name this mixed
   artifact actually launches). A cpu-trace dump would have reported
   `"other"` instead of those kernels.

So the session, when `HAWKING_TRACE_DISPATCH=1`, enables the existing
TCB structural kernel trace (`enable_structural_kernel_trace`) and
harvests the exact `dispatch_threads` labels. That is the runtime set:
the string the GPU pipeline is bound to, recorded at encode time.
Default (`HAWKING_TRACE_DISPATCH` unset) does not enable it;
`MetalContext::new_with_trace(false)` is the historical constructor.

The probe forces `HAWKING_TRACE_DISPATCH=1` and `HAWKING_TCB_TRACE=0`
so the structural path is the one that runs.

## Probe identity

| | uniform-q4-v1 | mixed-q3mlp-q3attn-v1 |
|---|---|---|
| artifact | `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1` | `.../mixed-q3mlp-q3attn-v1` |
| tokenizer | `.../qwen38-27b/bf16/tokenizer.json` | same |
| prompt | chat-templated `Say hi.` (11 ids) | same |
| max-new | 4 | 4 |
| new token ids | `[248068, 198, 760, 1156]` | `[248068, 198, 760, 1156]` |
| fallbacks | 0 | 0 |
| load | 755 catalog tensors (HQ30UQ4 + f32v2) | 851 HQ38M20 rows; census `binary=0 residual=0 hgravs=0 uniform=498 q4=0 f32=353` |

Both runs: `HAWKING_RMSNORM_TG` default 1024, `HAWKING_DN_RMSNORM_TG`
default 256, `HAWKING_ROPE_TG` default 256, `HAWKING_DN_VI_SIMD`
default on, `deltanet_vi_parallel=true`, `HAWKING_ARGMAX_TWO_PASS`
unset, `HAWKING_QWEN38_RECON_FUSE` default on.

## Runtime-dispatched sets (MEASURED)

### uniform-q4-v1 (13)

```
gk_swiglu_f32
mha_decode_f32
qwen38_attention_apply_sigmoid_gate
qwen38_gated_delta_decode_vi_simd
qwen38_gqa_qk_norm_rope_cache_tg
qwen38_qkvz_rearrange_conv_l2_f32
qwen80_ba_to_decay_beta_f32
qwen80_deltanet_gated_rmsnorm_tg
qwen80_residual_rmsnorm_tg
qwen_next_add_residual
qwen_uniform_q4_embedding_lookup
qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
sample_argmax_f32
```

### mixed-q3mlp-q3attn-v1 (16)

```
gk_swiglu_f32
mha_decode_f32
qwen38_attention_apply_sigmoid_gate
qwen38_fuse_split_ba_f32
qwen38_fuse_split_qkvz_f32
qwen38_gated_delta_decode_vi_simd
qwen38_gqa_qk_norm_rope_cache_tg
qwen38_hgravu_embedding_lookup
qwen38_qkvz_rearrange_conv_l2_f32
qwen80_ba_to_decay_beta_f32
qwen80_deltanet_gated_rmsnorm_tg
qwen80_residual_rmsnorm_tg
qwen_next_add_residual
qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
sample_argmax_f32
```

### union (18)

The 11 shared names plus:

- uniform-only: `qwen_uniform_q4_embedding_lookup`,
  `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
- mixed-only: `qwen38_fuse_split_ba_f32`, `qwen38_fuse_split_qkvz_f32`,
  `qwen38_hgravu_embedding_lookup`,
  `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128`,
  `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128`

Mixed fires the split-fuse pair because this pack keeps split
`in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a` rather than
fused `in_proj_qkvz` / `in_proj_ba`. Embed is HGRAVU01
(`qwen38_hgravu_embedding_lookup`). GEMVs are HGRAVU01 q3 (attention +
MLP) and HGRAVU01 q4 (embed/lm_head leftovers from mixed-2p0).

## NX declared set (MEASURED from `./tools/nx_genome.py --seal --out /tmp/nx16.json`)

`kernel_binding.count = 38` of `declared_in_tree = 554`.
Extraction (SOURCE): string literals in
`crates/hawking-core/src/model/qwen38_hybrid_decode.rs` intersected
with `kernel void` names under `crates/hawking-core/shaders`.

```
q80_binary_group_csr_matvec_bytes
q80_binary_group_csr_matvec_tg256
q80_binary_group_matvec_simd_bytes
q80_binary_group_matvec_tg256
q80_hgravs01_factor_matvec_simd
q80_hgravs01_factor_matvec_simd3
q80_sparse_q1_apply_csr
q80_uniform8_matvec_simd_bytes
q80_uniform8_matvec_tg256
qwen38_attention_apply_sigmoid_gate
qwen38_f32_stream_probe
qwen38_fuse_split_ba_f32
qwen38_fuse_split_qkvz_f32
qwen38_gated_delta_decode_vi
qwen38_gated_delta_decode_vi_simd
qwen38_gqa_qk_norm_rope_cache_f32
qwen38_gqa_qk_norm_rope_cache_tg
qwen38_hgravu_embedding_lookup
qwen38_qkvz_rearrange_conv_l2_f32
qwen80_ba_to_decay_beta_f32
qwen80_deltanet_gated_rmsnorm_f32
qwen80_deltanet_gated_rmsnorm_tg
qwen80_gated_delta_decode_tg
qwen80_residual_rmsnorm_f32
qwen80_residual_rmsnorm_tg
qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
qwen_uniform_q4_embedding_lookup
qwen_uniform_q4_group128_matvec_geo_tpr64_tg128
qwen_uniform_q4_group64_matvec
qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe
qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe
qwen_uniform_q4_group64_matvec_vecgroup
qwen_uniform_q4_group64_matvec_vecgroup_r4
qwen_uniform_q4_group64_matvec_vecgroup_x64
sample_argmax_f32_pass1
sample_argmax_f32_pass2
```

Sealed for Apple M3 Ultra, 60 GPU cores, 103.1 GB, Metal 4.
genome digest `c61afb5cce7ba294cd4bc3b6c19aeba4`.

## Set difference both directions

### dispatched-union − declared (4) — NX miss

These ran on **both** artifacts. They are not in the seal because the
name is not a string literal in `qwen38_hybrid_decode.rs`:

| kernel | call site |
|---|---|
| `gk_swiglu_f32` | `decode_family::swiglu_f32()` |
| `mha_decode_f32` | `mha_decode_f32_tcb` in `kernels/mod.rs` |
| `qwen_next_add_residual` | `qwen_next_add_residual_tcb` |
| `sample_argmax_f32` | `sample_argmax_f32_tcb` (default argmax). Decode.rs only literals are `sample_argmax_f32_pass1` / `pass2`, gated on `HAWKING_ARGMAX_TWO_PASS=1` |

### declared − dispatched-union (24) — dead-declared on this pair

**Other codecs, not present on either artifact** (mixed census
`binary=0 residual=0 hgravs=0`; uniform is HQ30UQ4 only). They would
fire on a pack that actually carries HGRAVB / HGRAVR / HGRAVS /
uniform-q8.

- `q80_binary_group_csr_matvec_bytes`
- `q80_binary_group_csr_matvec_tg256`
- `q80_binary_group_matvec_simd_bytes`
- `q80_binary_group_matvec_tg256`
- `q80_hgravs01_factor_matvec_simd`
- `q80_hgravs01_factor_matvec_simd3`
- `q80_sparse_q1_apply_csr`
- `q80_uniform8_matvec_simd_bytes`
- `q80_uniform8_matvec_tg256`

**Diagnostic / unused launch retargets** (literals exist so a probe
can retarget `matvec_kernel`; default is `GeoTpr64Tg128` / no probe):

- `qwen38_f32_stream_probe`
- `qwen_uniform_q4_group64_matvec`
- `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe`
- `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe`
- `qwen_uniform_q4_group64_matvec_vecgroup`
- `qwen_uniform_q4_group64_matvec_vecgroup_r4`
- `qwen_uniform_q4_group64_matvec_vecgroup_x64`
- `qwen_uniform_q4_group128_matvec_geo_tpr64_tg128` (group-128; neither artifact uses it)

**Default-off siblings of kernels that did fire** (same operator,
other env / flag):

- `qwen38_gated_delta_decode_vi` — `HAWKING_DN_VI_SIMD=0`; default is `_vi_simd`
- `qwen38_gqa_qk_norm_rope_cache_f32` — `HAWKING_ROPE_TG=0`; default is `_tg`
- `qwen80_deltanet_gated_rmsnorm_f32` — `HAWKING_DN_RMSNORM_TG=0`; default is `_tg`
- `qwen80_residual_rmsnorm_f32` — `HAWKING_RMSNORM_TG=0`; default is `_tg`
- `qwen80_gated_delta_decode_tg` — `deltanet_vi_parallel=false`
- `sample_argmax_f32_pass1`, `sample_argmax_f32_pass2` — `HAWKING_ARGMAX_TWO_PASS=1`

### declared ∩ union (14) — codec-relevant declared that actually ran

```
qwen38_attention_apply_sigmoid_gate
qwen38_fuse_split_ba_f32
qwen38_fuse_split_qkvz_f32
qwen38_gated_delta_decode_vi_simd
qwen38_gqa_qk_norm_rope_cache_tg
qwen38_hgravu_embedding_lookup
qwen38_qkvz_rearrange_conv_l2_f32
qwen80_ba_to_decay_beta_f32
qwen80_deltanet_gated_rmsnorm_tg
qwen80_residual_rmsnorm_tg
qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
qwen_uniform_q4_embedding_lookup
qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
```

"Codec-relevant declared" for this pair = those 14. The runtime union
is those 14 plus the 4 NX-invisible helpers. Equality fails by exactly
those 4.

## Default (trace-off) path

`qwen38_trace_dispatch_enabled()` is `env_on("HAWKING_TRACE_DISPATCH")`
(`== "1"` only). Unset → `MetalContext::new_with_trace(false)`, which
is what `MetalContext::new()` already called. Structural trace stays
off; `seen_kernels` is `None`; `step` does not allocate a name set.
`ascension_qwen38_hybrid_greedy` still compiles
(`release-fast`, 0.07 s incremental).

## What this does not claim

- Not a numeric / greedy-id authority. The matching token ids across
  artifacts are noted, not sealed.
- Not a census of every mixed codec in the tree. binary / residual /
  hgravs kernels are dead on *these two* artifacts; they are live
  source binds for other HQ38M20 packs.
- Not a fix to `nx_genome.py`. The seal still claims "kernels ACTUALLY
  dispatched". This probe shows that claim is false by 4 misses and
  24 over-claims on the default G0 + q3 vehicle.
