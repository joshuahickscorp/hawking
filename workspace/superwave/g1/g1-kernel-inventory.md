# G1 kernel inventory — Qwen3.8 decode token

Lane: 14-kernel-inventory. Base: `2eee9a004`. No GPU run. No inference.
Every number is labeled `SOURCE` (derived from this tree), `RECEIPT` (quoted),
or `UNVERIFIED_CLAIM` (campaign folklore, not re-measured here).

G0 organism vehicle (resident Genesis body): `qwen38-27b/uniform-q4-v1`
via `Qwen38HybridWeights::load` + `Qwen38HybridDecodeSession::step`.
Evidence: `receipts/ascent-2026-08-16/genesis-resident/GENESIS_RESIDENT.json`
(`artifact` = `.../uniform-q4-v1`; stderr `opening Metal + 755 catalog tensors`);
`tools/agentos/genesis_body/src/main.rs` uses `Qwen38HybridDecodeSession`.

`qwen_dense.rs` / generic `Engine` / Q4K / FlashMoE / DSV4F / Q80 MoE waves
do **not** participate in a Qwen3.8 token. Grep of `qwen_dense.rs` + `engine.rs`
for `qwen38|Qwen38|qwen3_5` is empty.

---

## 1. Model geometry (SOURCE)

`crates/hawking-core/src/model/qwen38_geometry.rs`:

| constant | value | lines |
|---|---:|---|
| layers | 64 | 20 |
| DeltaNet / GQA | 48 / 16 | 21–22 |
| GQA rule | `(layer+1)%4==0` → layers 3,7,…,63 | 82–93 |
| hidden | 5120 | 24 |
| intermediate | 17408 | 25 |
| vocab | 248320 | 26 |
| rms eps | 1e-6 | 27 |
| rope θ | 1e7 | 28 |
| linear heads | 16 key / 48 value / vpk=3 / dim 128 | 31–36 |
| GQA | 24 q / 4 kv / dim 256 / rotary 64 | 38–41 |
| QKVZ / BA rows | 16384 / 96 | 47–48 |
| q / kv / o | 12288 / 1024 / 5120×6144 | 49–52 |
| dense | no `num_experts` | 166–169 |
| vision | skipped at pack | `qwen38_pack.rs:1-2` |

G0 pack schema `hawking.ascent.qwen38_language_uniform_q4.v1`:
402 HQ30UQ4 + 353 f32v2 = 755 tensors (`qwen38_pack.rs:27-34`).
In-proj is fused at pack (`fuse_in_proj_qkvz` / `fuse_in_proj_ba`).

Complete physical BPW of this vehicle: `4.252735126866492`
(`qwen38_token_ns_ledger.rs:30` `UNIFORM_Q4_V1_BPW`). Matches the
unverified G0 “~4.2527” claim; this lane did not re-hash the artifact.

---

## 2. Production schedule (SOURCE)

`crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs`.

Per layer = 9 mixer + 6 dense-MLP = 15. Token = 1 embed + 64×15 + 3 terminal
= **964** (`qwen38_token_ns_ledger.rs:56-59`, test `production_dispatch_count_is_964`
at 716–721).

### 2.1 DeltaNet mixer prefix × 48 (layers not 3 mod 4)

| # | kernel | file |
|---|---|---|
| 1 | `qwen80_residual_rmsnorm_f32` | `shaders/qwen80_device_activations.metal:24` |
| 2 | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` QKVZ | `shaders/qwen_uniform_q4.metal:183` |
| 3 | same, BA | same |
| 4 | `qwen38_qkvz_rearrange_conv_l2_f32` | `shaders/qwen38_device_activations.metal:32` |
| 5 | `qwen80_ba_to_decay_beta_f32` | `qwen80_device_activations.metal:155` |
| 6 | `qwen38_gated_delta_decode_vi` | `qwen38_device_activations.metal:199` |
| 7 | `qwen80_deltanet_gated_rmsnorm_f32` | `qwen80_device_activations.metal:179` |
| 8 | geo_tpr64 out_proj | `qwen_uniform_q4.metal:183` |
| 9 | `qwen_next_add_residual` | `shaders/qwen_next.metal:328` |

Default `deltanet_vi_parallel=true` (`qwen38_hybrid_decode.rs:871-875,912`).
If false, #6 becomes `qwen80_gated_delta_decode_tg` (`qwen80_device_activations.metal:207`).

### 2.2 GQA mixer prefix × 16 (layers 3,7,…,63)

| # | kernel | file |
|---|---|---|
| 1 | `qwen80_residual_rmsnorm_f32` | as above |
| 2–4 | geo_tpr64 q / k / v | `qwen_uniform_q4.metal:183` |
| 5 | `qwen38_gqa_qk_norm_rope_cache_f32` | `qwen38_device_activations.metal:108` |
| 6 | `mha_decode_f32` | `shaders/mha.metal:602` |
| 7 | `qwen38_attention_apply_sigmoid_gate` | `qwen38_device_activations.metal:250` |
| 8 | geo_tpr64 o_proj | `qwen_uniform_q4.metal:183` |
| 9 | `qwen_next_add_residual` | as above |

### 2.3 Dense SwiGLU suffix × 64

| # | kernel | file |
|---|---|---|
| 1 | `qwen80_residual_rmsnorm_f32` | post-attn |
| 2–3 | geo_tpr64 gate + up | |
| 4 | `gk_swiglu_f32` (default) or `qwen80_silu_mul_f32` | `gk_family.metal:578` / `qwen80_device_activations.metal:50` |
| 5 | geo_tpr64 down | |
| 6 | `qwen_next_add_residual` | |

SwiGLU name: `decode_family::swiglu_f32()` → `gk_swiglu_f32` unless
`HAWKING_DECODE_FAMILY` is `0/false/off/no` (`decode_family.rs:168-169,103-105`;
`lib.rs:207-214` `env_opt_out` default **on**). Same math as `qwen80_silu_mul_f32`.

### 2.4 Embed + terminal

| when | kernel | file |
|---|---|---|
| token start | `qwen_uniform_q4_embedding_lookup` | `qwen_uniform_q4.metal:589` |
| final | `qwen80_residual_rmsnorm_f32` | |
| final | geo_tpr64 `lm_head` | |
| final | `sample_argmax_f32` | `shaders/sample.metal:48` |

`qwen_uniform_q4_embedding_lookup_device_token` exists (`qwen_uniform_q4.metal:608`)
and is **not** called from `qwen38_hybrid_decode.rs` (grep empty). Next-token id
is a host `u32` `set_bytes` after the previous wait.

---

## 3. Kernel table — G0 uniform-Q4 token

Representation consumed: **HQ30UQ4 group-64 + f32v2 vectors**. Layout
(`qwen_uniform_q4.metal:4-11`, `uniform_q4.rs:4-8,17-20`):

- magic `HQ30UQ4\0`, group 64, 32 code bytes/group
- nibble `q = nibble-8` ∈ [-8,7]; one IEEE f16 scale/group
- even local → low nibble, odd → high
- activations / KV / DeltaNet state: f32 device buffers
- norms, conv1d, A_log, dt_bias, q/k_norm: f32 (MLX residual-norm stored as
  delta-from-one at pack, `qwen38_pack.rs` `mlx_residual_norm_to_delta`)

Launch geometry is `dispatch_threads(grid, threadgroup)`. Metal simdgroup
width on this device is 32. Occupancy notes below are **launch-derived**,
not hardware counters (`qwen38_token_ns_ledger.rs:174-186`;
G024 `unresolved[0]`).

`times` = dispatches per **one** production token. All are on the single
serial CB → all are on the token critical path unless a later serial-group
collapse is opted in (production does not).

### 3.1 Weight GEMVs — `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`

Source: `qwen_uniform_q4.metal:181-221`. Host launch
`Qwen38MatvecKernel::GeoTpr64Tg128` (`qwen38_hybrid_decode.rs:264-268`):
`tg=(128,1,1)`, `grid=(rows.div_ceil(2)*128, 1, 1)`.

Kernel: 4 simdgroups / TG; 2 rows / TG; 64 threads / row (2 simdgroups);
`simd_sum` then TG add of the two halves. Packed decode stays in registers.
**Never writes a dense W.**

| organ | rows×cols | TGs | simdgroups/TG | times | bytes class (SOURCE, payload) |
|---|---|---:|---:|---:|---|
| in_proj_qkvz | 16384×5120 | 8192 | 4 | 48 | in `linear_attn_bytes` |
| in_proj_ba | 96×5120 | 48 | 4 | 48 | in `linear_attn_bytes` |
| linear out | 5120×6144 | 2560 | 4 | 48 | in `linear_attn_bytes` |
| q_proj | 12288×5120 | 6144 | 4 | 16 | in `full_attn_bytes` |
| k_proj | 1024×5120 | 512 | 4 | 16 | in `full_attn_bytes` |
| v_proj | 1024×5120 | 512 | 4 | 16 | in `full_attn_bytes` |
| o_proj | 5120×6144 | 2560 | 4 | 16 | in `full_attn_bytes` |
| gate | 17408×5120 | 8704 | 4 | 64 | in `mlp_bytes` |
| up | 17408×5120 | 8704 | 4 | 64 | in `mlp_bytes` |
| down | 5120×17408 | 2560 | 4 | 64 | in `mlp_bytes` |
| lm_head | 248320×5120 | 124160 | 4 | 1 | `lm_head_bytes` 675_430_400 |

GEMV count: 48×3 + 16×4 + 64×3 + 1 = **401**.
Theoretical Q4 payload (`theoretical_weight_bytes`,
`qwen38_token_ns_ledger.rs:74-103`, test 700–712):

| class | bytes |
|---|---:|
| mlp | 9_091_153_920 |
| linear attn | 2_953_789_440 |
| full attn | 891_289_600 |
| lm_head | 675_430_400 |
| norms f32 | 6_475_776 |
| embed one row | 2_720 |
| active (excl. embed table) | 13_618_141_856 |
| embed table (not streamed) | 675_430_440 |

RECEIPT `QWEN38_ACTIVE_BUDGET_MEASURED.json`:
`active_bytes_per_token = 13_622_264_240` (manifest-sum, includes headers).
Ledger constant `ACTIVE_BUDGET_BYTES = 13_622_264_240`
(`qwen38_token_ns_ledger.rs:31`).

Shipped-but-unused Q4 launch retargets (same file, **not** on `step`
unless `matvec_kernel` is overwritten by a diagnostic):
`vecgroup` / `vecgroup_x64` / `vecgroup_r4` (`qwen38_hybrid_decode.rs:244-258`).

### 3.2 Embed

`qwen_uniform_q4_embedding_lookup` (`qwen_uniform_q4.metal:589-602`).
Grid `(5120,1,1)` TG `(256,1,1)`. 1-D, one thread/hidden dim.
Consumes one HQ30UQ4 row of `embed_tokens`. Times: **1**.
`qwen38_hybrid_decode.rs:2522-2546`.

### 3.3 RMSNorm

`qwen80_residual_rmsnorm_f32` (`qwen80_device_activations.metal:24-48`).
Grid `(256,1,1)` TG `(256,1,1)`. One TG; 256-thread tree reduce;
`out = x * rsqrt(mean(x²)+eps) * (1+w)`. TG mem 256×f32.
Times: **129** (64 input + 64 post + 1 final).
`qwen38_hybrid_decode.rs:2498-2519`.

### 3.4 DeltaNet activations

`qwen38_qkvz_rearrange_conv_l2_f32` (`qwen38_device_activations.metal:32-105`).
Grid `(256, 16, 1)` TG `(256,1,1)`. One TG per key-head. Hard-fails unless
`key_heads==16 && vpk==3 && dims==128 && conv_kernel==4`.
TG mem `4*256*4` bytes. Consumes f32 QKVZ + f32 conv weights + f32 conv state.
Times: **48**. Encode: `qwen38_hybrid_decode.rs:2630-2654`.

`qwen80_ba_to_decay_beta_f32` (`qwen80_device_activations.metal:155-177`).
Grid `(48,1,1)` TG `(16,1,1)`. 1-D over 48 value heads. f32 BA + A_log + dt_bias.
Times: **48**. Encode: `2660-2673`.

`qwen38_gated_delta_decode_vi` (`qwen38_device_activations.metal:199-248`).
Grid `(128, 48, 128)` TG `(128,1,1)` = **786_432 TGs / layer**.
One TG per (value-head, value-dim); 128 threads walk the key axis;
serial 0..127 reduce in tid0. TG mem 128×f32.
Consumes / writes f32 recurrent state `[48,128,128]` per layer slot.
Times: **48**. Encode: `1623-1655`.
Alt (off): `qwen80_gated_delta_decode_tg` grid `(128,48,1)` TG `(128,1,1)` —
same arithmetic, 128 vi loop inside the TG (`qwen80_device_activations.metal:207-259`).

`qwen80_deltanet_gated_rmsnorm_f32` (`qwen80_device_activations.metal:179-202`).
Grid `(48,1,1)` TG `(16,1,1)`. Per-head serial RMS + silu(z). f32.
Times: **48**. Encode: `2676-2690`.

### 3.5 GQA activations

`qwen38_gqa_qk_norm_rope_cache_f32` (`qwen38_device_activations.metal:108-193`).
Grid `(24,1,1)` TG `(24,1,1)`. One thread per q-head (24); first 4 also
do k/v cache write. Hard-fails unless 24/4/256/64/θ=1e7/eps=1e-6.
Writes f32 K/V cache at `sequence_slot`. Times: **16**.
Encode: `2753-2778`.

`mha_decode_f32` (`mha.metal:602+`). Wrapper `mha_decode_f32_tcb`
(`kernels/mod.rs:10506-10560`): grid `(n_heads*128,1,1)=(3072,1,1)`,
TG `(128,1,1)` = 24 TGs. One TG per q-head. Consumes f32 Q + f32 K/V cache.
TG mem `(seq_len+128)*4` bytes. Not flash. Not int4 KV.
Times: **16**. Encode: `2780-2792`.

`qwen38_attention_apply_sigmoid_gate` (`qwen38_device_activations.metal:250-266`).
Grid `(6144,1,1)` TG `(256,1,1)`. `gated = attn * σ(q_proj[head, 256:])`.
Hard-fails unless `head_dim==256 && elements==24*256`. Times: **16**.
Encode: `2795-2806`.

### 3.6 MLP glue + residual + sample

`gk_swiglu_f32` (`gk_family.metal:578-584`). Grid `(17408,1,1)` TG `(256,1,1)`.
`silu(gate)*up`, f32. Times: **64**. Encode: `2575-2585`.

`qwen_next_add_residual` (`qwen_next.metal:328-335`). Grid `(5120,1,1)`
TG `(256,1,1)` (`kernels/mod.rs:438,13295-13305`). `out=a+b`. Times: **128**
(64 mixer + 64 mlp).

`sample_argmax_f32` (`sample.metal:48-77`). Grid `(256,1,1)` TG `(256,1,1)`.
One TG over vocab 248320. TG mem 256×f32 + 256×u32. Times: **1**.
Wrapper `kernels/mod.rs:14233-14248`.

### 3.7 Per-token dispatch census (SOURCE, G0)

| family | times | notes |
|---|---:|---|
| embed lookup | 1 | |
| geo_tpr64 GEMV | 401 | includes lm_head |
| rmsnorm | 129 | |
| rearrange / ba / gated_delta_vi / gated_rms | 48 each = 192 | |
| rope / mha / sigmoid | 16 each = 48 | |
| swiglu | 64 | |
| add_residual | 128 | |
| argmax | 1 | |
| **total** | **964** | `1+64*15+3` |

---

## 4. One-token dispatch shape (G0 production `step`)

Source: `qwen38_hybrid_decode.rs:3292-3310` +
`TokenCommandBuffer::dispatch_threads_inner` (`metal/mod.rs:4801-4884`) +
`commit_and_wait_split` (`metal/mod.rs:5330-5371`).

| quantity | value | class | evidence |
|---|---:|---|---|
| command buffers | **1** | SOURCE + RECEIPT | `TokenCommandBuffer::new` once; ledger `production_command_buffers: 1`; `QWEN38_TOKEN_NS_LEDGER.json` `dispatches.production_command_buffers` |
| compute encoders | **964** | SOURCE (derived) | default Off mode: each `dispatch_threads` does `cmd.new_compute_command_encoder()` + `end_encoding()` (`metal/mod.rs:4861-4873`). `step` never calls `begin_serial_group` / `begin_concurrent_group`. `concurrent_independent` default false (`qwen38_hybrid_decode.rs:867-870,911`). No receipt field counts encoders. |
| blit encoders | **0** | SOURCE | `copy_buffer_bytes` / `fill_buffer_bytes` not called on this path |
| compute dispatches | **964** | SOURCE + RECEIPT | schedule + ledger + G024 genome string |
| host GPU fences | **1** | SOURCE | `cmd.commit(); cmd.wait_until_completed()` (`metal/mod.rs:5358-5371`) |
| host sampled-id read | **1** × 4 bytes | SOURCE | `*(workspace.sampled.contents() as *const u32)` after wait (`qwen38_hybrid_decode.rs:3308`). Unified-memory load, not a blit. |
| host writes mid-token | **0** | SOURCE | token id is `set_bytes` on embed; no CPU GEMV |
| staging / copies mid-token | **none** | SOURCE | no blit; no reconstruct-to-float W |
| load-time copies (not per token) | one `new_buffer_with_bytes_checked` per tensor | SOURCE | `Qwen38HybridWeights::load` 508–580. 755 host→device uploads. |

Diagnostic modes that **change** this shape (not G0 production):

| lever | effect | evidence |
|---|---|---|
| `HAWKING_TCB_TRACE=gpu` (`SplitCbGpu`) | 964 CBs, 964 waits | `metal/mod.rs:4763-4768,4964-4998` |
| `step_decomposed` | 1+64+64+1 = 130 CBs | `qwen38_hybrid_decode.rs:2457-2496` |
| `concurrent_independent=true` | concurrent encoder around gate+up / qkvz+ba / qkv | `1593-1609,2727-2750`; default off |
| `begin_serial_group` | 1 encoder / CB | exists; **unused** by `step` |
| `commit_no_wait` | no host wait | exists; **unused** by `step` |

RECEIPT `QWEN38_TOKEN_NS_LEDGER.json` (DIRTY_ENGINEERING, not re-run):

```
vehicle: qwen38-27b/uniform-q4-v1
bpw: 4.252735126866492
kernel_runtime_genome: Qwen38HybridDecodeSession + geo_tpr64_tg128
  + qwen38_gated_delta_decode_vi + qwen38_qkvz_rearrange_conv_l2_f32
  + qwen38_gqa_qk_norm_rope_cache_f32
  deltanet_vi_parallel=true concurrent_independent=false
  1 production CB / 964 dispatches
median_gpu_ns: 33_912_333
median_wait_ns: 34_296_583
median_encode_ns: 919_250
median_submit_ns: 12_084
median_wall_ns: 35_227_917
wait_minus_gpu_ns: 384_250
greedy_matches_oracle: true
fallbacks: 0
```

RECEIPT `QWEN38_COMPLETE_TOKEN_WALL.json` (separate complete-wall harness,
DIRTY_ENGINEERING):

```
headline_complete_wall_ns_per_token: 38_216_792
headline_complete_tps: 26.166508167404526
headline_gpu_ns_per_token: 36_987_458
```

G0 folklore `TOKEN_NS ~ 37_900_000`, `TPS ~ 26.4` is **UNVERIFIED_CLAIM**
relative to this lane; the two receipts already disagree by ~3 ms
(35.2 vs 38.2). Do not treat either as BASE_TRUE.

---

## 5. Critical path (RECEIPT, not re-measured)

All 964 dispatches sit on one serial CB, so GPU time is the union.
G024 ranks **wall attribution**, not “can this skip”:

RECEIPT `receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json`
`ranked_by_ns` (against wall 35_227_917 ns):

| rank | component | ns | % wall | triage |
|---:|---|---:|---:|---|
| 1 | weight_addressing | 21_293_103 | 60.44 | EXISTENTIAL |
| 2 | deltanet | 3_732_795 | 10.60 | research |
| 3 | gqa | 2_443_471 | 6.94 | research |
| 4 | normalization | 2_367_415 | 6.72 | research |
| 5 | weight_decode_reconstruction | 1_808_227 | 5.13 | research |
| 6 | dense_swiglu | 1_004_198 | 2.85 | research |
| 7 | host_preparation | 919_250 | 2.61 | below 1 ms |
| 8 | kv_state | 537_665 | 1.53 | below 1 ms |
| 9 | synchronization | 384_250 | 1.09 | below 1 ms |
| 10 | terminal_head | 383_535 | 1.09 | below 1 ms |
| 11 | unattributed_residual | 341_925 | 0.97 | named |
| 12 | command_submission | 12_084 | 0.03 | noise |

Method for #1/#5: isolated class GEMV GPU × (addr_probe / full) and
(decode_probe − addr_probe) / full, then scaled onto production GPU
(`qwen38_token_ns_ledger.rs` `seal_components`; G024 `unresolved` warns
these probes are diagnostic kernels). **Component microbenchmark ≠ a
new token-level claim.**

G024 `top_three_attacks[0]`: addr_probe is 87% of MLP, 91% of DeltaNet
GEMVs, 83% of GQA GEMVs, 92% of lm_head. Production token claimed
401.6 / 411.51 GB/s of `HONEST_DECODE_CEILING_GB_S` (`qwen38_token_ns_ledger.rs:28`).
That ceiling is a **named decode-shape band**, not 819 peak. This lane
did not remeasure bandwidth.

Implication for G1 representation work: a low-BPW mechanism that
**expands to Q4/float then hits this same geo_tpr64 GEMV** pays the
60% addressing bucket again. Binding in the contract stands.

---

## 6. Mixed / heterogeneous artifact load — the prior finding

### 6.1 Historical claim (RECEIPT)

`receipts/ascent-2026-08-16/QWEN38_NO_NATIVE_MIXED_READER.json`:

> THE_REAL_BLOCKER: The Qwen3.8 runtime cannot read mixed-codec artifacts
> AT ALL. ascension_qwen38_hybrid_greedy speaks only HQ30UQ4 and f32v2.
> Grep finds NO HGRAVB01/HGRAVR02/HGRAVS01 reader in qwen38_hybrid_decode.rs.
> So every coherence test of a mixed artifact must first RECONSTRUCT it
> back to Q4 — a second lossy quantisation — which confounds the result.

Status of that sentence **at HEAD 2eee9a004: REFUTED**.

### 6.2 What HEAD actually does (SOURCE)

`Qwen38HybridWeights::load` (`qwen38_hybrid_decode.rs:508-513`):

```
if root.join("catalog.hq38m20").is_file() { return Self::load_mixed(root); }
```

`load_mixed` (`583-673`) parses magic `HQ38M20\0` version 1
(`31-34,96-107`), then per-tensor codec:

| catalog codec | magic | GPU object | kernel (recon-fuse ON, default) |
|---:|---|---|---|
| 0 | `HGRAVB01` | `GpuBinary` signs+f16 scales | `q80_binary_group_matvec_tg256` |
| 1 | `HGRAVR02` | binary + host-expanded CSR | `q80_binary_group_csr_matvec_tg256` |
| 2 | `HGRAVS01` | two factors, **only r160_b3** | two × `q80_hgravs01_factor_matvec_simd3` |
| 3 | `HGRAVU01` matrix | `GpuUniform` | `q80_hgravs01_factor_matvec_simd` (or simd3 if bits==3; or q8 tiles if bits==8) |
| 3 | `HGRAVU01` vector ≤65536 els | **host dequant → f32 buffer** | consumed as f32 (norms) |
| 3 | `HQ30UQ4\0` | `Q4Weight` | geo_tpr64 (same as G0) |
| other | — | **refuse** | no silent fallback |

`mixed_gpu_layout` (`q80_mixed_decode.rs:1174-1332`) only accepts codecs 0–3;
`other => Err("unknown mixed codec")`.

MLP role lock `assert_mixed_mlp_native` (`qwen38_hybrid_decode.rs:958-1003`):
**every** layer `gate` must be Binary, `up` Residual, `down` Hgravs.
Missing or wrong kind → refuse. No Q4 MLP fallback once `catalog.hq38m20` is present.

Rice indices of HGRAVR02 are **expanded on the host at load**
(`expand_rice_indices`, `upload_mixed` 1071–1114), then CSR lives on GPU.
That is load-time CPU work, not a per-token reconstruct-to-Q4.

`HAWKING_QWEN38_RECON_FUSE` default ON (`43-45`). OFF selects serial
`gk_matvec_binary` / `gk_matvec_hgravs` (or legacy `q80_*` if
`HAWKING_DECODE_FAMILY=0`) and splits residual into binary + `q80_sparse_q1_apply_csr`.

Split in_proj (if fused QKVZ/BA names are absent): 4 GEMVs +
`qwen38_fuse_split_qkvz_f32` + `qwen38_fuse_split_ba_f32`
(`qwen38_device_activations.metal:271-321`; encode `1522-1562`).
Activation interleave, not weight reconstruct.

Embed mixed: `qwen38_hgravu_embedding_lookup` (`qwen38_device_activations.metal:325-342`)
or HQ30UQ4 lookup. Anything else refuses (`2858-2908`).

### 6.3 Mixed path extra kernels (only if `catalog.hq38m20` present)

Not on the G0 uniform-q4 token.

| kernel | file | representation | launch (default fuse) | when |
|---|---|---|---|---|
| `q80_binary_group_matvec_tg256` | `q80_mixed_decode.metal:701` | HGRAVB01 1-bit + group scale | grid `rows*256`, TG 256; 8 simdgroups/TG; 1 TG/row | every Binary GEMV |
| `q80_binary_group_csr_matvec_tg256` | `q80_mixed_decode.metal:743` | HGRAVR02 binary + CSR q1 | same + CSR tail on lid0 | every Residual GEMV |
| `q80_hgravs01_factor_matvec_simd3` | `q80_mixed_decode.metal:845` | 3-bit grouped factor | grid `ceil(rows/8)*256`, TG 256; 8 SG/TG, 1 row/SG | HGRAVS L and R (2×) |
| `q80_hgravs01_factor_matvec_simd` | `q80_mixed_decode.metal:499` | n-bit factor, not 3, not wide-8 | same 8-row TG | HGRAVU01 bits∉{3,8} or narrow |
| `q80_uniform8_matvec_tg256` | `q80_mixed_decode.metal:992` | uniform-8, cols≥2048 | grid `rows*256`, TG 256 | HGRAVU01 bits==8 wide |
| `q80_uniform8_matvec_simd_bytes` | `q80_mixed_decode.metal:939` | uniform-8, cols<2048 | grid `ceil(rows/8)*256`, TG 256 | HGRAVU01 bits==8 narrow |
| `qwen38_hgravu_embedding_lookup` | `qwen38_device_activations.metal:325` | HGRAVU01 embed row | grid `(5120,)`, TG 256 | mixed embed |
| `qwen38_fuse_split_qkvz_f32` | `:271` | f32 activations | grid `(16384,)`, TG 256 | split in_proj only |
| `qwen38_fuse_split_ba_f32` | `:305` | f32 activations | grid `(96,)`, TG 32 | split in_proj only |
| `q80_sparse_q1_apply_csr` | `q80_mixed_decode.metal:157` | CSR leftover | grid `(rows,)`, TG 256 | fuse OFF only |
| `gk_matvec_binary` / `gk_matvec_hgravs` | `gk_family.metal` | same codecs | 1 thread/row, TG 256 | fuse OFF |

HGRAVS mid is a 160-f32 workspace (`hgravs_mid`, rank lock
`QWEN38_MIXED_HGRAVS_RANK=160`, `bits=3`, `group=64`;
`qwen38_hybrid_decode.rs:37-39,1116-1134,1418-1454`).
Two dispatches: `R @ x → mid[160]`, then `L @ mid`.

Mixed dispatch count is **not 964**. DERIVED (no receipt of a mixed
production census in this tree):

- each HGRAVS down = 2 factor dispatches vs 1 Q4 → **+64** if all 64 downs are HGRAVS
- fused QKVZ/BA HGRAVU01 still 1 each
- split in_proj → +4 per DeltaNet layer (**+192**)
- fuse OFF residual → +1 per Residual tensor

So a fused mixed-2p0-style graph is about **1028** dispatches, still 1 CB,
still 1 wait, still 1 sampled-id read. Not measured here.

### 6.4 Native mixed generate already ran (RECEIPT)

`QWEN38_NATIVE_MIXED_READER.json` status `SHIPPED`, blocker_closed =
`QWEN38_NO_NATIVE_MIXED_READER.json`.

`QWEN38_NATIVE_MIXED_2P0_GENERATE.json` + reader receipt
`mixed_2p0_v1_native_generate`:

- artifact `.../qwen38-27b/mixed-2p0-v1`, catalog `catalog.hq38m20`
- `reconstruct_to_q4: false`, `fallbacks_total: 0`, `dense_w_materialized_total: 0`
- 6 prompts → newline / `)` / `.` salad
- `coherence_verdict: INCOHERENT`

So: mixed artifacts **can** be loaded and generate-tested natively.
Coherence of mixed-2p0-v1 is attributable to the packed representation
(plus tile numeric error), not to a second Q4 quantisation.

### 6.5 What the runtime still CANNOT do (SOURCE)

These bound every G1 representation proposal.

1. **No codec outside {0,1,2,3}.** Unknown catalog codec refuses
   (`qwen38_hybrid_decode.rs:659-663`; `mixed_gpu_layout` 1331).
2. **No reconstruct-to-Q4 production path.** Comments + fail-loud
   (`qwen38_hybrid_decode.rs:4-7,652-656,1219-1220`).
3. **No arbitrary per-tensor MLP roles.** Under HQ38M20, gate/up/down
   kinds are locked for all 64 layers. A “Q4 down + binary gate on layer 7
   only” catalog will not open.
4. **No mixed catalog that omits mixed MLP.** Missing gate/up/down refuses
   (`958-1003`). Cannot open HQ38M20 that is “attention-only mixed”.
5. **No HGRAVS geometry other than r160_b3.** (`1116-1134`)
6. **No HGRAVU01 host-dequant of large tensors.** Vectors >65536 elements
   refuse (`194-206`). Embed/lm_head/GEMV names are forced off the
   dequant path (`938-956`).
7. **Uniform-Q4 catalog cannot carry mixed tensors.** Without
   `catalog.hq38m20`, kind must be `q4` or `f32` (`568-571`).
8. **No TQ / strand bitslice / Q4K / Q6K / Q8_0 / GGML mixed store
   on this token.** `mixed_quant_store.rs` is a MoE GGML cache
   (`StoreKey::{routed,shared}`), unused by Qwen38 hybrid decode.
   `strand_bitslice.metal` is `tq`-feature only (`metal/mod.rs:456-459`).
9. **No device-resident autoregressive feedback on `step`.** The
   device-token embed kernel exists and is unused. Host must wait + read
   the u32 + rebind it on the next embed.
10. **No encoder collapse / ICB replay / megakernel on `step`.**
    `begin_serial_group`, `execute_replayable_graph`,
    `qwen3b_megakernel_*` exist in the library and are not called.
11. **No flash / int4 / f16 KV.** `mha_decode_f32` only. f32 cache.
12. **No MoE / expert table / routed wave.** Dense. Schedule test
    forbids `expert|router|moe` in the MLP suffix
    (`qwen38_64_layer_execution_schedule.rs:176-179`).
13. **No vision tower.** Pack skips `vision_tower.*`.
14. **No generic Engine GEMV fallback** if a named weight is missing
    (`encode_named_matvec` 1219-1220).
15. **`qwen_dense.rs` cannot run this model.** Wrong architecture
    (`qwen38_accept_config` refuses `qwen3` / `qwen3_next` / MoE).

`mixed_quant_store.rs` being present in the crate is **not** a Qwen3.8
mixed reader.

---

## 7. Compiled but not on a Qwen3.8 token

The production Metal library concatenates ~30 shader files
(`metal/mod.rs:419-460`), including Q80 MoE waves, DSV4F, RWKV, gravity PQ,
megakernel, fused gate-up, etc. Those kernels are in the same `.metallib`.
They are **not dispatched** by `Qwen38HybridDecodeSession::step`.

Diagnostic-only on this vehicle (same shaders, not in `step`):

- `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_{addr,decode}_probe`
- `qwen38_f32_stream_probe`
- Q4 `vecgroup*` retargets
- `qwen80_gated_delta_decode_tg` (only if `deltanet_vi_parallel=false`)
- `qwen_uniform_q4_embedding_lookup_device_token`

Do not design G1 lanes against Q80 routed-expert waves or DSV4F worklists
for this model. SUPERWAVE_STATE G023 axis cut still holds: Qwen3.8 is
dense + DeltaNet/GQA, no experts.

---

## 8. Binding for G1 representation work

Preferred shape already exists as a pattern: packed bytes stay packed,
representation-specific kernel (`geo_tpr64` for HQ30UQ4; Q80 occupancy
tiles for HGRAV*). There is **no** production “decode to float, then
generic GEMV” on this token.

A new codec that the catalog cannot name (not 0–3, not HQ30UQ4/f32v2)
is currently unloadable. That is a **runtime** bound, not a kernel-quality
bound. Extending `load_mixed` + a tile is the cheapest experiment that
makes a new codec’s coherence measurable. Without that, you re-open
`QWEN38_NO_NATIVE_MIXED_READER` for that codec even though codecs 0–3
are already readable.

Cheapest experiment this lane did **not** run (GPU lane owns measurement):
open `Qwen38HybridDecodeSession` on a candidate `catalog.hq38m20`,
`enable_structural_kernel_trace()`, `step()` once, dump
`structural_kernel_names()` + `dispatch_count()`. That seals mixed
dispatch count instead of the DERIVED +64/+192 above.

---

## 9. Evidence appendix (command / file)

### 9.1 Schedule constants

```
crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:12-54
QWEN38_MIXER_PREFIX_DISPATCHES = 9
QWEN38_DENSE_MLP_SUFFIX_DISPATCHES = 6
QWEN38_FULL_LAYER_DISPATCHES = 15
QWEN38_DELTANET_MIXER_PREFIX_KERNELS = [
  qwen80_residual_rmsnorm_f32,
  qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 ×2,
  qwen38_qkvz_rearrange_conv_l2_f32,
  qwen80_ba_to_decay_beta_f32,
  qwen38_gated_delta_decode_vi,
  qwen80_deltanet_gated_rmsnorm_f32,
  qwen_uniform_q4_group64_matvec_geo_tpr64_tg128,
  qwen_next_add_residual]
QWEN38_GQA_MIXER_PREFIX_KERNELS = [
  qwen80_residual_rmsnorm_f32,
  geo_tpr64 ×3,
  qwen38_gqa_qk_norm_rope_cache_f32,
  mha_decode_f32,
  qwen38_attention_apply_sigmoid_gate,
  geo_tpr64,
  qwen_next_add_residual]
QWEN38_DENSE_MLP_SUFFIX_KERNELS = [
  qwen80_residual_rmsnorm_f32, geo_tpr64 ×2,
  qwen80_silu_mul_f32, geo_tpr64, qwen_next_add_residual]
QWEN38_TERMINAL_HEAD_KERNELS = [
  qwen80_residual_rmsnorm_f32, geo_tpr64, sample_argmax_f32]
```

Note: suffix array still names `qwen80_silu_mul_f32`; live dispatch uses
`decode_family::swiglu_f32()` → `gk_swiglu_f32` by default. Same ALU.

### 9.2 `step` + wait

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:3292-3310
  TokenCommandBuffer::new
  encode_embed; encode_layers; encode_terminal
  commit_and_wait_timed
  read sampled u32 from pinned buffer

crates/hawking-core/src/metal/mod.rs:4861-4873
  enc = cmd.new_compute_command_encoder(); … enc.end_encoding();

crates/hawking-core/src/metal/mod.rs:5358-5371
  cmd.commit(); cmd.wait_until_completed();
```

### 9.3 Mixed load branch

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:508-513
  if catalog.hq38m20 exists → load_mixed
:601-663 codec 0|1|2 upload packed; 3 HGRAVU01 or HQ30UQ4; else refuse
:958-1003 assert_mixed_mlp_native
```

### 9.4 Receipts quoted (via `git show HEAD:…`, not on sparse disk)

- `receipts/ascent-2026-08-16/QWEN38_NO_NATIVE_MIXED_READER.json`
- `receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_READER.json`
- `receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json`
- `receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`
- `receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json`
- `receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL.json`
- `receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json`
- `receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.md`
- `receipts/ascent-2026-08-16/genesis-resident/GENESIS_RESIDENT.json`

### 9.5 Tests this lane ran

See completion report. No `cargo test` / no GPU.

---

## Completion report

### STATUS
SUPPORTED

### CLAIMS

1. G0 Qwen3.8 decode is `Qwen38HybridDecodeSession::step` on
   `uniform-q4-v1`, not `qwen_dense` / Engine.
   Evidence: `genesis_body/src/main.rs` uses that type;
   `GENESIS_RESIDENT.json` artifact path + `opening Metal + 755 catalog tensors`.

2. One G0 token = 1 CB, 964 compute dispatches, 964 compute encoders
   (derived), 0 blits, 1 `waitUntilCompleted`, 1 four-byte host read.
   Evidence: `qwen38_hybrid_decode.rs:3292-3310`;
   `qwen38_token_ns_ledger.rs:56-59,716-721`;
   `metal/mod.rs:4861-4873,5358-5371`;
   `QWEN38_TOKEN_NS_LEDGER.json` `dispatches`.

3. 401 of 964 dispatches are `geo_tpr64_tg128` GEMVs consuming HQ30UQ4
   in-register. TG=128, 4 simdgroups, 2 rows/TG, 64 threads/row.
   Evidence: `qwen_uniform_q4.metal:181-221`;
   `qwen38_hybrid_decode.rs:264-268,1577-1588`;
   ledger `weight_addressing.dispatches=401`.

4. Default DeltaNet kernel is `qwen38_gated_delta_decode_vi` at
   grid (128,48,128) = 786432 TGs/layer × 48.
   Evidence: `qwen38_hybrid_decode.rs:871-875,912,1632-1636`;
   `qwen38_device_activations.metal:199-248`;
   ledger occupancy note.

5. Weight addressing is the existential wall bucket (21.293 ms, 60.44%)
   on the sealed DIRTY_ENGINEERING ledger. Not re-measured here.
   Evidence: `G024_QWEN38_TOKEN_NS.json` `ranked_by_ns[0]`.

6. Prior “runtime cannot read mixed artifacts” is **false at this commit**.
   `catalog.hq38m20` opens codecs 0–3 packed (HGRAVB01 / HGRAVR02 /
   HGRAVS01 r160_b3 / HGRAVU01 / HQ30UQ4). Native generate of mixed-2p0-v1
   ran with 0 fallbacks and was INCOHERENT.
   Evidence: `qwen38_hybrid_decode.rs:508-673`;
   `QWEN38_NATIVE_MIXED_READER.json`;
   `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`.

7. Runtime still cannot load a general heterogeneous artifact: unknown
   codecs refuse; HQ38M20 forces MLP roles; no reconstruct-to-Q4; no
   TQ/Q4K/GGML-store path on this token.
   Evidence: §6.5 pointers.

### EVIDENCE
File excerpts and receipt fields cited inline above. Commands that
produced receipt text: `git show HEAD:receipts/ascent-2026-08-16/<name>`.
No GPU command was run.

### CHANGES
Created `workspace/superwave/g1/g1-kernel-inventory.md` only.

### TESTS
See final message.

### RISKS
- Encoder count is derived, not sealed by a physical-encoder receipt.
- Mixed dispatch increment (+64 / +192) is derived, not traced.
- Token-ns numbers are DIRTY_ENGINEERING receipts; two harnesses
  disagree (35.2 vs 38.2 ms). This lane must not promote them to
  BASE_TRUE.
- `HAWKING_DECODE_FAMILY=0` or `HAWKING_QWEN38_RECON_FUSE=0` or
  `HAWKING_TCB_TRACE=gpu` changes names and/or CB shape. G0 resident
  stderr showed `HAWKING_TCB_TRACE="(unset)" → mode=Off`.

### UNRESOLVED
- Exact mixed-2p0 structural kernel list / dispatch_count (needs one
  `step` with `enable_structural_kernel_trace`; GPU lane).
- Hardware occupancy counters (G024 already flags this).
- Whether the live G0 process still points at `uniform-q4-v1` today
  (receipt is 2026-08-16; this lane did not inspect the live pid).

### NEXT
Representation lanes: consume HQ30UQ4 or HGRAV* directly; do not expand
to Q4 then geo_tpr64 unless a complete-token receipt shows a net win.
Runtime lanes that invent a new codec must extend `load_mixed` first or
coherence is again unmeasurable for that codec.
