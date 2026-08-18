# G1 token anatomy — Qwen3.8 G0 complete-token budget

Status of this lane: on-disk evidence only. No GPU run, no inference, no live-process touch.
Every number is **measured**, **inferred**, or **claimed**. A microbenchmark is not a token.

## 1. Which complete-token number

Three different "complete token" figures are on disk. They are not the same quantity.

| label | ns/token | TPS | what it is | source |
|---|---:|---:|---|---|
| campaign brief (unverified) | ~37,900,000 | ~26.4 | not a receipt field | task contract |
| G0 seated identity | 35,227,918 | 28.386576805362157 | `encode+submit+wait` median, 3 dirty-engineering generates | `lab/lineage/identity.py` `GENESIS_COMPLETE_TOKEN_NS`; `receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json` `slots.CURRENT.complete_token_ns` |
| G024 TOKEN_NS ledger wall | 35,227,917 | 28.38657761115992 | same definition; 1 ns rounding vs seating | `receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` `closure.total_token_ns` |
| G002 family-ab TOKEN_NS | 37,543,083 | 26.636 | same definition, different session | `receipts/ascent-2026-08-16/g002-family-ab/TOKEN_NS_QWEN38.json` `TOTAL_TOKEN_NS` |
| complete-token wall headline | 38,216,792 | 26.166508167404526 | decode-only, 6 A/B reps × 31 steps, every recurring host cost | `receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` `headline_32_new_tokens.complete_wall_ns_per_token` |
| complete-token wall 16-tok confirm | 38,543,084 | 25.944991843413465 | same definition, 15 decode steps | same file `g015_length_confirmation_16_new_tokens` |
| ICB wall (later genome) | 36,683,916 | 27.259903222981976 | same complete-token definition after ICB replay | `receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json` `complete_token_wall.after_headline_ms` |
| uninstrumented generate_greedy | 38,997,006 | 25.643 | decode wall, no per-token tokenizer | `QWEN38_COMPLETE_TOKEN_WALL.json` `control_uninstrumented_generate_greedy.decode_wall_ns_per_token` |
| G015 GPU (not a wall) | 33,535,999 | — | GPU only; includes leftover prefill steps | `receipts/ascent-2026-08-16/G015_NATIVE_LEG_VERIFY_ON_MAIN.json` `measured.median_gpu_ns_steady` |
| RUNG card | 38,216,792 | 26.1665 | copies the complete-token wall | `receipts/ascent-2026-08-16/RUNG_QWEN38_MEASURED.json` `ns_per_token` |

`1e9/26.4 = 37,878,788`. The brief's 37.9 ms / 26.4 TPS is an **unverified rounding** of the complete-token wall (38.217 ms / 26.167 TPS). It is **not** G0's seated number.

G0 bound itself to the G024 TOKEN_NS wall:

```
lab/lineage/identity.py (git HEAD, not materialized)
23  GENESIS_COMPLETE_TOKEN_NS = 35_227_918
24  GENESIS_BPW = 4.2527
25  GENESIS_TPS = 28.4
```

```
tools/genesis_seat.py (git HEAD)
8  Generation 0 is Qwen3.8 uniform-q4-v1 at 4.2527 BPW / 35,227,918 ns.
```

Primary closed budget below sums to **35,227,917 ns** (G024 identity). The 1 ns vs 35,227,918 is the ledger's own rounding (`closure.residual_ns = -1`). A second closed budget sums to the complete-token wall **38,216,792 ns**. No budget is fabricated onto 37,900,000.

All timing labels on these receipts are `DIRTY_ENGINEERING`. None is `BASE_TRUE` or `CLEAN_CANDIDATE`.

## 2. Genome this anatomy is of

**Vehicle.** `qwen38-27b/uniform-q4-v1`. Complete physical BPW **4.252735126866492** (measured artifact). Role: `PROFILING_ORACLE_ONLY`. Source `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` rev `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

**Architecture (census, not a timing).** 64 layers, hidden 5120, intermediate 17408, vocab 248320, dense SwiGLU every layer, no experts. Mixer rule `(layer+1)%4==0` → 16 GQA + 48 Gated-DeltaNet. GQA 24:4, head_dim 256. Linear: 16 key heads × 128, 48 value heads × 128, conv k=4. Vision tower skipped.

Source: `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json` `text`, `hybrid_attention`.
Code: `crates/hawking-core/src/model/qwen38_geometry.rs:20-41`, `qwen38_mixer_kind` at line 83.

**Runtime genome (G024).**

```
Qwen38HybridDecodeSession
+ qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
+ qwen38_gated_delta_decode_vi
+ qwen38_qkvz_rearrange_conv_l2_f32
+ qwen38_gqa_qk_norm_rope_cache_f32
deltanet_vi_parallel=true
concurrent_independent=false
1 production CB / 964 dispatches
```

Field: `QWEN38_TOKEN_NS_LEDGER.json` `kernel_runtime_genome`. Seated kernel string: `genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64` (`lab/lineage/identity.py` `make_qwen38_genesis`).

**Dispatch schedule (code, not a timing).** 15 dispatches/layer × 64 + embed + 3 terminal = 964.

```17:54:crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs
pub const QWEN38_DELTANET_MIXER_PREFIX_KERNELS: [&str; QWEN38_MIXER_PREFIX_DISPATCHES] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",  // qkvz
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",  // ba
    "qwen38_qkvz_rearrange_conv_l2_f32",
    "qwen80_ba_to_decay_beta_f32",
    "qwen38_gated_delta_decode_vi",
    "qwen80_deltanet_gated_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",  // out
    "qwen_next_add_residual",
];
// GQA prefix: rmsnorm, q, k, v, rope_cache, mha_decode, sigmoid, o, residual
// MLP suffix: rmsnorm, gate, up, silu, down, residual
// Terminal:   rmsnorm, lm_head, argmax
```

Ledger dispatch census (`QWEN38_TOKEN_NS_LEDGER.json` `dispatches`): embed 1, mixer_prefix 576, mlp_suffix 384, terminal 3, total 964, production_command_buffers 1.

**Encoder ceremony (code).** `TokenCommandBuffer::new` starts with `ordered_encoder_enabled=false`, `serial_group_active=false` (`crates/hawking-core/src/metal/mod.rs:2958-2960`). Default `dispatch_threads` therefore does `new_compute_command_encoder` + set pipeline + dispatch + `end_encoding` per call (`mod.rs:3353-3367`). G0 production path is **964 compute encoders inside 1 CB**.

**GPU timestamp authority (all receipts below).** `MTLCommandBuffer.GPUEndTime − GPUStartTime` after `waitUntilCompleted`. Never a CPU-wait proxy.

**Not G0.** The ICB genome (`HAWKING_QWEN38_ICB`, 964 commands replayed, encode 91 µs) is a later measured path. It is not what `GENESIS_COMPLETE_TOKEN_NS` binds. See §8.

## 3. Closed budget A — G0 seated TOKEN_NS (35,227,917 ns)

Identity (`qwen38_token_ns_ledger.rs:10`, `seal_components`):

`sum(12 serial components) == complete_token_wall`
`wall = encode + submit + wait`
`wait = gpu + synchronization`

Receipt: `receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` + projection `TOKEN_NS_QWEN38.json` + seal `G024_QWEN38_TOKEN_NS.json`.
Commit on the ledger: `57ee82ccef7aba803416ec3562c8981277120fd4`.
Regime: warm after 4 discarded tokens; 3 paired production generates; isolated/probe suite after rep 1; `measurement_label=DIRTY_ENGINEERING`.
Steady sample: 45 decode steps. `median_gpu_ns=33912333`, `median_wait_ns=34296583`, `median_encode_ns=919250`, `median_submit_ns=12084`, `median_wall_ns=35227917`, `wait_minus_gpu_ns=384250`, `gpu_spread_ns=[33393666, 34229458]`.
`fallbacks=0`. `greedy_matches_oracle=true` (16 ids starting `248068, 198, 760, …`).

| # | line | ns | % wall | class | kind | label | method |
|---:|---|---:|---:|---|---|---|---|
| 1 | weight_addressing | 21,293,102.52 | 60.444 | gpu | **traffic** | measured | addr_probe/full × isolated class GEMV GPU (mlp+dn+gqa+lm_head) |
| 2 | deltanet | 3,732,794.93 | 10.596 | gpu | **compute** | measured | isolated rearrange+ba+gated_delta+gated_rmsnorm+dn FMA + 48/64 mixer residual − kv streams |
| 3 | gqa | 2,443,470.71 | 6.936 | gpu | **compute** | measured | isolated rope+mha+sigmoid+gqa FMA + 16/64 mixer residual − kv stream |
| 4 | normalization | 2,367,415.00 | 6.720 | gpu | **compute** | measured | isolated input 64 + post 64 + final 1 RMSNorm CBs |
| 5 | weight_decode_reconstruction | 1,808,227.35 | 5.133 | gpu | **compute** | measured | (decode_probe − addr_probe)/full × isolated GEMV GPU |
| 6 | dense_swiglu | 1,004,197.53 | 2.851 | gpu | **compute** | measured | isolated silu + mlp residual + MLP FMA remainder |
| 7 | host_preparation | 919,250.00 | 2.609 | cpu | **ceremony** | measured | host Instant around encode of the production CB (964 pipeline lookup + set_buffer + dispatch) |
| 8 | kv_state | 537,665.00 | 1.526 | gpu | **traffic** | measured | sequential f32 stream of rec+conv+GQA, capped by fused parent |
| 9 | synchronization | 384,250.00 | 1.091 | sync | **ceremony** | measured | production wait_ns − gpu_ns |
| 10 | terminal_head | 383,534.95 | 1.089 | gpu | **compute** | measured | isolated argmax + lm_head FMA remainder (lm_head *bytes* live in addressing) |
| 11 | unattributed_residual | 341,925.00 | 0.971 | gpu | **ceremony** | measured | named: embed 4,999 + intra-CB encoder-transition gap 336,926 + host_tail 0 |
| 12 | command_submission | 12,084.00 | 0.034 | cpu | **ceremony** | measured | host Instant around `MTLCommandBuffer.commit` |
| | **sum** | **35,227,917.00** | **100.000** | | | | `identity_holds=true` |

`closure.residual_ns = -1` is rounding against the seated 35,227,918. It is not missing work.

### 3.1 Probe split behind rows 1 and 5 (measured, isolated GEMV CBs)

`QWEN38_TOKEN_NS_LEDGER.json` `probes`:

| class | full GPU ns | addr frac | decode−addr frac | FMA frac |
|---|---:|---:|---:|---:|
| mlp | 15,850,166 | 0.871692 | 0.083585 | 0.044724 |
| dn | 5,536,666 | 0.905087 | 0.059226 | 0.035686 |
| gqa | 1,894,625 | 0.830265 | 0.063843 | 0.105892 |
| lm_head | 1,052,874 | 0.915707 | 0.037081 | 0.047212 |

Applied to isolated production-shaped GEMV families (`mlp_matvecs_64=15,853,666`, `dn_gemvs=5,560,749`, `gqa_gemvs=1,817,416`, `lm_head=1,017,458`) this is how 21.293 ms addressing and 1.808 ms decode are obtained. Diagnostic kernels, same launch geometry, **not inside the timed token**. Isolated exclusive GPU sum 33,575,407 vs production GPU 33,912,333; scale=1.0 (no overcount).

### 3.2 Isolated family GPU medians (measured, separate CBs)

`QWEN38_TOKEN_NS_LEDGER.json` `isolated` (3 reps each, median):

| family | median GPU ns | dispatches | what |
|---|---:|---:|---|
| mlp_matvecs_64 | 15,853,666 | 192 | 64 × (gate+up+down) Q4 GEMV |
| mlp_full_64 | 17,195,166 | 384 | MLP suffix including silu/residual/norms mix |
| dn_gemvs | 5,560,749 | 144 | 48 × (qkvz+ba+out) |
| gated_delta_48 | 2,146,166 | 48 | vi kernel, grid `(kd, heads, vd)=(128,48,128)` |
| rope_cache_16 | 1,562,625 | 16 | qk_norm + rope + cache write |
| gated_rmsnorm_48 | 1,295,500 | 48 | 16-wide reductions |
| post_norms | 1,210,874 | 64 | post-attn RMSNorm |
| input_norms | 1,137,250 | 64 | input RMSNorm |
| lm_head | 1,017,458 | 1 | vocab 248320 × 5120 Q4 GEMV |
| gqa_gemvs | 1,817,416 | 64 | 16 × (q+k+v+o) |
| mha_16 | 666,500 | 16 | `mha_decode_f32` TG=128, seq≈19 |
| stream_rec_state | 467,374 | 1 | 151,000,000 B resident × 2 |
| argmax | 335,499 | 1 | `sample_argmax_f32` |
| rearrange_48 | 350,999 | 48 | qkvz rearrange + depthwise conv + L2 |
| silu_64 | 160,958 | 64 | |
| ba_to_decay_48 | 139,374 | 48 | |
| mixer_residual_64 | 118,250 | 64 | |
| mlp_residual_64 | 134,208 | 64 | |
| sigmoid_16 | 43,625 | 16 | attn output gate |
| stream_gqa_value | 26,916 | 1 | |
| stream_gqa_key | 24,375 | 1 | |
| final_norm | 19,291 | 1 | |
| stream_conv_state | 19,000 | 1 | |
| embed | 4,999 | 1 | one Q4 row gather, 2720 B |

A component microbenchmark (one family, one CB) is **not** a token. These numbers are used only as a partition of production GPU, which the ledger already closed.

## 4. Phase remapping onto the same 35,227,917 ns

No production-fused per-layer timestamp exists for the G0 (vi-parallel) genome. `step_decomposed` records `layer_mixer_gpu_ns` / `layer_mlp_gpu_ns` but the only persisted vectors are on the **serial** DeltaNet genome (`receipts/ascent-2026-08-16/qwen38-layer-dense-q4-swiglu.json` `decomposed`). Those are a different genome (A GPU 42.73 ms). See §7.

The table below is an **organ regrouping of the isolated medians in §3.2**, plus the production ceremony rows. Isolated organ GPU sums to 33,575,407. Production GPU − that sum = 336,926 (intra-CB encoder gap). Plus encode+submit+sync closes the wall.

| phase | ns | % of 35,227,917 | kind | label | how |
|---|---:|---:|---|---|---|
| embedding (row gather) | 4,999 | 0.014 | traffic | measured (isolated) | `isolated.embed`; 2720 B; parked in residual in the 12-row cover |
| per-layer DeltaNet attention × 48 | 9,581,475.5 | 27.199 | mixed | inferred (equal split of measured family) | rearrange+ba+gated_delta+gated_rmsnorm+dn_gemvs+48/64 mixer_res = 9,581,475.5 → **199,614 ns/DN-layer** |
| ↳ of which Q4 GEMV | 5,560,749 | 15.785 | traffic | measured (family) | 115,849 ns/layer |
| ↳ of which gated_delta_vi | 2,146,166 | 6.092 | compute | measured (family) | 44,712 ns/layer; includes rec-state stream |
| ↳ of which gated_rmsnorm | 1,295,500 | 3.678 | compute | measured (family) | 26,990 ns/layer; 223× its byte floor in the parent deltanet row |
| ↳ of which rearrange/conv/L2 | 350,999 | 0.996 | compute | measured (family) | 7,312 ns/layer |
| ↳ of which ba_to_decay | 139,374 | 0.396 | compute | measured (family) | 2,904 ns/layer |
| per-layer GQA attention × 16 | 4,119,728.5 | 11.695 | mixed | inferred (equal split) | rope+mha+sigmoid+gqa_gemvs+16/64 mixer_res = 4,119,728.5 → **257,483 ns/GQA-layer** |
| ↳ of which Q4 GEMV | 1,817,416 | 5.159 | traffic | measured (family) | 113,589 ns/layer |
| ↳ of which rope+qk_norm+cache | 1,562,625 | 4.436 | compute | measured (family) | 97,664 ns/layer; 24 threads |
| ↳ of which mha_decode | 666,500 | 1.892 | compute | measured (family) | 41,656 ns/layer; grows with seq (unmeasured beyond seq≈19) |
| ↳ of which sigmoid gate | 43,625 | 0.124 | compute | measured (family) | 2,727 ns/layer |
| per-layer MLP (dense SwiGLU) × 64 | 16,148,832 | 45.841 | mixed | inferred (equal split) | mlp_matvecs+silu+mlp_res = 16,148,832 → **252,326 ns/MLP-layer** |
| ↳ of which Q4 GEMV gate+up+down | 15,853,666 | 45.003 | traffic | measured (family) | 247,713 ns/layer |
| ↳ of which silu | 160,958 | 0.457 | compute | measured (family) | 2,515 ns/layer |
| ↳ of which mlp residual add | 134,208 | 0.381 | compute | measured (family) | 2,097 ns/layer |
| RMSNorms (input+post+final) | 2,367,415 | 6.720 | compute | measured (family) | input 17,770 ns/layer; post 18,920 ns/layer; final 19,291 once |
| lm_head + argmax | 1,352,957 | 3.841 | mixed | measured (family) | GEMV 1,017,458 (traffic) + argmax 335,499 (compute) |
| encode (host prepare) | 919,250 | 2.609 | ceremony | measured (production) | 953.6 ns/dispatch × 964 |
| wait − gpu | 384,250 | 1.091 | ceremony | measured (production) | |
| intra-CB encoder-transition gap | 336,926 | 0.956 | ceremony | measured (production − isolated) | 349.5 ns/encoder × 964 |
| submit | 12,084 | 0.034 | ceremony | measured (production) | |
| **sum** | **35,227,917** | **100** | | | organs 33,575,407 + ceremony 1,652,510 = wall |

Equal-split assumption: layers of a type share launch geometry. Not proven per layer on the vi-parallel genome.

Full layer (inferred) = mixer + input norm + post norm + MLP:

- DeltaNet layer: 199,614 + 17,770 + 18,920 + 252,326 = **488,629 ns**
- GQA layer: 257,483 + 17,770 + 18,920 + 252,326 = **546,498 ns**
- 48×DN + 16×GQA + embed + final_norm + lm_head+argmax = 33,575,407 (checks).

## 5. Closed budget B — complete-token wall (38,216,792 ns)

This is the definition that matches the English "complete token": one native decode step plus every recurring per-token host cost. Prefill excluded (Q80 rule: last prefill emits new-token[0]; denominator is decode steps only). Session open, prompt encode, chat template excluded.

Receipts: `QWEN38_COMPLETE_TOKEN_WALL.json`, `QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json`.
Binary: `target/release/examples/ascension_qwen38_hybrid_greedy` (lto=fat, codegen-units=1).
Lock: `./tools/gpu_lane_lock.sh qwen38-complete-wall`.
Set: discarded cold generate + 3 A/B pairs = 6 warm generates, 31 decode steps each, 186 pooled steps.
Headline = median of 6 per-rep medians.

```
headline_complete_wall_ns_per_token = 38,216,792
headline_gpu_ns_per_token           = 36,987,458
wall_minus_gpu_ns                   =  1,229,334
gpu_as_fraction_of_wall             = 0.96783
complete_tps                        = 26.1665
rep_median_wall_ns                  = [38245417, 38149458, 38142333, 37922375, 38216792, 38700583]
spread                              = min 37,922,375 / median 38,216,792 / max 38,700,583
pooled_186 wall                     = min 37,614,375 / median 38,210,417 / max 42,126,875
dispatches                          = 964
command_buffers                     = 1
fallbacks                           = 0
```

GPU is **not** an honest proxy for this wall (`is_33537_gpu_an_honest_proxy_for_wall.answer = "NO"`). Two gaps: (1) this session's GPU is 36.987 ms, not G015's 33.537 ms; (2) wait ≈ gpu+0.43 ms is true but wait is not wall — encode of 964 dispatches is ~0.89 ms, 67% of wall−gpu.

### 5.1 Named host (measured on means, not the headline)

`authority_decomposition.basis`: "mean of the 6 warm-rep per-step means, so sum(components)+gpu == mean complete wall".

Mean wall 38,327,186.13. Mean GPU 36,994,961.88. Mean wall−gpu 1,332,224.25. `named_sum_plus_gpu_minus_wall_ns ≈ 0`. `unattributed_ns = 0`. Residual named `instant_inter_phase_gap`. Max abs per-step residual 1,084 ns.

| line | mean ns | % of mean wall | % of wall−gpu | kind | label |
|---|---:|---:|---:|---|---|
| gpu | 36,994,961.88 | 96.524 | — | mixed (see A) | measured |
| encode_host_prepare | 886,210.36 | 2.312 | 66.521 | ceremony | measured |
| wait_minus_gpu | 425,929.41 | 1.111 | 31.971 | ceremony | measured |
| submit | 10,456.98 | 0.027 | 0.785 | ceremony | measured |
| tokenizer_decode_new_token | 6,306.01 | 0.016 | 0.473 | ceremony | measured |
| commit_epilogue_gpu_timestamp_and_status | 1,789.03 | 0.005 | 0.134 | ceremony | measured |
| sample_readback | 877.42 | 0.002 | 0.066 | ceremony | measured |
| instant_inter_phase_gap | 550.88 | 0.001 | 0.041 | ceremony | measured |
| bookkeeping | 80.38 | 0.000 | 0.006 | ceremony | measured |
| state_update | 23.77 | 0.000 | 0.002 | ceremony | measured |
| **sum** | **38,327,186.13** | **100** | **100** | | closes on **mean**, not headline |

Headline 38,216,792 vs mean 38,327,186 because one B3 step at 42.13 ms pulls the mean (`authority_decomposition.headline_is_median_of_rep_medians_not_mean_of_means`).

Headline split that does close:

| line | ns | label |
|---|---:|---|
| gpu | 36,987,458 | measured (headline) |
| wall − gpu | 1,229,334 | measured (headline) |
| **complete wall** | **38,216,792** | measured |

The 1,229,334 ns host blob is the same ceremony set as the mean table (encode dominates, then wait−gpu). Exact per-row headline split was not published; A3 (the headline rep) has `median_encode_ns=804125`, `median_gpu_ns=37014416`, `median_wait_minus_gpu_ns=395501`, `median_wall_minus_gpu_ns=1193710`.

### 5.2 What this wall adds over budget A

TOKEN_NS wall = encode + submit + wait = 35,227,917.
Complete wall also folds commit-epilogue, sample readback, position update, tokenizer decode, bookkeeping, Instant gaps, and a **different session's GPU** (36.99 ms vs 33.91 ms).

Session GPU movement is 3.075 ms — larger than all named ceremony on either receipt. It is **not** a new organ. Complete-wall authority says G015 33.537 ms is "a different session's GPU sample, not this complete wall." Organ fractions from G024 **must not** be scaled onto 36.99 ms and called measured.

Instrumentation does not dominate: uninstrumented `generate_greedy` decode wall is 38,997,006 ns, **0.78 ms slower** than the instrumented complete wall (`instrumentation_does_not_dominate`).

Request-level, excluded: `session_open_ns=6,284,320,375`, `prompt_encode_ns=27,542`.

Cold first step, excluded from headline: `complete_wall_ns=272,284,750`, `gpu_ns=142,669,124` (graph-cold pipeline first-touch).

## 6. Ceremony inventory (explicit check)

History in this repo: Q80/DSV4F walls were ceremony-dominated (host expert-table bind, 98 CBs, slab I/O). Qwen3.8 G0 is the opposite.

| ceremony | G024 TOKEN_NS | complete-wall mean | ICB genome | kind | label |
|---|---:|---:|---:|---|---|
| encode / host prepare (964 encoder create+bind+dispatch+end) | 919,250 | 886,210 | 90,981 | dispatch+encoding | measured |
| wait − gpu | 384,250 | 425,929 | 561,994 | synchronization | measured |
| intra-CB encoder gap | 336,926 | (inside GPU) | n/a (1 executeCommandsInBuffer) | dispatch | measured (A only) |
| submit / commit | 12,084 | 10,457 | 9,420 | command-buffer submission | measured |
| tokenizer decode | not in A | 6,306 | 6,831 | host | measured (B, ICB) |
| commit epilogue (GPU timestamps) | not in A | 1,789 | 1,708 | host | measured (B, ICB) |
| sample readback (4 B) | inside sync | 877 | (in named fixed) | host round trip | measured (B) |
| Instant inter-phase gap | 0 host_tail | 551 | — | host | measured (B) |
| bookkeeping | — | 80 | — | host | measured (B) |
| position / state update | — | 24 | — | host | measured (B) |
| allocation (per token) | — | — | — | — | **unmeasured**; no receipt names a per-token alloc |
| staging copy | — | — | — | — | **unmeasured**; unified memory, no discrete staging clocked |
| index / embed lookup | 4,999 GPU | — | — | index lookup | measured isolated; ≪ 0.02% |
| **ceremony total** | **1,652,510 (4.69%)** | **1,332,224 (3.48% of mean wall)** | **~670,934 named fixed (1.83%)** | | |

Encode tax: 919,250 / 964 = **953.6 ns per dispatch** (measured host Instant).

Nop-dispatch floor from a **different** (Q80 gather) control, **not** this workload: 1,155 nops in one CB = 1,483,875 ns GPU → 1,284.7 ns/dispatch; ×964 = 1,238,490 ns GPU issue floor (`Q80_DECODE_SHAPE_BANDWIDTH.json` `dispatch_and_cb.eleven55_nops_one_cb`, `RUNG_QWEN38_MEASURED.json` `dispatch_floor`). That is a **projected** GPU-issue floor, not a Qwen3.8 measurement. It sits in the same order of magnitude as the 336,926 ns intra-CB gap plus part of the tiny-kernel times.

ICB deleted 795,219 ns of encode and **raised** wait−gpu by 136,094 ns. Net named-fixed drop 0.744 ms. Complete wall 38.217 → 36.684 ms. Ceremony is compressible; it is not the wall.

`concurrent_independent=true` on isolated MLP won the organ (14.13 vs 15.34 ms) and **lost the token** (44.3 vs 42.7 ms). Source: `qwen38-layer-dense-q4-swiglu.json` `concurrent_isolated_mlp.complete_token`.

## 7. Bytes, bandwidth, reconstruction

**Active bytes / token (geometry, inferred from Q4 group-64 layout).**
`QWEN38_TOKEN_NS_LEDGER.json` `weight_bytes` + `qwen38_token_ns_ledger.rs:74-104`:

| organ | bytes | share of 13,618,141,856 |
|---|---:|---:|
| mlp (64 × gate+up+down) | 9,091,153,920 | 66.76% |
| linear attn (48 × qkvz+ba+out) | 2,953,789,440 | 21.69% |
| full attn (16 × q+k+v+o) | 891,289,600 | 6.54% |
| lm_head | 675,430,400 | 4.96% |
| norms (f32) | 6,475,776 | 0.05% |
| embed row (not the table) | 2,720 | 0.00% |
| embed table (excluded) | 675,430,440 | not moved |

**Active bytes / token (measured manifest).**
`QWEN38_ACTIVE_BUDGET_MEASURED.json`: 13,622,264,240 B. Embed table excluded because exactly one row is gathered.

**State traffic (inferred from geometry, seq-dependent).**
`state_bytes` at the ledger's seq≈ prompt+8: conv R+W 11,796,480; rec R+W 301,989,888; GQA write 131,072; GQA read 2,490,368; total R+W 316,407,808.

**Achieved bandwidth (bytes / measured GPU). Not a floor.**

| session | bytes | GPU ns | GB/s | label |
|---|---:|---:|---:|---|
| G024 production | 13,618,141,856 | 33,912,333 | 401.57 | measured |
| G015 / BW receipt | 13,621,829,601 | 33,536,999 | 406.2 | measured |
| complete-wall headline | 13,618,141,856 | 36,987,458 | 368.18 | measured |
| RUNG card | 13,622,000,000 | 36,987,000 | 368.29 | measured (copies wall) |

Per-class GEMV (isolated family GPU in denominator — **component**, not token):

| class | bytes | isolated GEMV ns | GB/s |
|---|---:|---:|---:|
| mlp | 9,091,153,920 | 15,853,666 | 573.4 |
| dn | 2,953,789,440 | 5,560,749 | 531.2 |
| gqa | 891,289,600 | 1,817,416 | 490.4 |
| lm_head | 675,430,400 | 1,017,458 | 663.8 |

These exceed 411.51 because (a) isolated sequential GEMV CBs are a friendlier shape than a mixed 964-dispatch token, and (b) 411.51 is **not this workload**.

**Do not treat 411.51 GB/s as a Qwen3.8 floor.** It is the Q80 unique-once 512 MiB no-model control (`Q80_DECODE_SHAPE_BANDWIDTH.json` `honest_control.unique_once_full_occupancy_gbps["512_mib"]=411.51358589633037`). The same receipt's claim_boundary: `dense_weight_not_materialized_as_decode=true`, `no_q4_kernel=true`. `QWEN38_ACTIVE_BUDGET_MEASURED.json` `CORRECTION_TO_MY_OWN_CLAIM`: the 320–411 gather control is the wrong shape; whether 406 GB/s is Qwen3.8's ceiling is **UNMEASURED**. Missing experiment: a no-model dense sequential bandwidth control on this box.

Reuse band 536–637 GB/s is a 64 MiB cache-resident roofline, **not** decode (`Q80_DECODE_SHAPE_BANDWIDTH.json` `reuse_64mib_x_4096_gbps.not_the_decode_ceiling`). Published 819 GB/s was not achieved.

**Reconstruction is not the wall.** `QWEN38_RECONSTRUCTION_IS_FREE.json`: at production tpr64, q4/q3/q2/binary/ternary/… clocks the same as uncompressed f32 (gate ~15,125 ns, down ~7,083 ns). 32/33 variants recon_excess_ns=0. This is a **component microbenchmark** on one organ, real captured activation, not a token. It licenses "codec choice is not constrained by unpack ALU" and does not move TOKEN_NS by itself. Consistent with addr_probe being 83–92% of every GEMV class.

**Density root cause (storage, not latency).** `QWEN38_DENSITY_ROOT_CAUSE.json`: on mixed-2p0-v1, MLP is already 0.848 BPW; attention+embed+norms sit at 4.250 BPW and 74% of that artifact. G0 uniform-q4-v1 does not have that split (everything is ~4.25). The 60% addressing bucket is **MLP-majority by bytes** (9.09/13.61 GB).

## 8. Other measured genomes (not G0)

**Serial DeltaNet (pre-vi).** `qwen38-layer-dense-q4-swiglu.json`:
- A (serial gated_delta) production GPU median **42,734,499 ns**
- B (`deltanet_vi_parallel`) production GPU median **33,449,499 ns**
- isolated gated_delta_48: serial 11,485,249 vs vi 2,141,916
- G0 is B.

Per-layer isolated-CB GPU on the **A** genome (`decomposed.layer_mixer_gpu_ns`, `layer_mlp_gpu_ns`; extra CB gaps in wait, not GPU):

| class | n | min | median | max | sum |
|---|---:|---:|---:|---:|---:|
| DN mixer CB | 48 | 410,541 | 413,562 | 503,874 | 20,140,643 |
| GQA mixer CB | 16 | 258,375 | 261,374 | 264,500 | 4,180,616 |
| MLP suffix CB | 64 | 259,166 | 261,875 | 418,458 | 17,253,803 |

These are **not** G0 per-layer times (serial gated_delta still in the mixer). MLP is the same organ; isolated per-layer MLP ~262 µs vs family-fused 252 µs, ~10 µs isolated-CB GPU tax. Mixer times on A are ~2× the vi-parallel family split because serial gated_delta is +9.3 ms.

**G002 family-ab TOKEN_NS** (`g002-family-ab/TOKEN_NS_QWEN38.json`): wall 37,543,083; GPU 36,375,708; addressing 20,134,235 (53.63%); **deltanet 7,035,913 (18.74%)**; scale 0.996. Isolated `gated_delta_48` in `qwen38_family_p3.json` is 5,321,833 — vi path not in the same state as G024. Session variance on the DN row is first-class. Do not average G002 and G024 into one genome.

**ICB replay** (`QWEN38_FIXED_OVERHEAD_DELETED.json`, commit `9c87c500`): encode 886,200 → 90,981; wait−gpu 425,900 → 561,994; complete wall 38.217 → 36.684 ms; TPS 27.26. 964 ICB commands, 64 KiB scalar slab, 3 u32s written per token (token, position, mha_seq_len). Coherence seal PASS, 0 fallbacks. This is a measured ceremony cut on a **later** genome. It is not seated as G0.

## 9. Three largest single line items

On the **closed G0 12-row cover** (the only partition that sums to the seated number):

1. **weight_addressing — 21.293 ms — 60.44% — TRAFFIC.** Sequential Q4 DRAM of 13.612 GB of GEMV weights. addr_probe is 87% of MLP, 91% of DN GEMVs, 83% of GQA GEMVs, 92% of lm_head. Launch geometry on the 17,408-row gate is 8,704 TGs / 60 cores = 145 TGs/core (derived, not a hardware occupancy counter). Kernel headroom on the GEMV is gone **on this genome**; the lever is fewer bytes. `G024_QWEN38_TOKEN_NS.json` `ranked_by_ns[0].triage = "EXISTENTIAL"`.

2. **deltanet — 3.733 ms — 10.60% — COMPUTE.** Isolated activation tails + DN FMA remainder after traffic is attributed out. 223× its byte floor (`measured_over_floor=223.23`). Dominated by gated_rmsnorm (1.296 ms / 48 launches of 16-wide reductions) and gated_delta leftover after rec-state is parked in kv_state. This is launch + low-occupancy ALU, not DRAM. Ceremony-adjacent compute.

3. **gqa — 2.443 ms — 6.94% — COMPUTE.** 313× its byte floor. Isolated rope+qk_norm is 1.563 ms / 16 layers at 24 threads — encoder-per-tiny-kernel tax plus an occupancy disaster. MHA itself is 0.667 ms. Normalization is 2.367 ms (6.72%), 0.076 ms behind GQA; if "single line item" is allowed to be the RMSNorm family it is essentially tied for third and is the same kind (122× byte floor, 129 launches).

On the **organ remapping** of the same isolated GPU (not the 12-row cover):

1. MLP SwiGLU (64 layers) 16.149 ms — **traffic** (15.854 ms of it is Q4 GEMV)
2. DeltaNet attention (48 layers) 9.581 ms — **mixed** (5.561 traffic + 4.020 compute/state)
3. GQA attention (16 layers) 4.120 ms — **mixed** (1.817 traffic + 2.303 compute)

Ceremony's largest single line is encode at 0.919 ms (rank 7). Ceremony is **not** the dominant lever on G0 Qwen3.8. Traffic is.

## 10. What this does and does not license

- G0 is bandwidth-heavy on the present Q4 genome. It is **not** proven to be at a physical roof. 368–406 GB/s achieved; sequential-dense ceiling unmeasured; 411.51 is the wrong control.
- `ms_at_target = measured_ms * (target_bpw / 4.252735)` is a **projection**, and it is wrong if applied to the whole wall: encode, wait−gpu, tokenizer, ICB wait tail do not scale with bytes (`QWEN38_COMPLETE_WALL_ESTIMATE.json` `projection_method_correction`; complete-wall `density_ladder`). Using it on GPU only, 2.0/4.2527 × 36.987 ms = 17.39 ms GPU; plus 1.23 ms host ≈ 18.6 ms → ~54 TPS. Using it on the 38.217 ms wall naively gives 17.97 ms / 55.6 TPS (`gpu_proxy_verdict.arithmetic`). RUNG B (10 ms / 100 TPS) is **not** reachable by density alone at current ceremony (`QWEN38_RESIDUAL_IS_UNRECORDED.json` `why_it_matters`; ICB still leaves 36.68 ms at 4.25 BPW).
- Reconstruction-is-free does **not** mean "pack anything." It means unpack ALU is not the token. Binding still forbids low-bpw → expand-to-float/Q4 → generic GEMV unless a complete-token measurement shows a net win.
- Isolated family GPU is a partition tool. Isolated CBs over-count launch tax when they exceed production GPU (G002 scale 0.996). G024 did not over-count.
- No energy (pJ/weight) was wrapped (`G024_QWEN38_TOKEN_NS.json` `unresolved`).
- Occupancy is launch-geometry derived, not a hardware counter.

## 11. Cheapest experiments that would close remaining holes

Not run here (GPU lane is serialized elsewhere).

1. **Dense sequential no-model bandwidth control** on this box, unique-once, 1–2 GB working set, same tpr64 launch. Settles whether 406 GB/s is a ceiling or has ~2× headroom. Named as missing by `QWEN38_ACTIVE_BUDGET_MEASURED.json`.
2. **`step_decomposed` on the vi-parallel genome**, persist `layer_mixer_gpu_ns` and `layer_mlp_gpu_ns` under GPU lock, 3 paired reps. Converts §4 from inferred equal-split to measured per-layer. Code already exists (`qwen38_hybrid_decode.rs:2457-2496`).
3. **Re-seat G0's complete_token_ns** against the complete-token wall definition (38,216,792) or re-run that wall on the current binary. The seated 35,227,918 and the campaign 37,900,000 are different quantities; G1 promotion will fight the wrong parent if this stays implicit.
4. **Hardware occupancy counter** on `gated_delta_vi` (786,432 TGs/layer) and `mha_decode` / rope (24 threads). Launch-geometry occupancy is not evidence.

## 12. Evidence index

| claim cluster | path | field / lines |
|---|---|---|
| G0 seated ns/TPS/BPW | `lab/lineage/identity.py` (git HEAD) | `GENESIS_COMPLETE_TOKEN_NS`, `GENESIS_BPW`, `GENESIS_TPS` |
| G0 slot | `receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json` | `slots.CURRENT` |
| 12-row closure | `receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` | `components`, `closure`, `isolated`, `probes`, `median_*` |
| same, ranked | `receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json` | `ranked_by_ns`, `measurement`, `top_three_attacks` |
| TOKEN_NS v1 projection | `receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json` | `TOTAL_TOKEN_NS`, `components` |
| complete wall | `receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL.json` | `authority`, `authority_decomposition` |
| complete wall headline | `receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` | `headline_32_new_tokens`, `wall_minus_gpu_named` |
| ICB | `receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json` | `named_fixed_components_ns`, `complete_token_wall` |
| ICB raw | `receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_ICB_WALL.json` | `authority.headline_*` |
| per-layer A-genome | `receipts/ascent-2026-08-16/qwen38-layer-dense-q4-swiglu.json` | `decomposed.layer_*`, `gated_delta_vi_parallel` |
| G002 session | `receipts/ascent-2026-08-16/g002-family-ab/TOKEN_NS_QWEN38.json` | `TOTAL_TOKEN_NS`, `components` |
| bytes measured | `receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json` | `active_bytes_per_token`, `achieved_gb_s` |
| recon free | `receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json` | `claim`, `evidence` |
| wrong ceiling | `receipts/ascent-2026-08-16/Q80_DECODE_SHAPE_BANDWIDTH.json` | `honest_control`, `dispatch_and_cb` |
| encode was dropped | `receipts/ascent-2026-08-16/QWEN38_RESIDUAL_IS_UNRECORDED.json` | `finding`, `arithmetic` |
| schedule | `crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs` | 12–54 |
| seal math | `crates/hawking-core/src/model/qwen38_token_ns_ledger.rs` | 333–464, 509–608 |
| step + complete step | `crates/hawking-core/src/model/qwen38_hybrid_decode.rs` | 3292–3355 |
| 964 encoders | `crates/hawking-core/src/metal/mod.rs` | 2958–2960, 3353–3367 |
| wall = encode+submit+wait | `crates/hawking-core/examples/ascension_qwen38_token_ns.rs` | 365–371 |

## 13. Required test outputs

Recorded after write, same turn.

```
$ test -s workspace/superwave/g1/g1-token-anatomy.md && echo PASS
PASS
```

```
$ wc -l workspace/superwave/g1/g1-token-anatomy.md
     515 workspace/superwave/g1/g1-token-anatomy.md
```

```
$ git status --porcelain
?? workspace/superwave/g1/g1-token-anatomy.md
```

Exactly one new untracked path.

---

```
STATUS
SUPPORTED

CLAIMS
1. G0's seated complete_token_ns is 35,227,918 (TPS 28.386576805362157, BPW 4.2527), not the campaign brief's ~37,900,000 / 26.4. Evidence: lab/lineage/identity.py GENESIS_COMPLETE_TOKEN_NS; receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json slots.CURRENT.complete_token_ns; 1e9/26.4 = 37,878,788 is an unverified rounding of the 38,216,792 complete-token wall.
2. A 12-component ledger closes on 35,227,917 ns (1 ns rounding vs seating). Sum of named rows = wall. Evidence: receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json closure.identity_holds=true, components[*].ns_per_token; this file §3.
3. The three largest G0 line items are weight_addressing 21.293 ms TRAFFIC (60.44%), deltanet 3.733 ms COMPUTE (10.60%), gqa 2.443 ms COMPUTE (6.94%). Ceremony is 1.653 ms / 4.69% and is not the wall. Evidence: G024_QWEN38_TOKEN_NS.json ranked_by_ns; TOKEN_NS_QWEN38.json components; this file §6 and §9.
4. The more complete definition (decode + every recurring host cost, 6 A/B reps) measures 38,216,792 ns / 26.1665 TPS, GPU 36,987,458, wall−gpu 1,229,334 of which encode is ~67%. Evidence: QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json headline_32_new_tokens and authority_decomposition.
5. No production-fused per-layer timestamp exists for the vi-parallel G0 genome. Per-layer ns in §4 are inferred equal-splits of measured isolated families. Evidence: absence of layer_* vectors on G024 receipts; qwen38-layer-dense-q4-swiglu.json decomposed is the serial-DeltaNet genome (A GPU 42.73 ms).
6. 411.51 GB/s is not a Qwen3.8 floor. Achieved production bandwidth is 368–406 GB/s measured. Sequential-dense ceiling unmeasured. Evidence: QWEN38_ACTIVE_BUDGET_MEASURED.json CORRECTION_TO_MY_OWN_CLAIM; Q80_DECODE_SHAPE_BANDWIDTH.json claim_boundary.dense_weight_not_materialized_as_decode.

EVIDENCE
test -s workspace/superwave/g1/g1-token-anatomy.md && echo PASS
PASS
wc -l workspace/superwave/g1/g1-token-anatomy.md
     515 workspace/superwave/g1/g1-token-anatomy.md
git status --porcelain
?? workspace/superwave/g1/g1-token-anatomy.md
git show HEAD:lab/lineage/identity.py | sed -n '23,25p'
GENESIS_COMPLETE_TOKEN_NS = 35_227_918
GENESIS_BPW = 4.2527
GENESIS_TPS = 28.4
QWEN38_TOKEN_NS_LEDGER.json closure.total_token_ns = 35227917
QWEN38_TOKEN_NS_LEDGER.json components.weight_addressing.ns_per_token = 21293102.524500456
QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json headline_32_new_tokens.complete_wall_ns_per_token = 38216792
G024_QWEN38_TOKEN_NS.json ranked_by_ns[0] = weight_addressing 21293103 ns EXISTENTIAL
crates/hawking-core/src/metal/mod.rs:3353-3367 default dispatch_threads = one encoder per dispatch
crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:12-54 15 dispatches/layer + embed + terminal = 964

CHANGES
created workspace/superwave/g1/g1-token-anatomy.md
no tracked file modified
no GPU / inference / live Genesis touch

TESTS
test -s workspace/superwave/g1/g1-token-anatomy.md → PASS
wc -l workspace/superwave/g1/g1-token-anatomy.md → 515
git status --porcelain → ?? workspace/superwave/g1/g1-token-anatomy.md

RISKS
DIRTY_ENGINEERING on every receipt. G024 GPU 33.91 ms and complete-wall GPU 36.99 ms are different sessions; organ fractions from one must not be scaled onto the other and called measured. G002 family-ab (37.54 ms, deltanet 18.7%) shows session variance on the DN row. ICB is a later genome, not G0.

UNRESOLVED
Sequential-dense no-model bandwidth ceiling. Hardware occupancy counters. Per-layer vi-parallel timestamps. Energy. Which complete-token definition G1 promotion will bind (35.23 vs 38.22). Live resident process not re-measured (forbidden).

NEXT
GPU-authority lane: dense sequential bandwidth control; persist step_decomposed on vi-parallel; decide which wall G1 inherits.
```
