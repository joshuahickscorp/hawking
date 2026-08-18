# G1 — Qwen3.8 lm_head and tails

Lane: `17-lm-head-and-tails`. No GPU. No inference. No artifact. Read of
existing receipts + geometry + the G0 decode path.

**Regime.** Qwen3.8 is in the small-tail regime, not the 70% regime.
On the G0 vehicle (`qwen38-27b/uniform-q4-v1`, complete physical BPW
4.252735126866492) the four non-layer organs consume:

| organ | per-token **bytes** (derived) | % of active 13,618,141,856 B | per-token **time** (isolated GPU ns, prior-lane measured) | % of complete-wall 38,216,792 ns |
|---|---:|---:|---:|---:|
| embed lookup | 2,720 | 0.000020% | 4,999 | 0.0131% |
| final RMSNorm | 20,480 | 0.000150% | 19,291 | 0.0505% |
| lm_head GEMV | 675,430,400 | 4.9598% | 1,017,458 | 2.6623% |
| sample (device argmax) | 993,284 write | 0.0073% | 335,499 | 0.8779% |
| **four tails** | **675,453,600 read + 993,284 write** | **4.960% read** | **1,377,247** | **3.6038%** |

lm_head is the largest **single** tensor (1,271,398,400 params, same
shape as the untied embed table) and is real traffic. It is **not** 70%
of decode. It is ~5% of active bytes and ~2.7–3.0% of token time.
Deleting it entirely would move complete-wall TPS from 26.17 to 26.88.
That is not a G1 TPS lever.

It **is** a complete-BPW lever: embed+lm_head are 9.45% of language
params. If both stay at 4.25 BPW, the rest of the model must average
**1.213 BPW** to hit complete 1.5.

Labels used below: **measured** = prior-lane GPU timestamp on this box;
**derived** = arithmetic from geometry / packed sizes; **projected** =
scaled from a measured number; **cited** = in-repo claim not re-derived;
**unmeasured**.

---

## 1. Genome this is conditioned on

G0 resident body is `uniform-q4-v1` via `Qwen38HybridDecodeSession`.

```
receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.md
  artifact: workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1
  resident_weight_bytes = 14,297,675,776
```

```
receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json
  vehicle: qwen38-27b/uniform-q4-v1
  bpw: 4.252735126866492
  kernel_runtime_genome: Qwen38HybridDecodeSession
    + qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
    + qwen38_gated_delta_decode_vi
    + qwen38_qkvz_rearrange_conv_l2_f32
    + qwen38_gqa_qk_norm_rope_cache_f32
    deltanet_vi_parallel=true concurrent_independent=false
    1 production CB / 964 dispatches
  measurement_label: DIRTY_ENGINEERING
  gpu_timestamp_authority: completed MTLCommandBuffer GPUStartTime/GPUEndTime
```

Production token path (`crates/hawking-core/src/model/qwen38_hybrid_decode.rs`):

```
2522:2546  encode_embed
           kernel qwen_uniform_q4_embedding_lookup
           grid (5120,1,1) tg (256,1,1)
           host token u32 → one Q4 row → hidden f32
2833:2855  encode_terminal
           1. qwen80_residual_rmsnorm_f32   (final norm, 5120-wide)
           2. qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
              on language_model.lm_head.weight
              in=normalized, out=logits[248320]
           3. sample_argmax_f32_tcb
              grid/tg (256,1,1) → sampled u32
3300:3308  step() = embed + 64 layers + terminal; then host read of sampled
```

Schedule names the same three terminal kernels:

```
crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:50-54
pub const QWEN38_TERMINAL_HEAD_KERNELS: [&str; 3] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "sample_argmax_f32",
];
```

964 = 1 embed + 64×15 layer + 3 terminal
(`qwen38_token_ns_ledger.rs` `production_dispatches_per_token`, test
`production_dispatch_count_is_964`).

G0 sampling is **greedy device argmax**. No temperature / top-p / top-k
on this path. Host sample_readback is a 4-byte pinned-buffer load after
`waitUntilCompleted`.

Embed and lm_head are **untied**, same shape, both packed HQ30UQ4
group-64 (`qwen38_pack.rs:474-479` and `:660-664`).
`tie_word_embeddings: false` (`QWEN38_ARCH_CENSUS.json` `text`).

Geometry authority:

```
crates/hawking-core/src/model/qwen38_geometry.rs:20-26
QWEN38_LAYERS = 64
QWEN38_HIDDEN = 5120
QWEN38_INTERMEDIATE = 17408
QWEN38_VOCAB = 248320
```

lm_head name `language_model.lm_head.weight`, embed
`language_model.model.embed_tokens.weight`, final norm
`language_model.model.norm.weight` (`qwen38_geometry.rs` helpers).

---

## 2. Per-token bytes

### 2.1 Formula (derived)

`qwen38_token_ns_ledger.rs:34,52-53,74-103`:

```
Q4_BYTES_PER_GROUP = 64/2 + 2 = 34          # 32 B codes + 2 B f16 scale
q4_matrix_bytes(rows, cols) = rows * ceil(cols/64) * 34
lm_head = q4_matrix_bytes(248320, 5120) = 248320 * 80 * 34 = 675,430,400
embed_row = q4_matrix_bytes(1, 5120)     = 1 * 80 * 34      = 2,720
```

Independent recomputation of that function (this lane, CPU):

```
lm_head_bytes           675430400
embed_row_bytes         2720
mlp_bytes               9091153920
linear_attn_bytes       2953789440
full_attn_bytes         891289600
norms_all_bytes         6475776
final_norm_f32_bytes    20480
logits_f32_bytes        993280
sampled_u32_bytes       4
active_bytes            13618141856
```

Matches the receipt field-for-field:

```
receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json  weight_bytes
  mlp_bytes: 9091153920
  linear_attn_bytes: 2953789440
  full_attn_bytes: 891289600
  lm_head_bytes: 675430400
  norms_bytes: 6475776
  embed_row_bytes: 2720
  embed_table_excluded_bytes: 675430440
  active_bytes: 13618141856
  note: "Q4 group-64 codes+f16 scales from geometry; norms are f32 scales.
         Embed table excluded except one gathered row."
```

Manifest class bytes include a ~40 B HQ30UQ4 header
(`qwen38_token_ns_ledger.rs` test comment at `:707-711`):

```
receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json
  by_class_bytes.lm_head:     675430440
  by_class_bytes.embed_table: 675430440
  active_bytes_per_token:     13622264240
  embed_excluded_why: "Dense model reads every weight per token EXCEPT
                       the embedding table, from which exactly one row
                       is gathered."
```

40 B header is 0.000006% of the tensor. Use 675,430,400 for traffic
arithmetic; 675,430,440 for storage.

### 2.2 Fractions of **active** per-token bytes (derived)

Active = every GEMV + all f32 norms + one embed row. Not the embed table.

| item | bytes | % of 13,618,141,856 |
|---|---:|---:|
| embed gather | 2,720 | 0.000020% |
| final-norm scale | 20,480 | 0.000150% |
| lm_head weights | 675,430,400 | **4.9598%** |
| logits+id write | 993,284 | 0.0073% |
| embed **table** (excluded) | 675,430,440 | 4.9598% of active **if wrongly counted** |

lm_head / MLP bytes = 9,091,153,920 / 675,430,400 = **13.46×**.
The 70% intuition (vocab GEMV dominates a small model) is false here
because this is a 26.89 G language-param dense model, not a 0.5 B.

### 2.3 Storage vs traffic (derived + measured artifact)

| tensor | stored bytes | moved per token |
|---|---:|---:|
| `embed_tokens.weight` | 675,430,440 | 2,720 |
| `lm_head.weight` | 675,430,440 | 675,430,400 |
| `model.norm.weight` | 20,480 | 20,480 |

Same packed layout, opposite economics. The Q80 terminal-head design
already named this trap and applied it to Qwen3.8:

```
receipts/ascent-2026-08-16/Q80_TERMINAL_HEAD_PACK_DESIGN.json
  table row "Qwen3.8 language_model.lm_head.weight"
    "675 MB is 5.0% of Qwen3.8's 13.62 GB active bytes and is REAL traffic
     (unlike the same-sized embed table)."
  table row "Qwen3.8 language_model.model.embed_tokens.weight"
    "LEAVE. The table is 675,430,440 B; the per-token gather is 2,720 B."
```

BPW-descent mass fraction independently: `embed_lm_head = 0.0945`
(`QWEN38_BPW_DESCENT.json` `mass_fractions`). 2 × 1,271,398,400 /
26,894,971,904 = 0.094545.

### 2.4 Params (derived)

```
lm_head_params          1,271,398,400   4.727% of language
embed_params            1,271,398,400   4.727%
mlp_params             17,112,760,320
linear_params           5,560,074,240
full_params             1,677,721,600
norm_params                 1,618,944
total_language_params  26,894,971,904
```

Census said "~1.27 G params on its own" (`QWEN38_ARCH_CENSUS.json`
`CENSUS_FINDINGS`). Matches.

---

## 3. Per-token time

Two walls exist. They are different sessions. Do not collapse them.

### 3.1 Production token walls (measured, prior lanes)

**Ledger wall** (component identity, 3 paired generates, first-step
dropped as leftover prefill):

```
receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json
  median_gpu_ns:  33,912,333
  median_wall_ns: 35,227,917
  wait_minus_gpu:    384,250
  identity_holds: true
```

**Complete-wall authority** (6 warm-rep medians, 32 new tokens, every
recurring host cost named):

```
receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json
  headline_complete_wall_ns_per_token: 38,216,792
  headline_gpu_ns_per_token:           36,987,458
  complete_tps:                        26.1665
  wall_minus_gpu_named.components_ms:
    encode_host_prepare                 0.8862 ms   2.312%
    wait_minus_gpu                      0.4259 ms   1.111%
    submit                              0.0105 ms   0.027%
    tokenizer_decode_new_token          0.0063 ms   0.016%
    commit_epilogue                     0.0018 ms   0.005%
    sample_readback                     0.0009 ms   0.002%
```

Campaign G0 claim TOKEN_NS ~37,900,000 / TPS ~26.4 is **unverified by
this lane**. It sits next to the complete-wall 38.22 ms / 26.17 TPS.

### 3.2 Isolated family GPU (measured component microbenchmarks)

Same receipt, `isolated[]`. Separate CBs **after** a production generate.
GPUEnd−GPUStart. Not inside the timed token. Used by the ledger with
`gpu_scale_applied = 1.0` because isolated GPU sum 33,575,407 <
production GPU 33,912,333.

```
name            median_gpu_ns   wait_ns_median  dispatches  reps
embed                  4,999         218,750           1    [4999, 5249, 4458]
final_norm            19,291         213,333           1    [24041, 19291, 18791]
lm_head            1,017,458       1,296,416           1    [1013791, 1044458, 1017458]
argmax               335,499         562,709           1    [433666, 335499, 332124]
lm_head_full_probe 1,052,874       1,325,708           1    [1052874, 1072874, 1039041]
lm_head_addr_probe   964,124       1,253,709           1    [984458, 964124, 952083]
lm_head_decode_probe 1,003,166     1,270,042           1    [991124, 1003166, 1003416]
input_norms        1,137,250       (64 RMSNorms, not final)
post_norms         1,210,874       (64 RMSNorms, not final)
```

**Which of these is a token-level claim.**

- `lm_head` 1.017 ms: large sequential GEMV, same kernel and launch as
  production. Isolated GPU ≈ production contribution. **High confidence
  token-level.**
- `embed` 5.0 µs and `final_norm` 19.3 µs: launch-bound. Isolated wait
  is 200+ µs of CB overhead that production does not pay (already inside
  the one CB). Isolated **GPU** is the right number; it is noise on the
  token. **Token-level: <0.06%.**
- `argmax` 335 µs: single 256-thread TG walking 248,320 f32s
  (`sample.metal:48-75`, `kernels/mod.rs:14232-14254`). Isolated CB may
  inflate a small kernel. Ledger still books the full 335 µs into
  `terminal_head`. **Medium confidence token-level.** Production-fused
  last-dispatch timestamp is the cheap missing measurement (GPU lane).

Host sample_readback **0.877 µs** (complete-wall authority
`sample_readback` mean 877.42 ns) is not the sampler. The sampler is
the device argmax. The 4-byte visibility cost sits inside
`wait_minus_gpu` (0.38–0.43 ms for the whole CB, not argmax-only).

### 3.3 How the ledger hides lm_head traffic

`qwen38_token_ns_ledger.rs:392-417,628-641`:

```
(head_addr, head_dec, head_fma) = split(lm_head, lm_head probe)
weight_addressing includes head_addr          # 91.57% of lm_head GPU
weight_decode_reconstruction includes head_dec #  3.71%
terminal_head = argmax + head_fma              #  4.72% + 335 µs
method: "isolated argmax + lm_head FMA remainder.
         lm_head weight traffic lives in addressing/decode."
```

Probe split (`QWEN38_TOKEN_NS_LEDGER.json` `probes[class=lm_head]`):

```
full_median_gpu_ns:        1,052,874
addr_median_gpu_ns:          964,124
decode_median_gpu_ns:      1,003,166
addr_frac_of_full:         0.915707
decode_minus_addr_frac:    0.037081
fma_remainder_frac:        0.047212
```

Applied to isolated family 1,017,458 ns: addr 931,693 / dec 37,729 /
fma 48,036.

That is why G024 ranks `terminal_head` at **1.09% / 0.384 ms** and
calls it "below 1 ms". That row is FMA remainder + argmax, **not** the
organ. The organ is 1.017 ms.

G024 `terminal_head` row (`G024_QWEN38_TOKEN_NS.json` `ranked_by_ns[9]`):

```
component: terminal_head
ns: 383535
ms: 0.384
pct: 1.09
triage: "below 1 ms"
```

Do not cite 1.09% as "lm_head is 1% of decode". Cite 2.66–3.00% for the
GEMV, 3.60–4.06% for all four tails.

`normalization` in the ledger is **129** RMSNorms (64 input + 64 post +
1 final) at 2.367 ms / 6.72%. Final alone is 19 µs / 0.05%. G024 attack
#3 ("Collapse 129 RMSNorms") is a layer-norm fusion story, not a final-norm
story. Isolated split: input 1.137 ms, post 1.211 ms, final 0.019 ms.

### 3.4 Fractions

Isolated GPU ns over each wall (this lane, arithmetic):

```
                    ledger_wall   ledger_gpu   complete_wall  complete_gpu
embed                   0.0142%      0.0147%         0.0131%       0.0135%
final_norm              0.0548%      0.0569%         0.0505%       0.0522%
lm_head                 2.8882%      3.0003%         2.6623%       2.7508%
argmax                  0.9524%      0.9893%         0.8779%       0.9071%
four_tails              3.9095%      4.0612%         3.6038%       3.7236%
```

If lm_head is deleted from the complete wall (projected, serial
subtraction, no second-order effect): 38,216,792 − 1,017,458 =
37,199,334 ns → **26.88 TPS**. Four tails deleted: 36,839,545 ns →
**27.15 TPS**. G1 target is 100 TPS / 10,000,000 ns.

### 3.5 Bandwidth on this organ (component microbenchmark, not a roof)

```
isolated lm_head family  675,430,400 B / 1,017,458 ns = 663.84 GB/s
lm_head_full_probe                               641.51 GB/s
lm_head_addr_probe                               700.56 GB/s
```

These are **not** a token-level GB/s and **not** a floor. They show the
organ is already a friendly sequential GEMV, well above the ledger's
quoted honest decode ceiling 411.51 GB/s (`qwen38_token_ns_ledger.rs:28`)
and at 81% of the published 819 GB/s peak. A "bytes / 411.51 GB/s"
lm_head floor would be 1.64 ms and is **false for this shape** — the
same receipt already measured 1.02 ms.

`QWEN38_BANDWIDTH_BOUND.json` `achieved_bandwidth_CORRECTED` counts
lm_head in the 13.622 GB / 406 GB/s end of its range. Correct to count
it. The 352 GB/s end is the bring-up floor that omitted lm_head
(`QWEN38_ACTIVE_BUDGET_MEASURED.json` `lane_floor_was_wrong`).

Reconstruction on this launch is small: addr is 91.6% of the lm_head
GEMV. `QWEN38_RECONSTRUCTION_IS_FREE.json` (Qwen3.8 tpr64, all tested
codecs) is consistent. The binding (low-bpw must be consumed by a
matching kernel, not expanded to float/Q4 then generic GEMV) is already
how G0 runs this organ: `geo_tpr64_tg128` unpacks nibbles in-register
(`qwen_uniform_q4.metal:183-224`).

---

## 4. The 4% prior vs the 70% assumption

**Cited, different model.** `crates/hawking-core/src/vocab_prune.rs:12-16`:

```
Slicing the LM head from [102400, 2048] → [23628, 2048] is a 76.9%
reduction in output-projection compute …
LM head ≈ 4% of decode-time per the v1.1.0 findings
```

That is V2-Lite-Chat, hidden 2048, vocab 102400. This lane did not find
the v1.1.0 receipt behind the sentence. Treat as **cited, not
re-verified**.

**Q80, measured in a prior design lane, different vehicle.**

```
receipts/ascent-2026-08-16/Q80_LM_HEAD_NEGATIVE.json
  "lm_head is 14.7% of per-token bytes but only ~1.25 ms of GPU."

receipts/ascent-2026-08-16/Q80_TERMINAL_HEAD_PACK_DESIGN.json
  time_share_honesty:
    q4_vehicle_terminal_gpu_ns_median: 1,253,249
    lm_head_pct_of_q4_token:   0.56     # 1.25 ms / 225 ms
    lm_head_pct_of_mixed_token: 0.21
    byte_share_q4_vehicle:     0.147
```

Q80 Q4 theoretical: lm_head 165,306,368 / total 1,892,511,808 = **8.73%**
of that ledger's weight bytes (`receipts/QWEN80_TOKEN_NS_LEDGER.json`
`theoretical_weight_bytes`). The 14.7% is the mixed-sub655 *active*
budget (NS-005: 165,329,552 / 1,121,230,144).

**Qwen3.8, this vehicle, this box.** ~5% of active bytes, ~2.7–3.0% of
token time. Same qualitative regime as the V2-Lite 4% citation and the
Q80 "already <1% of a much slower token". Not 70%.

The 70% number that *does* exist in-tree is the opposite organ:
`kernels/mod.rs:2733` "bandwidth-bound Q4_K GEMV (the
profiling-confirmed ~76%-of-decode-time wall)" — layer GEMV, not
lm_head. On Qwen3.8 the analog is `weight_addressing` 60.44% of the
ledger wall (G024 rank 1), of which lm_head is only the 675 MB / 13.61
GB slice.

---

## 5. Does it matter for G1

G1 targets: complete effective BPW < 1.5, TOKEN_NS ≤ 10,000,000, TPS ≥ 100.
Capability preserved.

### 5.1 TOKEN_NS / TPS — **KILLS** as a primary lever

Measured complete wall 38.22 ms. Four tails 1.38 ms. Zeroing them
leaves 36.8 ms. The existential bucket is layer GEMV traffic
(G024: weight_addressing 21.293 ms / 60.44%).

`QWEN38_BANDWIDTH_BOUND.json` / complete-wall density ladder: TPS
scales with BPW, not with a tail kernel. 2.0 / 4.2527 × 38.217 ms =
17.97 ms = 55.6 TPS (complete-wall authority `density_ladder`). Still
not 100. 1.5 / 4.2527 × 38.217 = 13.48 ms = 74 TPS **projected**,
byte-count invariant, layers+head scaled together.

A perfect vocab-sparse skip of the whole lm_head GEMV recovers ~1.02 ms
today. After a real layer-BPW cut the **share** grows (projected):

```
layers stay 4.25 BPW   lm_head =  4.96% of active
layers at 2.00 BPW     lm_head =  9.98%
layers at 1.50 BPW     lm_head = 12.88%
layers at 1.00 BPW     lm_head = 18.15%
```

**REOPEN_IF** layer active BPW is actually on disk and a new token-ns
ledger shows lm_head ≥ ~10% of the new wall. Then sparse eval / a
shape-specific kernel become first-class. Not before.

### 5.2 Complete BPW — **matters**, and embed ≠ lm_head

Complete physical BPW = `8 * tensor_payload_bytes / source_weight_elements`
(`qwen38_pack.rs:673-676`). Embed table **counts**. Traffic does not
care; the G1 BPW number does.

If embed and lm_head stay 4.25 BPW (G0 codec):

```
rest_params = 24,352,175,104
complete 1.5  requires rest BPW 1.213
complete 1.0  requires rest BPW 0.661
```

If embed is crushed to 1.5 BPW (gather-native) and lm_head stays 4.25
(capability-sensitive): rest may be 1.356 BPW.

BPW-descent already refused to score lm_head on output ids
(`QWEN38_BPW_DESCENT.json` `claim_boundary.lm_head_not_output_scored: true`)
and kept `q4_emb` in every candidate name. Mixed-2p0 left
attention+embed+norms at 4.250 BPW (`QWEN38_DENSITY_ROOT_CAUSE.json`
`nonmlp_physical_bpw: 4.250142713483966`).

Q80 behavioural screen (not a Qwen3.8 certificate):

```
receipts/ascent-2026-08-16/Q80_LM_HEAD_NEGATIVE.json
  q4: 11-13 greedy flips vs BF16
  q3: 37
  q2_or_binary: ~100
  svd_r256: 349
  "A logit error becomes the emitted token with no downstream layer
   to absorb it."

receipts/ascent-2026-08-16/CROSS_LANE_CONFLICT_LMHEAD.json
  RESOLUTION: lm_head STAYS Q8 on mixed-1p5; embed STAYS Q8.
  Q4 overlay FORBIDDEN as a silent capability change.
```

`qwen38_quality_not_measured: true`
(`Q80_TERMINAL_HEAD_PACK_DESIGN.json` `claim_boundary`). G0 incumbent
is already Q4 full eval. Further full-eval crush of Qwen3.8 lm_head is
a capability change until a Qwen3.8 generate-id gate says otherwise.

---

## 6. What is available if it matters

### 6.1 Vocabulary-sparse lm_head

Three shapes exist in-tree. None is wired for Qwen3.8 G0.

**A. Static vocab prune** (`crates/hawking-core/src/vocab_prune.rs`).
Slice `[orig_vocab, hidden] → [keep, hidden]` at load. Sampler maps
pruned id → original. Built for V2-Lite 102400→23628. Explicitly out of
scope: tied embeddings; pruning the **input** embed (prompt can be any
id). On Qwen3.8 this is a capability change: vocab 248,320 includes
image/video specials (`image_token_id` 248056, `video_token_id` 248057,
`QWEN38_ARCH_CENSUS.json`). A developer-model whitelist would drop
ids the organism is allowed to emit. **KILLS** as a silent G1
mechanism. **REOPEN_IF** a closed allowed-id set is an explicit
capability contract.

**B. Two-pass uniform draft + exact rescore** (Q80 design, not shipped).

```
receipts/ascent-2026-08-16/Q80_TERMINAL_HEAD_PACK_DESIGN.json
  pack_design.optional_two_pass_layout_not_to_ship_without_generate_gate:
    y_draft = draft @ h
    idx = topk(y_draft, k=32)
    y_exact = gather_rows(authority, idx) @ h
    greedy(y_exact) with reserved tail masked
  Q3 draft + k=8 covered BF16 and Q8 argmax 384/384 on Q80 L47-proxy
  SVD r256 and prefix-d512 drafts: NO k<=1024 reached 100%
  "Uniform-quant drafts work; low-rank and row-prefix drafts do not."
  "lm_head is HIGH RANK." r256 captures 29.92% of ||W||_F^2.
  sampling_gate: top-p/temperature UNMEASURED
  qwen38_quality_not_measured: true
```

On Qwen3.8, a Q3 draft is still a full-vocab GEMV:

```
derived: q3 group-64 ≈ 3.25 bpw → 516,505,600 B
projected time at the measured 664 GB/s: 0.78 ms
plus k=32 exact Q4 rows: 32 * 2720 = 87,040 B (noise)
save vs Q4 full eval: ~0.24 ms today
```

Storage **increases** if authority is kept (Q3 + Q4/Q8). Storage
decreases only if the authority matrix is dropped, which is a
capability change.

This is the only identity-preserving sparse shape in the record.
It is not a G1 TPS closer. It becomes interesting only under §5.1
REOPEN_IF, and only after a Qwen3.8 generate-id identity gate on more
than one prompt. The 12-token reverse-string / 16-token "Say hi."
oracle (`greedy_16_ids` in the complete-wall authority) is explicitly
too small (`Q80_LM_HEAD_NEGATIVE.json` `A_CAUTION_ABOUT_MY_OWN_GATE`).

**C. Cheap structured draft (SVD / prefix / hash).** Q80 screen:
capability change at every k tried. Do not port. **KILLS.**
**REOPEN_IF** a Qwen3.8 hidden-set screen shows some cheap basis
covering greedy argmax at k≤32.

Preferred production shape if B is ever built: draft codes consumed
directly by a Q3 (or whatever) `geo_*` kernel, then a gather-GEMV of k
authority rows. Not: dequant draft to f32, generic GEMV, dequant
authority to f32, generic GEMV.

### 6.2 Specialized kernel for [248320, 5120]

Incumbent: `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`.
Comment at `qwen38_hybrid_decode.rs:239-242` and
`qwen_uniform_q4.metal:183-186`: geometry-sweep winner on Q80
**512×2048**, 64 threads/row, 128-thread TG, 2 rows/TG. Launch
`ceil(rows/2)*128` → **124,160 TGs** for 248,320 rows
(`QWEN38_TOKEN_NS_LEDGER.json` `occupancy[1]`).

Alternate launches already exist and are **not** the default
(`Qwen38MatvecKernel::{Vecgroup, VecgroupX64, VecgroupR4}`). No receipt
of a Qwen3.8-shape sweep on 248320×5120.

Already in-tree, **not wired** on Qwen3.8:

- `qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8`
  (`qwen_uniform_q4.metal:632+`). Built for the **151936×2048** MAC
  bucket. Reconstructs 2048-wide RMSNorm into threadgroup (8 KB + 1 KB
  reduce). Qwen3.8 hidden is 5120 (20 KB + 1 KB = 21 KB, fits 32 KB TG)
  but the kernel is not parameterized for it and is "Not bit-identical
  to the serial oracle." Fusion of a 19 µs final-norm into a 1.02 ms
  GEMV is not a G1 lever.
- `qwen_uniform_q4_embedding_lookup_device_token`
  (`qwen_uniform_q4.metal:608+`). Device-side sampled id, no host
  token. **Not referenced** from `qwen38_hybrid_decode.rs` (grep empty).
  Would save the 0.877 µs host readback, not the 0.4 ms wait-minus-gpu
  (next token still depends on this CB finishing). Noise.

Argmax is the only tail kernel that looks **shape-wrong**: one 256-thread
TG, 993,280 B logits, isolated 335 µs → 3.0 GB/s. A multi-TG reduction,
or fusing running-argmax into the lm_head GEMV so 248,320 logits are
never materialized, is the obvious specialized form. Save ≤ 0.34 ms
today, and only if isolated 335 µs survives a production-fused
timestamp.

Specialized-kernel ceiling vs published peak: (819 − 664) / 664 = 23%
of lm_head = **0.23 ms**. Unmeasured. Not G1-load-bearing.

**KILLS** as a path to 10 ms. **REOPEN_IF** §5.1, or if a GPU-lane
shape sweep on 248320×5120 beats `geo_tpr64_tg128` by a measured
complete-token delta, not a microbench.

### 6.3 Different representation for the embed table

G0 already obeys the binding for embed: the table is **not** expanded
to float and then GEMVed. Consumer is a gather kernel
(`qwen_uniform_q4_embedding_lookup`, and mixed
`qwen38_hgravu_embedding_lookup` at
`qwen38_device_activations.metal:325-342`: "Never a dense W.").

Per-token traffic 2,720 B / 5 µs. Further crushing the **row** cannot
move TPS. Crushing the **table** moves complete BPW and RSS
(resident 14.30 GB includes the table).

What is allowed, if a later lane owns pack:

- Any gather-native low-bpw layout (HGRAVU bits<4, PQ, row-cluster)
  **iff** a matching lookup kernel reads it directly.
- Not: low-bpw → expand to f16/f32 table → gather. That spends
  reconstruction on 1 KB of traffic and still stores the expanded table
  at decode time.
- Not: binary/Q2 as a silent overlay. Q80 embed row-cosine tail:
  Q2 min 0.693, binary min 0.312
  (`Q80_TERMINAL_HEAD_PACK_DESIGN.json` embed rows). Token identity
  after 64 layers **unmeasured** for any Qwen3.8 embed requant.
- Not: cluster-interpolate a row. That is not the source row.

Q80 GLM R0 sub-bit embed is already in the graveyard as catastrophic
(same design receipt). Do not resurrect.

**IMPLEMENT_READY as a BPW-only design**, not as a TPS design. Quality
gate is generate-id across multiple prompts, not row cosine.

---

## 7. Negative results (first class)

| id | mechanism | verdict | REOPEN_IF |
|---|---|---|---|
| T1 | "lm_head is ~70% of Qwen3.8 decode" | **KILLS**. 2.7–3.0% time, 5.0% active bytes. | never on this geometry |
| T2 | Attack tails to reach TOKEN_NS 10 ms / 100 TPS | **KILLS**. Four tails 1.38 ms of 38.22 ms. | layer BPW cut makes lm_head ≥10% of a new ledger |
| T3 | Static vocab prune as a silent win | **KILLS** capability. | closed allowed-id contract |
| T4 | SVD / prefix / HGRAVS01 replacement of lm_head | **KILLS** on Q80 screen (9.11% greedy match at r256). | Qwen3.8 screen shows high-rank is false here |
| T5 | Silent Q8→Q4 or Q4→Q3 full-eval crush | **KILLS** as identity-preserving. Q80: 11–13 / 37 flips. Qwen3.8 unmeasured. | Qwen3.8 generate-id identity on >1 prompt |
| T6 | Specialized 248320×5120 kernel as G1 TPS closer | **KILLS** as primary. Ceiling ~0.23 ms vs peak, unmeasured. | measured complete-token win, or §5.1 |
| T7 | Compress embed table to move TPS | **KILLS**. 2,720 B / 5 µs. | never as a TPS lever |
| T8 | Low-rank or rice on lm_head because "reconstruction is free" | **KILLS** quality, not time. Reconstruction-free ≠ capability-free. | Qwen3.8 id-identity |

NS-002 (do not treat storage BPW as active BPW) is **Q80-MoE specific**.
Qwen3.8 is dense: every weight except the embed table is active. Storage
BPW ≈ active BPW here, plus the embed-table exception this lane exists
to keep visible.

---

## 8. Unmeasured on purpose

This lane did not run GPU, did not load weights, did not generate.
The Qwen3.8 artifact is not in the sparse checkout.

Cheapest experiment that would close each hole (for the GPU-authority
lane, or a CPU-only quality lane with the artifact widened):

1. **Qwen3.8 lm_head quality.** CPU: gather N real pre-lm_head hiddens
   from a capture (not L47-proxy), score greedy vs source BF16 at
   Q8/Q4/Q3/Q2 and Q3-draft+k∈{8,32} rescore. No GPU. Artifact path
   `workspace/campaign/records/runs/qwen38-27b/`. Blocker this worktree:
   path not materialized.
2. **Production argmax ns.** GPU: timestamp the last dispatch of the
   production CB, or a fused lm_head+argmax vs serial pair. Settles
   whether 335 µs is real or isolated-CB tax.
3. **248320×5120 launch sweep.** GPU: `GeoTpr64Tg128` vs
   `Vecgroup*` vs a fused-argmax variant, complete-token wall, not
   isolated. Only worth the lock if §5.1 has already fired or the
   sweep is otherwise free.

Sampling distributions (top-p / temperature) are unused on G0 and
unmeasured for any sparse draft.

---

## 9. Independent arithmetic (this lane)

Command: local python3, geometry constants only, plus receipt ns
copied from `QWEN38_TOKEN_NS_LEDGER.json` / complete-wall authority.
Output:

```
=== DERIVED BYTES (Q4 group-64 codes+f16 scales) ===
lm_head_bytes           675430400
embed_row_bytes         2720
embed_table_manifest    675430440
mlp_bytes               9091153920
linear_attn_bytes       2953789440
full_attn_bytes         891289600
norms_all_bytes         6475776
final_norm_f32_bytes    20480
logits_f32_bytes        993280
sampled_u32_bytes       4
active_bytes            13618141856

=== BYTE FRACTIONS of active_bytes ===
embed_lookup_row                          2720  0.000020%
final_norm_scale                         20480  0.000150%
lm_head_weights                      675430400  4.959784%
argmax_logits_write                     993284  0.007294%
four_tails_weight_plus_embed_row     675453600  4.959954%
embed_table_NOT_traffic              675430440  4.959784%   # excluded from active

=== PARAMS ===
lm_head_params          1271398400
embed_params            1271398400
total_language_params   26894971904
lm_head_frac_of_params  4.7273%
embed+lm_head_frac      9.4545%
lm_head_vs_mlp_bytes    13.460x

G1 BPW: heads stay 4.25, rest at t
  complete 1.5 requires rest BPW 1.2129
  complete 1.0 requires rest BPW 0.6606
embed crushed to 1.5, lm_head stays 4.25
  complete 1.5 requires rest BPW 1.3564

=== TIME ===
isolated embed          4999 ns
isolated final_norm     19291 ns
isolated lm_head        1017458 ns
isolated argmax         335499 ns
four_tails_sum          1377247 ns

vs complete_wall_38216792:  embed 0.0131%  final 0.0505%
                            lm_head 2.6623% argmax 0.8779%
                            four_tails 3.6038%
if lm_head deleted: 37199334 ns → 26.882 TPS
if four tails deleted: 36839545 ns → 27.145 TPS
```

Geometry-only complete BPW (no 40 B headers) came out 4.25167 vs the
packer's 4.252735126866492. The receipt / packer number is the
authority; the 0.001 gap is headers + any f32 mixer scales the
geometry function books separately.

---

## 10. Evidence index

| claim | pointer |
|---|---|
| shape / vocab / untied | `qwen38_geometry.rs:20-26`; `QWEN38_ARCH_CENSUS.json` `text` |
| G0 vehicle + BPW | `QWEN38_TOKEN_NS_LEDGER.json` `vehicle`,`bpw`; `GENESIS_RESIDENT_BODY.md` |
| production kernels | `qwen38_hybrid_decode.rs:2522-2546,2833-2855`; schedule `:50-54` |
| active / lm_head / embed bytes | ledger `weight_bytes`; `theoretical_weight_bytes()` `:74-103`; `QWEN38_ACTIVE_BUDGET_MEASURED.json` |
| isolated ns | ledger `isolated` names `embed`,`final_norm`,`lm_head`,`argmax` |
| probe split | ledger `probes[class=lm_head]` |
| ledger hides traffic | `qwen38_token_ns_ledger.rs:392-417,628-641`; G024 `terminal_head` 0.384 ms |
| complete wall + host sample | `QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` |
| 4% prior (other model) | `vocab_prune.rs:12-16` |
| Q80 14.7% bytes / 1.25 ms | `Q80_LM_HEAD_NEGATIVE.json`; `Q80_TERMINAL_HEAD_PACK_DESIGN.json` |
| Q4 flips / two-pass / SVD kill | same design receipt `measurement`,`table`,`claim_boundary` |
| embed is a table | design receipt embed rows; `qwen_uniform_q4.metal:589-601` |
| reconstruction free at tpr64 | `QWEN38_RECONSTRUCTION_IS_FREE.json` |
| BPW mass 9.45% | `QWEN38_BPW_DESCENT.json` `mass_fractions.embed_lm_head` |
| mixed-2p0 left heads at 4.25 | `QWEN38_DENSITY_ROOT_CAUSE.json` `nonmlp_physical_bpw` |

---

```
STATUS
SUPPORTED

CLAIMS
C1 SUPPORTED. Qwen3.8 lm_head is ~5.0% of active per-token bytes (675,430,400 / 13,618,141,856). Derived from q4_matrix_bytes; matches QWEN38_TOKEN_NS_LEDGER.json weight_bytes.lm_head_bytes and the ACTIVE_BUDGET_MEASURED manifest to a 40 B header. Evidence: §2, §9.
C2 SUPPORTED. Qwen3.8 lm_head GEMV is 2.66–3.00% of token time (1,017,458 ns isolated GPU / 38,216,792 ns complete wall = 2.66%; / 33,912,333 ns ledger GPU = 3.00%). Measured isolated family in QWEN38_TOKEN_NS_LEDGER.json; not the ledger's 1.09% terminal_head row, which is FMA remainder + argmax only. Evidence: §3.2–3.4.
C3 SUPPORTED. Embed lookup is 2,720 B and 4,999 ns (0.000020% bytes, 0.013% complete wall). Final norm is 20,480 B and 19,291 ns (0.00015% bytes, 0.050% wall). Device argmax is 335,499 ns isolated (0.88% wall, medium confidence) plus 877 ns host readback. Evidence: ledger isolated[]; complete-wall sample_readback; §3.
C4 SUPPORTED. Qwen3.8 is in the ~3–4% decode-time tail regime, not the 70% regime. Four tails sum to 3.60% of complete wall / 3.91% of ledger wall. The in-repo 4% sentence is V2-Lite (cited, vocab_prune.rs:14), not this model; Qwen3.8 lands in the same band by its own receipts. Evidence: §4.
C5 MEASURED_NEGATIVE as a G1 TPS lever. Deleting lm_head projects 26.88 TPS; deleting all four tails 27.15 TPS, against a 26.17 TPS complete-wall baseline and a 100 TPS target. Evidence: §3.4, §5.1.
C6 SUPPORTED. Embed table is 675,430,440 B stored and 2,720 B moved. Complete BPW cares; TPS does not. embed+lm_head = 9.45% of language params. Heads frozen at 4.25 BPW force rest = 1.213 BPW for complete 1.5. Evidence: §2.3, §5.2, §9.
C7 SUPPORTED as design, not as a ship. The only identity-preserving sparse lm_head in the record is Q3/Q4 uniform draft + exact k-rescore (Q80 screen). SVD/prefix drafts kill capability. Qwen3.8 quality unmeasured. Projected save today ~0.24 ms. Evidence: §6.1, Q80_TERMINAL_HEAD_PACK_DESIGN.json.
C8 SUPPORTED. G0 already uses a gather-native embed kernel and an in-register Q4 GEMV for lm_head. Preferred next shapes: gather-native low-bpw embed (BPW only); draft-native + gather-rescore lm_head (after a Qwen3.8 id gate); fused-argmax tall GEMV only if §5.1 fires. Expand-to-float-then-generic-GEMV is already rejected by the production path. Evidence: §6.2–6.3, qwen_uniform_q4.metal:183-224,589-601.

EVIDENCE
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json (weight_bytes, isolated, probes, components, closure)
- git show HEAD:receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json
- git show HEAD:receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_BANDWIDTH_BOUND.json
- git show HEAD:receipts/ascent-2026-08-16/Q80_LM_HEAD_NEGATIVE.json
- git show HEAD:receipts/ascent-2026-08-16/Q80_TERMINAL_HEAD_PACK_DESIGN.json
- git show HEAD:receipts/ascent-2026-08-16/CROSS_LANE_CONFLICT_LMHEAD.json
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json (mass_fractions, claim_boundary)
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json
- git show HEAD:receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.md
- git show HEAD:crates/hawking-core/src/model/qwen38_geometry.rs
- git show HEAD:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
- git show HEAD:crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
- git show HEAD:crates/hawking-core/src/model/qwen38_pack.rs
- git show HEAD:crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs
- git show HEAD:crates/hawking-core/src/vocab_prune.rs
- git show HEAD:crates/hawking-core/shaders/qwen_uniform_q4.metal
- git show HEAD:crates/hawking-core/shaders/sample.metal
- this-lane python recomputation in §9

CHANGES
created workspace/superwave/g1/g1-lm-head-and-tails.md
no other path touched

TESTS
see executor final message for exact test -s / wc -l / git status --porcelain output

RISKS
- Isolated argmax 335 µs may be isolated-CB-inflated; if it is, four-tail time is closer to 2.7% than 3.6%. Does not change T1/T2.
- Ledger 35.23 ms wall vs complete-wall 38.22 ms vs campaign claim 37.9 ms. Fractions move by ~0.3 pp. Regime does not.
- Qwen3.8 lm_head quality at Q3/Q4 vs BF16 is unmeasured. Q80 prior is not a certificate. Any pack that crushes this tensor without a generate-id gate is a capability change.
- G0 numbers in the task brief are unverified claims; this file uses the receipts, not the brief.

UNRESOLVED
- Qwen3.8 greedy-id screen for Q3/Q4/Q8 and for Q3-draft+k rescore. Artifact not in this sparse checkout.
- Production-fused argmax ns.
- 248320×5120 launch sweep (GPU lane).
- Sampling distributions under any sparse draft.

NEXT
Do not spend the GPU lock on lm_head until a layer-BPW vehicle exists and a new ledger shows the organ ≥10% of the wall. Spend CPU (once the artifact is visible) on the Qwen3.8 hidden-set id screen if a pack lane proposes to touch lm_head or embed. Embed-table gather-native crush is the only tail mechanism that moves complete BPW without pretending to move TPS.
```
