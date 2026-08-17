# G1 doctor tensor map — Qwen3.8-27B language path

Lane: `04-doctor-tensor-map`. One file. No GPU. No artifact produced.

## 0. Claim boundary

- **MEASURED this lane**: BF16 language-tensor distributions and outlier-channel counts (851/851, 0 errors, peak RSS 2.116 GB, 365.4 s); activation channel energy on the on-disk 256×5120×64 capture; Q4-artifact byte shares from `uniform-q4-v1/manifest.json`.
- **MEASURED by prior receipts, not re-run**: Q4-oracle generation coherence; mixed-2p0 and mixed-sub15 incoherence; organ hold-cosine screens on 6 layers; 19 attention-density probes on L0/L3/L32/L63 + lm_head.
- **PROJECTED**: any ms/token or TPS obtained by scaling a prior wall-clock by a BPW ratio. Not a token-level measurement. Not a bandwidth floor.
- **ESTIMATED**: collapse depth on tensors that were not organ-scored or generate-ablated, interpolated from same-class probes.
- **UNVERIFIED campaign claims, not used as baselines**: complete BPW ~4.2527 is actually MEASURED below; TPS ~26.4 and TOKEN_NS ~37.9e6 are *not* re-measured here. Receipt `RUNG_QWEN38_MEASURED.json` records 26.1665 tok/s and 38.216792 ms/token under a dirty-engineering label in sibling receipts.
- A component cosine is not a token-level claim. A whole-model generate is not a per-tensor ablation.
- Binding: mixed-sub15's generate vehicle was HQ30UQ4 of *reconstructed* weights. That path is expand-then-generic-GEMV. It is cited only as a coherence negative, not as a production shape.

## 1. What this lane opened

| object | path | status |
|---|---|---|
| BF16 source | `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` | 11 shards, 1184 tensors; 851 language analyzed |
| G0 artifact | `.../uniform-q4-v1/manifest.json` | 755 stored tensors (48 qkv/z fused, 48 a/b fused), 14,297,694,680 payload bytes |
| activation capture | `.../activation-capture-v1` | 256 rows × 5120 × 64 layers, real MLX bf16 forward |
| organ screen | `receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json` | 30 organs, layers {0,3,15,31,47,63}, roles gate/up/down/attn_in/attn_out |
| attention probes | `receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json` | 19 Qwen3.8 tensors |
| generation | Q4 oracle coherent; mixed-2p0 2.0856 BPW incoherent; mixed-sub15 1.291 BPW incoherent | receipts + on-disk GENERATE.json |
| mixed-q3mlp / mixed-q4down / mixed-floor-q7/q8 | packed on disk | **no GENERATE.json** — unused for token claims |

Vision (333 tensors) is skipped by G0 and by this map. Config has `mtp_num_hidden_layers: 1` but the weight map has no MTP tensors.

## 2. Architecture and current encoding

Source: `bf16/config.json` `text_config` + `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json` + `crates/hawking-core/src/model/qwen38_geometry.rs`.

- Dense Qwen3.5 text, **not MoE**. 64 layers, hidden 5120, intermediate 17408, vocab 248320, `tie_word_embeddings: false`.
- Hybrid mixer: ΔNet on 48 layers, GQA on 16 layers, rule `(layer+1)%4==0` → GQA (layers 3,7,…,63).
- GQA: 24 heads, 4 KV heads, head_dim 256, `attn_output_gate: true` so `q_proj` is 12288×5120 = 24×256×2 (q \| gate). `k,v` 1024×5120. `o` 5120×6144.
- ΔNet: `in_proj_qkv` 10240×5120, `in_proj_z` 6144×5120, `in_proj_a/b` 48×5120, `out_proj` 5120×6144, `conv1d` 10240×4×1, `A_log`/`dt_bias` length 48.
- G0 pack (`qwen38_pack.rs`): skip vision; fuse qkv+z and a+b at pack time; GEMVs as HQ30UQ4 group-64; vectors as f32.

MEASURED G0 ledger (`uniform-q4-v1/manifest.json` lines 2–15):

```
complete_physical_bpw: 4.252735126866492
source_weight_elements: 26895998464
tensor_payload_bytes: 14297694680
tensor_count: 755   q4_tensors: 402   f32_tensors: 353
skipped_vision_tensors: 333
```

`min_q4_cosine: 1.0` in that manifest is a reuse-path sentinel (all 402 stored q4 `cosine` fields are `null`). The pack-time quality number is in `receipts/ascent-2026-08-16/qwen38-native-bringup.json` `correctness.numeric_gate`: **Q4 pack min cosine 0.98948 vs BF16**.

## 3. Where the model actually needs information

### 3.1 Residual-stream channel 3994 (MEASURED this lane)

Activation capture, 256 tokens, 5 prompts, post-norm hidden width 5120:

- Energy grows L0 rms 0.0998 → L63 rms 1.167.
- Every layer has 1–3 channels at ≥10× median RMS. Cross-layer persistence:
  - ch **3994**: hot4 in 54 layers, hot10 in **54** layers, mean RMS 14.19. L6 xmed 46.93; L32 xmed 46.17; L63 xmed 30.35.
  - ch **3456**: hot4 in 63/64 layers, hot10 in 24, mean RMS 5.50.
  - ch **310**: hot4 in 52, hot10 in 32, mean RMS 4.07.
- File hash of `capture-result.json` = `01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe`. Field `sha256_self` = `fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512` (hashes the hidden payload, not the JSON).

Weight write-back (this lane, all 64 down + 48 lin_o + 16 o = 128 tensors):

- ch **3994 is in the top-5 output-row RMS of all 128 write tensors**.
- It is the sole ≥10× output-row on every 10×-outlier `lin_o`/`o` (L0,1,3,4,7,8,11) and on `down` L0 (11.47×, kurtosis 15.52).
- L0 `lin_o` kurtosis **149.36**, out-row 3994 = 20.70× median. Matches the prior probe (`qwen38.L0.linear_attn.out_proj` kurt 149.358, out_xmed 20.701) to 0.01.
- L3 `o` kurtosis **132.15**, out-row 3994 = 17.76×. Matches probe 132.137 / 17.759.

Mechanism: a single residual channel is both the hottest activation across depth and the fattest output row of every residual write. That row is the exact island. The other 5119 down/o rows are ordinary.

Naive AWQ column-scale on the captured X is **not** the correction. Probe `HGRAVU01_q4_g64_act_colscale` on L0 `out_proj` drops output cosine 0.99224 → **0.91865**. The capture site is `UNCONFIRMED_POST_NORM` for in-proj and a derived proxy for out-proj. Wrong-site scales are a KILL.

### 3.2 Mass that is *not* sensitive

MEASURED this lane, class medians:

- `gate` / `up`: kurtosis 0.07–0.51, **zero** 4× or 10× output-rows across all 128 tensors.
- `lin_z`: kurtosis ≤ 0.86, no 4× rows.
- `lin_qkv` / `q`: mostly platykurtic-to-mild (max kurt 1.54); three `q` tensors have a 4× row, none 10×.
- `embed`: kurt 0.397, out_xmed 1.28, n10=0.
- `lm_head`: kurt 0.570, out_xmed 1.73, n10=0; **in**-channel xmed 4.80 (the residual-hot dims, not fat vocab rows).

Late GQA `k`/`v` grow tails (L63 k kurt 6.06, v kurt 6.52) without 4× output-rows. That is a richer-levels problem, not an island.

### 3.3 Activation site honesty

On-disk schema name is `qwen38_bf16_post_swiglu_activation_capture.v1` but stored width is 5120, not 17408. Status field: `CAPTURED_REAL_BF16_POST_NORM_HIDDEN`. Attention-density receipt marks `site_is_attention_in_proj_input: UNCONFIRMED_POST_NORM`. down_proj X used in BPW descent was reconstructed as `silu(X@Wg.T)*(X@Wu.T)` from these hiddens + BF16 gate/up, not a stored SwiGLU intermediate. out_proj X in probes is a derived mixer proxy.

This capture is real and same-width. It is **not** a confirmed in-proj site, **not** a confirmed final-norm site for lm_head (`qwen38_L63_post_norm_hidden_NOT_confirmed_final_norm`), and **not** a Hessian / Wanda / per-token greedy-id map.

## 4. Class rollup (language path)

Param share = elements / 26,895,998,464. Byte share = G0 artifact bytes / 14,297,694,680. BPW = 8×bytes/elements at current encoding. Weight shape stats MEASURED this lane.

| class | n | elems | %params | G0 bytes | %bytes | G0 bpw | kurt med/max | out n10 | out xmed max | primary bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| embed | 1 | 1271398400 | 4.727 | 675430440 | 4.724 | 4.250 | 0.40/0.40 | 0 | 1.28 | need_richer_levels |
| q (GQA, includes swish gate) | 16 | 1006632960 | 3.743 | 534774400 | 3.740 | 4.250 | 0.74/1.54 | 0 | 4.82 | need_richer_levels |
| k | 16 | 83886080 | 0.312 | 44565120 | 0.312 | 4.250 | 1.88/6.06 | 0 | 1.83 | need_richer_levels |
| v | 16 | 83886080 | 0.312 | 44565120 | 0.312 | 4.250 | 1.02/6.52 | 0 | 2.29 | need_richer_levels |
| o (GQA) | 16 | 503316480 | 1.871 | 267387520 | 1.870 | 4.250 | 2.60/132.15 | 3 | 17.76 | need_richer_levels |
| ΔNet in_proj_qkv (~q,k,v) | 48 | 2516582400 | 9.357 | 1336935600 | 9.351 | 4.250 | 0.42/1.48 | 0 | 4.03 | need_richer_levels |
| ΔNet in_proj_z | 48 | 1509949440 | 5.614 | 802161360 | 5.610 | 4.250 | 0.30/0.86 | 0 | 2.17 | need_richer_levels |
| ΔNet out_proj (~o) | 48 | 1509949440 | 5.614 | 802162560 | 5.610 | 4.250 | 2.39/149.36 | 4 | 20.70 | need_richer_levels |
| ΔNet in_proj_a | 48 | 11796480 | 0.044 | 6267840 | 0.044 | 4.251 | 2.27/11.12 | 0 | 1.68 | need_richer_levels |
| ΔNet in_proj_b | 48 | 11796480 | 0.044 | 6267840 | 0.044 | 4.251 | 3.53/9.44 | 0 | 2.00 | need_richer_levels |
| ΔNet conv1d | 48 | 1966080 | 0.007 | 7864704 | 0.055 | 32.002 | 19.90/51.52 | 490 | 22.03 | exact_island |
| ΔNet A_log | 48 | 2304 | 0.000 | 9600 | 0.000 | 33.333 | -0.06/6.43 | 0 | 6.38 | exact_island |
| ΔNet dt_bias | 48 | 2304 | 0.000 | 9600 | 0.000 | 33.333 | -0.83/1.04 | 0 | 5.72 | exact_island |
| gate | 64 | 5704253440 | 21.209 | 3030387200 | 21.195 | 4.250 | 0.28/1.55 | 0 | 2.90 | cheap_to_crush |
| up | 64 | 5704253440 | 21.209 | 3030387200 | 21.195 | 4.250 | 0.17/0.51 | 0 | 3.82 | cheap_to_crush |
| down | 64 | 5704253440 | 21.209 | 3030387200 | 21.195 | 4.250 | 0.60/15.52 | 1 | 11.47 | cheap_to_crush |
| input RMSNorm | 64 | 327680 | 0.001 | 1311232 | 0.009 | 32.013 | 8.39/188.17 | 0 | 1.96 | exact_island |
| post-attn RMSNorm | 64 | 327680 | 0.001 | 1311232 | 0.009 | 32.013 | 4.98/82.01 | 0 | 1.78 | exact_island |
| q_norm | 16 | 4096 | 0.000 | 16512 | 0.000 | 32.250 | 26.80/47.98 | 0 | 1.56 | exact_island |
| k_norm | 16 | 4096 | 0.000 | 16512 | 0.000 | 32.250 | 16.04/30.40 | 0 | 1.94 | exact_island |
| ΔNet norm | 48 | 6144 | 0.000 | 24960 | 0.000 | 32.500 | 5.31/36.33 | 0 | 1.59 | exact_island |
| final RMSNorm | 1 | 5120 | 0.000 | 20488 | 0.000 | 32.013 | 7.57/7.57 | 0 | 1.38 | exact_island |
| lm_head | 1 | 1271398400 | 4.727 | 675430440 | 4.724 | 4.250 | 0.57/0.57 | 0 | 1.73 | need_richer_levels |

MLP gate+up+down = 63.63% of parameters and 63.58% of G0 bytes. Attention GEMVs (ΔNet qkv/z/o + GQA q/k/v/o + ba) = 26.90% params / 26.89% bytes. embed+lm_head = 9.45% params / 9.45% bytes. All f32 vectors together = 0.010% params / 0.074% bytes.

## 5. Four buckets

### 5.1 Cheap to crush

**gate, up, and the non-island rows of down**, to uniform-Q3 group-64 (**3.25 BPW**), ESTIMATED from organ hold-cosine, generation untested.

BPW descent, 6 layers, hold n=64 of the 256-token capture (`QWEN38_BPW_DESCENT.json`):

| layer | gate q3 hold | up q3 hold | down q3 hold | gate binary hold | up binary hold | down binary hold |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9821 | 0.9811 | 0.9923 | 0.8619 | 0.8574 | 0.9195 |
| 3 | 0.9772 | 0.9736 | 0.9796 | 0.8378 | 0.8198 | 0.8471 |
| 15 | 0.9777 | 0.9744 | 0.9756 | 0.8466 | 0.8208 | 0.8268 |
| 31 | 0.9766 | 0.9679 | 0.9742 | 0.8558 | 0.7639 | 0.8162 |
| 47 | 0.9781 | 0.9693 | 0.9729 | 0.8454 | 0.7858 | 0.7802 |
| 63 | 0.9940 | 0.9957 | 0.9727 | 0.9543 | 0.9601 | 0.7297 |

- Q3 hold min across those 18 MLP organs = 0.9679 (L31 up). Clears the descent 0.95 bar everywhere it was scored.
- Binary/Q2 does **not**: mid-depth up/down hold 0.73–0.82. L63 gate/up binary hold 0.954/0.960 is a last-layer exception — do not average it into a pass.
- Ternary 2.25 BPW hold stays ≥0.90 on gate/up/attn_in; L63 down dips to 0.843.
- **KILLS as a generation claim**: mixed-2p0 packed MLP to 0.848 BPW (binary gate + rice up + r160 down) with attention/embed/lm_head left at Q4, native generate, 0 fallbacks, 0 dense-W materialize, emitted newline/paren garbage (`QWEN38_COHERENCE_FLOOR_BRACKETED.json`, `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`, on-disk `mixed-2p0-v1/GENERATE.json`). The failure is the *recipe*, not isolated per organ. Cosine-said-fine is not token-safe.
- mixed-q3mlp (Q3 all MLP + Q4 rest, 3.614 BPW) is packed and has **no generate**. That is the cheapest missing token test.

### 5.2 Need correction

Not AWQ-from-this-capture. The correction is **per-output-row treatment of residual channel 3994** (and secondarily 310 / 3456) on every `down`, `lin_o`, and `o`.

- Store those 1–3 rows f16/f32 (exact island).
- Quantize the remaining rows with the class codec.
- Mass: 128 tensors × 1 row × in_dim. down row is 17408 weights; o/lin_o row is 6144. Upper bound ≈ 128×17408 ≈ 2.2 M weights if every down row is kept exact — **0.008% of params**. Real island is smaller (only L0 down is 10×; all 128 have 3994 in top-5 but most at 1.3–4×).
- Doctor6 already has `quant_outlier_channel` / `l3_outlier_residual` (`lab/operators/doctor6/rungs.py`). Steal the *idea*, not the Q80 5% column cut — here the island is **one output row**, not 5% of input columns.

### 5.3 Need richer levels

Attention GEMVs and the token-facing tables.

Attention-density verdict (`QWEN_ATTENTION_DENSITY_VERDICT.json`): *“Attention GEMVs cannot be cheaply compressed below uniform-Q4 at Q4-equivalent output quality. Existing Gravity expert families (HGRAVB01 / HGRAVR02 / HGRAVS01) do not transfer.”*

Probed Qwen3.8 output cosines vs BF16 W on captured X (n=256):

| tensor | Q8 | Q5 | Q4 | Q3 | Q2 | binary | proposed |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen38.L0.linear_attn.in_proj_qkv | 1.0000 | 0.9991 | 0.9961 | 0.9794 | 0.8387 | 0.8351 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L0.linear_attn.in_proj_z | 1.0000 | 0.9990 | 0.9954 | 0.9758 | 0.8142 | 0.8184 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L0.linear_attn.in_proj_qkvz_fused | 1.0000 | 0.9991 | 0.9959 | 0.9783 | 0.8307 | 0.8297 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L0.linear_attn.in_proj_ba_fused | 1.0000 | 0.9992 | 0.9967 | 0.9799 | 0.8539 | 0.8585 | HGRAVU01_q4_g64 @4.254 |
| qwen38.L0.linear_attn.out_proj | 1.0000 | 0.9983 | 0.9922 | 0.9532 | 0.7063 | 0.7688 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L32.linear_attn.in_proj_qkv | 1.0000 | 0.9988 | 0.9949 | 0.9755 | 0.8352 | 0.7359 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L32.linear_attn.in_proj_z | 1.0000 | 0.9994 | 0.9971 | 0.9848 | 0.8865 | 0.7940 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L32.linear_attn.in_proj_qkvz_fused | 1.0000 | 0.9991 | 0.9959 | 0.9797 | 0.8583 | 0.7525 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L32.linear_attn.in_proj_ba_fused | 1.0000 | 0.9989 | 0.9948 | 0.9737 | 0.8639 | 0.7645 | HGRAVU01_q4_g64 @4.254 |
| qwen38.L32.linear_attn.out_proj | 1.0000 | 0.9986 | 0.9938 | 0.9678 | 0.7686 | 0.7955 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L3.self_attn.q_proj | 1.0000 | 0.9993 | 0.9970 | 0.9838 | 0.8691 | 0.8442 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L3.self_attn.k_proj | 1.0000 | 0.9994 | 0.9971 | 0.9849 | 0.8724 | 0.8588 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L3.self_attn.v_proj | 1.0000 | 0.9991 | 0.9961 | 0.9797 | 0.8442 | 0.8382 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L3.self_attn.o_proj | 1.0000 | 0.9993 | 0.9967 | 0.9834 | 0.8110 | 0.8610 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L63.self_attn.q_proj | 1.0000 | 0.9996 | 0.9983 | 0.9909 | 0.9084 | 0.9304 | HGRAVU01_q3_g64 @3.250 |
| qwen38.L63.self_attn.k_proj | 1.0000 | 0.9983 | 0.9924 | 0.9633 | 0.7650 | 0.8055 | HGRAVU01_q4_g64 @4.250 |
| qwen38.L63.self_attn.v_proj | 1.0000 | 0.9997 | 0.9984 | 0.9925 | 0.9395 | 0.9462 | HGRAVU01_q3_g64 @3.250 |
| qwen38.L63.self_attn.o_proj | 1.0000 | 0.9983 | 0.9925 | 0.9603 | 0.7233 | 0.7610 | HGRAVU01_q4_g64 @4.250 |
| qwen38.lm_head | 1.0000 | — | 0.9989 | 0.9942 | — | 0.8002 | HGRAVU01_q3_g64 @3.250 |

- 0.99 bar is the attention bar (Qwen3.8 Q4-vs-bf16 min cosine 0.98948), not the 0.8604 expert bar.
- L0 `lin_o` Q3 = 0.953 (min-row 0.916). L63 `o` Q3 = 0.960. L63 `k` Q3 = 0.963. These fail 0.99.
- L63 `q` Q3 = 0.9909 and L63 `v` Q3 = 0.9925 clear 0.99 — late q/v are the only probed attention GEMVs that look Q3-legal. ESTIMATED, generation untested, 16 layers × those two tensors = 3.7%+0.31% of params.
- Hadamard-Q4 saves 2.9% bits (4.125 vs 4.250) at similar cosine. Not the mass.
- SVD/HGRAVS01 on attention: typical 0.66–0.91, often negative min-row on late o/k. Attention is not a low-rank organ.
- `lin_a`/`lin_b` (fused ba, 0.09% params): stay Q4. Ignore.

embed + lm_head (9.45% params, 1.351 GB together at Q4):

- G0 Q4 is part of the coherent oracle.
- Probe lm_head Q3 cosine 0.99419 but min-row 0.982; top-1 vs BF16 0.8906 (Q4) → 0.8438 (Q3), n=128 (`QWEN_ATTENTION_LMHEAD_TOPK.json`).
- Q80 steal (`Q80_LM_HEAD_NEGATIVE.json`, `CROSS_LANE_CONFLICT_LMHEAD.json`): greedy-id outranks cosine; Q4 already flips 11–13 tokens vs BF16 on Q80. That is Q80 evidence, not Qwen3.8. For Qwen3.8 the only token-level fact is “Q4 whole model is coherent”. Isolated Q3 lm_head/embed generate does not exist.
- Do not drop embed/lm_head below Q4 until a multi-prompt greedy-id screen vs the Q4 oracle says otherwise.

### 5.4 Need exact islands

| island | why | mass |
|---|---|---|
| all RMSNorms (input, post-attn, q_norm, k_norm, lin_norm, final) | already f32; 1D scales; high kurtosis of the scale vector but max/median < 2 | 0.0025% params, 0.019% bytes |
| ΔNet `A_log`, `dt_bias` | recurrence parameters; already f32; 48×48 each | 4608 weights |
| ΔNet `conv1d` | already f32; 0.007% params; 4-tap rows make 10× counts noisy (490 “n10” across 48 tensors) | 1.97 M weights, 7.86 MB |
| residual **output row 3994** on every `down` / `lin_o` / `o` | §3.1 | ≪ 0.01% params |
| optionally rows 310 and 3456 on early write tensors | activation-persistent, appear in write top-5 | even smaller |

## 6. How far each class can collapse before tokens drift

Token-level facts (whole model, not per tensor):

| artifact | complete BPW | recipe | generate | label |
|---|---:|---|---|---|
| `uniform-q4-v1` | 4.2527 | all GEMV Q4, vectors f32 | coherent English, greedy-id identical across reps, 0 fallbacks | MEASURED |
| `mixed-q3mlp-v1` | 3.614 | Q3 MLP + Q4 rest | **not run** | PACKED only |
| `mixed-q4down-v1` | 2.959 | binary gate + rice up + Q4 down + Q4 attn | **not run** | PACKED only |
| `mixed-floor-q7/q8` | 3.18 / 3.54 | 0.848 MLP + Q7/Q8 attn | **not run** | PACKED only |
| `mixed-2p0-v1` | 2.0856 | 0.848 MLP + Q4 attn/embed/lm_head | INCOHERENT, 0 fallbacks, native reader | MEASURED |
| `mixed-sub15-v1` | 1.291 | same MLP + rice attention + Q4 embed/lm_head | INCOHERENT (space/` a` cycle) | MEASURED |

Therefore the **token** coherence floor with *current codecs* is bracketed **(2.0856, 4.2527]** and not located. Nothing between those two points has a generate. `QWEN38_COHERENCE_FLOOR_BRACKETED.json` says this explicitly.

Per-class collapse (labels required):

| class | cosine-legal cheap floor | token-legal floor |
|---|---|---|
| gate, up | ESTIMATED 3.25 BPW Q3 (hold ≥ 0.9679 on 6 layers). 2.25 ternary hold ≥ 0.90. 1.125 binary FAILS mid-depth. | UNKNOWN. 1.125-in-combo KILLS. Q3-in-isolation untested. |
| down body | ESTIMATED 3.25 Q3 (hold ≥ 0.9727). r160 @ 0.13 was in the KILLS recipe. | UNKNOWN. Keep row 3994 exact regardless. |
| ΔNet in_proj_qkv / z, GQA q | ESTIMATED Q4 to hold 0.99; L63 q/v Q3 clears 0.99. | UNKNOWN below Q4. Q4 works as part of oracle. |
| GQA k, late o, ΔNet out_proj | Q4 last cheap codec at 0.99. Q3 0.953–0.963. | Q4 only MEASURED (as part of oracle). |
| embed, lm_head | Q3 cosine-legal, top-1 regresses. | Q4 only MEASURED. |
| norms / A_log / dt_bias / conv1d | n/a (already exact) | keep exact |

A 1.5 complete-BPW G1 target is **not reachable with the codec set that has been generate-tested**. Attention at 4.25 plus crushed MLP already failed at 2.09. Getting under 1.5 requires an attention codec that does not exist in the generate-tested set, or a new family. That is a prior finding (`QWEN_ATTENTION_DENSITY_VERDICT`, `QWEN38_DENSITY_ROOT_CAUSE`) and this lane's weight map does not overturn it.

## 7. Per-tensor table

All 851 language tensors. `%par` = 100×elems/26895998464. `%byt` = 100×G0-bytes/14297694680. Fused ΔNet qkv/z and a/b bytes are *row-fraction* of the stored fused tensor (flag `F`). `out_n10` / `out_xmed` / `top_ch` MEASURED this lane on BF16. `collapse` is ESTIMATED unless marked MEASURED.

| L | mix | class | shape | %par | %byt | bpw | kurt | n10 | xmed | top_ch | buckets | collapse |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 00 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 15.52 | 1 | 11.47 | 3994 | cheap_to_crush,exact_island,need_correction | ESTIMATED Q3 3.25 BPW (hold cos 0.9923 on L0); Q2/binary not intact mid-depth |
| 00 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.31 | 0 | 2.22 | 12355 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9821 on L0); Q2/binary not intact mid-depth |
| 00 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.12 | 0 | 1.23 | 1764 | exact_island | keep f32; do not quantize |
| 00 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.84 | 0 | 1.60 | 24 | exact_island | keep f32; do not quantize |
| 00 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 0.78 | 0 | 1.60 | 30 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 00 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 0.54 | 0 | 1.42 | 10 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 00 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 26.56 | 90 | 17.50 | 3692 | exact_island | keep f32; do not quantize |
| 00 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.07 | 0 | 5.72 | 8 | exact_island | keep f32; do not quantize |
| 00 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.06 | 0 | 1.07 | 92 | exact_island | keep f32; do not quantize |
| 00 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 149.36 | 1 | 20.70 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9532) |
| 00 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 1.48 | 0 | 4.03 | 4209 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9794) |
| 00 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.29 | 0 | 1.71 | 5663 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9758) |
| 00 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 49.72 | 0 | 1.27 | 3212 | exact_island | keep f32; do not quantize |
| 00 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.64 | 8800 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9811 on L0); Q2/binary not intact mid-depth |
| 01 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.37 | 0 | 3.62 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 01 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 2.90 | 9537 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 01 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.55 | 0 | 1.42 | 3849 | exact_island | keep f32; do not quantize |
| 01 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.93 | 0 | 2.72 | 3 | exact_island | keep f32; do not quantize |
| 01 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.62 | 0 | 1.61 | 37 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 01 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.50 | 0 | 1.54 | 37 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 01 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 13.71 | 1 | 12.84 | 218 | exact_island | keep f32; do not quantize |
| 01 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.60 | 0 | 3.37 | 13 | exact_island | keep f32; do not quantize |
| 01 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 21.11 | 0 | 1.09 | 45 | exact_island | keep f32; do not quantize |
| 01 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 87.98 | 1 | 15.43 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 01 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.26 | 0 | 2.02 | 6624 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 01 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.29 | 0 | 2.03 | 2812 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 01 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 82.01 | 0 | 1.20 | 3849 | exact_island | keep f32; do not quantize |
| 01 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.10 | 0 | 1.36 | 10960 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 02 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 2.30 | 0 | 4.89 | 3994 | cheap_to_crush,exact_island,need_correction | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 02 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.13 | 0 | 2.29 | 7390 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 02 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 11.45 | 0 | 1.26 | 207 | exact_island | keep f32; do not quantize |
| 02 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.42 | 0 | 2.16 | 46 | exact_island | keep f32; do not quantize |
| 02 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.03 | 0 | 1.68 | 22 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 02 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.73 | 0 | 1.41 | 42 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 02 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 11.55 | 0 | 9.66 | 214 | exact_island | keep f32; do not quantize |
| 02 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.66 | 0 | 1.54 | 5 | exact_island | keep f32; do not quantize |
| 02 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 31.03 | 0 | 1.05 | 114 | exact_island | keep f32; do not quantize |
| 02 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 27.60 | 0 | 8.71 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 02 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.19 | 0 | 1.82 | 3539 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 02 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.20 | 0 | 1.99 | 4517 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 02 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 12.49 | 0 | 1.19 | 2304 | exact_island | keep f32; do not quantize |
| 02 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.06 | 0 | 1.84 | 7390 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 03 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.70 | 0 | 3.14 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9796 on L3); Q2/binary not intact mid-depth |
| 03 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.95 | 6135 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9772 on L3); Q2/binary not intact mid-depth |
| 03 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.16 | 0 | 1.53 | 310 | exact_island | keep f32; do not quantize |
| 03 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.72 | 0 | 1.83 | 290 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9849) |
| 03 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 11.15 | 0 | 1.40 | 24 | exact_island | keep f32; do not quantize |
| 03 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 132.15 | 1 | 17.76 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9834) |
| 03 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 17.84 | 0 | 1.21 | 839 | exact_island | keep f32; do not quantize |
| 03 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.64 | 0 | 2.30 | 8452 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9838) |
| 03 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 9.81 | 0 | 1.19 | 56 | exact_island | keep f32; do not quantize |
| 03 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.05 | 0 | 1.24 | 7506 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9736 on L3); Q2/binary not intact mid-depth |
| 03 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.45 | 0 | 1.36 | 753 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9797) |
| 04 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.14 | 0 | 2.05 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 04 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 2.17 | 9980 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 04 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 36.71 | 0 | 1.28 | 4152 | exact_island | keep f32; do not quantize |
| 04 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.85 | 0 | 2.50 | 35 | exact_island | keep f32; do not quantize |
| 04 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.08 | 0 | 1.50 | 7 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 04 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.46 | 0 | 1.19 | 4 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 04 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 37.34 | 3 | 22.03 | 2658 | exact_island | keep f32; do not quantize |
| 04 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.07 | 0 | 2.12 | 24 | exact_island | keep f32; do not quantize |
| 04 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 26.74 | 0 | 1.04 | 81 | exact_island | keep f32; do not quantize |
| 04 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 36.70 | 1 | 10.67 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 04 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.19 | 0 | 1.87 | 5014 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 04 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.22 | 0 | 1.50 | 1774 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 04 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 15.11 | 0 | 1.51 | 291 | exact_island | keep f32; do not quantize |
| 04 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.20 | 11906 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 05 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.27 | 0 | 2.97 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 05 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.72 | 10493 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 05 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.91 | 0 | 1.34 | 4615 | exact_island | keep f32; do not quantize |
| 05 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.06 | 0 | 4.61 | 29 | exact_island | keep f32; do not quantize |
| 05 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.91 | 0 | 1.33 | 43 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 05 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.80 | 0 | 1.37 | 16 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 05 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 17.04 | 4 | 12.56 | 718 | exact_island | keep f32; do not quantize |
| 05 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.13 | 0 | 2.76 | 7 | exact_island | keep f32; do not quantize |
| 05 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 31.12 | 0 | 1.07 | 36 | exact_island | keep f32; do not quantize |
| 05 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 23.58 | 0 | 9.60 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 05 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.22 | 0 | 1.67 | 8335 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 05 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.23 | 0 | 1.77 | 6026 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 05 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 15.85 | 0 | 1.14 | 3396 | exact_island | keep f32; do not quantize |
| 05 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.27 | 6005 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 06 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.90 | 0 | 2.99 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 06 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.29 | 0 | 1.91 | 7261 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 06 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.16 | 0 | 1.26 | 3684 | exact_island | keep f32; do not quantize |
| 06 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 1.11 | 0 | 6.38 | 31 | exact_island | keep f32; do not quantize |
| 06 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.56 | 0 | 1.38 | 13 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 06 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.76 | 0 | 1.32 | 45 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 06 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 16.18 | 2 | 17.45 | 8167 | exact_island | keep f32; do not quantize |
| 06 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.99 | 0 | 2.25 | 16 | exact_island | keep f32; do not quantize |
| 06 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 22.37 | 0 | 1.09 | 97 | exact_island | keep f32; do not quantize |
| 06 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 20.41 | 0 | 9.27 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 06 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.25 | 0 | 1.62 | 8167 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 06 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.31 | 0 | 2.17 | 3995 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 06 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 16.13 | 0 | 1.12 | 2479 | exact_island | keep f32; do not quantize |
| 06 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.17 | 0 | 1.38 | 10064 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 07 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.17 | 0 | 2.17 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 07 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.64 | 11009 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 07 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.89 | 0 | 1.96 | 310 | exact_island | keep f32; do not quantize |
| 07 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.97 | 0 | 1.57 | 801 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 07 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 12.89 | 0 | 1.35 | 21 | exact_island | keep f32; do not quantize |
| 07 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 28.45 | 1 | 11.71 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 07 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.80 | 0 | 1.09 | 2921 | exact_island | keep f32; do not quantize |
| 07 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.55 | 0 | 1.88 | 11287 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 07 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 13.25 | 0 | 1.11 | 22 | exact_island | keep f32; do not quantize |
| 07 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.33 | 7329 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 07 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.49 | 0 | 1.37 | 840 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 08 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.22 | 0 | 2.68 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 08 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.96 | 5821 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 08 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 26.70 | 0 | 1.25 | 2823 | exact_island | keep f32; do not quantize |
| 08 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.95 | 0 | 2.21 | 38 | exact_island | keep f32; do not quantize |
| 08 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.14 | 0 | 1.23 | 31 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 08 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.13 | 0 | 1.38 | 13 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 08 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 18.78 | 1 | 12.04 | 8024 | exact_island | keep f32; do not quantize |
| 08 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.06 | 0 | 1.92 | 46 | exact_island | keep f32; do not quantize |
| 08 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 28.18 | 0 | 1.07 | 106 | exact_island | keep f32; do not quantize |
| 08 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 18.29 | 1 | 10.02 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 08 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.22 | 0 | 1.69 | 1577 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 08 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.23 | 0 | 1.60 | 2612 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 08 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.89 | 0 | 1.24 | 2631 | exact_island | keep f32; do not quantize |
| 08 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.06 | 0 | 1.22 | 8592 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 09 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 2.83 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 09 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.10 | 0 | 1.76 | 757 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 09 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 15.43 | 0 | 1.26 | 2823 | exact_island | keep f32; do not quantize |
| 09 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 1.11 | 0 | 6.17 | 29 | exact_island | keep f32; do not quantize |
| 09 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.82 | 0 | 1.46 | 16 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 09 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.54 | 0 | 1.35 | 40 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 09 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 17.88 | 6 | 12.47 | 262 | exact_island | keep f32; do not quantize |
| 09 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.90 | 0 | 2.12 | 23 | exact_island | keep f32; do not quantize |
| 09 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 17.30 | 0 | 1.05 | 18 | exact_island | keep f32; do not quantize |
| 09 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 14.46 | 0 | 8.57 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 09 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.20 | 0 | 1.86 | 7711 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 09 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.21 | 0 | 1.65 | 3073 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 09 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.71 | 0 | 1.18 | 2631 | exact_island | keep f32; do not quantize |
| 09 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.06 | 0 | 1.35 | 9554 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 10 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.64 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 10 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.19 | 0 | 2.40 | 6795 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 10 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.93 | 0 | 1.29 | 2631 | exact_island | keep f32; do not quantize |
| 10 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.77 | 0 | 3.61 | 23 | exact_island | keep f32; do not quantize |
| 10 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.53 | 0 | 1.61 | 26 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 10 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.95 | 0 | 1.57 | 28 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 10 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 35.57 | 5 | 21.33 | 2722 | exact_island | keep f32; do not quantize |
| 10 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.24 | 0 | 1.65 | 35 | exact_island | keep f32; do not quantize |
| 10 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 36.33 | 0 | 1.06 | 4 | exact_island | keep f32; do not quantize |
| 10 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 10.50 | 0 | 7.76 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 10 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.23 | 0 | 1.67 | 7915 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 10 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.22 | 0 | 1.67 | 3806 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 10 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.23 | 0 | 1.48 | 2631 | exact_island | keep f32; do not quantize |
| 10 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.43 | 4897 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 11 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.29 | 0 | 2.96 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 11 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.14 | 0 | 2.75 | 5762 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 11 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.88 | 0 | 1.94 | 310 | exact_island | keep f32; do not quantize |
| 11 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.91 | 0 | 1.25 | 139 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 11 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 7.85 | 0 | 1.23 | 56 | exact_island | keep f32; do not quantize |
| 11 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 17.42 | 1 | 10.56 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 11 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.55 | 0 | 1.31 | 2631 | exact_island | keep f32; do not quantize |
| 11 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.94 | 0 | 4.82 | 420 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 11 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 14.93 | 0 | 1.10 | 21 | exact_island | keep f32; do not quantize |
| 11 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.32 | 14 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 11 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.46 | 0 | 1.22 | 846 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 12 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 2.05 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 12 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 2.04 | 8950 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 12 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.60 | 0 | 1.37 | 2631 | exact_island | keep f32; do not quantize |
| 12 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 3.36 | 0 | 4.66 | 29 | exact_island | keep f32; do not quantize |
| 12 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.28 | 0 | 1.45 | 0 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 12 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.66 | 0 | 1.53 | 46 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 12 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 22.91 | 9 | 13.22 | 2681 | exact_island | keep f32; do not quantize |
| 12 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.89 | 0 | 1.89 | 5 | exact_island | keep f32; do not quantize |
| 12 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 11.96 | 0 | 1.11 | 89 | exact_island | keep f32; do not quantize |
| 12 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 7.54 | 0 | 8.00 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 12 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.25 | 0 | 2.09 | 7102 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 12 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.23 | 0 | 1.80 | 4422 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 12 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.96 | 0 | 1.48 | 2631 | exact_island | keep f32; do not quantize |
| 12 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.06 | 0 | 1.32 | 11141 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 13 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.63 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 13 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 2.28 | 3301 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 13 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 13.11 | 0 | 1.36 | 2631 | exact_island | keep f32; do not quantize |
| 13 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.85 | 0 | 2.88 | 44 | exact_island | keep f32; do not quantize |
| 13 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.76 | 0 | 1.42 | 20 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 13 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.60 | 0 | 1.41 | 13 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 13 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 16.41 | 6 | 15.50 | 6164 | exact_island | keep f32; do not quantize |
| 13 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.58 | 0 | 1.82 | 27 | exact_island | keep f32; do not quantize |
| 13 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 23.42 | 0 | 1.09 | 66 | exact_island | keep f32; do not quantize |
| 13 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 3.45 | 0 | 5.62 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 13 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.16 | 0 | 1.59 | 7252 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 13 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.19 | 0 | 1.90 | 2132 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 13 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.19 | 0 | 1.40 | 3532 | exact_island | keep f32; do not quantize |
| 13 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.06 | 0 | 1.40 | 6851 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 14 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.60 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 14 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.08 | 0 | 1.46 | 4574 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 14 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.03 | 0 | 1.42 | 3532 | exact_island | keep f32; do not quantize |
| 14 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.56 | 0 | 3.45 | 7 | exact_island | keep f32; do not quantize |
| 14 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.30 | 0 | 1.33 | 30 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 14 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.42 | 0 | 1.35 | 30 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 14 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 19.25 | 8 | 16.54 | 5153 | exact_island | keep f32; do not quantize |
| 14 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.54 | 0 | 1.78 | 31 | exact_island | keep f32; do not quantize |
| 14 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 26.93 | 0 | 1.10 | 12 | exact_island | keep f32; do not quantize |
| 14 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 7.13 | 0 | 6.71 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 14 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.27 | 0 | 1.74 | 5495 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 14 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.21 | 0 | 1.75 | 1537 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 14 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 5.32 | 0 | 1.56 | 3532 | exact_island | keep f32; do not quantize |
| 14 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.06 | 0 | 1.29 | 4364 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 15 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.13 | 0 | 2.01 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9756 on L15); Q2/binary not intact mid-depth |
| 15 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.53 | 4914 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9777 on L15); Q2/binary not intact mid-depth |
| 15 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.67 | 0 | 1.87 | 310 | exact_island | keep f32; do not quantize |
| 15 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.90 | 0 | 1.29 | 612 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 15 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 16.04 | 0 | 1.31 | 24 | exact_island | keep f32; do not quantize |
| 15 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.71 | 0 | 5.72 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 15 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.22 | 0 | 1.43 | 3532 | exact_island | keep f32; do not quantize |
| 15 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.31 | 0 | 1.86 | 4129 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 15 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 47.33 | 0 | 1.08 | 21 | exact_island | keep f32; do not quantize |
| 15 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.30 | 4914 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9744 on L15); Q2/binary not intact mid-depth |
| 15 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.23 | 0 | 1.25 | 302 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 16 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 2.55 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 16 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 1.88 | 13279 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 16 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.59 | 0 | 1.35 | 3532 | exact_island | keep f32; do not quantize |
| 16 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.29 | 0 | 2.11 | 13 | exact_island | keep f32; do not quantize |
| 16 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 0.97 | 0 | 1.58 | 30 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 16 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.49 | 0 | 1.50 | 23 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 16 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 26.14 | 8 | 16.44 | 2575 | exact_island | keep f32; do not quantize |
| 16 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.58 | 0 | 1.55 | 41 | exact_island | keep f32; do not quantize |
| 16 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 12.96 | 0 | 1.08 | 63 | exact_island | keep f32; do not quantize |
| 16 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 5.25 | 0 | 6.15 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 16 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.23 | 0 | 1.74 | 9433 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 16 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.23 | 0 | 1.84 | 5015 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 16 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.26 | 0 | 1.50 | 3532 | exact_island | keep f32; do not quantize |
| 16 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.26 | 8905 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 17 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.19 | 0 | 2.79 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 17 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.15 | 0 | 1.57 | 7830 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 17 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.74 | 0 | 1.30 | 3532 | exact_island | keep f32; do not quantize |
| 17 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.66 | 0 | 2.65 | 27 | exact_island | keep f32; do not quantize |
| 17 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.13 | 0 | 1.37 | 23 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 17 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.48 | 0 | 1.22 | 26 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 17 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 19.47 | 6 | 13.08 | 9249 | exact_island | keep f32; do not quantize |
| 17 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.08 | 0 | 1.84 | 9 | exact_island | keep f32; do not quantize |
| 17 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 20.15 | 0 | 1.07 | 44 | exact_island | keep f32; do not quantize |
| 17 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.84 | 0 | 4.71 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 17 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.23 | 0 | 1.64 | 1545 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 17 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.24 | 0 | 1.94 | 2234 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 17 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.27 | 0 | 1.40 | 3532 | exact_island | keep f32; do not quantize |
| 17 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.35 | 9787 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 18 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 2.19 | 0 | 4.18 | 3994 | cheap_to_crush,exact_island,need_correction | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 18 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.47 | 0 | 1.79 | 14590 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 18 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.72 | 0 | 1.34 | 3532 | exact_island | keep f32; do not quantize |
| 18 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.75 | 0 | 2.16 | 4 | exact_island | keep f32; do not quantize |
| 18 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 0.93 | 0 | 1.64 | 10 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 18 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.82 | 0 | 1.23 | 27 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 18 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 30.93 | 32 | 16.83 | 7796 | exact_island | keep f32; do not quantize |
| 18 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.72 | 0 | 2.00 | 1 | exact_island | keep f32; do not quantize |
| 18 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 12.85 | 0 | 1.09 | 45 | exact_island | keep f32; do not quantize |
| 18 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 6.24 | 0 | 5.68 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 18 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.27 | 0 | 1.80 | 8507 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 18 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.30 | 0 | 1.97 | 30 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 18 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 24.05 | 0 | 1.46 | 3532 | exact_island | keep f32; do not quantize |
| 18 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.37 | 11391 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 19 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.55 | 0 | 4.01 | 3994 | cheap_to_crush,exact_island,need_correction | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 19 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 1.62 | 5266 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 19 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.25 | 0 | 1.58 | 310 | exact_island | keep f32; do not quantize |
| 19 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.24 | 0 | 1.27 | 32 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 19 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 18.66 | 0 | 1.23 | 53 | exact_island | keep f32; do not quantize |
| 19 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 3.50 | 0 | 4.73 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 19 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.96 | 0 | 1.24 | 2316 | exact_island | keep f32; do not quantize |
| 19 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.33 | 0 | 2.02 | 11808 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 19 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 47.98 | 0 | 1.12 | 21 | exact_island | keep f32; do not quantize |
| 19 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.08 | 0 | 1.39 | 681 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 19 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.36 | 0 | 1.26 | 789 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 20 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 3.21 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 20 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.77 | 15346 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 20 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 30.51 | 0 | 1.19 | 2316 | exact_island | keep f32; do not quantize |
| 20 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.24 | 0 | 2.42 | 7 | exact_island | keep f32; do not quantize |
| 20 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.33 | 0 | 1.27 | 12 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 20 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.84 | 0 | 1.29 | 25 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 20 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 24.36 | 42 | 14.75 | 5663 | exact_island | keep f32; do not quantize |
| 20 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.39 | 0 | 1.65 | 31 | exact_island | keep f32; do not quantize |
| 20 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.75 | 0 | 1.10 | 73 | exact_island | keep f32; do not quantize |
| 20 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 3.68 | 0 | 5.68 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 20 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.31 | 0 | 1.91 | 7760 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 20 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.26 | 0 | 1.71 | 5796 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 20 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.50 | 0 | 1.22 | 2316 | exact_island | keep f32; do not quantize |
| 20 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.08 | 0 | 1.18 | 5453 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 21 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.15 | 0 | 1.79 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 21 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.10 | 0 | 1.67 | 16097 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 21 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 16.60 | 0 | 1.24 | 2316 | exact_island | keep f32; do not quantize |
| 21 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.65 | 0 | 1.54 | 19 | exact_island | keep f32; do not quantize |
| 21 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.49 | 0 | 1.16 | 16 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 21 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.78 | 0 | 1.24 | 17 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 21 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 26.72 | 24 | 13.26 | 5796 | exact_island | keep f32; do not quantize |
| 21 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.08 | 0 | 1.80 | 40 | exact_island | keep f32; do not quantize |
| 21 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 5.31 | 0 | 1.08 | 54 | exact_island | keep f32; do not quantize |
| 21 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 4.69 | 0 | 6.03 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 21 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.38 | 0 | 1.78 | 8945 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 21 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.30 | 0 | 1.65 | 5693 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 21 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 5.51 | 0 | 1.19 | 2316 | exact_island | keep f32; do not quantize |
| 21 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.07 | 0 | 1.34 | 3026 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 22 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.73 | 0 | 3.10 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 22 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.29 | 0 | 1.94 | 8585 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 22 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 15.34 | 0 | 1.23 | 2316 | exact_island | keep f32; do not quantize |
| 22 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 2.21 | 0 | 1.44 | 25 | exact_island | keep f32; do not quantize |
| 22 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.03 | 0 | 1.21 | 38 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 22 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.75 | 0 | 1.17 | 29 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 22 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 51.52 | 37 | 14.12 | 9852 | exact_island | keep f32; do not quantize |
| 22 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.93 | 0 | 4.54 | 45 | exact_island | keep f32; do not quantize |
| 22 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 1.11 | 0 | 1.10 | 76 | exact_island | keep f32; do not quantize |
| 22 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 3.53 | 0 | 5.55 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 22 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.23 | 0 | 1.70 | 9587 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 22 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.21 | 0 | 1.68 | 5010 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 22 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.36 | 0 | 1.19 | 2316 | exact_island | keep f32; do not quantize |
| 22 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.21 | 0 | 1.20 | 10142 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 23 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.68 | 0 | 2.15 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 23 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.50 | 3797 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 23 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 13.01 | 0 | 1.58 | 310 | exact_island | keep f32; do not quantize |
| 23 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.34 | 0 | 1.20 | 409 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 23 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 18.44 | 0 | 1.48 | 25 | exact_island | keep f32; do not quantize |
| 23 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.89 | 0 | 4.26 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 23 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.18 | 0 | 1.13 | 2316 | exact_island | keep f32; do not quantize |
| 23 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.35 | 0 | 2.03 | 9728 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 23 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 32.58 | 0 | 1.21 | 57 | exact_island | keep f32; do not quantize |
| 23 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.09 | 0 | 1.26 | 11164 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 23 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.75 | 0 | 1.43 | 554 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 24 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.03 | 0 | 2.59 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 24 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.41 | 5432 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 24 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 27.67 | 0 | 1.31 | 3986 | exact_island | keep f32; do not quantize |
| 24 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -1.04 | 0 | 2.18 | 38 | exact_island | keep f32; do not quantize |
| 24 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.66 | 0 | 1.44 | 13 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 24 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.98 | 0 | 1.40 | 13 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 24 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 22.23 | 4 | 12.01 | 2135 | exact_island | keep f32; do not quantize |
| 24 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.97 | 0 | 1.91 | 46 | exact_island | keep f32; do not quantize |
| 24 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 11.61 | 0 | 1.16 | 12 | exact_island | keep f32; do not quantize |
| 24 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.77 | 0 | 5.09 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 24 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.36 | 0 | 1.59 | 1577 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 24 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.30 | 0 | 1.67 | 1271 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 24 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.05 | 0 | 1.29 | 2631 | exact_island | keep f32; do not quantize |
| 24 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.13 | 0 | 1.21 | 9906 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 25 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.50 | 0 | 2.76 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 25 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 1.29 | 1439 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 25 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 19.03 | 0 | 1.34 | 1089 | exact_island | keep f32; do not quantize |
| 25 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 2.06 | 0 | 3.50 | 29 | exact_island | keep f32; do not quantize |
| 25 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.01 | 0 | 1.46 | 3 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 25 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.82 | 0 | 1.45 | 21 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 25 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 18.47 | 1 | 10.98 | 1316 | exact_island | keep f32; do not quantize |
| 25 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.98 | 0 | 2.21 | 21 | exact_island | keep f32; do not quantize |
| 25 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 11.28 | 0 | 1.14 | 37 | exact_island | keep f32; do not quantize |
| 25 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.11 | 0 | 4.69 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 25 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.35 | 0 | 1.61 | 7033 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 25 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.26 | 0 | 1.62 | 3508 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 25 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.23 | 0 | 1.24 | 2316 | exact_island | keep f32; do not quantize |
| 25 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.14 | 0 | 1.27 | 4365 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 26 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.98 | 0 | 3.46 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 26 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.19 | 0 | 1.47 | 2347 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 26 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 14.16 | 0 | 1.42 | 2631 | exact_island | keep f32; do not quantize |
| 26 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.77 | 0 | 2.82 | 23 | exact_island | keep f32; do not quantize |
| 26 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.14 | 0 | 1.41 | 26 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 26 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.73 | 0 | 1.43 | 28 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 26 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 22.64 | 7 | 17.32 | 7659 | exact_island | keep f32; do not quantize |
| 26 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.22 | 0 | 1.62 | 35 | exact_island | keep f32; do not quantize |
| 26 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 13.01 | 0 | 1.30 | 125 | exact_island | keep f32; do not quantize |
| 26 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.31 | 0 | 4.72 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 26 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.42 | 0 | 1.57 | 2722 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 26 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.32 | 0 | 1.73 | 4989 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 26 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.58 | 0 | 1.46 | 2631 | exact_island | keep f32; do not quantize |
| 26 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 1.45 | 13624 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 27 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.52 | 0 | 2.61 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 27 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.15 | 0 | 1.48 | 1497 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 27 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.46 | 0 | 1.70 | 310 | exact_island | keep f32; do not quantize |
| 27 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 2.08 | 0 | 1.29 | 134 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 27 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 14.45 | 0 | 1.66 | 56 | exact_island | keep f32; do not quantize |
| 27 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 0.93 | 0 | 3.37 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 27 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.40 | 0 | 1.36 | 1689 | exact_island | keep f32; do not quantize |
| 27 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.80 | 0 | 3.11 | 9273 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 27 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 17.30 | 0 | 1.53 | 24 | exact_island | keep f32; do not quantize |
| 27 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.47 | 6896 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 27 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.02 | 0 | 1.52 | 845 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 28 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.35 | 0 | 2.17 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 28 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.19 | 0 | 1.80 | 9165 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 28 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.80 | 0 | 1.56 | 2631 | exact_island | keep f32; do not quantize |
| 28 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 2.88 | 0 | 3.23 | 29 | exact_island | keep f32; do not quantize |
| 28 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.97 | 0 | 1.44 | 1 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 28 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 5.33 | 0 | 1.45 | 22 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 28 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 27.11 | 18 | 16.85 | 3849 | exact_island | keep f32; do not quantize |
| 28 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.70 | 0 | 1.87 | 3 | exact_island | keep f32; do not quantize |
| 28 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 9.93 | 0 | 1.21 | 120 | exact_island | keep f32; do not quantize |
| 28 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.38 | 0 | 4.73 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 28 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.54 | 0 | 1.71 | 1870 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 28 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.38 | 0 | 1.81 | 4422 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 28 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.89 | 0 | 1.55 | 2631 | exact_island | keep f32; do not quantize |
| 28 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.37 | 7194 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 29 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.41 | 0 | 2.47 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 29 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.17 | 0 | 1.68 | 10855 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 29 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.13 | 0 | 1.59 | 2631 | exact_island | keep f32; do not quantize |
| 29 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.94 | 0 | 2.45 | 42 | exact_island | keep f32; do not quantize |
| 29 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.56 | 0 | 1.50 | 16 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 29 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.86 | 0 | 1.49 | 20 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 29 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 16.95 | 5 | 15.30 | 4425 | exact_island | keep f32; do not quantize |
| 29 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.56 | 0 | 1.91 | 29 | exact_island | keep f32; do not quantize |
| 29 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 7.25 | 0 | 1.21 | 53 | exact_island | keep f32; do not quantize |
| 29 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.61 | 0 | 4.21 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 29 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.33 | 0 | 1.52 | 355 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 29 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.32 | 0 | 1.93 | 4057 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 29 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.57 | 0 | 1.59 | 3532 | exact_island | keep f32; do not quantize |
| 29 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.42 | 12981 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 30 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.75 | 0 | 3.01 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 30 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.22 | 0 | 1.68 | 3557 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 30 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.61 | 0 | 1.73 | 3532 | exact_island | keep f32; do not quantize |
| 30 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.85 | 0 | 2.63 | 7 | exact_island | keep f32; do not quantize |
| 30 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.44 | 0 | 1.40 | 37 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 30 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.00 | 0 | 1.47 | 7 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 30 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 23.27 | 6 | 15.75 | 3278 | exact_island | keep f32; do not quantize |
| 30 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.51 | 0 | 1.75 | 31 | exact_island | keep f32; do not quantize |
| 30 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 1.88 | 0 | 1.30 | 60 | exact_island | keep f32; do not quantize |
| 30 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.40 | 0 | 5.26 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 30 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.47 | 0 | 1.81 | 146 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 30 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.44 | 0 | 1.77 | 4459 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 30 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.52 | 0 | 1.78 | 3532 | exact_island | keep f32; do not quantize |
| 30 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.22 | 0 | 1.59 | 16280 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 31 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.61 | 0 | 2.80 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9742 on L31); Q2/binary not intact mid-depth |
| 31 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.26 | 0 | 1.84 | 9047 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9766 on L31); Q2/binary not intact mid-depth |
| 31 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.98 | 0 | 1.62 | 3532 | exact_island | keep f32; do not quantize |
| 31 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.91 | 0 | 1.46 | 854 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 31 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 30.40 | 0 | 1.82 | 24 | exact_island | keep f32; do not quantize |
| 31 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.04 | 0 | 3.36 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 31 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.38 | 0 | 1.76 | 3532 | exact_island | keep f32; do not quantize |
| 31 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.53 | 0 | 2.56 | 11778 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 31 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 37.53 | 0 | 1.56 | 56 | exact_island | keep f32; do not quantize |
| 31 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.22 | 0 | 1.49 | 14156 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9679 on L31); Q2/binary not intact mid-depth |
| 31 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.74 | 0 | 1.41 | 827 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 32 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.67 | 0 | 3.10 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 32 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.35 | 0 | 2.04 | 4227 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 32 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.32 | 0 | 1.65 | 3532 | exact_island | keep f32; do not quantize |
| 32 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.32 | 0 | 1.95 | 13 | exact_island | keep f32; do not quantize |
| 32 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.17 | 0 | 1.51 | 32 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 32 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 8.20 | 0 | 1.49 | 4 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 32 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 27.67 | 18 | 20.13 | 2575 | exact_island | keep f32; do not quantize |
| 32 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.73 | 0 | 1.50 | 31 | exact_island | keep f32; do not quantize |
| 32 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.50 | 0 | 1.20 | 100 | exact_island | keep f32; do not quantize |
| 32 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.07 | 0 | 4.73 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9678) |
| 32 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.49 | 0 | 1.68 | 493 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9755) |
| 32 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.37 | 0 | 1.75 | 5015 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9848) |
| 32 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.51 | 0 | 1.74 | 3532 | exact_island | keep f32; do not quantize |
| 32 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.24 | 0 | 1.53 | 12153 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 33 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.42 | 0 | 2.67 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 33 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.49 | 0 | 2.77 | 10848 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 33 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.82 | 0 | 1.60 | 2205 | exact_island | keep f32; do not quantize |
| 33 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.75 | 0 | 2.18 | 27 | exact_island | keep f32; do not quantize |
| 33 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.53 | 0 | 1.58 | 10 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 33 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.93 | 0 | 1.79 | 9 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 33 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 26.11 | 23 | 18.58 | 2109 | exact_island | keep f32; do not quantize |
| 33 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.30 | 0 | 1.88 | 9 | exact_island | keep f32; do not quantize |
| 33 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 1.61 | 0 | 1.17 | 115 | exact_island | keep f32; do not quantize |
| 33 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.65 | 0 | 4.20 | 3994 | need_richer_levels,exact_island,need_correction | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 33 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.49 | 0 | 1.64 | 60 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 33 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.40 | 0 | 1.87 | 1997 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 33 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.35 | 0 | 1.64 | 3532 | exact_island | keep f32; do not quantize |
| 33 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.28 | 0 | 1.55 | 10363 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 34 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.14 | 0 | 3.46 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 34 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 1.48 | 0 | 2.56 | 2618 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 34 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.54 | 0 | 1.56 | 2205 | exact_island | keep f32; do not quantize |
| 34 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 1.42 | 0 | 1.86 | 40 | exact_island | keep f32; do not quantize |
| 34 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 1.57 | 0 | 1.53 | 10 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 34 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 9.44 | 0 | 1.76 | 9 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 34 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 21.67 | 21 | 14.94 | 3646 | exact_island | keep f32; do not quantize |
| 34 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.67 | 0 | 2.04 | 1 | exact_island | keep f32; do not quantize |
| 34 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 2.12 | 0 | 1.13 | 45 | exact_island | keep f32; do not quantize |
| 34 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.15 | 0 | 3.94 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 34 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.48 | 0 | 1.78 | 6208 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 34 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.75 | 0 | 1.73 | 4456 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 34 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.24 | 0 | 1.65 | 3532 | exact_island | keep f32; do not quantize |
| 34 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.34 | 0 | 1.60 | 15989 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 35 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.69 | 0 | 2.68 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 35 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.62 | 0 | 2.29 | 1875 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 35 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.49 | 0 | 1.51 | 4316 | exact_island | keep f32; do not quantize |
| 35 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.60 | 0 | 1.31 | 401 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 35 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 22.43 | 0 | 1.67 | 25 | exact_island | keep f32; do not quantize |
| 35 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.23 | 0 | 3.59 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 35 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.14 | 0 | 1.46 | 2316 | exact_island | keep f32; do not quantize |
| 35 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.74 | 0 | 2.77 | 11296 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 35 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 38.13 | 0 | 1.34 | 57 | exact_island | keep f32; do not quantize |
| 35 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.22 | 0 | 1.62 | 3015 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 35 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.43 | 0 | 1.33 | 589 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 36 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.24 | 0 | 2.10 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 36 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.60 | 0 | 2.51 | 3838 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 36 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 14.57 | 0 | 1.32 | 2482 | exact_island | keep f32; do not quantize |
| 36 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.50 | 0 | 1.84 | 7 | exact_island | keep f32; do not quantize |
| 36 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.09 | 0 | 1.29 | 12 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 36 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 5.47 | 0 | 1.45 | 25 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 36 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 17.11 | 7 | 11.27 | 8762 | exact_island | keep f32; do not quantize |
| 36 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.44 | 0 | 1.71 | 31 | exact_island | keep f32; do not quantize |
| 36 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | -0.38 | 0 | 1.17 | 93 | exact_island | keep f32; do not quantize |
| 36 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.72 | 0 | 3.60 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 36 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.45 | 0 | 1.98 | 1832 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 36 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.32 | 0 | 1.48 | 5727 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 36 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.49 | 0 | 1.49 | 2316 | exact_island | keep f32; do not quantize |
| 36 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.16 | 0 | 1.60 | 9109 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 37 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 1.67 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 37 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.28 | 0 | 2.06 | 7949 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 37 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.79 | 0 | 1.38 | 2768 | exact_island | keep f32; do not quantize |
| 37 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.64 | 0 | 1.52 | 14 | exact_island | keep f32; do not quantize |
| 37 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.25 | 0 | 1.55 | 33 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 37 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.62 | 0 | 1.39 | 38 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 37 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 16.74 | 2 | 10.41 | 7051 | exact_island | keep f32; do not quantize |
| 37 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.23 | 0 | 1.92 | 40 | exact_island | keep f32; do not quantize |
| 37 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 2.18 | 0 | 1.14 | 113 | exact_island | keep f32; do not quantize |
| 37 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.80 | 0 | 3.59 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 37 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.55 | 0 | 1.63 | 4683 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 37 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.36 | 0 | 1.50 | 5693 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 37 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.31 | 0 | 1.38 | 2316 | exact_island | keep f32; do not quantize |
| 37 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.12 | 0 | 1.56 | 13801 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 38 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.23 | 0 | 3.41 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 38 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.57 | 0 | 1.97 | 8172 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 38 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.00 | 0 | 1.35 | 2768 | exact_island | keep f32; do not quantize |
| 38 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 4.35 | 0 | 1.26 | 34 | exact_island | keep f32; do not quantize |
| 38 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.73 | 0 | 1.31 | 36 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 38 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.34 | 0 | 1.34 | 12 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 38 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 28.14 | 7 | 11.61 | 6651 | exact_island | keep f32; do not quantize |
| 38 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.97 | 0 | 3.48 | 45 | exact_island | keep f32; do not quantize |
| 38 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 2.05 | 0 | 1.24 | 76 | exact_island | keep f32; do not quantize |
| 38 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.53 | 0 | 3.30 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 38 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.29 | 0 | 1.58 | 1515 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 38 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.25 | 0 | 1.51 | 5656 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 38 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.65 | 0 | 1.35 | 2316 | exact_island | keep f32; do not quantize |
| 38 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.48 | 4981 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 39 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.50 | 0 | 2.42 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 39 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.23 | 0 | 1.69 | 3797 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 39 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 20.18 | 0 | 1.61 | 4316 | exact_island | keep f32; do not quantize |
| 39 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.85 | 0 | 1.26 | 354 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 39 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 18.57 | 0 | 1.73 | 25 | exact_island | keep f32; do not quantize |
| 39 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.35 | 0 | 3.65 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 39 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.98 | 0 | 1.27 | 2316 | exact_island | keep f32; do not quantize |
| 39 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.60 | 0 | 2.40 | 9249 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 39 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 26.80 | 0 | 1.42 | 57 | exact_island | keep f32; do not quantize |
| 39 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.58 | 2728 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 39 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.01 | 0 | 1.95 | 341 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 40 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.74 | 0 | 2.79 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 40 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.27 | 0 | 1.91 | 10149 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 40 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.12 | 0 | 1.43 | 2631 | exact_island | keep f32; do not quantize |
| 40 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -1.00 | 0 | 1.93 | 36 | exact_island | keep f32; do not quantize |
| 40 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 5.14 | 0 | 1.54 | 32 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 40 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.22 | 0 | 1.62 | 47 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 40 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 22.37 | 9 | 13.30 | 8179 | exact_island | keep f32; do not quantize |
| 40 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.12 | 0 | 1.99 | 46 | exact_island | keep f32; do not quantize |
| 40 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 6.96 | 0 | 1.26 | 12 | exact_island | keep f32; do not quantize |
| 40 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.16 | 0 | 3.99 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 40 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.53 | 0 | 1.70 | 8100 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 40 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.45 | 0 | 1.51 | 3702 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 40 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.33 | 0 | 1.44 | 2631 | exact_island | keep f32; do not quantize |
| 40 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.14 | 0 | 1.27 | 15960 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 41 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.62 | 0 | 2.85 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 41 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.28 | 0 | 1.63 | 6631 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 41 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.76 | 0 | 1.36 | 2631 | exact_island | keep f32; do not quantize |
| 41 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 1.53 | 0 | 2.74 | 27 | exact_island | keep f32; do not quantize |
| 41 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.89 | 0 | 1.38 | 40 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 41 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 5.88 | 0 | 1.40 | 27 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 41 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 20.51 | 6 | 12.31 | 3788 | exact_island | keep f32; do not quantize |
| 41 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.06 | 0 | 2.20 | 21 | exact_island | keep f32; do not quantize |
| 41 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 5.92 | 0 | 1.28 | 119 | exact_island | keep f32; do not quantize |
| 41 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.18 | 0 | 3.54 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 41 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.36 | 0 | 1.69 | 265 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 41 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.31 | 0 | 1.64 | 5507 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 41 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.08 | 0 | 1.40 | 2316 | exact_island | keep f32; do not quantize |
| 41 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.17 | 0 | 1.33 | 8154 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 42 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.89 | 0 | 3.29 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 42 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.41 | 0 | 2.19 | 10050 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 42 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.03 | 0 | 1.53 | 2631 | exact_island | keep f32; do not quantize |
| 42 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.85 | 0 | 2.64 | 14 | exact_island | keep f32; do not quantize |
| 42 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.70 | 0 | 1.39 | 28 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 42 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.62 | 0 | 2.00 | 16 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 42 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 21.80 | 12 | 14.03 | 2125 | exact_island | keep f32; do not quantize |
| 42 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.22 | 0 | 1.66 | 35 | exact_island | keep f32; do not quantize |
| 42 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 9.67 | 0 | 1.32 | 125 | exact_island | keep f32; do not quantize |
| 42 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.02 | 0 | 3.81 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 42 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.45 | 0 | 1.81 | 2722 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 42 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.37 | 0 | 1.62 | 3827 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 42 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.82 | 0 | 1.63 | 2631 | exact_island | keep f32; do not quantize |
| 42 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.27 | 0 | 1.70 | 11305 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 43 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.72 | 0 | 2.65 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 43 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.28 | 0 | 2.20 | 5762 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 43 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.07 | 0 | 1.62 | 310 | exact_island | keep f32; do not quantize |
| 43 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 3.17 | 0 | 1.47 | 134 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 43 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 20.99 | 0 | 1.88 | 56 | exact_island | keep f32; do not quantize |
| 43 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.15 | 0 | 3.22 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 43 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.43 | 0 | 1.44 | 1689 | exact_island | keep f32; do not quantize |
| 43 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 1.11 | 0 | 3.32 | 7680 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 43 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 25.89 | 0 | 1.45 | 24 | exact_island | keep f32; do not quantize |
| 43 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.18 | 0 | 1.40 | 14765 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 43 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.55 | 0 | 1.51 | 845 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 44 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.39 | 0 | 2.34 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 44 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.33 | 0 | 1.98 | 8950 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 44 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.36 | 0 | 1.55 | 2631 | exact_island | keep f32; do not quantize |
| 44 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 2.79 | 0 | 3.16 | 27 | exact_island | keep f32; do not quantize |
| 44 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.06 | 0 | 1.61 | 1 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 44 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.52 | 0 | 1.47 | 1 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 44 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 19.90 | 20 | 15.12 | 3849 | exact_island | keep f32; do not quantize |
| 44 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.80 | 0 | 1.94 | 3 | exact_island | keep f32; do not quantize |
| 44 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 3.24 | 0 | 1.25 | 120 | exact_island | keep f32; do not quantize |
| 44 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.30 | 0 | 3.62 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 44 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.48 | 0 | 1.90 | 1519 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 44 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.51 | 0 | 1.76 | 3037 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 44 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.26 | 0 | 1.62 | 1689 | exact_island | keep f32; do not quantize |
| 44 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 1.64 | 13897 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 45 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.35 | 0 | 2.37 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 45 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.30 | 0 | 1.85 | 7010 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 45 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.32 | 0 | 1.54 | 3532 | exact_island | keep f32; do not quantize |
| 45 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.94 | 0 | 2.30 | 42 | exact_island | keep f32; do not quantize |
| 45 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.35 | 0 | 1.67 | 16 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 45 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.09 | 0 | 1.51 | 14 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 45 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 17.94 | 10 | 15.79 | 740 | exact_island | keep f32; do not quantize |
| 45 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.55 | 0 | 1.97 | 27 | exact_island | keep f32; do not quantize |
| 45 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 3.33 | 0 | 1.25 | 62 | exact_island | keep f32; do not quantize |
| 45 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.17 | 0 | 3.15 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 45 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.35 | 0 | 1.61 | 355 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 45 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.28 | 0 | 1.67 | 4057 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 45 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.00 | 0 | 1.59 | 3532 | exact_island | keep f32; do not quantize |
| 45 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.19 | 0 | 1.49 | 12981 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 46 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.60 | 0 | 3.02 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 46 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.33 | 0 | 2.13 | 10720 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 46 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.79 | 0 | 1.63 | 2205 | exact_island | keep f32; do not quantize |
| 46 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.75 | 0 | 2.50 | 7 | exact_island | keep f32; do not quantize |
| 46 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.72 | 0 | 1.52 | 44 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 46 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.40 | 0 | 1.44 | 6 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 46 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 22.27 | 10 | 14.13 | 3278 | exact_island | keep f32; do not quantize |
| 46 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.50 | 0 | 1.88 | 31 | exact_island | keep f32; do not quantize |
| 46 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 1.12 | 0 | 1.38 | 60 | exact_island | keep f32; do not quantize |
| 46 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.70 | 0 | 3.66 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 46 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.46 | 0 | 1.86 | 146 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 46 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.50 | 0 | 1.61 | 4459 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 46 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.03 | 0 | 1.70 | 3532 | exact_island | keep f32; do not quantize |
| 46 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.23 | 0 | 1.74 | 16280 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 47 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.67 | 0 | 2.35 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9729 on L47); Q2/binary not intact mid-depth |
| 47 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.39 | 0 | 2.11 | 8788 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9781 on L47); Q2/binary not intact mid-depth |
| 47 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.92 | 0 | 1.52 | 4316 | exact_island | keep f32; do not quantize |
| 47 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 2.21 | 0 | 1.70 | 1008 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 47 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 22.14 | 0 | 1.94 | 24 | exact_island | keep f32; do not quantize |
| 47 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 0.76 | 0 | 2.93 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 47 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.02 | 0 | 1.57 | 3532 | exact_island | keep f32; do not quantize |
| 47 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.93 | 0 | 3.33 | 11778 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 47 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 33.12 | 0 | 1.45 | 56 | exact_island | keep f32; do not quantize |
| 47 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.23 | 0 | 1.51 | 16338 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9693 on L47); Q2/binary not intact mid-depth |
| 47 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 0.82 | 0 | 1.61 | 133 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 48 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.60 | 0 | 2.49 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 48 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.70 | 0 | 2.15 | 3782 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 48 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.39 | 0 | 1.48 | 1689 | exact_island | keep f32; do not quantize |
| 48 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.36 | 0 | 1.83 | 13 | exact_island | keep f32; do not quantize |
| 48 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.27 | 0 | 1.44 | 30 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 48 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.38 | 0 | 1.41 | 4 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 48 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 21.63 | 6 | 18.39 | 2575 | exact_island | keep f32; do not quantize |
| 48 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.83 | 0 | 1.49 | 4 | exact_island | keep f32; do not quantize |
| 48 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | -0.04 | 0 | 1.24 | 100 | exact_island | keep f32; do not quantize |
| 48 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.76 | 0 | 3.76 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 48 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.46 | 0 | 1.74 | 9433 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 48 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.38 | 0 | 1.62 | 938 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 48 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.74 | 0 | 1.59 | 3532 | exact_island | keep f32; do not quantize |
| 48 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.26 | 0 | 1.43 | 15255 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 49 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.60 | 0 | 2.54 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 49 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 1.19 | 0 | 2.01 | 2018 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 49 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.72 | 0 | 1.42 | 2205 | exact_island | keep f32; do not quantize |
| 49 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.48 | 0 | 2.27 | 27 | exact_island | keep f32; do not quantize |
| 49 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.08 | 0 | 1.62 | 26 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 49 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.16 | 0 | 1.28 | 45 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 49 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 17.67 | 7 | 14.11 | 2068 | exact_island | keep f32; do not quantize |
| 49 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.30 | 0 | 1.96 | 11 | exact_island | keep f32; do not quantize |
| 49 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.77 | 0 | 1.30 | 115 | exact_island | keep f32; do not quantize |
| 49 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.46 | 0 | 3.67 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 49 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.48 | 0 | 2.03 | 9118 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 49 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.29 | 0 | 1.61 | 2253 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 49 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.16 | 0 | 1.51 | 1689 | exact_island | keep f32; do not quantize |
| 49 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.28 | 0 | 1.50 | 15176 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 50 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.20 | 0 | 3.09 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 50 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 1.55 | 0 | 2.13 | 2618 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 50 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.19 | 0 | 1.39 | 2205 | exact_island | keep f32; do not quantize |
| 50 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 1.37 | 0 | 1.88 | 4 | exact_island | keep f32; do not quantize |
| 50 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.57 | 0 | 1.38 | 40 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 50 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.88 | 0 | 1.64 | 11 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 50 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 20.21 | 6 | 12.80 | 837 | exact_island | keep f32; do not quantize |
| 50 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.68 | 0 | 2.27 | 1 | exact_island | keep f32; do not quantize |
| 50 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.04 | 0 | 1.21 | 24 | exact_island | keep f32; do not quantize |
| 50 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.98 | 0 | 2.98 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 50 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.58 | 0 | 2.09 | 289 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 50 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.44 | 0 | 1.65 | 1229 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 50 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 0.63 | 0 | 1.39 | 3532 | exact_island | keep f32; do not quantize |
| 50 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.40 | 0 | 1.55 | 13352 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 51 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.82 | 0 | 2.29 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 51 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.85 | 0 | 1.94 | 4748 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 51 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.39 | 0 | 1.43 | 4316 | exact_island | keep f32; do not quantize |
| 51 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.88 | 0 | 1.57 | 514 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 51 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 13.42 | 0 | 1.62 | 24 | exact_island | keep f32; do not quantize |
| 51 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.53 | 0 | 3.42 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 51 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 2.96 | 0 | 1.40 | 1689 | exact_island | keep f32; do not quantize |
| 51 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 1.07 | 0 | 3.25 | 7680 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 51 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 27.42 | 0 | 1.42 | 21 | exact_island | keep f32; do not quantize |
| 51 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.29 | 0 | 1.46 | 13306 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 51 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 2.60 | 0 | 1.54 | 486 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 52 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.56 | 0 | 1.62 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 52 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.74 | 0 | 1.98 | 3058 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 52 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 6.74 | 0 | 1.30 | 4349 | exact_island | keep f32; do not quantize |
| 52 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 2.16 | 0 | 1.63 | 7 | exact_island | keep f32; do not quantize |
| 52 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.31 | 0 | 1.61 | 1 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 52 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.69 | 0 | 1.40 | 26 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 52 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 11.23 | 1 | 10.11 | 277 | exact_island | keep f32; do not quantize |
| 52 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.51 | 0 | 1.83 | 31 | exact_island | keep f32; do not quantize |
| 52 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.05 | 0 | 1.19 | 36 | exact_island | keep f32; do not quantize |
| 52 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.81 | 0 | 2.77 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 52 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.59 | 0 | 2.03 | 1832 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 52 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.39 | 0 | 1.64 | 2026 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 52 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.93 | 0 | 1.51 | 1689 | exact_island | keep f32; do not quantize |
| 52 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.21 | 0 | 1.36 | 9529 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 53 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.43 | 0 | 1.99 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 53 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.71 | 0 | 1.87 | 12507 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 53 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 1.94 | 0 | 1.30 | 3986 | exact_island | keep f32; do not quantize |
| 53 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 0.70 | 0 | 1.37 | 13 | exact_island | keep f32; do not quantize |
| 53 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.95 | 0 | 1.21 | 38 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 53 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.65 | 0 | 1.44 | 6 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 53 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 8.05 | 0 | 6.57 | 131 | exact_island | keep f32; do not quantize |
| 53 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.05 | 0 | 1.85 | 40 | exact_island | keep f32; do not quantize |
| 53 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | -0.47 | 0 | 1.15 | 13 | exact_island | keep f32; do not quantize |
| 53 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.87 | 0 | 2.99 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 53 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.42 | 0 | 1.82 | 1072 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 53 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.31 | 0 | 1.70 | 4775 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 53 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 4.83 | 0 | 1.41 | 1689 | exact_island | keep f32; do not quantize |
| 53 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 1.32 | 16335 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 54 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.07 | 0 | 2.33 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 54 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.63 | 0 | 1.58 | 280 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 54 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.53 | 0 | 1.44 | 3986 | exact_island | keep f32; do not quantize |
| 54 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 6.43 | 0 | 1.18 | 10 | exact_island | keep f32; do not quantize |
| 54 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.05 | 0 | 1.21 | 19 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 54 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.45 | 0 | 1.34 | 36 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 54 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 10.24 | 0 | 6.32 | 10096 | exact_island | keep f32; do not quantize |
| 54 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.84 | 0 | 3.29 | 45 | exact_island | keep f32; do not quantize |
| 54 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | -0.34 | 0 | 1.13 | 37 | exact_island | keep f32; do not quantize |
| 54 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.56 | 0 | 2.81 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 54 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.37 | 0 | 1.74 | 1043 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 54 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.23 | 0 | 1.38 | 4613 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 54 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 3.78 | 0 | 1.29 | 1270 | exact_island | keep f32; do not quantize |
| 54 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.21 | 0 | 1.46 | 16302 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 55 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.71 | 0 | 1.90 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 55 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.40 | 0 | 1.75 | 4162 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 55 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.51 | 0 | 1.69 | 4316 | exact_island | keep f32; do not quantize |
| 55 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 2.98 | 0 | 1.64 | 461 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 55 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 9.93 | 0 | 1.69 | 25 | exact_island | keep f32; do not quantize |
| 55 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 5.36 | 0 | 3.88 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 55 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.42 | 0 | 1.23 | 2316 | exact_island | keep f32; do not quantize |
| 55 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 0.74 | 0 | 2.93 | 5665 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 55 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 17.12 | 0 | 1.41 | 24 | exact_island | keep f32; do not quantize |
| 55 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.15 | 0 | 1.33 | 8791 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 55 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 1.59 | 0 | 1.49 | 1008 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 56 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.39 | 0 | 1.65 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 56 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.34 | 0 | 1.52 | 14477 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 56 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 15.93 | 0 | 1.53 | 4615 | exact_island | keep f32; do not quantize |
| 56 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.42 | 0 | 1.21 | 40 | exact_island | keep f32; do not quantize |
| 56 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.39 | 0 | 1.36 | 10 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 56 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.53 | 0 | 1.43 | 21 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 56 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 7.12 | 0 | 6.63 | 3235 | exact_island | keep f32; do not quantize |
| 56 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -1.20 | 0 | 2.04 | 13 | exact_island | keep f32; do not quantize |
| 56 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | -0.44 | 0 | 1.15 | 97 | exact_island | keep f32; do not quantize |
| 56 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.30 | 0 | 3.27 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 56 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.43 | 0 | 1.86 | 1833 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 56 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.31 | 0 | 1.68 | 1777 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 56 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.28 | 0 | 1.30 | 2067 | exact_island | keep f32; do not quantize |
| 56 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.32 | 15224 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 57 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.26 | 0 | 1.35 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 57 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.31 | 0 | 1.69 | 9825 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 57 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 25.80 | 0 | 1.67 | 4615 | exact_island | keep f32; do not quantize |
| 57 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 3.47 | 0 | 1.34 | 1 | exact_island | keep f32; do not quantize |
| 57 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 8.57 | 0 | 1.13 | 19 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 57 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.97 | 0 | 1.35 | 17 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 57 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 6.51 | 0 | 5.73 | 10014 | exact_island | keep f32; do not quantize |
| 57 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.06 | 0 | 3.91 | 13 | exact_island | keep f32; do not quantize |
| 57 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 8.99 | 0 | 1.59 | 117 | exact_island | keep f32; do not quantize |
| 57 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.39 | 0 | 2.98 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 57 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.66 | 0 | 1.82 | 566 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 57 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.25 | 0 | 1.48 | 2647 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 57 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 10.20 | 0 | 1.37 | 2067 | exact_island | keep f32; do not quantize |
| 57 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.11 | 0 | 1.41 | 9825 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 58 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.71 | 0 | 1.81 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 58 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.34 | 0 | 1.43 | 6275 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 58 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 33.59 | 0 | 1.73 | 4615 | exact_island | keep f32; do not quantize |
| 58 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.50 | 0 | 1.35 | 36 | exact_island | keep f32; do not quantize |
| 58 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 11.12 | 0 | 1.15 | 41 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 58 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 2.82 | 0 | 1.27 | 9 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 58 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 7.10 | 0 | 6.21 | 5509 | exact_island | keep f32; do not quantize |
| 58 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | 0.64 | 0 | 4.34 | 16 | exact_island | keep f32; do not quantize |
| 58 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.34 | 0 | 1.20 | 3 | exact_island | keep f32; do not quantize |
| 58 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.62 | 0 | 2.63 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 58 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.52 | 0 | 1.72 | 1122 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 58 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.23 | 0 | 1.35 | 4537 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 58 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 8.37 | 0 | 1.35 | 2067 | exact_island | keep f32; do not quantize |
| 58 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.13 | 0 | 1.29 | 8580 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 59 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.04 | 0 | 1.93 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 59 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.35 | 0 | 1.56 | 5254 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 59 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.82 | 0 | 1.86 | 1501 | exact_island | keep f32; do not quantize |
| 59 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 4.05 | 0 | 1.82 | 400 | need_richer_levels | Q4 4.25 ESTIMATED; L63 k Q3=0.963 fails 0.99 |
| 59 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 9.46 | 0 | 1.70 | 56 | exact_island | keep f32; do not quantize |
| 59 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.55 | 0 | 3.50 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 59 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 14.75 | 0 | 1.32 | 2067 | exact_island | keep f32; do not quantize |
| 59 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 1.15 | 0 | 2.91 | 5120 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 59 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 15.68 | 0 | 1.51 | 24 | exact_island | keep f32; do not quantize |
| 59 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.20 | 0 | 1.51 | 5254 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 59 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 4.26 | 0 | 2.29 | 265 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 60 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.33 | 0 | 1.76 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 60 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.49 | 0 | 1.63 | 6800 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 60 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 163.29 | 0 | 1.91 | 4615 | exact_island | keep f32; do not quantize |
| 60 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | 3.18 | 0 | 1.25 | 31 | exact_island | keep f32; do not quantize |
| 60 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 10.78 | 0 | 1.18 | 34 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 60 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.46 | 0 | 1.64 | 39 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 60 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 5.31 | 0 | 6.53 | 1812 | exact_island | keep f32; do not quantize |
| 60 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | 1.04 | 0 | 4.97 | 9 | exact_island | keep f32; do not quantize |
| 60 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.22 | 0 | 1.23 | 84 | exact_island | keep f32; do not quantize |
| 60 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.78 | 0 | 2.90 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 60 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 1.08 | 0 | 3.99 | 1812 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 60 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.46 | 0 | 1.73 | 4355 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 60 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 23.15 | 0 | 1.24 | 1671 | exact_island | keep f32; do not quantize |
| 60 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.17 | 0 | 1.71 | 14445 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 61 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.33 | 0 | 1.48 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 61 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.58 | 0 | 1.57 | 9695 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 61 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 188.17 | 0 | 1.92 | 4615 | exact_island | keep f32; do not quantize |
| 61 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.76 | 0 | 1.15 | 27 | exact_island | keep f32; do not quantize |
| 61 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 9.82 | 0 | 1.33 | 8 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 61 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 3.77 | 0 | 1.59 | 9 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 61 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 7.52 | 0 | 6.37 | 8097 | exact_island | keep f32; do not quantize |
| 61 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.36 | 0 | 2.95 | 13 | exact_island | keep f32; do not quantize |
| 61 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.04 | 0 | 1.16 | 89 | exact_island | keep f32; do not quantize |
| 61 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 1.32 | 0 | 2.88 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 61 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 0.53 | 0 | 1.95 | 232 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 61 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.39 | 0 | 1.43 | 4850 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 61 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 26.79 | 0 | 1.19 | 1671 | exact_island | keep f32; do not quantize |
| 61 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.25 | 0 | 1.74 | 9456 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 62 | ΔNet | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 0.90 | 0 | 1.82 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9727–0.9923); generation untested |
| 62 | ΔNet | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.66 | 0 | 1.66 | 424 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9766–0.9940); generation untested |
| 62 | ΔNet | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 56.61 | 0 | 1.88 | 3986 | exact_island | keep f32; do not quantize |
| 62 | ΔNet | lin_A_log | 48 | 0.0000 | 0.0000 | 33.333 | -0.87 | 0 | 1.63 | 37 | exact_island | keep f32; do not quantize |
| 62 | ΔNet | lin_a† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 6.86 | 0 | 1.48 | 1 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 62 | ΔNet | lin_b† | 48×5120 | 0.0009 | 0.0009 | 4.251 | 4.27 | 0 | 1.87 | 39 | need_richer_levels | tiny; stay Q4 (fused ba 96×5120). Not worth a new codec. |
| 62 | ΔNet | lin_conv | 10240×4×1 | 0.0002 | 0.0011 | 32.002 | 7.21 | 0 | 7.07 | 3151 | exact_island | keep f32; do not quantize |
| 62 | ΔNet | lin_dt_bias | 48 | 0.0000 | 0.0000 | 33.333 | -0.38 | 0 | 2.73 | 23 | exact_island | keep f32; do not quantize |
| 62 | ΔNet | lin_norm | 128 | 0.0000 | 0.0000 | 32.500 | 0.11 | 0 | 1.13 | 65 | exact_island | keep f32; do not quantize |
| 62 | ΔNet | lin_o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.81 | 0 | 3.28 | 3994 | need_richer_levels | Q4 4.25 ESTIMATED class default; Q3 fails 0.99 on probed L0 lin_o (0.953) and L63 o (0.960) |
| 62 | ΔNet | lin_qkv† | 10240×5120 | 0.1949 | 0.1948 | 4.250 | 1.14 | 0 | 3.67 | 1537 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 62 | ΔNet | lin_z† | 6144×5120 | 0.1170 | 0.1169 | 4.250 | 0.86 | 0 | 1.66 | 4774 | need_richer_levels | Q4 4.25 ESTIMATED class default from L0/L3/L32/L63 probes |
| 62 | ΔNet | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 35.71 | 0 | 1.22 | 3321 | exact_island | keep f32; do not quantize |
| 62 | ΔNet | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.33 | 0 | 1.83 | 11415 | cheap_to_crush | ESTIMATED Q3 3.25 BPW by class interpolation (6-layer hold range 0.9679–0.9957); generation untested |
| 63 | GQA | down | 5120×17408 | 0.3314 | 0.3312 | 4.250 | 1.49 | 0 | 3.26 | 3994 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9727 on L63); Q2/binary not intact mid-depth |
| 63 | GQA | gate | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.85 | 0 | 2.04 | 12095 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9940 on L63); Q2/binary not intact mid-depth; L63-only binary hold 0.9543 — do not average into a global pass |
| 63 | GQA | input_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.63 | 0 | 1.87 | 2479 | exact_island | keep f32; do not quantize |
| 63 | GQA | k | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 6.06 | 0 | 1.50 | 1 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9633) |
| 63 | GQA | k_norm | 256 | 0.0000 | 0.0000 | 32.250 | 13.56 | 0 | 1.74 | 56 | exact_island | keep f32; do not quantize |
| 63 | GQA | o | 5120×6144 | 0.1170 | 0.1169 | 4.250 | 2.60 | 0 | 3.32 | 3994 | need_richer_levels | Q4 4.25 is last cheap codec clearing 0.99 (Q3 out_cos 0.9603) |
| 63 | GQA | post_attn_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 9.13 | 0 | 1.23 | 2479 | exact_island | keep f32; do not quantize |
| 63 | GQA | q | 12288×5120 | 0.2339 | 0.2338 | 4.250 | 1.54 | 0 | 2.81 | 71 | need_richer_levels | ESTIMATED Q3 3.25 (probe out_cos 0.9909 ≥ 0.99); generation untested |
| 63 | GQA | q_norm | 256 | 0.0000 | 0.0000 | 32.250 | 25.32 | 0 | 1.47 | 20 | exact_island | keep f32; do not quantize |
| 63 | GQA | up | 17408×5120 | 0.3314 | 0.3312 | 4.250 | 0.51 | 0 | 3.82 | 6844 | cheap_to_crush | ESTIMATED Q3 3.25 BPW (hold cos 0.9957 on L63); Q2/binary not intact mid-depth; L63-only binary hold 0.9601 — do not average into a global pass |
| 63 | GQA | v | 1024×5120 | 0.0195 | 0.0195 | 4.250 | 6.52 | 0 | 1.94 | 465 | need_richer_levels | ESTIMATED Q3 3.25 (probe out_cos 0.9925 ≥ 0.99); generation untested |
| — | — | embed | 248320×5120 | 4.7271 | 4.7241 | 4.250 | 0.40 | 0 | 1.28 | 72964 | need_richer_levels | MEASURED-as-part-of-Q4-oracle: stay 4.25 BPW. Q3 cosine-legal, top-1 risky. No isolated generate. |
| — | — | final_norm | 5120 | 0.0000 | 0.0001 | 32.013 | 7.57 | 0 | 1.38 | 471 | exact_island | keep f32; do not quantize |
| — | — | lm_head | 248320×5120 | 4.7271 | 4.7241 | 4.250 | 0.57 | 0 | 1.73 | 248050 | need_richer_levels | MEASURED-as-part-of-Q4-oracle: stay 4.25 BPW. Q3 cosine-legal, top-1 risky. No isolated generate. |

† = G0 stores a fused tensor; bytes are an element-share of that payload, not a standalone file.

### 7.1 Layer-wise write-island (channel 3994 xmed on residual writes)

| L | mix | lin_o/o xmed | down xmed | act ch3994 rms | act n10 |
|---:|---|---:|---:|---:|---:|
| 00 | ΔNet | 20.70 | 11.47 | (not in top5) | 2 |
| 01 | ΔNet | 15.43 | 3.62 | (not in top5) | 1 |
| 02 | ΔNet | 8.71 | 4.89 | (not in top5) | 1 |
| 03 | GQA | 17.76 | 3.14 | (not in top5) | 1 |
| 04 | ΔNet | 10.67 | 2.05 | (not in top5) | 1 |
| 05 | ΔNet | 9.60 | 2.97 | (not in top5) | 1 |
| 06 | ΔNet | 9.27 | 2.99 | 8.374 | 2 |
| 07 | GQA | 11.71 | 2.17 | (not in top5) | 1 |
| 08 | ΔNet | 10.02 | 2.68 | (not in top5) | 1 |
| 09 | ΔNet | 8.57 | 2.83 | (not in top5) | 1 |
| 10 | ΔNet | 7.76 | 1.64 | (not in top5) | 1 |
| 11 | GQA | 10.56 | 2.96 | 5.043 | 2 |
| 12 | ΔNet | 8.00 | 2.05 | 4.493 | 2 |
| 13 | ΔNet | 5.62 | 1.63 | 4.210 | 2 |
| 14 | ΔNet | 6.71 | 1.60 | 2.872 | 2 |
| 15 | GQA | 5.72 | 2.01 | 4.962 | 2 |
| 16 | ΔNet | 6.15 | 2.55 | 5.733 | 2 |
| 17 | ΔNet | 4.71 | 2.79 | 6.461 | 2 |
| 18 | ΔNet | 5.68 | 4.18 | 10.170 | 3 |
| 19 | GQA | 4.73 | 4.01 | 10.832 | 3 |
| 20 | ΔNet | 5.68 | 3.21 | 9.249 | 2 |
| 21 | ΔNet | 6.03 | 1.79 | 8.319 | 1 |
| 22 | ΔNet | 5.55 | 3.10 | 13.011 | 2 |
| 23 | GQA | 4.26 | 2.15 | 12.666 | 2 |
| 24 | ΔNet | 5.09 | 2.59 | 16.209 | 2 |
| 25 | ΔNet | 4.69 | 2.76 | 17.184 | 2 |
| 26 | ΔNet | 4.72 | 3.46 | 21.405 | 3 |
| 27 | GQA | 3.37 | 2.61 | 22.127 | 2 |
| 28 | ΔNet | 4.73 | 2.17 | 22.027 | 2 |
| 29 | ΔNet | 4.21 | 2.47 | 21.793 | 2 |
| 30 | ΔNet | 5.26 | 3.01 | 23.297 | 3 |
| 31 | GQA | 3.36 | 2.80 | 23.485 | 2 |
| 32 | ΔNet | 4.73 | 3.10 | 24.498 | 1 |
| 33 | ΔNet | 4.20 | 2.67 | 24.668 | 1 |
| 34 | ΔNet | 3.94 | 3.46 | 27.255 | 3 |
| 35 | GQA | 3.59 | 2.68 | 24.259 | 2 |
| 36 | ΔNet | 3.60 | 2.10 | 19.757 | 2 |
| 37 | ΔNet | 3.59 | 1.67 | 16.036 | 1 |
| 38 | ΔNet | 3.30 | 3.41 | 21.558 | 2 |
| 39 | GQA | 3.65 | 2.42 | 16.911 | 2 |
| 40 | ΔNet | 3.99 | 2.79 | 19.610 | 2 |
| 41 | ΔNet | 3.54 | 2.85 | 16.736 | 2 |
| 42 | ΔNet | 3.81 | 3.29 | 16.966 | 2 |
| 43 | GQA | 3.22 | 2.65 | 16.152 | 2 |
| 44 | ΔNet | 3.62 | 2.34 | 15.631 | 2 |
| 45 | ΔNet | 3.15 | 2.37 | 15.323 | 2 |
| 46 | ΔNet | 3.66 | 3.02 | 15.698 | 2 |
| 47 | GQA | 2.93 | 2.35 | 16.783 | 2 |
| 48 | ΔNet | 3.76 | 2.49 | 16.370 | 2 |
| 49 | ΔNet | 3.67 | 2.54 | 16.243 | 2 |
| 50 | ΔNet | 2.98 | 3.09 | 19.545 | 2 |
| 51 | GQA | 3.42 | 2.29 | 16.129 | 2 |
| 52 | ΔNet | 2.77 | 1.62 | 15.157 | 2 |
| 53 | ΔNet | 2.99 | 1.99 | 14.784 | 2 |
| 54 | ΔNet | 2.81 | 2.33 | 21.736 | 2 |
| 55 | GQA | 3.88 | 1.90 | 13.558 | 1 |
| 56 | ΔNet | 3.27 | 1.65 | 15.853 | 1 |
| 57 | ΔNet | 2.98 | 1.35 | 19.227 | 1 |
| 58 | ΔNet | 2.63 | 1.81 | 23.160 | 1 |
| 59 | GQA | 3.50 | 1.93 | 23.425 | 1 |
| 60 | ΔNet | 2.90 | 1.76 | 23.392 | 1 |
| 61 | ΔNet | 2.88 | 1.48 | 25.948 | 1 |
| 62 | ΔNet | 3.28 | 1.82 | 29.082 | 1 |
| 63 | GQA | 3.32 | 3.26 | 30.440 | 1 |

When act ch3994 is not in that layer's top-5, it is still in the write-tensor top-5 (true for all 64 downs and all 64 out-projs). The activation list only stores top-5.

## 8. What is not on disk, and the cheapest capture that would answer it

The map above is honest with what exists. These holes block a *token-level* per-tensor floor:

1. **No isolated generate between 2.0856 and 4.2527 BPW.** mixed-q3mlp (3.614), mixed-q4down (2.959), mixed-floor-q7/q8 sit packed with no GENERATE. Cheapest: run the existing Q4-oracle greedy prompts on `mixed-q3mlp-v1` without packing a new artifact. If that is coherent, the MLP-Q3 hypothesis lives; if not, Q3-MLP cosine is another false friend.
2. **No per-class generate ablation.** Replace one class at a time in the Q4 oracle (gate-only Q3, up-only Q3, down-only Q3, attn-only Q3, lm_head-only Q3) and compare greedy ids to the Q4 oracle on ≥5 prompts × ≥32 new tokens. That is the cheapest experiment that turns §6's UNKNOWN column into MEASURED.
3. **Activation site not confirmed.** Capture run that would fix it, CPU-side, no GPU lane:
   - Model: `.../qwen38-27b/bf16` via the same `mlx_lm.qwen3_5_text` forward.
   - Tokens: ≥4096, ≥32 prompts, mix of the 5 existing plus code/math/long-context. Fit/hold 3:1.
   - Sites to store, per layer, f32, width stated in the filename:
     - `input_layernorm` output (true attn in-proj X), 256+ × 5120
     - mixer output *before* `o`/`out_proj` (true o X; for ΔNet the real `v*silu(z)` after conv/recurrence, not a proxy), 256+ × 6144
     - `post_attention_layernorm` output (true MLP in-proj X), 256+ × 5120
     - SwiGLU intermediate `silu(x@Wgate)* (x@Wup)`, 256+ × 17408 (true down X; do not reconstruct later)
     - final RMSNorm output (true lm_head X), 256+ × 5120
   - Also store greedy teacher tokens and full lm_head logits for those rows (or at least top-32 + argmax) so lm_head screens are token-id, not cosine.
   - Schema name must match width. Do not call a 5120 tensor `post_swiglu`.
4. **No Hessian / Fisher / Wanda.** Not required if (2)+(3) exist. Do not spend a Hessian capture before the Q3-MLP generate.
5. **lm_head site is L63 post-norm, not confirmed final-norm.** Item 3 last bullet.
6. **attn_out in BPW descent is weight-scored only** (`quality_space: weight_only`). Organ hold cosine for o/lin_o does not exist except via the 4-layer attention probes.

## 9. Evidence appendix (verbatim)

### 9.1 G0 artifact ledger

Path: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`

```
     2|  "complete_physical_bpw": 4.252735126866492,
     3|  "f32_tensors": 353,
     8|  "q4_tensors": 402,
    10|  "skipped_vision_tensors": 333,
    12|  "source_weight_elements": 26895998464,
    14|  "tensor_count": 755,
    15|  "tensor_payload_bytes": 14297694680,
    19|      "bytes": 675430440,
    23|      "name": "language_model.model.embed_tokens.weight",
```

### 9.2 Activation capture header

Path: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json`

```
     2|  "schema": "hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1",
     3|  "status": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN",
     7|    "forward": "mlx_lm.qwen3_5_text",
    10|  "n_tokens": 256,
    11|  "n_layers": 64,
    12|  "hidden": 5120,
```

`sha256sum` of that JSON file: `01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe`.

### 9.3 Weight-tensor analyzer (this lane)

Command: `python3 /tmp/g1-doctor/analyze_weights.py`
Source: 11 BF16 shards under `.../qwen38-27b/bf16`. Language tensors only.

```
language tensors=851 shards=11
[20/851] language_model.model.layers.0.linear_attn.out_proj.weight kurt=149.3597766824085 out10x=1 maxrss=1974206464
[117/851] language_model.model.layers.3.self_attn.o_proj.weight kurt=132.14839201639205 out10x=1
[27/851] language_model.model.embed_tokens.weight kurt=0.39657541954778086 out10x=0
[851/851] language_model.lm_head.weight kurt=0.5704912188779523 out10x=0 maxrss=2115567616 elapsed=365.3s
wrote /tmp/g1-doctor/weight_stats.json n=851 errors=0 elapsed=365.4s maxrss=2115567616
```

Element total of the 851 records = 26,895,998,464, matching `source_weight_elements`.

### 9.4 Coherence bracket

`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json`:

```
     7|    "4.2527_BPW_q4_oracle": "COHERENT",
     8|    "2.0856_BPW_mixed-2p0-v1": "INCOHERENT (native, verified twice - lane and controller)",
     9|    "1.2910_BPW_mixed-sub15-v1": "INCOHERENT",
    10|    "conclusion": "Qwen3.8's coherence floor with current codecs lies between 2.0856 and 4.2527 BPW."
```

`mixed-2p0-v1/PACK_REPORT.json` lines 10–12: `complete_physical_bpw` 2.0855934079220506, `mlp_physical_bpw` 0.8480504639008466, `nonmlp_physical_bpw` 4.250142713483966.

### 9.5 Q4-oracle pack quality

`receipts/ascent-2026-08-16/qwen38-native-bringup.json` `correctness.numeric_gate`: `Q4 pack min cosine 0.98948 vs BF16 source`.

### 9.6 lm_head top-1

`receipts/ascent-2026-08-16/QWEN_ATTENTION_LMHEAD_TOPK.json` `qwen38_lm_head`: n=128, q4 top1_agree=0.890625, q3 top1_agree=0.84375, both have ref_top1_in_pred_top5=1.0.

### 9.7 Geometry authority

`crates/hawking-core/src/model/qwen38_geometry.rs` lines 20–52, 82–90: 64 layers, 48 ΔNet / 16 GQA, hidden 5120, intermediate 17408, vocab 248320, GQA iff `(layer+1)%4==0`.

STATUS

SUPPORTED

CLAIMS

- C1. G0 language artifact is 26,895,998,464 weights / 14,297,694,680 bytes / 4.252735 BPW. Evidence: `uniform-q4-v1/manifest.json` lines 2,12,15; this-lane sum of 851 BF16 tensors equals `source_weight_elements`.
- C2. Residual channel 3994 is the information island: activation-hot in 54/64 layers at ≥10× and the top-5 output row of all 128 write tensors (down+lin_o+o). Evidence: `/tmp/g1-doctor/act_stats.json` `cross_layer`; `/tmp/g1-doctor/weight_stats.json` `out_ch.top5`.
- C3. gate/up (and down body) are distributionally cheap (kurtosis < 0.6 median, zero 10× rows except down L0 row 3994). Evidence: this-lane class rollup §4.
- C4. Attention GEMVs and embed/lm_head need richer levels; Q4 is the last generate-proven cheap codec. Evidence: `QWEN_ATTENTION_DENSITY_VERDICT.json`; `QWEN38_COHERENCE_FLOOR_BRACKETED.json`; probe table §5.3.
- C5. Token coherence with current codecs is bracketed (2.0856, 4.2527], not located. Evidence: `QWEN38_COHERENCE_FLOOR_BRACKETED.json` lines 7–10; mixed-q3mlp has no GENERATE.
- C6. mixed-2p0 MLP-crush to 0.848 BPW KILLS tokens even with Q4 attention. Evidence: `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`; `mixed-2p0-v1/GENERATE.json`; 0 fallbacks.
- C7. Naive act-column-scale on this capture is a KILL on L0 out_proj (0.992→0.919). Evidence: `QWEN_ATTENTION_DENSITY_PROBE.json` candidate `HGRAVU01_q4_g64_act_colscale`.

EVIDENCE

Weight analyzer stdout: `n=851 errors=0 elapsed=365.4s maxrss=2115567616`. Activation analyzer wrote `/tmp/g1-doctor/act_stats.json`. Receipts cited by path+field in §§2–9. Q4 manifest lines 2–15 quoted. Capture-result lines 2–12 quoted.

CHANGES

Created `workspace/superwave/g1/g1-doctor-tensor-map.md` only.

TESTS

```
$ test -s workspace/superwave/g1/g1-doctor-tensor-map.md; echo exit:$?
exit:0

$ wc -l workspace/superwave/g1/g1-doctor-tensor-map.md
    1311 workspace/superwave/g1/g1-doctor-tensor-map.md

$ git status --porcelain
?? workspace/superwave/g1/g1-doctor-tensor-map.md
```

RISKS

256-token / 5-prompt capture can invent persistent channels that are prompt-set artifacts. Channel 3994 is corroborated by the weight write-back (independent of X), which reduces that risk but does not eliminate a domain shift. Organ hold-cosine over-predicted mixed-2p0. Interpolation from 4–6 layers onto 64 is ESTIMATED.

UNRESOLVED

Per-tensor token-drift floors. Location of the coherence floor inside (2.0856, 4.2527]. Confirmed in-proj / mixer / SwiGLU / final-norm sites. Isolated Q3-MLP generate. Isolated Q3-lm_head greedy-id. Vision unused. MTP tensors absent from the snapshot.

NEXT

1) Generate `mixed-q3mlp-v1` against the Q4-oracle prompt set. 2) One-class Q3 overlays on the Q4 oracle. 3) Capture run in §8.3 if (1) is coherent and a site-correct out_proj island pack is next. Do not expand-to-Q4-then-GEMV as a production path.

