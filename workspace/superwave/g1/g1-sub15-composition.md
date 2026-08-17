# G1 mixed-sub15-v1 composition

Lane: `60-sub15-composition`. CPU/numpy. No GPU, no generate, no pack, no resident touch.
Tags: **MEASURED** this process; **RECEIPT** on-disk field; **SOURCE** file:line; **PROXY** underdetermined activation number; **PREDICTED** not a generate result.

Artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1`
Parent: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16`
Sibling: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1`
Capture: `activation-capture-v1` sha256_self `fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512`, 256 × 5120 post-norm hidden. mixer_x absent.

---

## 0. Verdict

sub15 **repeats** mixed-2p0's `down_proj` starvation byte-identically (HGRAVS01 r160_b3, 93_847_197 B, physical BPW **0.13161714918473189**) and **inverts** the attention allocation (HGRAVU01 q4 **4.250** → HGRAVR02 rice_q1_rms_2pct **1.2877935788805008**). Embed/lm_head stay HQ30UQ4 **4.250**. Small tensors go the other way (2p0 HGRAVU01 ~4.3–47 → sub15 f32 **32**).

It does not invert the 2p0 policy. It copies the MLP recipe and takes the entire 0.7945152149158003 complete-BPW cut from attention GEMVs (saved 2_680_063_944 B = 0.7971636219677024 BPW; ~0.0026 BPW given back by storing small tensors as f32).

2p0 did **not** prove `down_proj` cannot be starved: both of its INCOHERENT paths were confounded (MLX expand; native reader on 2048-col tiles). Do not treat that as a floor. Independently, this lane **MEASURED** the down weight matrix is gone (mean weight cosine **0.173122**, rel-L2 **0.985619**). 62/64 down headers set `distribution_local_only=true`. `n_hold_tokens = n_fit_tokens = 256` on every down. That is fit-set leakage, not a holdout.

**Most likely to break first: `mlp.down_proj`.** Weight space is destroyed; the only decent-looking number is output cosine on the same 256 tokens used to fit, at rows-per-dim **0.014705882352941176** (worse than Q80 NS-014 0.0449). If that crush is fatal, sub15 dies the same way 2p0 would have, plus rice attention at weight cosine ~0.84.

`dn.out_proj` / `gqa.o_proj` cannot be ranked in output space. mixer_x (in-dim 6144) was never captured. Weight space is the wrong metric for those organs (`g1-out-proj-forensics.md`: |W|-hottest and |X|-hottest 42 columns on L0 out_proj are disjoint, overlap 0/42).

---

## 1. Ledger identity

Definition (`qwen38_pack.rs:673-679`): `complete_physical_bpw = 8 * tensor_payload_bytes / source_weight_elements`.

MEASURED this process, sum of 851 language tensors:

```
tensors                  851
source_weight_elements   26895998464
payload_bytes            4340604637
complete_physical_bpw    8 * 4340604637 / 26895998464
                       = 1.2910781930062503
```

Equals PACK_REPORT.json:23. Integrity: attn `lstat` vs `attn_rows.json` mismatches **0**; mlp `packed_bytes` vs mixed-2p0 catalog `(codec,nbytes)` mismatches **0** missing **0**.

This is the **packed** ledger. `manifest.json` still says 4.252735126866492 — that is the expand-to-Q4 vehicle, not this number (`g1-sub15-native-gap.md`, PACK_REPORT.json:78-81).

PACK_REPORT.json:75-76 `projected_tps` 79.44 is **PROJECTED** from bytes against a 4.2527 wall. Retired as a speed claim.

---

## 2. Per-organ composition and cumulative BPW

Share = `8 * class_bytes / 26895998464` (contribution to complete BPW). Physical BPW = `8 * class_bytes / class_elements`.

Sorted by share descending. Cum is running complete BPW.

| class | n | codec | bits | g | scale | bytes | phys BPW | share | cum | 2p0 codec | 2p0 phys BPW |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| `mlp.up_proj` | 64 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 918,036,000 | 1.2875108158 | 0.273062 | 0.273062 | HGRAVR02 | 1.287511 |
| `mlp.gate_proj` | 64 | HGRAVB01 | 1 | 128 | mean_abs f16 / 128 | 802,177,344 | 1.1250234267 | 0.238601 | 0.511664 | HGRAVB01 | 1.125023 |
| `embed` | 1 | HQ30UQ4 | 4 | 64 | absmax /7 f16 /64 | 675,430,440 | 4.2500002517 | 0.200901 | 0.712565 | HGRAVU01 | 4.250002 |
| `lm_head` | 1 | HQ30UQ4 | 4 | 64 | absmax /7 f16 /64 | 675,430,440 | 4.2500002517 | 0.200901 | 0.913467 | HGRAVU01 | 4.250002 |
| `dn.in_proj_qkv` | 48 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 405,054,144 | 1.2876324463 | 0.120480 | 1.033947 | HGRAVU01 | 4.250043 |
| `dn.out_proj` | 48 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 243,056,165 | 1.2877579000 | 0.072295 | 1.106242 | HGRAVU01 | 4.250070 |
| `dn.in_proj_z` | 48 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 243,042,038 | 1.2876830525 | 0.072291 | 1.178533 | HGRAVU01 | 4.250070 |
| `gqa.q_proj` | 16 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 162,049,269 | 1.2878518820 | 0.048200 | 1.226733 | HGRAVU01 | 4.250035 |
| `mlp.down_proj` | 64 | HGRAVS01 | 3 | 64 | factor absmax /7-of-3bit g64 | 93,847,197 | 0.1316171492 | 0.027914 | 1.254647 | HGRAVS01 | 0.131617 |
| `gqa.o_proj` | 16 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 81,022,343 | 1.2878154596 | 0.024099 | 1.278746 | HGRAVU01 | 4.250070 |
| `gqa.k_proj` | 16 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 13,511,256 | 1.2885337830 | 0.004019 | 1.282765 | HGRAVU01 | 4.250418 |
| `gqa.v_proj` | 16 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 13,510,872 | 1.2884971619 | 0.004019 | 1.286784 | HGRAVU01 | 4.250418 |
| `dn.conv1d` | 48 | f32v2 | 32 | — | identity f32 | 7,864,704 | 32.0015625000 | 0.002339 | 1.289123 | HGRAVU01 | 4.301953 |
| `dn.in_proj_a` | 48 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 1,926,160 | 1.3062608507 | 0.000573 | 1.289696 | HGRAVU01 | 4.258691 |
| `dn.in_proj_b` | 48 | HGRAVR02 | 1 | 128 | mean_abs f16 /128 + residual rms f16 | 1,926,129 | 1.3062398275 | 0.000573 | 1.290269 | HGRAVU01 | 4.258691 |
| `norm.input` | 64 | f32v2 | 32 | — | identity f32 (HF δ) | 1,311,232 | 32.0125000000 | 0.000390 | 1.290659 | HGRAVU01 | 4.651562 |
| `norm.post_attn` | 64 | f32v2 | 32 | — | identity f32 (HF δ) | 1,311,232 | 32.0125000000 | 0.000390 | 1.291049 | HGRAVU01 | 4.651562 |
| `dn.norm` | 48 | f32v2 | 32 | — | identity f32 (full-scale) | 24,960 | 32.5000000000 | 0.000007 | 1.291057 | HGRAVU01 | 19.875000 |
| `norm.final` | 1 | f32v2 | 32 | — | identity f32 (HF δ) | 20,488 | 32.0125000000 | 0.000006 | 1.291063 | HGRAVU01 | 4.651562 |
| `gqa.q_norm` | 16 | f32v2 | 32 | — | identity f32 (HF δ) | 16,512 | 32.2500000000 | 0.000005 | 1.291068 | HGRAVU01 | 12.093750 |
| `gqa.k_norm` | 16 | f32v2 | 32 | — | identity f32 (HF δ) | 16,512 | 32.2500000000 | 0.000005 | 1.291072 | HGRAVU01 | 12.093750 |
| `dn.A_log` | 48 | f32v2 | 32 | — | identity f32 | 9,600 | 33.3333333333 | 0.000003 | 1.291075 | HGRAVU01 | 47.166667 |
| `dn.dt_bias` | 48 | f32v2 | 32 | — | identity f32 | 9,600 | 33.3333333333 | 0.000003 | 1.291078 | HGRAVU01 | 47.166667 |

Mass-weighted class BPW (RECEIPT PACK_REPORT.json:26-68, MEASURED equal):

| bucket | bytes | elements | phys BPW | share of 1.291078 |
|---|---:|---:|---:|---:|
| MLP gate+up+down | 1,814,060,541 | 17,112,760,320 | 0.8480504639 | 0.539578 |
| attention GEMV | 1,165,098,376 | 7,237,795,840 | 1.2877935789 | 0.346549 |
| embed+lm_head | 1,350,860,880 | 2,542,796,800 | 4.2500002517 | 0.401803 |
| small f32 | 10,584,840 | 2,645,504 | 32.0085397716 | 0.003148 |
| **complete** | **4,340,604,637** | **26,895,998,464** | **1.2910781930** | **1.291078** |

Embed+lm_head are 9.454% of N and **31.12%** of the complete-BPW budget. `down_proj` is 21.209% of N and **2.16%** of the budget.

Cumulative (organs that actually spend the 1.291):

```
up          0.273  ################
+gate       0.512  ##############################
+embed      0.713  ##########################################
+lm_head    0.913  ######################################################
+qkv        1.034  #############################################################
+dn.out     1.106
+dn.z       1.179
+gqa.q      1.227
+down       1.255   <-- 0.028 BPW, 21% of parameters
+rest       1.291
```

---

## 3. Per-layer

Embed, lm_head, final norm are not in a layer (share 0.401809). Layer sum share 0.889269. 0.401809+0.889269=1.291078.

DN layer: 14 tensors, 383,273,184 els, ~47.20 MB, phys BPW **0.98510–0.98517**.
GQA layer: 11 tensors, 372,255,232 els, ~45.27 MB, phys BPW **0.97279–0.97293**.
Almost flat. No layer is the budget. Tables are.

2p0 same layers sit at DN **1.8767** / GQA **1.8065** because attention is still q4.

| L | kind | n | bytes | phys BPW | share | 2p0 phys BPW |
|---:|---|---:|---:|---:|---:|---:|
| 0 | DN | 14 | 47,204,110 | 0.985283854 | 0.014040486 | 1.876721686 |
| 1 | DN | 14 | 47,195,893 | 0.985112342 | 0.014038042 | 1.876717950 |
| 2 | DN | 14 | 47,195,384 | 0.985101718 | 0.014037890 | 1.876713671 |
| 3 | GQA | 11 | 45,268,505 | 0.972848758 | 0.013464755 | 1.806451827 |
| 4 | DN | 14 | 47,195,102 | 0.985095832 | 0.014037806 | 1.876713441 |
| 5 | DN | 14 | 47,195,510 | 0.985104348 | 0.014037928 | 1.876717031 |
| 6 | DN | 14 | 47,195,935 | 0.985113219 | 0.014038054 | 1.876717323 |
| 7 | GQA | 11 | 45,267,404 | 0.972825097 | 0.013464428 | 1.806455760 |
| 8 | DN | 14 | 47,195,258 | 0.985099088 | 0.014037853 | 1.876715320 |
| 9 | DN | 14 | 47,195,210 | 0.985098086 | 0.014037838 | 1.876715758 |
| 10 | DN | 14 | 47,195,384 | 0.985101718 | 0.014037890 | 1.876715654 |
| 11 | GQA | 11 | 45,267,254 | 0.972821873 | 0.013464383 | 1.806454879 |
| 12 | DN | 14 | 47,195,521 | 0.985104578 | 0.014037931 | 1.876716718 |
| 13 | DN | 14 | 47,195,125 | 0.985096312 | 0.014037813 | 1.876717866 |
| 14 | DN | 14 | 47,195,536 | 0.985104891 | 0.014037935 | 1.876717219 |
| 15 | GQA | 11 | 45,265,949 | 0.972793828 | 0.013463995 | 1.806456233 |
| 16 | DN | 14 | 47,195,456 | 0.985103221 | 0.014037912 | 1.876718847 |
| 17 | DN | 14 | 47,195,491 | 0.985103951 | 0.014037922 | 1.876717887 |
| 18 | DN | 14 | 47,196,068 | 0.985115995 | 0.014038094 | 1.876722228 |
| 19 | GQA | 11 | 45,266,896 | 0.972814179 | 0.013464277 | 1.806453998 |
| 20 | DN | 14 | 47,195,964 | 0.985113824 | 0.014038063 | 1.876716008 |
| 21 | DN | 14 | 47,196,221 | 0.985119189 | 0.014038139 | 1.876716781 |
| 22 | DN | 14 | 47,195,321 | 0.985100403 | 0.014037871 | 1.876715737 |
| 23 | GQA | 11 | 45,266,561 | 0.972806980 | 0.013464177 | 1.806455158 |
| 24 | DN | 14 | 47,195,644 | 0.985107145 | 0.014037967 | 1.876718346 |
| 25 | DN | 14 | 47,195,686 | 0.985108022 | 0.014037980 | 1.876715570 |
| 26 | DN | 14 | 47,196,436 | 0.985123676 | 0.014038203 | 1.876725944 |
| 27 | GQA | 11 | 45,268,379 | 0.972846050 | 0.013464718 | 1.806464786 |
| 28 | DN | 14 | 47,196,598 | 0.985127058 | 0.014038251 | 1.876723356 |
| 29 | DN | 14 | 47,196,499 | 0.985124991 | 0.014038222 | 1.876723210 |
| 30 | DN | 14 | 47,197,102 | 0.985137577 | 0.014038401 | 1.876726674 |
| 31 | GQA | 11 | 45,268,538 | 0.972849467 | 0.013464765 | 1.806466419 |
| 32 | DN | 14 | 47,197,284 | 0.985141376 | 0.014038455 | 1.876727175 |
| 33 | DN | 14 | 47,196,651 | 0.985128164 | 0.014038267 | 1.876726654 |
| 34 | DN | 14 | 47,197,305 | 0.985141815 | 0.014038462 | 1.876735274 |
| 35 | GQA | 11 | 45,268,956 | 0.972858450 | 0.013464890 | 1.806464249 |
| 36 | DN | 14 | 47,196,012 | 0.985114826 | 0.014038077 | 1.876719452 |
| 37 | DN | 14 | 47,196,404 | 0.985123008 | 0.014038194 | 1.876717240 |
| 38 | DN | 14 | 47,195,470 | 0.985103513 | 0.014037916 | 1.876722333 |
| 39 | GQA | 11 | 45,267,928 | 0.972836358 | 0.013464584 | 1.806455330 |
| 40 | DN | 14 | 47,195,763 | 0.985109629 | 0.014038003 | 1.876717824 |
| 41 | DN | 14 | 47,195,729 | 0.985108919 | 0.014037993 | 1.876718993 |
| 42 | DN | 14 | 47,196,331 | 0.985121485 | 0.014038172 | 1.876722312 |
| 43 | GQA | 11 | 45,270,154 | 0.972884196 | 0.013465246 | 1.806463045 |
| 44 | DN | 14 | 47,196,811 | 0.985131503 | 0.014038315 | 1.876725255 |
| 45 | DN | 14 | 47,196,301 | 0.985120858 | 0.014038163 | 1.876725025 |
| 46 | DN | 14 | 47,196,966 | 0.985134739 | 0.014038361 | 1.876726633 |
| 47 | GQA | 11 | 45,270,338 | 0.972888150 | 0.013465301 | 1.806470309 |
| 48 | DN | 14 | 47,197,442 | 0.985144674 | 0.014038502 | 1.876733855 |
| 49 | DN | 14 | 47,196,959 | 0.985134593 | 0.014038359 | 1.876734794 |
| 50 | DN | 14 | 47,197,949 | 0.985155257 | 0.014038653 | 1.876746170 |
| 51 | GQA | 11 | 45,271,575 | 0.972914734 | 0.013465669 | 1.806483010 |
| 52 | DN | 14 | 47,196,745 | 0.985130126 | 0.014038295 | 1.876730160 |
| 53 | DN | 14 | 47,196,131 | 0.985117310 | 0.014038112 | 1.876724712 |
| 54 | DN | 14 | 47,195,637 | 0.985106999 | 0.014037965 | 1.876725339 |
| 55 | GQA | 11 | 45,269,290 | 0.972865628 | 0.013464989 | 1.806460617 |
| 56 | DN | 14 | 47,195,863 | 0.985111716 | 0.014038033 | 1.876719703 |
| 57 | DN | 14 | 47,195,982 | 0.985114200 | 0.014038068 | 1.876721456 |
| 58 | DN | 14 | 47,195,677 | 0.985107834 | 0.014037977 | 1.876723460 |
| 59 | GQA | 11 | 45,269,248 | 0.972864725 | 0.013464976 | 1.806461047 |
| 60 | DN | 14 | 47,197,910 | 0.985154443 | 0.014038641 | 1.876724337 |
| 61 | DN | 14 | 47,195,759 | 0.985109545 | 0.014038002 | 1.876728449 |
| 62 | DN | 14 | 47,198,598 | 0.985168803 | 0.014038846 | 1.876740346 |
| 63 | GQA | 11 | 45,272,161 | 0.972927327 | 0.013465843 | 1.806514150 |

L0 (MEASURED, typical DN):

| tensor | codec | bytes | phys BPW | share |
|---|---|---:|---:|---:|
| mlp.gate_proj | HGRAVB01 | 12,534,021 | 1.1250234842 | 0.003728 |
| mlp.up_proj | HGRAVR02 | 14,344,242 | 1.2875044346 | 0.004267 |
| mlp.down_proj | HGRAVS01 | 1,466,363 | 0.1316172034 | 0.000436 |
| dn.in_proj_qkv | HGRAVR02 | 8,443,806 | 1.2884225464 | 0.002512 |
| dn.in_proj_z | HGRAVR02 | 5,063,449 | 1.2877016703 | 0.001506 |
| dn.in_proj_a | HGRAVR02 | 40,142 | 1.3067057292 | 0.000012 |
| dn.in_proj_b | HGRAVR02 | 40,133 | 1.3064127604 | 0.000012 |
| dn.out_proj | HGRAVR02 | 5,066,210 | 1.2884038289 | 0.001507 |
| small ×6 | f32v2 | 205,744 | 32.00–33.33 | 0.000061 |

---

## 4. Codec / geometry lock (every tensor)

Header-parsed this process. No exceptions inside a class except rice/hgravs payload size.

| class | n | magic | schema | bits | group | groups (large) | scale rule | extra |
|---|---:|---|---|---:|---:|---|---|---|
| mlp.gate_proj | 64 | `HGRAVB01` | `hawking.gravity.binary_sign_scale.v1` | 1 | 128 | 696320 | per-group mean\|W\| → f16 | sign bits little-endian |
| mlp.up_proj | 64 | `HGRAVR02` | `hawking.gravity.binary_outlier_residual.v2` | 1+1 | 128 | 696320 | binary mean-abs f16 + one global residual RMS f16 | rice_k=5, outlier_ratio=0.02, value_bits=1, codebook sign×scale |
| mlp.down_proj | 64 | `HGRAVS01` | `hawking.gravity.activation_weighted_svd_low_rank.v1` | 3 | 64 | factors | uniform absmax / 7 on each factor | rank=160, n_fit=256, n_hold=256 |
| attn GEMV (304) | 304 | `HGRAVR02` | same as up | 1+1 | 128 | cols/128 | same as up | rice_k=5 ×304, outlier_count_sum=144756064 |
| embed, lm_head | 2 | `HQ30UQ4\0` | uniform q4 v1 | 4 | 64 | n/64 | per-group absmax / 7 → f16 | inode-shared with uniform-q4-v1 (embed ino 314847693) |
| small 353 | 353 | f32v2 | u64 numel + f32 LE | 32 | — | — | identity | residual RMSNorm stored as HF δ |

368/368 rice (64 up + 304 attn): `index_mode=rice`, `value_bits=1`, `value_scale=rms`, `rice_k=5`, `group_size=128`.
64/64 down: rank 160, factor_bits 3, factor_group_size 64.
64/64 gate: group 128, groups 696320, every file **12,534,021** B.

Binary scale is **mean-abs**, not absmax (`_binary_parts`, `ascension_dual_gravity_worker.py:640`). Residual 1-bit values are **sign × stored global RMS** (`residual_compact_codec.py:219-241`). HQ30UQ4 is **absmax / 7** (`uniform_q4.rs`, packer `pack_hq30uq4`).

Full 851-row table: appendix A. Bytes vary inside rice/hgravs; codec/geometry do not.

---

## 5. Versus mixed-2p0-v1

| | mixed-2p0-v1 | mixed-sub15-v1 |
|---|---:|---:|
| complete BPW | 2.0855934079220506 (artifact; tensor 2.0855385872764454) | **1.2910781930062503** |
| MLP bytes | 1,814,060,541 | 1,814,060,541 (delta **0**) |
| MLP phys BPW (on 17.11e9) | 0.8480504639008466 | 0.8480504639008466 |
| gate | HGRAVB01 1.1250234267 | same bytes, same codec |
| up | HGRAVR02 1.2875108158 | same bytes, same codec |
| down | HGRAVS01 r160_b3 0.1316171492 | same bytes, same codec |
| attention GEMV | HGRAVU01 q4 **4.2500920501** (3,845,162,320 B) | HGRAVR02 rice **1.2877935789** (1,165,098,376 B) |
| embed+lm_head | HGRAVU01 675,430,686 B each | HQ30UQ4 675,430,440 B each (oracle Q4, hardlinked) |
| small | HGRAVU01 (conv1d 4.302, RMS 4.652, A_log 47.17) | f32v2 32.00–33.33 |
| attention role lock | none needed (all U01) | none (rice attn is legal; MLP lock already satisfied) |

2p0 left attention at 4.250 and crushed down to 0.1316. sub15 leaves that crush untouched and crushes attention too.

That is **not** an inversion and **not** something else entirely. It is the 2p0 MLP plus a 2.68 GB rice rewrite of every attention GEMV.

2p0 INCOHERENT is not a down-starvation proof (expand / wrong tile). It is also not an attention-compression proof (attention was not compressed). `g1-out-proj-forensics.md:71-75`.

If a later confound-free generate shows 0.1316 down is fatal, sub15 is already dead. If down at 0.1316 is survivable, the new risk is rice attention at weight cosine 0.84.

---

## 6. Weight-space reconstruction vs BF16

MEASURED: decode packed codec → compare to BF16 parent (`load_tensor` BF16→f32). Wall 499.2 s. Peak RSS 17111 MB (lm_head stream; over the 15 GB cap, see Risks).

Attention MEASURED cosine matches pack-time `cosine_vs_bf16` to max \|Δ\| 4.8e-15 (independent decode, same floats).
Down MEASURED weight cosine vs encoder header: max \|Δ\| 3.52e-5 (decode path agrees).

| class | n | weight cosine min / mean / max | weight rel-L2 mean | output cosine mean | output space |
|---|---:|---|---:|---:|---|
| `mlp.up_proj` | 64 | 0.840642 / 0.841593 / 0.843064 | 0.541397 | 0.843006 | PROXY_underdetermined_256x5120 |
| `mlp.gate_proj` | 64 | 0.790170 / 0.796640 / 0.798939 | 0.604446 | 0.861605 | PROXY_underdetermined_256x5120 |
| `embed` | 1 | 0.994213 / 0.994213 / 0.994213 | 0.108100 | — | N/A_embedding_table |
| `lm_head` | 1 | 0.993758 / 0.993758 / 0.993758 | 0.112304 | 0.999158 | PROXY_underdetermined_256x5120_L63_hidden |
| `dn.in_proj_qkv` | 48 | 0.839754 / 0.841666 / 0.847267 | 0.541441 | 0.874154 | PROXY_underdetermined_256x5120 |
| `dn.out_proj` | 48 | 0.836874 / 0.841539 / 0.844217 | 0.541799 | — | UNAVAILABLE_mixer_x_missing |
| `dn.in_proj_z` | 48 | 0.841731 / 0.842571 / 0.843550 | 0.539992 | 0.900958 | PROXY_underdetermined_256x5120 |
| `gqa.q_proj` | 16 | 0.842602 / 0.843259 / 0.845617 | 0.539063 | 0.908645 | PROXY_underdetermined_256x5120 |
| `mlp.down_proj` | 64 | 0.152506 / 0.173122 / 0.211188 | 0.985619 | 0.955953 | PROXY_underdetermined_256x17408_bf16_swiglu_isolated |
| `gqa.o_proj` | 16 | 0.836396 / 0.841679 / 0.844159 | 0.541692 | — | UNAVAILABLE_mixer_x_missing |
| `gqa.k_proj` | 16 | 0.840688 / 0.842240 / 0.843809 | 0.541529 | 0.884712 | PROXY_underdetermined_256x5120 |
| `gqa.v_proj` | 16 | 0.838320 / 0.840012 / 0.842535 | 0.544608 | 0.845545 | PROXY_underdetermined_256x5120 |
| `dn.conv1d` | 48 | 1.000000 / 1.000000 / 1.000000 | 0.000000 | — | N/A_vector |
| `dn.in_proj_a` | 48 | 0.834410 / 0.842915 / 0.847443 | 0.540909 | 0.934294 | PROXY_underdetermined_256x5120 |
| `dn.in_proj_b` | 48 | 0.839366 / 0.842237 / 0.846261 | 0.542393 | 0.937267 | PROXY_underdetermined_256x5120 |
| `norm.input` | 64 | -0.588250 / 0.537216 / 0.986201 | 0.941396 | — | N/A_vector |
| `norm.post_attn` | 64 | -0.976747 / 0.071851 / 0.992923 | 0.985280 | — | N/A_vector |
| `dn.norm` | 48 | 1.000000 / 1.000000 / 1.000000 | 0.000000 | — | N/A_vector |
| `norm.final` | 1 | 0.997238 / 0.997238 / 0.997238 | 0.513088 | — | N/A_vector |
| `gqa.q_norm` | 16 | 0.959869 / 0.977582 / 0.991443 | 0.669391 | — | N/A_vector |
| `gqa.k_norm` | 16 | 0.904974 / 0.960345 / 0.976759 | 0.670504 | — | N/A_vector |
| `dn.A_log` | 48 | 1.000000 / 1.000000 / 1.000000 | 0.000000 | — | N/A_vector |
| `dn.dt_bias` | 48 | 1.000000 / 1.000000 / 1.000000 | 0.000000 | — | N/A_vector |

Notes, MEASURED:

- `mlp.down_proj`: weight cosine **0.1525–0.2112**, mean **0.1731**. rel-L2 mean **0.9856**. The matrix is not there. L0 header `weight_cosine=0.211223`, `weight_relative_l2=0.978454`, `distribution_local_only=true`. L8 and L62 are the two headers with `distribution_local_only=false` (header output cosine 0.877 and 0.900).
- `mlp.gate_proj` binary: weight cosine mean **0.7966**. Honest. Not activation-fit.
- rice (up + 304 attn): weight cosine **0.834–0.847**. Uniformly ~0.84.
- embed / lm_head oracle Q4: 0.994213 / 0.993758. Incumbent quality, not this pack's experiment.
- residual RMSNorm (`input`, `post_attn`, `final`, `q_norm`, `k_norm`): raw vs BF16 is the δ vs 1+δ affine. `hat+1` cosine = **1.0** exactly. Stored as HF δ. Not reconstruction error.
- `dn.norm`: raw cosine **1.0**, rel-L2 **0**. Full-scale, not δ.
- `conv1d`, `A_log`, `dt_bias`: rel-L2 **0**. Exact f32 of BF16.

L0 down (MEASURED):

```
weight cosine  0.21118785178934474
weight rel-L2  0.9786166191143807
weight rmse    0.010406163905527272
hat_rms/ref    0.002755 / 0.010634   (energy collapsed)
output cosine  0.974196911639411     PROXY, frobenius, BF16-SwiGLU X
output rel-L2  0.2327164650458956
rows/dim       0.014705882352941176  (256 / 17408)
header output  0.9190489856950793    RECEIPT, mean-row cosine, fit X
n_fit=n_hold   256
```

L54 down is the smoking gun for leakage: **worst** weight cosine 0.152506, **near-best** isolated output cosine 0.988784. Same 256 tokens.

---

## 7. Output-space ranking — and why most of it is the wrong number

Activation capture is 256 tokens. Established: 256/6144 = 0.0417, 256/17408 = 0.0147, Q80 catastrophe 0.0449. Every activation-derived number here is **PROXY / underdetermined**. mixer_x missing ⇒ out_proj / o_proj have **no** output number.

Score used: `elements * (rel_L2)^2`. Output rel-L2 when a legal X exists; else weight rel-L2.

| rank | class | score | kind |
|---:|---|---:|---|
| 1 | `mlp.up_proj` | 1.692775e+09 | output_rel_l2_sq_times_elements_PROXY |
| 2 | `mlp.gate_proj` | 1.555468e+09 | output_rel_l2_sq_times_elements_PROXY |
| 3 | `dn.in_proj_qkv` | 5.984423e+08 | output_rel_l2_sq_times_elements_PROXY |
| 4 | `mlp.down_proj` | 5.301810e+08 | output_rel_l2_sq_times_elements_PROXY |
| 5 | `dn.out_proj` | 4.432460e+08 | weight_only_WRONG_METRIC_mixer_x_missing |
| 6 | `dn.in_proj_z` | 2.952273e+08 | output_rel_l2_sq_times_elements_PROXY |
| 7 | `gqa.q_proj` | 1.859194e+08 | output_rel_l2_sq_times_elements_PROXY |
| 8 | `gqa.o_proj` | 1.476943e+08 | weight_only_WRONG_METRIC_mixer_x_missing |
| 9 | `gqa.v_proj` | 2.440996e+07 | output_rel_l2_sq_times_elements_PROXY |
| 10 | `gqa.k_proj` | 1.850689e+07 | output_rel_l2_sq_times_elements_PROXY |
| 11 | `embed` | 1.485698e+07 | weight_rel_l2_sq_times_elements |
| 12 | `lm_head` | 2.145881e+06 | output_rel_l2_sq_times_elements_PROXY |
| 13 | `dn.in_proj_a` | 1.716610e+06 | output_rel_l2_sq_times_elements_PROXY |
| 14 | `dn.in_proj_b` | 1.676339e+06 | output_rel_l2_sq_times_elements_PROXY |
| 15 | `norm.post_attn` | 3.218018e+05 | weight_rel_l2_sq_times_elements |
| 16 | `norm.input` | 2.912259e+05 | weight_rel_l2_sq_times_elements |
| 17 | `gqa.k_norm` | 1.856662e+03 | weight_rel_l2_sq_times_elements |
| 18 | `gqa.q_norm` | 1.850065e+03 | weight_rel_l2_sq_times_elements |
| 19 | `norm.final` | 1.347886e+03 | weight_rel_l2_sq_times_elements |
| 20 | `dn.conv1d` | 0.000000e+00 | weight_rel_l2_sq_times_elements |
| 21 | `dn.A_log` | 0.000000e+00 | weight_rel_l2_sq_times_elements |
| 22 | `dn.dt_bias` | 0.000000e+00 | weight_rel_l2_sq_times_elements |
| 23 | `dn.norm` | 0.000000e+00 | weight_rel_l2_sq_times_elements |

The raw table puts `mlp.up_proj` first. That is what you get if you trust down's output error on its own fit set. Do not.

Contamination-aware rank (this lane's call):

1. **`mlp.down_proj`** — first to break. Weight cosine 0.17, 62/64 `distribution_local_only`, n_hold=n_fit, rows/dim 0.0147, identical 0.1316 crush as 2p0.
2. **`mlp.up_proj`** — largest *honest* output energy (rice, not fit to X). Mean output cosine 0.843 on 256×5120 (rows/dim 0.05, still underdetermined).
3. **`mlp.gate_proj`** — binary, weight cosine 0.797, output cosine 0.862.
4. **attention in-proj rice** — same 0.84 weight cosine as up, smaller mass.
5. **`dn.out_proj` / `gqa.o_proj`** — unknown. Weight ~0.842 is the wrong metric. mixer_x missing. Cheapest experiment: capture mixer-site X (v·silu(z) / gated V) at width 6144, then `||X (W−Ŵ)^T||`.

lm_head output cosine 0.99916 on L63 hidden is oracle Q4 and is not this pack's risk.

---

## 8. What this does not say

- Native generate of packed 1.291 BPW. Loader still eats the Q4 vehicle unless catalog+codec-4+K-complete bind land (`g1-sub15-native-gap.md`).
- A coherence floor. Two INCOHERENT receipts in this campaign are confounded. This file does not add a third.
- That weight-space ranking of out_proj predicts token drift. It does not.
- TPS. PACK_REPORT 79.44 is PROJECTED.

KILL: treat sub15 as an inversion of 2p0. It is the same down crush.
KILL: treat 2p0 INCOHERENT as proof that 0.1316 down is a floor.
REOPEN_IF: confound-free native generate of mixed-2p0 (K-complete tiles or fuse-off) is coherent. Then down 0.1316 is survivable and rice attention becomes the first question.
REOPEN_IF: mixer_x captured and out_proj output rel-L2 on a holdout exceeds down's *holdout* output rel-L2. Then out_proj could be first-break.

---

## 9. Evidence

### 9.1 PACK_REPORT.json (RECEIPT)

```
:12-19  recipe: gate HGRAVB01 / up HGRAVR02 / down HGRAVS01 / attn rice / tables HQ30UQ4 / small f32
:22     sibling_mixed_2p0_bpw 2.0855934079220506
:23-25  complete_physical_bpw 1.2910781930062503 ; bytes 4340604637 ; N 26895998464
:27-67  class bytes/bpw (re-summed this process, exact match)
:75-76  projected_tps 79.44  (PROJECTED, retired)
:77-81  generate_vehicle = HQ30UQ4 of reconstructed mixed/rice
```

### 9.2 mixed-2p0-v1/PACK_REPORT.json (RECEIPT)

```
:10     complete_physical_bpw 2.0855934079220506
:11-12  mlp 0.8480504639008466 ; nonmlp 4.250142713483966
:16-32  gate/up/down bytes identical to sub15
:34-38  attention_embed_norm_4bit 5197519789 B / 4.250142713483966
```

### 9.3 This process (MEASURED)

```
census tensors=851 bytes=4340604637 elems=26895998464
complete_bpw=1.2910781930062503
attn_size_mismatch=0 mlp_vs_mix={'missing': 0, 'nbytes_mismatch': 0, 'codec_mismatch': 0}
304/304 rice headers identical on (schema, g=128, value_bits=1, value_scale=rms, index_mode=rice, rice_k=5, outlier_ratio=0.02)
outlier_count_sum=144756064
368/368 rice (up+attn) geometry lock
64/64 down rank=160 bits=3 g=64 n_fit=256 n_hold=256
embed inode 314847693 shared with uniform-q4-v1
wall_s 499.2 peak_rss_mb 17111.0
decoder: lab.operators.residual_compact_codec.decode_residual_compact
         _decode_activation_weighted_svd_low_rank_codec
         binary mean-abs rebuild
         HQ30UQ4 nibble unpack (even low, odd high, q-8)*scale
parent:  lab.operators.qwen30b_gravity_pack.load_tensor BF16
```

Layer-00 log line:

```
[11:43:14] rss=4035MB layer 00 DN  8.1s errors=14
```

### 9.4 SOURCE already paid

```
tools/qwen38_sub15_pack.py:13-15,82-86,160-170,402-411
residual_compact_codec.py:640-scale is in worker:_binary_parts:640 mean-abs; :219-241 residual rms
ascension_dual_gravity_worker.py:1279-1293 HGRAVS01 decode left@right; :1333-1375 dlo predicate
g1-sub15-native-gap.md:56-67,170-184 MLP byte-identical to 2p0; 304 rice
g1-out-proj-forensics.md:17-33,71-75 weight vs mixer-output; 2p0 did not compress attention
g1-bit-budget-accounting.md:188-232 codec byte formulas
g1-artifact-inventory.md:332-357 2p0 / sub15 on-disk identity
```

---

## Appendix A — every tensor

851 rows. `share` is contribution to complete BPW `1.2910781930062503`.
`mix2p0_codec` 0=HGRAVB01 1=HGRAVR02 2=HGRAVS01 3=HGRAVU01.

```
name	class	L	codec	bits	g	bytes	els	bpw	share	w_cos	w_rel	mix2p0_bytes
layers.0.linear_attn.A_log	dn.A_log	0	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.0.linear_attn.conv1d.weight	dn.conv1d	0	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.0.linear_attn.dt_bias	dn.dt_bias	0	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.0.linear_attn.in_proj_a.weight	dn.in_proj_a	0	HGRAVR02	1	128	40142	245760	1.3067057292	0.0000119399	0.8474425216	0.5328323452	130827
layers.0.linear_attn.in_proj_b.weight	dn.in_proj_b	0	HGRAVR02	1	128	40133	245760	1.3064127604	0.0000119372	0.8462606420	0.5344511539	130827
layers.0.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	0	HGRAVR02	1	128	8443806	52428800	1.2884225464	0.0025115427	0.8472668985	0.5325623970	27853079
layers.0.linear_attn.in_proj_z.weight	dn.in_proj_z	0	HGRAVR02	1	128	5063449	31457280	1.2877016703	0.0015060825	0.8426322242	0.5398857047	16711957
layers.0.linear_attn.norm.weight	dn.norm	0	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.0.linear_attn.out_proj.weight	dn.out_proj	0	HGRAVR02	1	128	5066210	31457280	1.2884038289	0.0015069037	0.8368737923	0.5507239768	16711957
layers.0.mlp.down_proj.weight	mlp.down_proj	0	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.2111878518	0.9786166191	1466363
layers.0.mlp.gate_proj.weight	mlp.gate_proj	0	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7983613877	0.6021786235	12534021
layers.0.mlp.up_proj.weight	mlp.up_proj	0	HGRAVR02	1	128	14344242	89128960	1.2875044879	0.0042665803	0.8424793159	0.5399726608	14344242
layers.0.input_layernorm.weight	norm.input	0	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.5882498372	1.0335574002	2977
layers.0.post_attention_layernorm.weight	norm.post_attn	0	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.9767473731	1.2762289867	2977
layers.1.linear_attn.A_log	dn.A_log	1	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.1.linear_attn.conv1d.weight	dn.conv1d	1	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.1.linear_attn.dt_bias	dn.dt_bias	1	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.1.linear_attn.in_proj_a.weight	dn.in_proj_a	1	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8460582846	0.5355629378	130827
layers.1.linear_attn.in_proj_b.weight	dn.in_proj_b	1	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8453998942	0.5369137445	130827
layers.1.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	1	HGRAVR02	1	128	8438486	52428800	1.2876107788	0.0025099603	0.8427700728	0.5396470504	27853079
layers.1.linear_attn.in_proj_z.weight	dn.in_proj_z	1	HGRAVR02	1	128	5063171	31457280	1.2876309713	0.0015059998	0.8425830163	0.5399422148	16711957
layers.1.linear_attn.norm.weight	dn.norm	1	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.1.linear_attn.out_proj.weight	dn.out_proj	1	HGRAVR02	1	128	5063786	31457280	1.2877873739	0.0015061827	0.8411690988	0.5429504935	16711957
layers.1.mlp.down_proj.weight	mlp.down_proj	1	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1706760163	0.9864884580	1466361
layers.1.mlp.gate_proj.weight	mlp.gate_proj	1	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7984415785	0.6020722927	12534021
layers.1.mlp.up_proj.weight	mlp.up_proj	1	HGRAVR02	1	128	14344065	89128960	1.2874886008	0.0042665276	0.8421702357	0.5404452843	14344065
layers.1.input_layernorm.weight	norm.input	1	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0393690630	0.9996661324	2977
layers.1.post_attention_layernorm.weight	norm.post_attn	1	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.9630569545	1.1869256535	2977
layers.2.linear_attn.A_log	dn.A_log	2	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.2.linear_attn.conv1d.weight	dn.conv1d	2	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.2.linear_attn.dt_bias	dn.dt_bias	2	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.2.linear_attn.in_proj_a.weight	dn.in_proj_a	2	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8461632519	0.5356577998	130827
layers.2.linear_attn.in_proj_b.weight	dn.in_proj_b	2	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8443056631	0.5388170718	130827
layers.2.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	2	HGRAVR02	1	128	8438219	52428800	1.2875700378	0.0025098809	0.8424124637	0.5401557769	27853079
layers.2.linear_attn.in_proj_z.weight	dn.in_proj_z	2	HGRAVR02	1	128	5063159	31457280	1.2876279195	0.0015059962	0.8427073926	0.5396944722	16711957
layers.2.linear_attn.norm.weight	dn.norm	2	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.2.linear_attn.out_proj.weight	dn.out_proj	2	HGRAVR02	1	128	5063764	31457280	1.2877817790	0.0015061762	0.8404615079	0.5436335590	16711957
layers.2.mlp.down_proj.weight	mlp.down_proj	2	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1672304723	0.9868587399	1466363
layers.2.mlp.gate_proj.weight	mlp.gate_proj	2	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7989387889	0.6014123474	12534021
layers.2.mlp.up_proj.weight	mlp.up_proj	2	HGRAVR02	1	128	14343858	89128960	1.2874700210	0.0042664660	0.8420516138	0.5406146434	14343858
layers.2.input_layernorm.weight	norm.input	2	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.6858752326	0.9571602098	2977
layers.2.post_attention_layernorm.weight	norm.post_attn	2	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.8124495533	1.1713567691	2977
layers.3.self_attn.k_norm.weight	gqa.k_norm	3	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9049743337	0.8145770185	387
layers.3.self_attn.k_proj.weight	gqa.k_proj	3	HGRAVR02	1	128	844434	5242880	1.2885040283	0.0002511702	0.8432426434	0.5392654770	2785554
layers.3.self_attn.o_proj.weight	gqa.o_proj	3	HGRAVR02	1	128	5063792	31457280	1.2877888997	0.0015061845	0.8363961299	0.5508987560	16711957
layers.3.self_attn.q_norm.weight	gqa.q_norm	3	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9709204710	0.8115677098	387
layers.3.self_attn.q_proj.weight	gqa.q_proj	3	HGRAVR02	1	128	10128599	62914560	1.2879179637	0.0030126709	0.8456168689	0.5355307015	33423639
layers.3.self_attn.v_proj.weight	gqa.v_proj	3	HGRAVR02	1	128	844429	5242880	1.2884963989	0.0002511687	0.8425345001	0.5401620108	2785554
layers.3.mlp.down_proj.weight	mlp.down_proj	3	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1721649968	0.9860658866	1466361
layers.3.mlp.gate_proj.weight	mlp.gate_proj	3	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7986340738	0.6018169292	12534021
layers.3.mlp.up_proj.weight	mlp.up_proj	3	HGRAVR02	1	128	14343829	89128960	1.2874674180	0.0042664574	0.8420403320	0.5406267875	14343829
layers.3.input_layernorm.weight	norm.input	3	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9825286708	0.7995943426	2977
layers.3.post_attention_layernorm.weight	norm.post_attn	3	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.7558985133	1.1159691800	2977
layers.4.linear_attn.A_log	dn.A_log	4	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.4.linear_attn.conv1d.weight	dn.conv1d	4	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.4.linear_attn.dt_bias	dn.dt_bias	4	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.4.linear_attn.in_proj_a.weight	dn.in_proj_a	4	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8442890165	0.5385181046	130827
layers.4.linear_attn.in_proj_b.weight	dn.in_proj_b	4	HGRAVR02	1	128	40123	245760	1.3060872396	0.0000119343	0.8423219003	0.5423016305	130827
layers.4.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	4	HGRAVR02	1	128	8438238	52428800	1.2875729370	0.0025098865	0.8423052877	0.5403180201	27853079
layers.4.linear_attn.in_proj_z.weight	dn.in_proj_z	4	HGRAVR02	1	128	5063100	31457280	1.2876129150	0.0015059787	0.8422867342	0.5403482833	16711957
layers.4.linear_attn.norm.weight	dn.norm	4	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.4.linear_attn.out_proj.weight	dn.out_proj	4	HGRAVR02	1	128	5063538	31457280	1.2877243042	0.0015061089	0.8408432545	0.5430280962	16711957
layers.4.mlp.down_proj.weight	mlp.down_proj	4	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1678309506	0.9867426687	1466362
layers.4.mlp.gate_proj.weight	mlp.gate_proj	4	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7981502819	0.6024584031	12534021
layers.4.mlp.up_proj.weight	mlp.up_proj	4	HGRAVR02	1	128	14343848	89128960	1.2874691234	0.0042664631	0.8419232614	0.5408205670	14343848
layers.4.input_layernorm.weight	norm.input	4	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9373765674	0.9063081323	2977
layers.4.post_attention_layernorm.weight	norm.post_attn	4	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.6504678541	1.0927008864	2977
layers.5.linear_attn.A_log	dn.A_log	5	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.5.linear_attn.conv1d.weight	dn.conv1d	5	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.5.linear_attn.dt_bias	dn.dt_bias	5	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.5.linear_attn.in_proj_a.weight	dn.in_proj_a	5	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8428786713	0.5405866262	130827
layers.5.linear_attn.in_proj_b.weight	dn.in_proj_b	5	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8429056115	0.5410868367	130827
layers.5.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	5	HGRAVR02	1	128	8438290	52428800	1.2875808716	0.0025099020	0.8423759396	0.5402315446	27853079
layers.5.linear_attn.in_proj_z.weight	dn.in_proj_z	5	HGRAVR02	1	128	5063250	31457280	1.2876510620	0.0015060233	0.8424550579	0.5401005260	16711957
layers.5.linear_attn.norm.weight	dn.norm	5	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.5.linear_attn.out_proj.weight	dn.out_proj	5	HGRAVR02	1	128	5063570	31457280	1.2877324422	0.0015061185	0.8412244939	0.5423480247	16711957
layers.5.mlp.down_proj.weight	mlp.down_proj	5	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1672515077	0.9869074279	1466364
layers.5.mlp.gate_proj.weight	mlp.gate_proj	5	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7987878763	0.6016127730	12534021
layers.5.mlp.up_proj.weight	mlp.up_proj	5	HGRAVR02	1	128	14344018	89128960	1.2874843822	0.0042665136	0.8420842514	0.5405674235	14344018
layers.5.input_layernorm.weight	norm.input	5	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8686904378	0.9180574271	2977
layers.5.post_attention_layernorm.weight	norm.post_attn	5	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.6503299859	1.0877013485	2977
layers.6.linear_attn.A_log	dn.A_log	6	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.6.linear_attn.conv1d.weight	dn.conv1d	6	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.6.linear_attn.dt_bias	dn.dt_bias	6	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.6.linear_attn.in_proj_a.weight	dn.in_proj_a	6	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8438031241	0.5389753016	130827
layers.6.linear_attn.in_proj_b.weight	dn.in_proj_b	6	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8422227338	0.5426101348	130827
layers.6.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	6	HGRAVR02	1	128	8438316	52428800	1.2875848389	0.0025099097	0.8421027453	0.5406813575	27853079
layers.6.linear_attn.in_proj_z.weight	dn.in_proj_z	6	HGRAVR02	1	128	5063260	31457280	1.2876536051	0.0015060263	0.8425627761	0.5399785912	16711957
layers.6.linear_attn.norm.weight	dn.norm	6	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.6.linear_attn.out_proj.weight	dn.out_proj	6	HGRAVR02	1	128	5063944	31457280	1.2878275553	0.0015062297	0.8410216195	0.5427820125	16711957
layers.6.mlp.down_proj.weight	mlp.down_proj	6	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1611532924	0.9880560655	1466361
layers.6.mlp.gate_proj.weight	mlp.gate_proj	6	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7973715502	0.6034886999	12534021
layers.6.mlp.up_proj.weight	mlp.up_proj	6	HGRAVR02	1	128	14344035	89128960	1.2874859081	0.0042665187	0.8418178155	0.5410379898	14344035
layers.6.input_layernorm.weight	norm.input	6	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8428791277	0.9321374811	2977
layers.6.post_attention_layernorm.weight	norm.post_attn	6	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.7347579798	1.0974061498	2977
layers.7.self_attn.k_norm.weight	gqa.k_norm	7	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9572846240	0.7501189647	387
layers.7.self_attn.k_proj.weight	gqa.k_proj	7	HGRAVR02	1	128	844358	5242880	1.2883880615	0.0002511475	0.8425967827	0.5404003001	2785554
layers.7.self_attn.o_proj.weight	gqa.o_proj	7	HGRAVR02	1	128	5063950	31457280	1.2878290812	0.0015062315	0.8420460259	0.5412312763	16711957
layers.7.self_attn.q_norm.weight	gqa.q_norm	7	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9830368742	0.7488150788	387
layers.7.self_attn.q_proj.weight	gqa.q_proj	7	HGRAVR02	1	128	10127278	62914560	1.2877499898	0.0030122780	0.8437803399	0.5383034974	33423639
layers.7.self_attn.v_proj.weight	gqa.v_proj	7	HGRAVR02	1	128	844384	5242880	1.2884277344	0.0002511553	0.8418430797	0.5412596792	2785554
layers.7.mlp.down_proj.weight	mlp.down_proj	7	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1666870625	0.9868821857	1466362
layers.7.mlp.gate_proj.weight	mlp.gate_proj	7	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7985170599	0.6019721796	12534021
layers.7.mlp.up_proj.weight	mlp.up_proj	7	HGRAVR02	1	128	14344011	89128960	1.2874837539	0.0042665115	0.8419554410	0.5407737322	14344011
layers.7.input_layernorm.weight	norm.input	7	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7420649364	0.9069747057	2977
layers.7.post_attention_layernorm.weight	norm.post_attn	7	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.3302985154	1.0528615789	2977
layers.8.linear_attn.A_log	dn.A_log	8	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.8.linear_attn.conv1d.weight	dn.conv1d	8	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.8.linear_attn.dt_bias	dn.dt_bias	8	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.8.linear_attn.in_proj_a.weight	dn.in_proj_a	8	HGRAVR02	1	128	40122	245760	1.3060546875	0.0000119340	0.8425880718	0.5410447503	130827
layers.8.linear_attn.in_proj_b.weight	dn.in_proj_b	8	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8423729886	0.5419801649	130827
layers.8.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	8	HGRAVR02	1	128	8438286	52428800	1.2875802612	0.0025099008	0.8421436397	0.5405964898	27853079
layers.8.linear_attn.in_proj_z.weight	dn.in_proj_z	8	HGRAVR02	1	128	5063218	31457280	1.2876429240	0.0015060138	0.8422307073	0.5404625627	16711957
layers.8.linear_attn.norm.weight	dn.norm	8	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.8.linear_attn.out_proj.weight	dn.out_proj	8	HGRAVR02	1	128	5063439	31457280	1.2876991272	0.0015060795	0.8420564239	0.5409754800	16711957
layers.8.mlp.down_proj.weight	mlp.down_proj	8	HGRAVS01	3	64	1466365	89128960	0.1316173778	0.0004361586	0.1615851474	0.9879096435	1466365
layers.8.mlp.gate_proj.weight	mlp.gate_proj	8	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7985991817	0.6018632294	12534021
layers.8.mlp.up_proj.weight	mlp.up_proj	8	HGRAVR02	1	128	14343935	89128960	1.2874769323	0.0042664889	0.8420091099	0.5406858585	14343935
layers.8.input_layernorm.weight	norm.input	8	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8887306295	0.9200050069	2977
layers.8.post_attention_layernorm.weight	norm.post_attn	8	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.2878174900	1.0467386043	2977
layers.9.linear_attn.A_log	dn.A_log	9	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.9.linear_attn.conv1d.weight	dn.conv1d	9	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.9.linear_attn.dt_bias	dn.dt_bias	9	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.9.linear_attn.in_proj_a.weight	dn.in_proj_a	9	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8443965673	0.5382242401	130827
layers.9.linear_attn.in_proj_b.weight	dn.in_proj_b	9	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8429319214	0.5413877511	130827
layers.9.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	9	HGRAVR02	1	128	8438205	52428800	1.2875679016	0.0025098767	0.8421694148	0.5405378569	27853079
layers.9.linear_attn.in_proj_z.weight	dn.in_proj_z	9	HGRAVR02	1	128	5063235	31457280	1.2876472473	0.0015060188	0.8425449382	0.5399541892	16711957
layers.9.linear_attn.norm.weight	dn.norm	9	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.9.linear_attn.out_proj.weight	dn.out_proj	9	HGRAVR02	1	128	5063428	31457280	1.2876963298	0.0015060762	0.8417159068	0.5414758991	16711957
layers.9.mlp.down_proj.weight	mlp.down_proj	9	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1761386554	0.9853472662	1466363
layers.9.mlp.gate_proj.weight	mlp.gate_proj	9	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7985449141	0.6019352292	12534021
layers.9.mlp.up_proj.weight	mlp.up_proj	9	HGRAVR02	1	128	14343958	89128960	1.2874789967	0.0042664958	0.8420698106	0.5405849700	14343958
layers.9.input_layernorm.weight	norm.input	9	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8080335125	0.9338065596	2977
layers.9.post_attention_layernorm.weight	norm.post_attn	9	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.3883102877	1.0565741424	2977
layers.10.linear_attn.A_log	dn.A_log	10	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.10.linear_attn.conv1d.weight	dn.conv1d	10	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.10.linear_attn.dt_bias	dn.dt_bias	10	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.10.linear_attn.in_proj_a.weight	dn.in_proj_a	10	HGRAVR02	1	128	40125	245760	1.3061523438	0.0000119349	0.8446686960	0.5375810160	130827
layers.10.linear_attn.in_proj_b.weight	dn.in_proj_b	10	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8432751687	0.5405685020	130827
layers.10.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	10	HGRAVR02	1	128	8438248	52428800	1.2875744629	0.0025098895	0.8421131641	0.5406469005	27853079
layers.10.linear_attn.in_proj_z.weight	dn.in_proj_z	10	HGRAVR02	1	128	5063173	31457280	1.2876314799	0.0015060004	0.8425228337	0.5399912384	16711957
layers.10.linear_attn.norm.weight	dn.norm	10	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.10.linear_attn.out_proj.weight	dn.out_proj	10	HGRAVR02	1	128	5063631	31457280	1.2877479553	0.0015061366	0.8417211055	0.5414733195	16711957
layers.10.mlp.down_proj.weight	mlp.down_proj	10	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1808468456	0.9845284473	1466364
layers.10.mlp.gate_proj.weight	mlp.gate_proj	10	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7980161906	0.6026360091	12534021
layers.10.mlp.up_proj.weight	mlp.up_proj	10	HGRAVR02	1	128	14343952	89128960	1.2874784582	0.0042664940	0.8420186480	0.5406764005	14343952
layers.10.input_layernorm.weight	norm.input	10	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.6295691695	0.9577135595	2977
layers.10.post_attention_layernorm.weight	norm.post_attn	10	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.6200845682	1.0858062422	2977
layers.11.self_attn.k_norm.weight	gqa.k_norm	11	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9668854038	0.7373896642	387
layers.11.self_attn.k_proj.weight	gqa.k_proj	11	HGRAVR02	1	128	844387	5242880	1.2884323120	0.0002511562	0.8427189267	0.5401935190	2785554
layers.11.self_attn.o_proj.weight	gqa.o_proj	11	HGRAVR02	1	128	5063891	31457280	1.2878140767	0.0015062139	0.8425798237	0.5402645740	16711957
layers.11.self_attn.q_norm.weight	gqa.q_norm	11	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9914427732	0.7339647572	387
layers.11.self_attn.q_proj.weight	gqa.q_proj	11	HGRAVR02	1	128	10127224	62914560	1.2877431234	0.0030122619	0.8435285304	0.5386075762	33423639
layers.11.self_attn.v_proj.weight	gqa.v_proj	11	HGRAVR02	1	128	844359	5242880	1.2883895874	0.0002511478	0.8412515109	0.5421443487	2785554
layers.11.mlp.down_proj.weight	mlp.down_proj	11	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1896123524	0.9826832916	1466362
layers.11.mlp.gate_proj.weight	mlp.gate_proj	11	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7985588333	0.6019167632	12534021
layers.11.mlp.up_proj.weight	mlp.up_proj	11	HGRAVR02	1	128	14343970	89128960	1.2874800738	0.0042664994	0.8420731410	0.5405895553	14343970
layers.11.input_layernorm.weight	norm.input	11	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.6090746396	0.9369398008	2977
layers.11.post_attention_layernorm.weight	norm.post_attn	11	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.3654880946	1.0459423571	2977
layers.12.linear_attn.A_log	dn.A_log	12	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.12.linear_attn.conv1d.weight	dn.conv1d	12	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.12.linear_attn.dt_bias	dn.dt_bias	12	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.12.linear_attn.in_proj_a.weight	dn.in_proj_a	12	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8457299825	0.5357876744	130827
layers.12.linear_attn.in_proj_b.weight	dn.in_proj_b	12	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8448531143	0.5379746286	130827
layers.12.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	12	HGRAVR02	1	128	8438373	52428800	1.2875935364	0.0025099267	0.8420371887	0.5407828745	27853079
layers.12.linear_attn.in_proj_z.weight	dn.in_proj_z	12	HGRAVR02	1	128	5063178	31457280	1.2876327515	0.0015060019	0.8424431657	0.5401279618	16711957
layers.12.linear_attn.norm.weight	dn.norm	12	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.12.linear_attn.out_proj.weight	dn.out_proj	12	HGRAVR02	1	128	5063582	31457280	1.2877354940	0.0015061220	0.8423433545	0.5404531268	16711957
layers.12.mlp.down_proj.weight	mlp.down_proj	12	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1866658347	0.9832177057	1466361
layers.12.mlp.gate_proj.weight	mlp.gate_proj	12	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7984236579	0.6020960575	12534021
layers.12.mlp.up_proj.weight	mlp.up_proj	12	HGRAVR02	1	128	14344006	89128960	1.2874833051	0.0042665101	0.8421453248	0.5404736754	14344006
layers.12.input_layernorm.weight	norm.input	12	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.6667134771	0.9543419836	2977
layers.12.post_attention_layernorm.weight	norm.post_attn	12	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.4076328505	1.0495732472	2977
layers.13.linear_attn.A_log	dn.A_log	13	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.13.linear_attn.conv1d.weight	dn.conv1d	13	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.13.linear_attn.dt_bias	dn.dt_bias	13	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.13.linear_attn.in_proj_a.weight	dn.in_proj_a	13	HGRAVR02	1	128	40122	245760	1.3060546875	0.0000119340	0.8444643577	0.5380163481	130827
layers.13.linear_attn.in_proj_b.weight	dn.in_proj_b	13	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8429909640	0.5407666438	130827
layers.13.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	13	HGRAVR02	1	128	8438066	52428800	1.2875466919	0.0025098354	0.8421373144	0.5405548093	27853079
layers.13.linear_attn.in_proj_z.weight	dn.in_proj_z	13	HGRAVR02	1	128	5063242	31457280	1.2876490275	0.0015060209	0.8427711089	0.5395860627	16711957
layers.13.linear_attn.norm.weight	dn.norm	13	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.13.linear_attn.out_proj.weight	dn.out_proj	13	HGRAVR02	1	128	5063382	31457280	1.2876846313	0.0015060625	0.8424269837	0.5401982343	16711957
layers.13.mlp.down_proj.weight	mlp.down_proj	13	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1907196351	0.9825353062	1466363
layers.13.mlp.gate_proj.weight	mlp.gate_proj	13	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7986622341	0.6017795575	12534021
layers.13.mlp.up_proj.weight	mlp.up_proj	13	HGRAVR02	1	128	14344059	89128960	1.2874880622	0.0042665258	0.8421615795	0.5404457060	14344059
layers.13.input_layernorm.weight	norm.input	13	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7421912317	0.9452457709	2977
layers.13.post_attention_layernorm.weight	norm.post_attn	13	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.4677848581	1.0628665524	2977
layers.14.linear_attn.A_log	dn.A_log	14	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.14.linear_attn.conv1d.weight	dn.conv1d	14	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.14.linear_attn.dt_bias	dn.dt_bias	14	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.14.linear_attn.in_proj_a.weight	dn.in_proj_a	14	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8447441495	0.5373220236	130827
layers.14.linear_attn.in_proj_b.weight	dn.in_proj_b	14	HGRAVR02	1	128	40123	245760	1.3060872396	0.0000119343	0.8432801753	0.5401397784	130827
layers.14.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	14	HGRAVR02	1	128	8438288	52428800	1.2875805664	0.0025099014	0.8422485374	0.5404621022	27853079
layers.14.linear_attn.in_proj_z.weight	dn.in_proj_z	14	HGRAVR02	1	128	5063186	31457280	1.2876347860	0.0015060043	0.8425757267	0.5399035631	16711957
layers.14.linear_attn.norm.weight	dn.norm	14	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.14.linear_attn.out_proj.weight	dn.out_proj	14	HGRAVR02	1	128	5063657	31457280	1.2877545675	0.0015061443	0.8417019714	0.5414993194	16711957
layers.14.mlp.down_proj.weight	mlp.down_proj	14	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1925485415	0.9822702544	1466363
layers.14.mlp.gate_proj.weight	mlp.gate_proj	14	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7986580006	0.6017851760	12534021
layers.14.mlp.up_proj.weight	mlp.up_proj	14	HGRAVR02	1	128	14344028	89128960	1.2874852798	0.0042665166	0.8422263790	0.5403500505	14344028
layers.14.input_layernorm.weight	norm.input	14	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.5714464975	0.9625959805	2977
layers.14.post_attention_layernorm.weight	norm.post_attn	14	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.5486393956	1.0821183361	2977
layers.15.self_attn.k_norm.weight	gqa.k_norm	15	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9651426675	0.7291401260	387
layers.15.self_attn.k_proj.weight	gqa.k_proj	15	HGRAVR02	1	128	844432	5242880	1.2885009766	0.0002511696	0.8429193327	0.5397709209	2785554
layers.15.self_attn.o_proj.weight	gqa.o_proj	15	HGRAVR02	1	128	5063739	31457280	1.2877754211	0.0015061687	0.8422816118	0.5404945200	16711957
layers.15.self_attn.q_norm.weight	gqa.q_norm	15	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9659750308	0.7278463148	387
layers.15.self_attn.q_proj.weight	gqa.q_proj	15	HGRAVR02	1	128	10125980	62914560	1.2875849406	0.0030118919	0.8427765972	0.5396815844	33423639
layers.15.self_attn.v_proj.weight	gqa.v_proj	15	HGRAVR02	1	128	844342	5242880	1.2883636475	0.0002511428	0.8411717798	0.5421150132	2785554
layers.15.mlp.down_proj.weight	mlp.down_proj	15	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.2035205440	0.9800124842	1466362
layers.15.mlp.gate_proj.weight	mlp.gate_proj	15	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7986801296	0.6017558065	12534021
layers.15.mlp.up_proj.weight	mlp.up_proj	15	HGRAVR02	1	128	14344033	89128960	1.2874857285	0.0042665181	0.8421440175	0.5404833037	14344033
layers.15.input_layernorm.weight	norm.input	15	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4572610976	0.9642753091	2977
layers.15.post_attention_layernorm.weight	norm.post_attn	15	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.4411283489	1.0544999201	2977
layers.16.linear_attn.A_log	dn.A_log	16	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.16.linear_attn.conv1d.weight	dn.conv1d	16	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.16.linear_attn.dt_bias	dn.dt_bias	16	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.16.linear_attn.in_proj_a.weight	dn.in_proj_a	16	HGRAVR02	1	128	40131	245760	1.3063476563	0.0000119366	0.8450113158	0.5367319016	130827
layers.16.linear_attn.in_proj_b.weight	dn.in_proj_b	16	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8440541141	0.5390977057	130827
layers.16.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	16	HGRAVR02	1	128	8438234	52428800	1.2875723267	0.0025098853	0.8419529745	0.5408970776	27853079
layers.16.linear_attn.in_proj_z.weight	dn.in_proj_z	16	HGRAVR02	1	128	5063182	31457280	1.2876337687	0.0015060031	0.8426273270	0.5398396547	16711957
layers.16.linear_attn.norm.weight	dn.norm	16	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.16.linear_attn.out_proj.weight	dn.out_proj	16	HGRAVR02	1	128	5063549	31457280	1.2877271016	0.0015061122	0.8421421371	0.5407298261	16711957
layers.16.mlp.down_proj.weight	mlp.down_proj	16	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1982867232	0.9810238560	1466361
layers.16.mlp.gate_proj.weight	mlp.gate_proj	16	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7982511292	0.6023247751	12534021
layers.16.mlp.up_proj.weight	mlp.up_proj	16	HGRAVR02	1	128	14344108	89128960	1.2874924604	0.0042665404	0.8421628029	0.5404661264	14344108
layers.16.input_layernorm.weight	norm.input	16	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.5273687062	0.9669357900	2977
layers.16.post_attention_layernorm.weight	norm.post_attn	16	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.5433357660	1.0564586519	2977
layers.17.linear_attn.A_log	dn.A_log	17	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.17.linear_attn.conv1d.weight	dn.conv1d	17	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.17.linear_attn.dt_bias	dn.dt_bias	17	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.17.linear_attn.in_proj_a.weight	dn.in_proj_a	17	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8444528892	0.5376602679	130827
layers.17.linear_attn.in_proj_b.weight	dn.in_proj_b	17	HGRAVR02	1	128	40119	245760	1.3059570312	0.0000119331	0.8420743076	0.5419171425	130827
layers.17.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	17	HGRAVR02	1	128	8438174	52428800	1.2875631714	0.0025098675	0.8418230533	0.5410899665	27853079
layers.17.linear_attn.in_proj_z.weight	dn.in_proj_z	17	HGRAVR02	1	128	5063365	31457280	1.2876803080	0.0015060575	0.8429533714	0.5393417454	16711957
layers.17.linear_attn.norm.weight	dn.norm	17	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.17.linear_attn.out_proj.weight	dn.out_proj	17	HGRAVR02	1	128	5063517	31457280	1.2877189636	0.0015061027	0.8420391429	0.5408572259	16711957
layers.17.mlp.down_proj.weight	mlp.down_proj	17	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1941196758	0.9818210361	1466362
layers.17.mlp.gate_proj.weight	mlp.gate_proj	17	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7979750006	0.6026905494	12534021
layers.17.mlp.up_proj.weight	mlp.up_proj	17	HGRAVR02	1	128	14344061	89128960	1.2874882418	0.0042665264	0.8420523408	0.5406489974	14344061
layers.17.input_layernorm.weight	norm.input	17	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.6021731014	0.9638094504	2977
layers.17.post_attention_layernorm.weight	norm.post_attn	17	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.5405367035	1.0495770480	2977
layers.18.linear_attn.A_log	dn.A_log	18	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.18.linear_attn.conv1d.weight	dn.conv1d	18	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.18.linear_attn.dt_bias	dn.dt_bias	18	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.18.linear_attn.in_proj_a.weight	dn.in_proj_a	18	HGRAVR02	1	128	40133	245760	1.3064127604	0.0000119372	0.8446199310	0.5373011841	130827
layers.18.linear_attn.in_proj_b.weight	dn.in_proj_b	18	HGRAVR02	1	128	40123	245760	1.3060872396	0.0000119343	0.8418512889	0.5419801540	130827
layers.18.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	18	HGRAVR02	1	128	8438355	52428800	1.2875907898	0.0025099213	0.8420033484	0.5408392413	27853079
layers.18.linear_attn.in_proj_z.weight	dn.in_proj_z	18	HGRAVR02	1	128	5063298	31457280	1.2876632690	0.0015060376	0.8426334161	0.5398564163	16711957
layers.18.linear_attn.norm.weight	dn.norm	18	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.18.linear_attn.out_proj.weight	dn.out_proj	18	HGRAVR02	1	128	5063763	31457280	1.2877815247	0.0015061759	0.8412662580	0.5422192564	16711957
layers.18.mlp.down_proj.weight	mlp.down_proj	18	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1835207089	0.9838693990	1466363
layers.18.mlp.gate_proj.weight	mlp.gate_proj	18	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7961874768	0.6050499995	12534021
layers.18.mlp.up_proj.weight	mlp.up_proj	18	HGRAVR02	1	128	14344268	89128960	1.2875068216	0.0042665880	0.8418419514	0.5410103520	14344268
layers.18.input_layernorm.weight	norm.input	18	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4567935063	0.9747717835	2977
layers.18.post_attention_layernorm.weight	norm.post_attn	18	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.6935818292	1.0522121984	2977
layers.19.self_attn.k_norm.weight	gqa.k_norm	19	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9671787244	0.7145695307	387
layers.19.self_attn.k_proj.weight	gqa.k_proj	19	HGRAVR02	1	128	844398	5242880	1.2884490967	0.0002511594	0.8424823144	0.5406830713	2785554
layers.19.self_attn.o_proj.weight	gqa.o_proj	19	HGRAVR02	1	128	5064706	31457280	1.2880213420	0.0015064564	0.8421021104	0.5409673106	16711957
layers.19.self_attn.q_norm.weight	gqa.q_norm	19	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9785290482	0.7133744214	387
layers.19.self_attn.q_proj.weight	gqa.q_proj	19	HGRAVR02	1	128	10126085	62914560	1.2875982920	0.0030119231	0.8426613769	0.5398649024	33423639
layers.19.self_attn.v_proj.weight	gqa.v_proj	19	HGRAVR02	1	128	844355	5242880	1.2883834839	0.0002511467	0.8401738366	0.5443439855	2785554
layers.19.mlp.down_proj.weight	mlp.down_proj	19	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1760826213	0.9850101423	1466364
layers.19.mlp.gate_proj.weight	mlp.gate_proj	19	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7983576683	0.6021835546	12534021
layers.19.mlp.up_proj.weight	mlp.up_proj	19	HGRAVR02	1	128	14343927	89128960	1.2874762142	0.0042664866	0.8418542283	0.5409398370	14343927
layers.19.input_layernorm.weight	norm.input	19	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1763761062	0.9885756935	2977
layers.19.post_attention_layernorm.weight	norm.post_attn	19	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.3354954767	1.0375881903	2977
layers.20.linear_attn.A_log	dn.A_log	20	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.20.linear_attn.conv1d.weight	dn.conv1d	20	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.20.linear_attn.dt_bias	dn.dt_bias	20	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.20.linear_attn.in_proj_a.weight	dn.in_proj_a	20	HGRAVR02	1	128	40122	245760	1.3060546875	0.0000119340	0.8430196930	0.5399907456	130827
layers.20.linear_attn.in_proj_b.weight	dn.in_proj_b	20	HGRAVR02	1	128	40124	245760	1.3061197917	0.0000119346	0.8410463895	0.5437505800	130827
layers.20.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	20	HGRAVR02	1	128	8438639	52428800	1.2876341248	0.0025100058	0.8421218620	0.5406962114	27853079
layers.20.linear_attn.in_proj_z.weight	dn.in_proj_z	20	HGRAVR02	1	128	5063618	31457280	1.2877446493	0.0015061327	0.8430605196	0.5392096099	16711957
layers.20.linear_attn.norm.weight	dn.norm	20	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.20.linear_attn.out_proj.weight	dn.out_proj	20	HGRAVR02	1	128	5063363	31457280	1.2876797994	0.0015060569	0.8415803321	0.5415762626	16711957
layers.20.mlp.down_proj.weight	mlp.down_proj	20	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1683533556	0.9864034073	1466362
layers.20.mlp.gate_proj.weight	mlp.gate_proj	20	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7986158028	0.6018411747	12534021
layers.20.mlp.up_proj.weight	mlp.up_proj	20	HGRAVR02	1	128	14343971	89128960	1.2874801636	0.0042664996	0.8417754221	0.5410549769	14343971
layers.20.input_layernorm.weight	norm.input	20	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8499758724	0.9174577905	2977
layers.20.post_attention_layernorm.weight	norm.post_attn	20	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.2235950920	1.0337743282	2977
layers.21.linear_attn.A_log	dn.A_log	21	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.21.linear_attn.conv1d.weight	dn.conv1d	21	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.21.linear_attn.dt_bias	dn.dt_bias	21	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.21.linear_attn.in_proj_a.weight	dn.in_proj_a	21	HGRAVR02	1	128	40123	245760	1.3060872396	0.0000119343	0.8416456787	0.5422000236	130827
layers.21.linear_attn.in_proj_b.weight	dn.in_proj_b	21	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8430128535	0.5402524296	130827
layers.21.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	21	HGRAVR02	1	128	8438476	52428800	1.2876092529	0.0025099573	0.8419677300	0.5409703687	27853079
layers.21.linear_attn.in_proj_z.weight	dn.in_proj_z	21	HGRAVR02	1	128	5063610	31457280	1.2877426147	0.0015061304	0.8429260057	0.5394504356	16711957
layers.21.linear_attn.norm.weight	dn.norm	21	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.21.linear_attn.out_proj.weight	dn.out_proj	21	HGRAVR02	1	128	5063750	31457280	1.2877782186	0.0015061720	0.8410979750	0.5424309457	16711957
layers.21.mlp.down_proj.weight	mlp.down_proj	21	HGRAVS01	3	64	1466360	89128960	0.1316169290	0.0004361571	0.1673002404	0.9865786983	1466360
layers.21.mlp.gate_proj.weight	mlp.gate_proj	21	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7985919322	0.6018728486	12534021
layers.21.mlp.up_proj.weight	mlp.up_proj	21	HGRAVR02	1	128	14344010	89128960	1.2874836641	0.0042665112	0.8417755378	0.5410531625	14344010
layers.21.input_layernorm.weight	norm.input	21	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9472417752	0.8793527274	2977
layers.21.post_attention_layernorm.weight	norm.post_attn	21	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.2243218006	1.0378239700	2977
layers.22.linear_attn.A_log	dn.A_log	22	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.22.linear_attn.conv1d.weight	dn.conv1d	22	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.22.linear_attn.dt_bias	dn.dt_bias	22	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.22.linear_attn.in_proj_a.weight	dn.in_proj_a	22	HGRAVR02	1	128	40123	245760	1.3060872396	0.0000119343	0.8412138896	0.5430618963	130827
layers.22.linear_attn.in_proj_b.weight	dn.in_proj_b	22	HGRAVR02	1	128	40124	245760	1.3061197917	0.0000119346	0.8413645751	0.5427244130	130827
layers.22.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	22	HGRAVR02	1	128	8438196	52428800	1.2875665283	0.0025098740	0.8418024397	0.5411235335	27853079
layers.22.linear_attn.in_proj_z.weight	dn.in_proj_z	22	HGRAVR02	1	128	5063466	31457280	1.2877059937	0.0015060875	0.8430600771	0.5391740327	16711957
layers.22.linear_attn.norm.weight	dn.norm	22	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.22.linear_attn.out_proj.weight	dn.out_proj	22	HGRAVR02	1	128	5063327	31457280	1.2876706441	0.0015060462	0.8412999843	0.5420141518	16711957
layers.22.mlp.down_proj.weight	mlp.down_proj	22	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1733672164	0.9856478786	1466363
layers.22.mlp.gate_proj.weight	mlp.gate_proj	22	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7974545860	0.6033789715	12534021
layers.22.mlp.up_proj.weight	mlp.up_proj	22	HGRAVR02	1	128	14343957	89128960	1.2874789070	0.0042664955	0.8413351980	0.5417972035	14343957
layers.22.input_layernorm.weight	norm.input	22	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8027057898	0.9163488848	2977
layers.22.post_attention_layernorm.weight	norm.post_attn	22	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.0845078663	1.0192095847	2977
layers.23.self_attn.k_norm.weight	gqa.k_norm	23	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9649856836	0.6911437536	387
layers.23.self_attn.k_proj.weight	gqa.k_proj	23	HGRAVR02	1	128	844392	5242880	1.2884399414	0.0002511577	0.8422886018	0.5410321577	2785554
layers.23.self_attn.o_proj.weight	gqa.o_proj	23	HGRAVR02	1	128	5063750	31457280	1.2877782186	0.0015061720	0.8426092131	0.5400723813	16711957
layers.23.self_attn.q_norm.weight	gqa.q_norm	23	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9738553730	0.6899575529	387
layers.23.self_attn.q_proj.weight	gqa.q_proj	23	HGRAVR02	1	128	10126576	62914560	1.2876607259	0.0030120692	0.8430238703	0.5393124620	33423639
layers.23.self_attn.v_proj.weight	gqa.v_proj	23	HGRAVR02	1	128	844437	5242880	1.2885086060	0.0002511710	0.8400445201	0.5442379474	2785554
layers.23.mlp.down_proj.weight	mlp.down_proj	23	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1705679033	0.9861031505	1466363
layers.23.mlp.gate_proj.weight	mlp.gate_proj	23	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7984863742	0.6020128821	12534021
layers.23.mlp.up_proj.weight	mlp.up_proj	23	HGRAVR02	1	128	14343982	89128960	1.2874811509	0.0042665029	0.8414675845	0.5415483623	14343982
layers.23.input_layernorm.weight	norm.input	23	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0613014394	0.9981507790	2977
layers.23.post_attention_layernorm.weight	norm.post_attn	23	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0238078440	1.0065960397	2977
layers.24.linear_attn.A_log	dn.A_log	24	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.24.linear_attn.conv1d.weight	dn.conv1d	24	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.24.linear_attn.dt_bias	dn.dt_bias	24	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.24.linear_attn.in_proj_a.weight	dn.in_proj_a	24	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8424676002	0.5415262513	130827
layers.24.linear_attn.in_proj_b.weight	dn.in_proj_b	24	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8414011687	0.5437261685	130827
layers.24.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	24	HGRAVR02	1	128	8438318	52428800	1.2875851440	0.0025099103	0.8414763491	0.5417310062	27853079
layers.24.linear_attn.in_proj_z.weight	dn.in_proj_z	24	HGRAVR02	1	128	5063367	31457280	1.2876808167	0.0015060581	0.8421600814	0.5406351594	16711957
layers.24.linear_attn.norm.weight	dn.norm	24	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.24.linear_attn.out_proj.weight	dn.out_proj	24	HGRAVR02	1	128	5063495	31457280	1.2877133687	0.0015060962	0.8418913668	0.5411109357	16711957
layers.24.mlp.down_proj.weight	mlp.down_proj	24	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1720509402	0.9858087867	1466363
layers.24.mlp.gate_proj.weight	mlp.gate_proj	24	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7979298144	0.6027503723	12534021
layers.24.mlp.up_proj.weight	mlp.up_proj	24	HGRAVR02	1	128	14344082	89128960	1.2874901267	0.0042665327	0.8411007805	0.5421327258	14344082
layers.24.input_layernorm.weight	norm.input	24	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4064746712	0.9767585035	2977
layers.24.post_attention_layernorm.weight	norm.post_attn	24	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1269668823	0.9921595538	2977
layers.25.linear_attn.A_log	dn.A_log	25	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.25.linear_attn.conv1d.weight	dn.conv1d	25	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.25.linear_attn.dt_bias	dn.dt_bias	25	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.25.linear_attn.in_proj_a.weight	dn.in_proj_a	25	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8439538524	0.5389756722	130827
layers.25.linear_attn.in_proj_b.weight	dn.in_proj_b	25	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8427332302	0.5421245750	130827
layers.25.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	25	HGRAVR02	1	128	8438346	52428800	1.2875894165	0.0025099186	0.8413427513	0.5419257701	27853079
layers.25.linear_attn.in_proj_z.weight	dn.in_proj_z	25	HGRAVR02	1	128	5063413	31457280	1.2876925151	0.0015060718	0.8424844880	0.5400916806	16711957
layers.25.linear_attn.norm.weight	dn.norm	25	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.25.linear_attn.out_proj.weight	dn.out_proj	25	HGRAVR02	1	128	5063591	31457280	1.2877377828	0.0015061247	0.8422973730	0.5404821639	16711957
layers.25.mlp.down_proj.weight	mlp.down_proj	25	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1809185189	0.9842820718	1466363
layers.25.mlp.gate_proj.weight	mlp.gate_proj	25	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7975436222	0.6032612790	12534021
layers.25.mlp.up_proj.weight	mlp.up_proj	25	HGRAVR02	1	128	14343949	89128960	1.2874781889	0.0042664931	0.8410052910	0.5422888447	14343949
layers.25.input_layernorm.weight	norm.input	25	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3306174669	0.9758575077	2977
layers.25.post_attention_layernorm.weight	norm.post_attn	25	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1411681541	0.9900967520	2977
layers.26.linear_attn.A_log	dn.A_log	26	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.26.linear_attn.conv1d.weight	dn.conv1d	26	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.26.linear_attn.dt_bias	dn.dt_bias	26	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.26.linear_attn.in_proj_a.weight	dn.in_proj_a	26	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8434606115	0.5396705621	130827
layers.26.linear_attn.in_proj_b.weight	dn.in_proj_b	26	HGRAVR02	1	128	40132	245760	1.3063802083	0.0000119369	0.8427105331	0.5417517517	130827
layers.26.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	26	HGRAVR02	1	128	8438337	52428800	1.2875880432	0.0025099160	0.8410008193	0.5425040380	27853079
layers.26.linear_attn.in_proj_z.weight	dn.in_proj_z	26	HGRAVR02	1	128	5063418	31457280	1.2876937866	0.0015060733	0.8422582696	0.5404761312	16711957
layers.26.linear_attn.norm.weight	dn.norm	26	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.26.linear_attn.out_proj.weight	dn.out_proj	26	HGRAVR02	1	128	5063849	31457280	1.2878033956	0.0015062015	0.8420680271	0.5409405877	16711957
layers.26.mlp.down_proj.weight	mlp.down_proj	26	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1847656101	0.9836651066	1466361
layers.26.mlp.gate_proj.weight	mlp.gate_proj	26	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7966751063	0.6044077887	12534021
layers.26.mlp.up_proj.weight	mlp.up_proj	26	HGRAVR02	1	128	14344448	89128960	1.2875229779	0.0042666415	0.8408325343	0.5426065237	14344448
layers.26.input_layernorm.weight	norm.input	26	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.0783010160	1.0104662217	2977
layers.26.post_attention_layernorm.weight	norm.post_attn	26	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0511058786	1.0036800459	2977
layers.27.self_attn.k_norm.weight	gqa.k_norm	27	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9700782052	0.6551844718	387
layers.27.self_attn.k_proj.weight	gqa.k_proj	27	HGRAVR02	1	128	844415	5242880	1.2884750366	0.0002511645	0.8420111979	0.5418860560	2785554
layers.27.self_attn.o_proj.weight	gqa.o_proj	27	HGRAVR02	1	128	5063963	31457280	1.2878323873	0.0015062354	0.8436677642	0.5384494404	16711957
layers.27.self_attn.q_norm.weight	gqa.q_norm	27	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9864934411	0.6536767647	387
layers.27.self_attn.q_proj.weight	gqa.q_proj	27	HGRAVR02	1	128	10127689	62914560	1.2878022512	0.0030124002	0.8427940142	0.5398634094	33423639
layers.27.self_attn.v_proj.weight	gqa.v_proj	27	HGRAVR02	1	128	844458	5242880	1.2885406494	0.0002511773	0.8386374704	0.5464926113	2785554
layers.27.mlp.down_proj.weight	mlp.down_proj	27	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1798677227	0.9844664960	1466363
layers.27.mlp.gate_proj.weight	mlp.gate_proj	27	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7971929004	0.6037246720	12534021
layers.27.mlp.up_proj.weight	mlp.up_proj	27	HGRAVR02	1	128	14344430	89128960	1.2875213623	0.0042666362	0.8410460783	0.5422632165	14344430
layers.27.input_layernorm.weight	norm.input	27	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.2603700556	1.0261465113	2977
layers.27.post_attention_layernorm.weight	norm.post_attn	27	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1974307667	0.9807847843	2977
layers.28.linear_attn.A_log	dn.A_log	28	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.28.linear_attn.conv1d.weight	dn.conv1d	28	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.28.linear_attn.dt_bias	dn.dt_bias	28	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.28.linear_attn.in_proj_a.weight	dn.in_proj_a	28	HGRAVR02	1	128	40132	245760	1.3063802083	0.0000119369	0.8448048733	0.5376024761	130827
layers.28.linear_attn.in_proj_b.weight	dn.in_proj_b	28	HGRAVR02	1	128	40132	245760	1.3063802083	0.0000119369	0.8407440431	0.5454895589	130827
layers.28.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	28	HGRAVR02	1	128	8438467	52428800	1.2876078796	0.0025099546	0.8405573117	0.5432311538	27853079
layers.28.linear_attn.in_proj_z.weight	dn.in_proj_z	28	HGRAVR02	1	128	5063453	31457280	1.2877026876	0.0015060837	0.8422330166	0.5405646471	16711957
layers.28.linear_attn.norm.weight	dn.norm	28	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.28.linear_attn.out_proj.weight	dn.out_proj	28	HGRAVR02	1	128	5063964	31457280	1.2878326416	0.0015062357	0.8433061325	0.5390257057	16711957
layers.28.mlp.down_proj.weight	mlp.down_proj	28	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1781484371	0.9847430662	1466364
layers.28.mlp.gate_proj.weight	mlp.gate_proj	28	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7966427626	0.6044504188	12534021
layers.28.mlp.up_proj.weight	mlp.up_proj	28	HGRAVR02	1	128	14344321	89128960	1.2875115787	0.0042666038	0.8410945754	0.5421916677	14344321
layers.28.input_layernorm.weight	norm.input	28	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.0229755169	1.0114551728	2977
layers.28.post_attention_layernorm.weight	norm.post_attn	28	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1418628358	0.9905686306	2977
layers.29.linear_attn.A_log	dn.A_log	29	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.29.linear_attn.conv1d.weight	dn.conv1d	29	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.29.linear_attn.dt_bias	dn.dt_bias	29	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.29.linear_attn.in_proj_a.weight	dn.in_proj_a	29	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8441976435	0.5388919183	130827
layers.29.linear_attn.in_proj_b.weight	dn.in_proj_b	29	HGRAVR02	1	128	40124	245760	1.3061197917	0.0000119346	0.8417820070	0.5440883281	130827
layers.29.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	29	HGRAVR02	1	128	8438257	52428800	1.2875758362	0.0025098922	0.8407085469	0.5428821292	27853079
layers.29.linear_attn.in_proj_z.weight	dn.in_proj_z	29	HGRAVR02	1	128	5063476	31457280	1.2877085368	0.0015060905	0.8426244795	0.5399184163	16711957
layers.29.linear_attn.norm.weight	dn.norm	29	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.29.linear_attn.out_proj.weight	dn.out_proj	29	HGRAVR02	1	128	5064073	31457280	1.2878603617	0.0015062681	0.8437785437	0.5382820927	16711957
layers.29.mlp.down_proj.weight	mlp.down_proj	29	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1861267952	0.9832558682	1466363
layers.29.mlp.gate_proj.weight	mlp.gate_proj	29	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7963740008	0.6048044733	12534021
layers.29.mlp.up_proj.weight	mlp.up_proj	29	HGRAVR02	1	128	14344315	89128960	1.2875110402	0.0042666020	0.8410769288	0.5422153340	14344315
layers.29.input_layernorm.weight	norm.input	29	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2444505997	0.9738153175	2977
layers.29.post_attention_layernorm.weight	norm.post_attn	29	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1182452908	0.9955759450	2977
layers.30.linear_attn.A_log	dn.A_log	30	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.30.linear_attn.conv1d.weight	dn.conv1d	30	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.30.linear_attn.dt_bias	dn.dt_bias	30	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.30.linear_attn.in_proj_a.weight	dn.in_proj_a	30	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8455875614	0.5361240073	130827
layers.30.linear_attn.in_proj_b.weight	dn.in_proj_b	30	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8424498809	0.5421505308	130827
layers.30.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	30	HGRAVR02	1	128	8438521	52428800	1.2876161194	0.0025099707	0.8403466772	0.5435337464	27853079
layers.30.linear_attn.in_proj_z.weight	dn.in_proj_z	30	HGRAVR02	1	128	5063460	31457280	1.2877044678	0.0015060857	0.8419934075	0.5409552481	16711957
layers.30.linear_attn.norm.weight	dn.norm	30	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.30.linear_attn.out_proj.weight	dn.out_proj	30	HGRAVR02	1	128	5064255	31457280	1.2879066467	0.0015063222	0.8427101733	0.5400813448	16711957
layers.30.mlp.down_proj.weight	mlp.down_proj	30	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1897203939	0.9826072862	1466363
layers.30.mlp.gate_proj.weight	mlp.gate_proj	30	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7956884418	0.6057061198	12534021
layers.30.mlp.up_proj.weight	mlp.up_proj	30	HGRAVR02	1	128	14344481	89128960	1.2875259399	0.0042666513	0.8409744356	0.5423994143	14344481
layers.30.input_layernorm.weight	norm.input	30	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0848188253	0.9995630716	2977
layers.30.post_attention_layernorm.weight	norm.post_attn	30	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0448562588	1.0097041646	2977
layers.31.self_attn.k_norm.weight	gqa.k_norm	31	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9741028734	0.6378119442	387
layers.31.self_attn.k_proj.weight	gqa.k_proj	31	HGRAVR02	1	128	844515	5242880	1.2886276245	0.0002511942	0.8426539837	0.5407629647	2785554
layers.31.self_attn.o_proj.weight	gqa.o_proj	31	HGRAVR02	1	128	5064072	31457280	1.2878601074	0.0015062678	0.8441585367	0.5377396961	16711957
layers.31.self_attn.q_norm.weight	gqa.q_norm	31	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9755593244	0.6384430282	387
layers.31.self_attn.q_proj.weight	gqa.q_proj	31	HGRAVR02	1	128	10127567	62914560	1.2877867381	0.0030123639	0.8430620141	0.5393368374	33423639
layers.31.self_attn.v_proj.weight	gqa.v_proj	31	HGRAVR02	1	128	844454	5242880	1.2885345459	0.0002511761	0.8383196474	0.5467982688	2785554
layers.31.mlp.down_proj.weight	mlp.down_proj	31	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1916490812	0.9822293406	1466362
layers.31.mlp.gate_proj.weight	mlp.gate_proj	31	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7953716923	0.6061219935	12534021
layers.31.mlp.up_proj.weight	mlp.up_proj	31	HGRAVR02	1	128	14344507	89128960	1.2875282736	0.0042666591	0.8409442151	0.5424497570	14344507
layers.31.input_layernorm.weight	norm.input	31	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0163358100	1.0084500761	2977
layers.31.post_attention_layernorm.weight	norm.post_attn	31	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1400012927	0.9911582531	2977
layers.32.linear_attn.A_log	dn.A_log	32	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.32.linear_attn.conv1d.weight	dn.conv1d	32	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.32.linear_attn.dt_bias	dn.dt_bias	32	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.32.linear_attn.in_proj_a.weight	dn.in_proj_a	32	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8443182016	0.5384963384	130827
layers.32.linear_attn.in_proj_b.weight	dn.in_proj_b	32	HGRAVR02	1	128	40131	245760	1.3063476563	0.0000119366	0.8397849478	0.5475356235	130827
layers.32.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	32	HGRAVR02	1	128	8438502	52428800	1.2876132202	0.0025099650	0.8397537001	0.5444607701	27853079
layers.32.linear_attn.in_proj_z.weight	dn.in_proj_z	32	HGRAVR02	1	128	5063444	31457280	1.2877003988	0.0015060810	0.8423974566	0.5403085122	16711957
layers.32.linear_attn.norm.weight	dn.norm	32	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.32.linear_attn.out_proj.weight	dn.out_proj	32	HGRAVR02	1	128	5064444	31457280	1.2879547119	0.0015063784	0.8442167817	0.5376908398	16711957
layers.32.mlp.down_proj.weight	mlp.down_proj	32	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1872462789	0.9830246123	1466364
layers.32.mlp.gate_proj.weight	mlp.gate_proj	32	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7945133040	0.6072467454	12534021
layers.32.mlp.up_proj.weight	mlp.up_proj	32	HGRAVR02	1	128	14344504	89128960	1.2875280044	0.0042666582	0.8408951101	0.5425396864	14344504
layers.32.input_layernorm.weight	norm.input	32	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1328900760	0.9917756163	2977
layers.32.post_attention_layernorm.weight	norm.post_attn	32	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0837680094	1.0005883980	2977
layers.33.linear_attn.A_log	dn.A_log	33	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.33.linear_attn.conv1d.weight	dn.conv1d	33	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.33.linear_attn.dt_bias	dn.dt_bias	33	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.33.linear_attn.in_proj_a.weight	dn.in_proj_a	33	HGRAVR02	1	128	40136	245760	1.3065104167	0.0000119381	0.8459331408	0.5362385036	130827
layers.33.linear_attn.in_proj_b.weight	dn.in_proj_b	33	HGRAVR02	1	128	40133	245760	1.3064127604	0.0000119372	0.8414782084	0.5447052632	130827
layers.33.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	33	HGRAVR02	1	128	8438372	52428800	1.2875933838	0.0025099264	0.8398075270	0.5443720541	27853079
layers.33.linear_attn.in_proj_z.weight	dn.in_proj_z	33	HGRAVR02	1	128	5063432	31457280	1.2876973470	0.0015060774	0.8423090809	0.5404521428	16711957
layers.33.linear_attn.norm.weight	dn.norm	33	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.33.linear_attn.out_proj.weight	dn.out_proj	33	HGRAVR02	1	128	5063970	31457280	1.2878341675	0.0015062374	0.8432210621	0.5391692529	16711957
layers.33.mlp.down_proj.weight	mlp.down_proj	33	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1771536022	0.9848479178	1466362
layers.33.mlp.gate_proj.weight	mlp.gate_proj	33	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7937407762	0.6082561798	12534021
layers.33.mlp.up_proj.weight	mlp.up_proj	33	HGRAVR02	1	128	14344481	89128960	1.2875259399	0.0042666513	0.8408191896	0.5426791800	14344481
layers.33.input_layernorm.weight	norm.input	33	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3115148041	0.9633869030	2977
layers.33.post_attention_layernorm.weight	norm.post_attn	33	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0620896781	1.0019528837	2977
layers.34.linear_attn.A_log	dn.A_log	34	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.34.linear_attn.conv1d.weight	dn.conv1d	34	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.34.linear_attn.dt_bias	dn.dt_bias	34	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.34.linear_attn.in_proj_a.weight	dn.in_proj_a	34	HGRAVR02	1	128	40133	245760	1.3064127604	0.0000119372	0.8458007300	0.5358605331	130827
layers.34.linear_attn.in_proj_b.weight	dn.in_proj_b	34	HGRAVR02	1	128	40123	245760	1.3060872396	0.0000119343	0.8393657341	0.5479818215	130827
layers.34.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	34	HGRAVR02	1	128	8438590	52428800	1.2876266479	0.0025099912	0.8398930292	0.5442502655	27853079
layers.34.linear_attn.in_proj_z.weight	dn.in_proj_z	34	HGRAVR02	1	128	5063342	31457280	1.2876744588	0.0015060507	0.8417313316	0.5414017154	16711957
layers.34.linear_attn.norm.weight	dn.norm	34	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.34.linear_attn.out_proj.weight	dn.out_proj	34	HGRAVR02	1	128	5064096	31457280	1.2878662109	0.0015062749	0.8420915653	0.5410474871	16711957
layers.34.mlp.down_proj.weight	mlp.down_proj	34	HGRAVS01	3	64	1466360	89128960	0.1316169290	0.0004361571	0.1650007869	0.9869696416	1466360
layers.34.mlp.gate_proj.weight	mlp.gate_proj	34	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7905339208	0.6124182558	12534021
layers.34.mlp.up_proj.weight	mlp.up_proj	34	HGRAVR02	1	128	14344896	89128960	1.2875631893	0.0042667748	0.8406419943	0.5429954737	14344896
layers.34.input_layernorm.weight	norm.input	34	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2552065375	0.9737602644	2977
layers.34.post_attention_layernorm.weight	norm.post_attn	34	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.1552502021	1.0234708792	2977
layers.35.self_attn.k_norm.weight	gqa.k_norm	35	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9734619209	0.6192621449	387
layers.35.self_attn.k_proj.weight	gqa.k_proj	35	HGRAVR02	1	128	844447	5242880	1.2885238647	0.0002511740	0.8419295967	0.5416842566	2785554
layers.35.self_attn.o_proj.weight	gqa.o_proj	35	HGRAVR02	1	128	5063653	31457280	1.2877535502	0.0015061432	0.8428015740	0.5397678601	16711957
layers.35.self_attn.q_norm.weight	gqa.q_norm	35	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9841001458	0.6196284845	387
layers.35.self_attn.q_proj.weight	gqa.q_proj	35	HGRAVR02	1	128	10128624	62914560	1.2879211426	0.0030126783	0.8427974852	0.5397112968	33423639
layers.35.self_attn.v_proj.weight	gqa.v_proj	35	HGRAVR02	1	128	844403	5242880	1.2884567261	0.0002511609	0.8389421018	0.5462034325	2785554
layers.35.mlp.down_proj.weight	mlp.down_proj	35	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1656272479	0.9867718601	1466363
layers.35.mlp.gate_proj.weight	mlp.gate_proj	35	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7945375412	0.6072150324	12534021
layers.35.mlp.up_proj.weight	mlp.up_proj	35	HGRAVR02	1	128	14344405	89128960	1.2875191184	0.0042666287	0.8409737423	0.5423969664	14344405
layers.35.input_layernorm.weight	norm.input	35	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2267652923	0.9817805351	2977
layers.35.post_attention_layernorm.weight	norm.post_attn	35	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0235590028	1.0046356467	2977
layers.36.linear_attn.A_log	dn.A_log	36	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.36.linear_attn.conv1d.weight	dn.conv1d	36	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.36.linear_attn.dt_bias	dn.dt_bias	36	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.36.linear_attn.in_proj_a.weight	dn.in_proj_a	36	HGRAVR02	1	128	40125	245760	1.3061523438	0.0000119349	0.8422479992	0.5419430598	130827
layers.36.linear_attn.in_proj_b.weight	dn.in_proj_b	36	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8414128050	0.5443159898	130827
layers.36.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	36	HGRAVR02	1	128	8438725	52428800	1.2876472473	0.0025100314	0.8413741894	0.5419410067	27853079
layers.36.linear_attn.in_proj_z.weight	dn.in_proj_z	36	HGRAVR02	1	128	5063360	31457280	1.2876790365	0.0015060560	0.8425222040	0.5400705619	16711957
layers.36.linear_attn.norm.weight	dn.norm	36	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.36.linear_attn.out_proj.weight	dn.out_proj	36	HGRAVR02	1	128	5063410	31457280	1.2876917521	0.0015060709	0.8416825207	0.5414346632	16711957
layers.36.mlp.down_proj.weight	mlp.down_proj	36	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1618572128	0.9874562311	1466364
layers.36.mlp.gate_proj.weight	mlp.gate_proj	36	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7956839492	0.6057120215	12534021
layers.36.mlp.up_proj.weight	mlp.up_proj	36	HGRAVR02	1	128	14344134	89128960	1.2874947941	0.0042665481	0.8410907947	0.5421713590	14344134
layers.36.input_layernorm.weight	norm.input	36	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7868342359	0.9081574283	2977
layers.36.post_attention_layernorm.weight	norm.post_attn	36	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.0268043413	1.0119135528	2977
layers.37.linear_attn.A_log	dn.A_log	37	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.37.linear_attn.conv1d.weight	dn.conv1d	37	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.37.linear_attn.dt_bias	dn.dt_bias	37	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.37.linear_attn.in_proj_a.weight	dn.in_proj_a	37	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8422978365	0.5420573525	130827
layers.37.linear_attn.in_proj_b.weight	dn.in_proj_b	37	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8415452028	0.5429259419	130827
layers.37.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	37	HGRAVR02	1	128	8438432	52428800	1.2876025391	0.0025099442	0.8412355352	0.5421817545	27853079
layers.37.linear_attn.in_proj_z.weight	dn.in_proj_z	37	HGRAVR02	1	128	5063483	31457280	1.2877103170	0.0015060926	0.8423543519	0.5403708305	16711957
layers.37.linear_attn.norm.weight	dn.norm	37	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.37.linear_attn.out_proj.weight	dn.out_proj	37	HGRAVR02	1	128	5064075	31457280	1.2878608704	0.0015062687	0.8401988664	0.5438310222	16711957
layers.37.mlp.down_proj.weight	mlp.down_proj	37	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1633759251	0.9871967380	1466363
layers.37.mlp.gate_proj.weight	mlp.gate_proj	37	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7968954012	0.6041173061	12534021
layers.37.mlp.up_proj.weight	mlp.up_proj	37	HGRAVR02	1	128	14344029	89128960	1.2874853695	0.0042665169	0.8414075548	0.5416538586	14344029
layers.37.input_layernorm.weight	norm.input	37	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9123375809	0.8980314979	2977
layers.37.post_attention_layernorm.weight	norm.post_attn	37	f32v2	32		20488	5120	32.0125000000	0.0000060940	-0.0514670739	1.0159289975	2977
layers.38.linear_attn.A_log	dn.A_log	38	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.38.linear_attn.conv1d.weight	dn.conv1d	38	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.38.linear_attn.dt_bias	dn.dt_bias	38	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.38.linear_attn.in_proj_a.weight	dn.in_proj_a	38	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8396643699	0.5465461999	130827
layers.38.linear_attn.in_proj_b.weight	dn.in_proj_b	38	HGRAVR02	1	128	40125	245760	1.3061523438	0.0000119349	0.8412804039	0.5436928450	130827
layers.38.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	38	HGRAVR02	1	128	8438147	52428800	1.2875590515	0.0025098595	0.8414510329	0.5416990180	27853079
layers.38.linear_attn.in_proj_z.weight	dn.in_proj_z	38	HGRAVR02	1	128	5063417	31457280	1.2876935323	0.0015060730	0.8428722944	0.5394878957	16711957
layers.38.linear_attn.norm.weight	dn.norm	38	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.38.linear_attn.out_proj.weight	dn.out_proj	38	HGRAVR02	1	128	5063254	31457280	1.2876520793	0.0015060245	0.8412471445	0.5420535654	16711957
layers.38.mlp.down_proj.weight	mlp.down_proj	38	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1680390306	0.9864657903	1466363
layers.38.mlp.gate_proj.weight	mlp.gate_proj	38	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7963457286	0.6048416988	12534021
layers.38.mlp.up_proj.weight	mlp.up_proj	38	HGRAVR02	1	128	14344273	89128960	1.2875072704	0.0042665895	0.8411961126	0.5420207067	14344273
layers.38.input_layernorm.weight	norm.input	38	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7581566082	0.9129462109	2977
layers.38.post_attention_layernorm.weight	norm.post_attn	38	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1937872988	0.9832982652	2977
layers.39.self_attn.k_norm.weight	gqa.k_norm	39	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9681490582	0.6194199816	387
layers.39.self_attn.k_proj.weight	gqa.k_proj	39	HGRAVR02	1	128	844437	5242880	1.2885086060	0.0002511710	0.8418696824	0.5419746539	2785554
layers.39.self_attn.o_proj.weight	gqa.o_proj	39	HGRAVR02	1	128	5063869	31457280	1.2878084819	0.0015062074	0.8426842358	0.5399880214	16711957
layers.39.self_attn.q_norm.weight	gqa.q_norm	39	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9774663974	0.6196228268	387
layers.39.self_attn.q_proj.weight	gqa.q_proj	39	HGRAVR02	1	128	10127761	62914560	1.2878114065	0.0030124216	0.8428691869	0.5396290739	33423639
layers.39.self_attn.v_proj.weight	gqa.v_proj	39	HGRAVR02	1	128	844447	5242880	1.2885238647	0.0002511740	0.8398875871	0.5446217125	2785554
layers.39.mlp.down_proj.weight	mlp.down_proj	39	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1657553688	0.9869096522	1466363
layers.39.mlp.gate_proj.weight	mlp.gate_proj	39	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7979617463	0.6027080980	12534021
layers.39.mlp.up_proj.weight	mlp.up_proj	39	HGRAVR02	1	128	14343990	89128960	1.2874818690	0.0042665053	0.8415222552	0.5414636532	14343990
layers.39.input_layernorm.weight	norm.input	39	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1545319989	0.9908734536	2977
layers.39.post_attention_layernorm.weight	norm.post_attn	39	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2117990201	0.9801156658	2977
layers.40.linear_attn.A_log	dn.A_log	40	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.40.linear_attn.conv1d.weight	dn.conv1d	40	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.40.linear_attn.dt_bias	dn.dt_bias	40	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.40.linear_attn.in_proj_a.weight	dn.in_proj_a	40	HGRAVR02	1	128	40125	245760	1.3061523438	0.0000119349	0.8422364736	0.5428171862	130827
layers.40.linear_attn.in_proj_b.weight	dn.in_proj_b	40	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8409118766	0.5454720307	130827
layers.40.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	40	HGRAVR02	1	128	8438457	52428800	1.2876063538	0.0025099517	0.8410538900	0.5424780906	27853079
layers.40.linear_attn.in_proj_z.weight	dn.in_proj_z	40	HGRAVR02	1	128	5063490	31457280	1.2877120972	0.0015060947	0.8423166044	0.5404791003	16711957
layers.40.linear_attn.norm.weight	dn.norm	40	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.40.linear_attn.out_proj.weight	dn.out_proj	40	HGRAVR02	1	128	5063379	31457280	1.2876838684	0.0015060617	0.8415735066	0.5415887408	16711957
layers.40.mlp.down_proj.weight	mlp.down_proj	40	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1677291854	0.9865082993	1466363
layers.40.mlp.gate_proj.weight	mlp.gate_proj	40	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7975147826	0.6032994045	12534021
layers.40.mlp.up_proj.weight	mlp.up_proj	40	HGRAVR02	1	128	14344057	89128960	1.2874878827	0.0042665252	0.8411204392	0.5421096987	14344057
layers.40.input_layernorm.weight	norm.input	40	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7086353888	0.9505328617	2977
layers.40.post_attention_layernorm.weight	norm.post_attn	40	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3124435444	0.9641744129	2977
layers.41.linear_attn.A_log	dn.A_log	41	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.41.linear_attn.conv1d.weight	dn.conv1d	41	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.41.linear_attn.dt_bias	dn.dt_bias	41	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.41.linear_attn.in_proj_a.weight	dn.in_proj_a	41	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8417301561	0.5435707578	130827
layers.41.linear_attn.in_proj_b.weight	dn.in_proj_b	41	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8397975855	0.5469588095	130827
layers.41.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	41	HGRAVR02	1	128	8438427	52428800	1.2876017761	0.0025099427	0.8412791520	0.5420297945	27853079
layers.41.linear_attn.in_proj_z.weight	dn.in_proj_z	41	HGRAVR02	1	128	5063493	31457280	1.2877128601	0.0015060956	0.8425906494	0.5399630579	16711957
layers.41.linear_attn.norm.weight	dn.norm	41	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.41.linear_attn.out_proj.weight	dn.out_proj	41	HGRAVR02	1	128	5063310	31457280	1.2876663208	0.0015060411	0.8418755161	0.5410848245	16711957
layers.41.mlp.down_proj.weight	mlp.down_proj	41	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1711932792	0.9859747806	1466363
layers.41.mlp.gate_proj.weight	mlp.gate_proj	41	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7967364695	0.6043268967	12534021
layers.41.mlp.up_proj.weight	mlp.up_proj	41	HGRAVR02	1	128	14344113	89128960	1.2874929092	0.0042665419	0.8409693834	0.5423624361	14344113
layers.41.input_layernorm.weight	norm.input	41	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.5117868609	0.9528551917	2977
layers.41.post_attention_layernorm.weight	norm.post_attn	41	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2581007135	0.9714321032	2977
layers.42.linear_attn.A_log	dn.A_log	42	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.42.linear_attn.conv1d.weight	dn.conv1d	42	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.42.linear_attn.dt_bias	dn.dt_bias	42	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.42.linear_attn.in_proj_a.weight	dn.in_proj_a	42	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8423821008	0.5419060893	130827
layers.42.linear_attn.in_proj_b.weight	dn.in_proj_b	42	HGRAVR02	1	128	40138	245760	1.3065755208	0.0000119387	0.8446623734	0.5392265043	130827
layers.42.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	42	HGRAVR02	1	128	8438455	52428800	1.2876060486	0.0025099511	0.8409178441	0.5426412083	27853079
layers.42.linear_attn.in_proj_z.weight	dn.in_proj_z	42	HGRAVR02	1	128	5063519	31457280	1.2877194722	0.0015061033	0.8425003888	0.5401457945	16711957
layers.42.linear_attn.norm.weight	dn.norm	42	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.42.linear_attn.out_proj.weight	dn.out_proj	42	HGRAVR02	1	128	5063693	31457280	1.2877637227	0.0015061551	0.8413344125	0.5420929712	16711957
layers.42.mlp.down_proj.weight	mlp.down_proj	42	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1741766507	0.9855222011	1466362
layers.42.mlp.gate_proj.weight	mlp.gate_proj	42	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7953244242	0.6061840152	12534021
layers.42.mlp.up_proj.weight	mlp.up_proj	42	HGRAVR02	1	128	14344273	89128960	1.2875072704	0.0042665895	0.8407209641	0.5427983474	14344273
layers.42.input_layernorm.weight	norm.input	42	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2621162931	0.9791324251	2977
layers.42.post_attention_layernorm.weight	norm.post_attn	42	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1823778649	0.9839406564	2977
layers.43.self_attn.k_norm.weight	gqa.k_norm	43	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9767588654	0.6093702192	387
layers.43.self_attn.k_proj.weight	gqa.k_proj	43	HGRAVR02	1	128	844480	5242880	1.2885742188	0.0002511838	0.8412240709	0.5437261765	2785554
layers.43.self_attn.o_proj.weight	gqa.o_proj	43	HGRAVR02	1	128	5064023	31457280	1.2878476461	0.0015062532	0.8425711559	0.5402835580	16711957
layers.43.self_attn.q_norm.weight	gqa.q_norm	43	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9855983906	0.6101661893	387
layers.43.self_attn.q_proj.weight	gqa.q_proj	43	HGRAVR02	1	128	10129445	62914560	1.2880255381	0.0030129225	0.8426019613	0.5401766715	33423639
layers.43.self_attn.v_proj.weight	gqa.v_proj	43	HGRAVR02	1	128	844433	5242880	1.2885025024	0.0002511699	0.8385081419	0.5469807772	2785554
layers.43.mlp.down_proj.weight	mlp.down_proj	43	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1716977879	0.9859165957	1466361
layers.43.mlp.gate_proj.weight	mlp.gate_proj	43	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7963343951	0.6048566203	12534021
layers.43.mlp.up_proj.weight	mlp.up_proj	43	HGRAVR02	1	128	14344351	89128960	1.2875142715	0.0042666127	0.8410983745	0.5421835741	14344351
layers.43.input_layernorm.weight	norm.input	43	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.0177440195	1.0022835377	2977
layers.43.post_attention_layernorm.weight	norm.post_attn	43	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3198529133	0.9604224911	2977
layers.44.linear_attn.A_log	dn.A_log	44	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.44.linear_attn.conv1d.weight	dn.conv1d	44	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.44.linear_attn.dt_bias	dn.dt_bias	44	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.44.linear_attn.in_proj_a.weight	dn.in_proj_a	44	HGRAVR02	1	128	40132	245760	1.3063802083	0.0000119369	0.8457883690	0.5367563094	130827
layers.44.linear_attn.in_proj_b.weight	dn.in_proj_b	44	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8410702333	0.5447212668	130827
layers.44.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	44	HGRAVR02	1	128	8438599	52428800	1.2876280212	0.0025099939	0.8406102087	0.5431406914	27853079
layers.44.linear_attn.in_proj_z.weight	dn.in_proj_z	44	HGRAVR02	1	128	5063541	31457280	1.2877250671	0.0015061098	0.8422009432	0.5406660371	16711957
layers.44.linear_attn.norm.weight	dn.norm	44	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.44.linear_attn.out_proj.weight	dn.out_proj	44	HGRAVR02	1	128	5063868	31457280	1.2878082275	0.0015062071	0.8423985404	0.5404435018	16711957
layers.44.mlp.down_proj.weight	mlp.down_proj	44	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1715965682	0.9858637415	1466361
layers.44.mlp.gate_proj.weight	mlp.gate_proj	44	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7956413948	0.6057679184	12534021
layers.44.mlp.up_proj.weight	mlp.up_proj	44	HGRAVR02	1	128	14344415	89128960	1.2875200159	0.0042666317	0.8411101150	0.5421784916	14344415
layers.44.input_layernorm.weight	norm.input	44	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1592025825	0.9882386205	2977
layers.44.post_attention_layernorm.weight	norm.post_attn	44	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2983072226	0.9640382941	2977
layers.45.linear_attn.A_log	dn.A_log	45	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.45.linear_attn.conv1d.weight	dn.conv1d	45	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.45.linear_attn.dt_bias	dn.dt_bias	45	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.45.linear_attn.in_proj_a.weight	dn.in_proj_a	45	HGRAVR02	1	128	40135	245760	1.3064778646	0.0000119378	0.8431469554	0.5412686498	130827
layers.45.linear_attn.in_proj_b.weight	dn.in_proj_b	45	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8399937178	0.5466427507	130827
layers.45.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	45	HGRAVR02	1	128	8438366	52428800	1.2875924683	0.0025099246	0.8406308339	0.5430190920	27853079
layers.45.linear_attn.in_proj_z.weight	dn.in_proj_z	45	HGRAVR02	1	128	5063402	31457280	1.2876897176	0.0015060685	0.8424096583	0.5402288771	16711957
layers.45.linear_attn.norm.weight	dn.norm	45	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.45.linear_attn.out_proj.weight	dn.out_proj	45	HGRAVR02	1	128	5063739	31457280	1.2877754211	0.0015061687	0.8428937069	0.5395910530	16711957
layers.45.mlp.down_proj.weight	mlp.down_proj	45	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1749851587	0.9853077537	1466362
layers.45.mlp.gate_proj.weight	mlp.gate_proj	45	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7959291723	0.6053897527	12534021
layers.45.mlp.up_proj.weight	mlp.up_proj	45	HGRAVR02	1	128	14344403	89128960	1.2875189388	0.0042666281	0.8411127585	0.5421681343	14344403
layers.45.input_layernorm.weight	norm.input	45	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3440014738	0.9605893478	2977
layers.45.post_attention_layernorm.weight	norm.post_attn	45	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2903463475	0.9646131870	2977
layers.46.linear_attn.A_log	dn.A_log	46	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.46.linear_attn.conv1d.weight	dn.conv1d	46	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.46.linear_attn.dt_bias	dn.dt_bias	46	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.46.linear_attn.in_proj_a.weight	dn.in_proj_a	46	HGRAVR02	1	128	40135	245760	1.3064778646	0.0000119378	0.8453327185	0.5371256548	130827
layers.46.linear_attn.in_proj_b.weight	dn.in_proj_b	46	HGRAVR02	1	128	40131	245760	1.3063476563	0.0000119366	0.8420979040	0.5424659185	130827
layers.46.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	46	HGRAVR02	1	128	8438630	52428800	1.2876327515	0.0025100031	0.8404841214	0.5433216990	27853079
layers.46.linear_attn.in_proj_z.weight	dn.in_proj_z	46	HGRAVR02	1	128	5063569	31457280	1.2877321879	0.0015061182	0.8420287517	0.5409216178	16711957
layers.46.linear_attn.norm.weight	dn.norm	46	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.46.linear_attn.out_proj.weight	dn.out_proj	46	HGRAVR02	1	128	5063894	31457280	1.2878148397	0.0015062148	0.8418365112	0.5413257992	16711957
layers.46.mlp.down_proj.weight	mlp.down_proj	46	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1758782369	0.9851394370	1466362
layers.46.mlp.gate_proj.weight	mlp.gate_proj	46	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7957590752	0.6056133207	12534021
layers.46.mlp.up_proj.weight	mlp.up_proj	46	HGRAVR02	1	128	14344480	89128960	1.2875258502	0.0042666510	0.8411899656	0.5420691169	14344480
layers.46.input_layernorm.weight	norm.input	46	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.2611705522	0.9734751410	2977
layers.46.post_attention_layernorm.weight	norm.post_attn	46	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.1938147909	0.9816774163	2977
layers.47.self_attn.k_norm.weight	gqa.k_norm	47	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9689060165	0.6073636263	387
layers.47.self_attn.k_proj.weight	gqa.k_proj	47	HGRAVR02	1	128	844645	5242880	1.2888259888	0.0002512329	0.8438094703	0.5391684455	2785554
layers.47.self_attn.o_proj.weight	gqa.o_proj	47	HGRAVR02	1	128	5063536	31457280	1.2877237956	0.0015061084	0.8421232351	0.5408400156	16711957
layers.47.self_attn.q_norm.weight	gqa.q_norm	47	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9796667915	0.6079645195	387
layers.47.self_attn.q_proj.weight	gqa.q_proj	47	HGRAVR02	1	128	10129582	62914560	1.2880429586	0.0030129633	0.8433016097	0.5389394987	33423639
layers.47.self_attn.v_proj.weight	gqa.v_proj	47	HGRAVR02	1	128	844464	5242880	1.2885498047	0.0002511791	0.8405608761	0.5433970772	2785554
layers.47.mlp.down_proj.weight	mlp.down_proj	47	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1777445777	0.9848759544	1466363
layers.47.mlp.gate_proj.weight	mlp.gate_proj	47	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7955532533	0.6058836696	12534021
layers.47.mlp.up_proj.weight	mlp.up_proj	47	HGRAVR02	1	128	14344687	89128960	1.2875444300	0.0042667126	0.8412435837	0.5419885140	14344687
layers.47.input_layernorm.weight	norm.input	47	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4448815882	0.9541475048	2977
layers.47.post_attention_layernorm.weight	norm.post_attn	47	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3442868612	0.9587046659	2977
layers.48.linear_attn.A_log	dn.A_log	48	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.48.linear_attn.conv1d.weight	dn.conv1d	48	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.48.linear_attn.dt_bias	dn.dt_bias	48	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.48.linear_attn.in_proj_a.weight	dn.in_proj_a	48	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8438964842	0.5391446422	130827
layers.48.linear_attn.in_proj_b.weight	dn.in_proj_b	48	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8420428555	0.5427473445	130827
layers.48.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	48	HGRAVR02	1	128	8438595	52428800	1.2876274109	0.0025099927	0.8404499926	0.5433793429	27853079
layers.48.linear_attn.in_proj_z.weight	dn.in_proj_z	48	HGRAVR02	1	128	5063539	31457280	1.2877245585	0.0015061092	0.8425774121	0.5400262151	16711957
layers.48.linear_attn.norm.weight	dn.norm	48	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.48.linear_attn.out_proj.weight	dn.out_proj	48	HGRAVR02	1	128	5064096	31457280	1.2878662109	0.0015062749	0.8425393802	0.5402394366	16711957
layers.48.mlp.down_proj.weight	mlp.down_proj	48	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1731921778	0.9856311347	1466362
layers.48.mlp.gate_proj.weight	mlp.gate_proj	48	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7941434448	0.6077303588	12534021
layers.48.mlp.up_proj.weight	mlp.up_proj	48	HGRAVR02	1	128	14344826	89128960	1.2875569063	0.0042667540	0.8412433713	0.5420144377	14344826
layers.48.input_layernorm.weight	norm.input	48	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4150788937	0.9568628024	2977
layers.48.post_attention_layernorm.weight	norm.post_attn	48	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.3738792199	0.9587157100	2977
layers.49.linear_attn.A_log	dn.A_log	49	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.49.linear_attn.conv1d.weight	dn.conv1d	49	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.49.linear_attn.dt_bias	dn.dt_bias	49	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.49.linear_attn.in_proj_a.weight	dn.in_proj_a	49	HGRAVR02	1	128	40141	245760	1.3066731771	0.0000119396	0.8471987972	0.5345914634	130827
layers.49.linear_attn.in_proj_b.weight	dn.in_proj_b	49	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8431719613	0.5406997918	130827
layers.49.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	49	HGRAVR02	1	128	8438660	52428800	1.2876373291	0.0025100120	0.8407989083	0.5428392279	27853079
layers.49.linear_attn.in_proj_z.weight	dn.in_proj_z	49	HGRAVR02	1	128	5063335	31457280	1.2876726786	0.0015060486	0.8425624896	0.5399941113	16711957
layers.49.linear_attn.norm.weight	dn.norm	49	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.49.linear_attn.out_proj.weight	dn.out_proj	49	HGRAVR02	1	128	5063699	31457280	1.2877652486	0.0015061568	0.8418758518	0.5412473816	16711957
layers.49.mlp.down_proj.weight	mlp.down_proj	49	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1676696114	0.9864744174	1466362
layers.49.mlp.gate_proj.weight	mlp.gate_proj	49	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7925098954	0.6098590540	12534021
layers.49.mlp.up_proj.weight	mlp.up_proj	49	HGRAVR02	1	128	14344871	89128960	1.2875609454	0.0042667673	0.8412306705	0.5420463485	14344871
layers.49.input_layernorm.weight	norm.input	49	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.5983699277	0.9322083348	2977
layers.49.post_attention_layernorm.weight	norm.post_attn	49	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4326945959	0.9570861902	2977
layers.50.linear_attn.A_log	dn.A_log	50	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.50.linear_attn.conv1d.weight	dn.conv1d	50	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.50.linear_attn.dt_bias	dn.dt_bias	50	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.50.linear_attn.in_proj_a.weight	dn.in_proj_a	50	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8453424041	0.5370427469	130827
layers.50.linear_attn.in_proj_b.weight	dn.in_proj_b	50	HGRAVR02	1	128	40134	245760	1.3064453125	0.0000119375	0.8430472959	0.5412129141	130827
layers.50.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	50	HGRAVR02	1	128	8438819	52428800	1.2876615906	0.0025100593	0.8409609848	0.5426597042	27853079
layers.50.linear_attn.in_proj_z.weight	dn.in_proj_z	50	HGRAVR02	1	128	5063458	31457280	1.2877039591	0.0015060852	0.8422427722	0.5405543639	16711957
layers.50.linear_attn.norm.weight	dn.norm	50	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.50.linear_attn.out_proj.weight	dn.out_proj	50	HGRAVR02	1	128	5063865	31457280	1.2878074646	0.0015062062	0.8408612188	0.5429629996	16711957
layers.50.mlp.down_proj.weight	mlp.down_proj	50	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1633835796	0.9871240131	1466361
layers.50.mlp.gate_proj.weight	mlp.gate_proj	50	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7901697938	0.6128879971	12534021
layers.50.mlp.up_proj.weight	mlp.up_proj	50	HGRAVR02	1	128	14345417	89128960	1.2876099530	0.0042669297	0.8411478185	0.5422571586	14345417
layers.50.input_layernorm.weight	norm.input	50	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.5721401310	0.9477735912	2977
layers.50.post_attention_layernorm.weight	norm.post_attn	50	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.4653799213	0.9601942610	2977
layers.51.self_attn.k_norm.weight	gqa.k_norm	51	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9722379542	0.5962181201	387
layers.51.self_attn.k_proj.weight	gqa.k_proj	51	HGRAVR02	1	128	844521	5242880	1.2886367798	0.0002511960	0.8429636502	0.5403468833	2785554
layers.51.self_attn.o_proj.weight	gqa.o_proj	51	HGRAVR02	1	128	5063625	31457280	1.2877464294	0.0015061348	0.8407583791	0.5430716036	16711957
layers.51.self_attn.q_norm.weight	gqa.q_norm	51	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9830752547	0.5985894834	387
layers.51.self_attn.q_proj.weight	gqa.q_proj	51	HGRAVR02	1	128	10130316	62914560	1.2881362915	0.0030131816	0.8432690607	0.5389422923	33423639
layers.51.self_attn.v_proj.weight	gqa.v_proj	51	HGRAVR02	1	128	844411	5242880	1.2884689331	0.0002511633	0.8390200272	0.5467189253	2785554
layers.51.mlp.down_proj.weight	mlp.down_proj	51	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1599570626	0.9876497920	1466362
layers.51.mlp.gate_proj.weight	mlp.gate_proj	51	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7933824213	0.6087235281	12534021
layers.51.mlp.up_proj.weight	mlp.up_proj	51	HGRAVR02	1	128	14345279	89128960	1.2875975665	0.0042668887	0.8415744683	0.5415348801	14345279
layers.51.input_layernorm.weight	norm.input	51	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7738247213	0.9401152229	2977
layers.51.post_attention_layernorm.weight	norm.post_attn	51	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.6368222458	0.9315079083	2977
layers.52.linear_attn.A_log	dn.A_log	52	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.52.linear_attn.conv1d.weight	dn.conv1d	52	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.52.linear_attn.dt_bias	dn.dt_bias	52	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.52.linear_attn.in_proj_a.weight	dn.in_proj_a	52	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8420915555	0.5425127173	130827
layers.52.linear_attn.in_proj_b.weight	dn.in_proj_b	52	HGRAVR02	1	128	40128	245760	1.3062500000	0.0000119358	0.8409341706	0.5443296008	130827
layers.52.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	52	HGRAVR02	1	128	8438913	52428800	1.2876759338	0.0025100873	0.8420787076	0.5409413614	27853079
layers.52.linear_attn.in_proj_z.weight	dn.in_proj_z	52	HGRAVR02	1	128	5063405	31457280	1.2876904806	0.0015060694	0.8426885063	0.5398179779	16711957
layers.52.linear_attn.norm.weight	dn.norm	52	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.52.linear_attn.out_proj.weight	dn.out_proj	52	HGRAVR02	1	128	5063395	31457280	1.2876879374	0.0015060664	0.8404647334	0.5434325637	16711957
layers.52.mlp.down_proj.weight	mlp.down_proj	52	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1571068307	0.9881504228	1466364
layers.52.mlp.gate_proj.weight	mlp.gate_proj	52	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7943218632	0.6074971420	12534021
layers.52.mlp.up_proj.weight	mlp.up_proj	52	HGRAVR02	1	128	14344647	89128960	1.2875408397	0.0042667007	0.8414378557	0.5416791883	14344647
layers.52.input_layernorm.weight	norm.input	52	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9297097029	0.8724492245	2977
layers.52.post_attention_layernorm.weight	norm.post_attn	52	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7802403108	0.8968938267	2977
layers.53.linear_attn.A_log	dn.A_log	53	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.53.linear_attn.conv1d.weight	dn.conv1d	53	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.53.linear_attn.dt_bias	dn.dt_bias	53	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.53.linear_attn.in_proj_a.weight	dn.in_proj_a	53	HGRAVR02	1	128	40125	245760	1.3061523438	0.0000119349	0.8396401057	0.5461920703	130827
layers.53.linear_attn.in_proj_b.weight	dn.in_proj_b	53	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8411077806	0.5435809543	130827
layers.53.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	53	HGRAVR02	1	128	8438610	52428800	1.2876296997	0.0025099972	0.8421576433	0.5407014991	27853079
layers.53.linear_attn.in_proj_z.weight	dn.in_proj_z	53	HGRAVR02	1	128	5063447	31457280	1.2877011617	0.0015060819	0.8428634239	0.5395479759	16711957
layers.53.linear_attn.norm.weight	dn.norm	53	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.53.linear_attn.out_proj.weight	dn.out_proj	53	HGRAVR02	1	128	5063308	31457280	1.2876658122	0.0015060405	0.8405025607	0.5433205172	16711957
layers.53.mlp.down_proj.weight	mlp.down_proj	53	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1584391044	0.9879076169	1466363
layers.53.mlp.gate_proj.weight	mlp.gate_proj	53	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7951422295	0.6064229834	12534021
layers.53.mlp.up_proj.weight	mlp.up_proj	53	HGRAVR02	1	128	14344387	89128960	1.2875175027	0.0042666234	0.8415097202	0.5415468360	14344387
layers.53.input_layernorm.weight	norm.input	53	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9298360778	0.8740726639	2977
layers.53.post_attention_layernorm.weight	norm.post_attn	53	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8278611822	0.8840719531	2977
layers.54.linear_attn.A_log	dn.A_log	54	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.54.linear_attn.conv1d.weight	dn.conv1d	54	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.54.linear_attn.dt_bias	dn.dt_bias	54	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.54.linear_attn.in_proj_a.weight	dn.in_proj_a	54	HGRAVR02	1	128	40125	245760	1.3061523438	0.0000119349	0.8379999487	0.5493373069	130827
layers.54.linear_attn.in_proj_b.weight	dn.in_proj_b	54	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8425907334	0.5417737557	130827
layers.54.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	54	HGRAVR02	1	128	8438410	52428800	1.2875991821	0.0025099377	0.8418390798	0.5411433309	27853079
layers.54.linear_attn.in_proj_z.weight	dn.in_proj_z	54	HGRAVR02	1	128	5063250	31457280	1.2876510620	0.0015060233	0.8427350148	0.5396838762	16711957
layers.54.linear_attn.norm.weight	dn.norm	54	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.54.linear_attn.out_proj.weight	dn.out_proj	54	HGRAVR02	1	128	5063180	31457280	1.2876332601	0.0015060025	0.8405172895	0.5432464708	16711957
layers.54.mlp.down_proj.weight	mlp.down_proj	54	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1525058592	0.9889897986	1466361
layers.54.mlp.gate_proj.weight	mlp.gate_proj	54	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7950547215	0.6065377069	12534021
layers.54.mlp.up_proj.weight	mlp.up_proj	54	HGRAVR02	1	128	14344419	89128960	1.2875203750	0.0042666329	0.8415595546	0.5414853259	14344419
layers.54.input_layernorm.weight	norm.input	54	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9178515386	0.8847835090	2977
layers.54.post_attention_layernorm.weight	norm.post_attn	54	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9076103803	0.8478620658	2977
layers.55.self_attn.k_norm.weight	gqa.k_norm	55	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9469652493	0.6258133049	387
layers.55.self_attn.k_proj.weight	gqa.k_proj	55	HGRAVR02	1	128	844443	5242880	1.2885177612	0.0002511728	0.8406877705	0.5444007747	2785554
layers.55.self_attn.o_proj.weight	gqa.o_proj	55	HGRAVR02	1	128	5063997	31457280	1.2878410339	0.0015062455	0.8420962565	0.5410050770	16711957
layers.55.self_attn.q_norm.weight	gqa.q_norm	55	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9691692855	0.6227469737	387
layers.55.self_attn.q_proj.weight	gqa.q_proj	55	HGRAVR02	1	128	10128805	62914560	1.2879441579	0.0030127322	0.8432946831	0.5389285498	33423639
layers.55.self_attn.v_proj.weight	gqa.v_proj	55	HGRAVR02	1	128	844385	5242880	1.2884292603	0.0002511556	0.8389721106	0.5463544482	2785554
layers.55.mlp.down_proj.weight	mlp.down_proj	55	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1565934217	0.9882880937	1466362
layers.55.mlp.gate_proj.weight	mlp.gate_proj	55	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7970433422	0.6039221064	12534021
layers.55.mlp.up_proj.weight	mlp.up_proj	55	HGRAVR02	1	128	14344237	89128960	1.2875040391	0.0042665788	0.8419068777	0.5408911031	14344237
layers.55.input_layernorm.weight	norm.input	55	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7392304732	0.9200748775	2977
layers.55.post_attention_layernorm.weight	norm.post_attn	55	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.8992429429	0.8569531851	2977
layers.56.linear_attn.A_log	dn.A_log	56	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.56.linear_attn.conv1d.weight	dn.conv1d	56	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.56.linear_attn.dt_bias	dn.dt_bias	56	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.56.linear_attn.in_proj_a.weight	dn.in_proj_a	56	HGRAVR02	1	128	40126	245760	1.3061848958	0.0000119352	0.8388332600	0.5481943084	130827
layers.56.linear_attn.in_proj_b.weight	dn.in_proj_b	56	HGRAVR02	1	128	40124	245760	1.3061197917	0.0000119346	0.8410555360	0.5440885132	130827
layers.56.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	56	HGRAVR02	1	128	8438619	52428800	1.2876310730	0.0025099998	0.8423567282	0.5403934955	27853079
layers.56.linear_attn.in_proj_z.weight	dn.in_proj_z	56	HGRAVR02	1	128	5063503	31457280	1.2877154032	0.0015060985	0.8431315323	0.5391316444	16711957
layers.56.linear_attn.norm.weight	dn.norm	56	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.56.linear_attn.out_proj.weight	dn.out_proj	56	HGRAVR02	1	128	5063216	31457280	1.2876424154	0.0015060132	0.8404738238	0.5433330035	16711957
layers.56.mlp.down_proj.weight	mlp.down_proj	56	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1553919968	0.9885548232	1466362
layers.56.mlp.gate_proj.weight	mlp.gate_proj	56	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7976268429	0.6031512409	12534021
layers.56.mlp.up_proj.weight	mlp.up_proj	56	HGRAVR02	1	128	14344148	89128960	1.2874960507	0.0042665523	0.8419428508	0.5408236808	14344148
layers.56.input_layernorm.weight	norm.input	56	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9515285877	0.8703025124	2977
layers.56.post_attention_layernorm.weight	norm.post_attn	56	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9318015056	0.8246048574	2977
layers.57.linear_attn.A_log	dn.A_log	57	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.57.linear_attn.conv1d.weight	dn.conv1d	57	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.57.linear_attn.dt_bias	dn.dt_bias	57	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.57.linear_attn.in_proj_a.weight	dn.in_proj_a	57	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8369801940	0.5515377630	130827
layers.57.linear_attn.in_proj_b.weight	dn.in_proj_b	57	HGRAVR02	1	128	40120	245760	1.3059895833	0.0000119334	0.8402591765	0.5455285553	130827
layers.57.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	57	HGRAVR02	1	128	8438964	52428800	1.2876837158	0.0025101025	0.8421962221	0.5407491272	27853079
layers.57.linear_attn.in_proj_z.weight	dn.in_proj_z	57	HGRAVR02	1	128	5063344	31457280	1.2876749674	0.0015060512	0.8428948443	0.5394579312	16711957
layers.57.linear_attn.norm.weight	dn.norm	57	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.57.linear_attn.out_proj.weight	dn.out_proj	57	HGRAVR02	1	128	5063068	31457280	1.2876047770	0.0015059692	0.8405871392	0.5432878441	16711957
layers.57.mlp.down_proj.weight	mlp.down_proj	57	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1573713813	0.9882916248	1466363
layers.57.mlp.gate_proj.weight	mlp.gate_proj	57	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7978294414	0.6028832245	12534021
layers.57.mlp.up_proj.weight	mlp.up_proj	57	HGRAVR02	1	128	14344231	89128960	1.2875035005	0.0042665770	0.8419594701	0.5407929670	14344231
layers.57.input_layernorm.weight	norm.input	57	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9639564678	0.8614402685	2977
layers.57.post_attention_layernorm.weight	norm.post_attn	57	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9568493174	0.7842768648	2977
layers.58.linear_attn.A_log	dn.A_log	58	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.58.linear_attn.conv1d.weight	dn.conv1d	58	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.58.linear_attn.dt_bias	dn.dt_bias	58	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.58.linear_attn.in_proj_a.weight	dn.in_proj_a	58	HGRAVR02	1	128	40119	245760	1.3059570312	0.0000119331	0.8344098744	0.5557312985	130827
layers.58.linear_attn.in_proj_b.weight	dn.in_proj_b	58	HGRAVR02	1	128	40124	245760	1.3061197917	0.0000119346	0.8409825339	0.5438657375	130827
layers.58.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	58	HGRAVR02	1	128	8438635	52428800	1.2876335144	0.0025100046	0.8421762965	0.5406995196	27853079
layers.58.linear_attn.in_proj_z.weight	dn.in_proj_z	58	HGRAVR02	1	128	5063182	31457280	1.2876337687	0.0015060031	0.8427650979	0.5396361098	16711957
layers.58.linear_attn.norm.weight	dn.norm	58	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.58.linear_attn.out_proj.weight	dn.out_proj	58	HGRAVR02	1	128	5063162	31457280	1.2876286825	0.0015059971	0.8411257710	0.5422517601	16711957
layers.58.mlp.down_proj.weight	mlp.down_proj	58	HGRAVS01	3	64	1466361	89128960	0.1316170188	0.0004361574	0.1627368353	0.9875296615	1466361
layers.58.mlp.gate_proj.weight	mlp.gate_proj	58	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7975063011	0.6033106163	12534021
layers.58.mlp.up_proj.weight	mlp.up_proj	58	HGRAVR02	1	128	14344329	89128960	1.2875122968	0.0042666061	0.8418868851	0.5409188163	14344329
layers.58.input_layernorm.weight	norm.input	58	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9630474317	0.8608491741	2977
layers.58.post_attention_layernorm.weight	norm.post_attn	58	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9704568957	0.7510437563	2977
layers.59.self_attn.k_norm.weight	gqa.k_norm	59	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9306461170	0.6571929721	387
layers.59.self_attn.k_proj.weight	gqa.k_proj	59	HGRAVR02	1	128	844441	5242880	1.2885147095	0.0002511722	0.8410735321	0.5443626152	2785554
layers.59.self_attn.o_proj.weight	gqa.o_proj	59	HGRAVR02	1	128	5063642	31457280	1.2877507528	0.0015061399	0.8410725689	0.5424938984	16711957
layers.59.self_attn.q_norm.weight	gqa.q_norm	59	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9598692504	0.6516681319	387
layers.59.self_attn.q_proj.weight	gqa.q_proj	59	HGRAVR02	1	128	10129026	62914560	1.2879722595	0.0030127979	0.8434350138	0.5389395214	33423639
layers.59.self_attn.v_proj.weight	gqa.v_proj	59	HGRAVR02	1	128	844459	5242880	1.2885421753	0.0002511776	0.8402253649	0.5453041945	2785554
layers.59.mlp.down_proj.weight	mlp.down_proj	59	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1540708791	0.9889618089	1466362
layers.59.mlp.gate_proj.weight	mlp.gate_proj	59	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7973085067	0.6035719884	12534021
layers.59.mlp.up_proj.weight	mlp.up_proj	59	HGRAVR02	1	128	14344257	89128960	1.2875058342	0.0042665847	0.8419973055	0.5407556059	14344257
layers.59.input_layernorm.weight	norm.input	59	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7095226092	0.8765224496	2977
layers.59.post_attention_layernorm.weight	norm.post_attn	59	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9801146828	0.7551058564	2977
layers.60.linear_attn.A_log	dn.A_log	60	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.60.linear_attn.conv1d.weight	dn.conv1d	60	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.60.linear_attn.dt_bias	dn.dt_bias	60	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.60.linear_attn.in_proj_a.weight	dn.in_proj_a	60	HGRAVR02	1	128	40127	245760	1.3062174479	0.0000119355	0.8358389631	0.5535329492	130827
layers.60.linear_attn.in_proj_b.weight	dn.in_proj_b	60	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8432589602	0.5412130846	130827
layers.60.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	60	HGRAVR02	1	128	8440561	52428800	1.2879273987	0.0025105775	0.8432462341	0.5392807110	27853079
layers.60.linear_attn.in_proj_z.weight	dn.in_proj_z	60	HGRAVR02	1	128	5063491	31457280	1.2877123515	0.0015060950	0.8435495257	0.5385668704	16711957
layers.60.linear_attn.norm.weight	dn.norm	60	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.60.linear_attn.out_proj.weight	dn.out_proj	60	HGRAVR02	1	128	5063104	31457280	1.2876139323	0.0015059799	0.8402177887	0.5437675225	16711957
layers.60.mlp.down_proj.weight	mlp.down_proj	60	HGRAVS01	3	64	1466362	89128960	0.1316171085	0.0004361577	0.1599304921	0.9878762364	1466362
layers.60.mlp.gate_proj.weight	mlp.gate_proj	60	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7964301765	0.6047304969	12534021
layers.60.mlp.up_proj.weight	mlp.up_proj	60	HGRAVR02	1	128	14344370	89128960	1.2875159768	0.0042666183	0.8420829625	0.5406295420	14344370
layers.60.input_layernorm.weight	norm.input	60	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9862014714	0.8207350372	2977
layers.60.post_attention_layernorm.weight	norm.post_attn	60	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9869564483	0.7497470274	2977
layers.61.linear_attn.A_log	dn.A_log	61	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.61.linear_attn.conv1d.weight	dn.conv1d	61	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.61.linear_attn.dt_bias	dn.dt_bias	61	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.61.linear_attn.in_proj_a.weight	dn.in_proj_a	61	HGRAVR02	1	128	40129	245760	1.3062825521	0.0000119361	0.8363556699	0.5527847933	130827
layers.61.linear_attn.in_proj_b.weight	dn.in_proj_b	61	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8428951656	0.5415059479	130827
layers.61.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	61	HGRAVR02	1	128	8438362	52428800	1.2875918579	0.0025099234	0.8417233987	0.5413812570	27853079
layers.61.linear_attn.in_proj_z.weight	dn.in_proj_z	61	HGRAVR02	1	128	5063237	31457280	1.2876477559	0.0015060194	0.8428104120	0.5396530115	16711957
layers.61.linear_attn.norm.weight	dn.norm	61	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.61.linear_attn.out_proj.weight	dn.out_proj	61	HGRAVR02	1	128	5063207	31457280	1.2876401265	0.0015060105	0.8412940963	0.5420185095	16711957
layers.61.mlp.down_proj.weight	mlp.down_proj	61	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1653544831	0.9870312581	1466364
layers.61.mlp.gate_proj.weight	mlp.gate_proj	61	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7954749061	0.6059865294	12534021
layers.61.mlp.up_proj.weight	mlp.up_proj	61	HGRAVR02	1	128	14344565	89128960	1.2875334796	0.0042666763	0.8420828172	0.5406605859	14344565
layers.61.input_layernorm.weight	norm.input	61	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9707442859	0.8764566106	2977
layers.61.post_attention_layernorm.weight	norm.post_attn	61	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9917078104	0.7349368183	2977
layers.62.linear_attn.A_log	dn.A_log	62	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.62.linear_attn.conv1d.weight	dn.conv1d	62	f32v2	32		163848	40960	32.0015625000	0.0000487353	1.0000000000	0.0000000000	22026
layers.62.linear_attn.dt_bias	dn.dt_bias	62	f32v2	32		200	48	33.3333333333	0.0000000595	1.0000000000	0.0000000000	283
layers.62.linear_attn.in_proj_a.weight	dn.in_proj_a	62	HGRAVR02	1	128	40130	245760	1.3063151042	0.0000119363	0.8387766371	0.5485802143	130827
layers.62.linear_attn.in_proj_b.weight	dn.in_proj_b	62	HGRAVR02	1	128	40137	245760	1.3065429688	0.0000119384	0.8442973058	0.5396440606	130827
layers.62.linear_attn.in_proj_qkv.weight	dn.in_proj_qkv	62	HGRAVR02	1	128	8440153	52428800	1.2878651428	0.0025104561	0.8423069210	0.5408573807	27853079
layers.62.linear_attn.in_proj_z.weight	dn.in_proj_z	62	HGRAVR02	1	128	5063608	31457280	1.2877421061	0.0015061298	0.8430469525	0.5395470016	16711957
layers.62.linear_attn.norm.weight	dn.norm	62	f32v2	32		520	128	32.5000000000	0.0000001547	1.0000000000	0.0000000000	318
layers.62.linear_attn.out_proj.weight	dn.out_proj	62	HGRAVR02	1	128	5063306	31457280	1.2876653035	0.0015060399	0.8398246287	0.5446070838	16711957
layers.62.mlp.down_proj.weight	mlp.down_proj	62	HGRAVS01	3	64	1466364	89128960	0.1316172880	0.0004361583	0.1551809472	0.9888519253	1466364
layers.62.mlp.gate_proj.weight	mlp.gate_proj	62	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7948573849	0.6067962901	12534021
layers.62.mlp.up_proj.weight	mlp.up_proj	62	HGRAVR02	1	128	14345135	89128960	1.2875846414	0.0042668459	0.8424859119	0.5400977258	14345135
layers.62.input_layernorm.weight	norm.input	62	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9475881710	0.8754301563	2977
layers.62.post_attention_layernorm.weight	norm.post_attn	62	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9929231462	0.7625216259	2977
layers.63.self_attn.k_norm.weight	gqa.k_norm	63	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9577671749	0.6634847613	387
layers.63.self_attn.k_proj.weight	gqa.k_proj	63	HGRAVR02	1	128	844511	5242880	1.2886215210	0.0002511931	0.8413643613	0.5448134744	2785554
layers.63.self_attn.o_proj.weight	gqa.o_proj	63	HGRAVR02	1	128	5064135	31457280	1.2878761292	0.0015062865	0.8369211175	0.5495051352	16711957
layers.63.self_attn.q_norm.weight	gqa.q_norm	63	f32v2	32		1032	256	32.2500000000	0.0000003070	0.9765499960	0.6622285729	387
layers.63.self_attn.q_proj.weight	gqa.q_proj	63	HGRAVR02	1	128	10128712	62914560	1.2879323324	0.0030127045	0.8433244090	0.5392426180	33423639
layers.63.self_attn.v_proj.weight	gqa.v_proj	63	HGRAVR02	1	128	844652	5242880	1.2888366699	0.0002512350	0.8400961313	0.5466003496	2785554
layers.63.mlp.down_proj.weight	mlp.down_proj	63	HGRAVS01	3	64	1466363	89128960	0.1316171983	0.0004361580	0.1812010060	0.9849306832	1466363
layers.63.mlp.gate_proj.weight	mlp.gate_proj	63	HGRAVB01	1	128	12534021	89128960	1.1250234267	0.0037281445	0.7944383811	0.6073447610	12534021
layers.63.mlp.up_proj.weight	mlp.up_proj	63	HGRAVR02	1	128	14346727	89128960	1.2877275355	0.0042673194	0.8430639111	0.5392906418	14346727
layers.63.input_layernorm.weight	norm.input	63	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.7368489311	0.8810376323	2977
layers.63.post_attention_layernorm.weight	norm.post_attn	63	f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9837108194	0.8192623515	2977
embed_tokens.weight	embed		HQ30UQ4	4	64	675430440	1271398400	4.2500002517	0.2009013916	0.9942128523	0.1080996927	675430686
lm_head.weight	lm_head		HQ30UQ4	4	64	675430440	1271398400	4.2500002517	0.2009013916	0.9937578776	0.1123040753	675430686
norm.weight	norm.final		f32v2	32		20488	5120	32.0125000000	0.0000060940	0.9972375486	0.5130877870	2977
```

---

```
STATUS
SUPPORTED

CLAIMS
1. mixed-sub15-v1 packed complete BPW is 1.2910781930062503 = 8*4340604637/26895998464 over 851 language tensors. Evidence: §1; PACK_REPORT.json:23-25; this-process census.
2. Per-organ codec/width/group/scale/bytes/share are the table in §2; every tensor is appendix A. MLP is byte-identical to mixed-2p0-v1. Evidence: mlp_vs_mix 0/0/0; PACK_REPORT class bytes; mixed-2p0 catalog.
3. Cumulative spend: up 0.273 + gate 0.512 + embed 0.713 + lm_head 0.913 of 1.291. down is 0.028 BPW. Tables are 31% of the budget. Evidence: §2.
4. Per-layer phys BPW is flat (DN 0.9851, GQA 0.9728). No layer is the budget. Evidence: §3.
5. vs 2p0: REPEATS down 0.13161714918473189 exactly; INVERTS attention 4.250→1.288; keeps tables at 4.25. Not an inversion of the 2p0 policy. Evidence: §5; both PACK_REPORTs.
6. 2p0 INCOHERENT is not a proof that down cannot be starved. Evidence: g1-out-proj-forensics.md:71-75; g1-sub15-native-gap.md §8.5; standing rule on confounded failures.
7. Weight-space: down mean cosine 0.173122 rel-L2 0.985619; rice organs ~0.84; gate 0.7966; embed/lm_head 0.9942/0.9938. Evidence: §6; independent decode vs BF16.
8. Weight space is the wrong metric for out_proj; mixer_x is missing. Evidence: g1-out-proj-forensics.md:17-33; capture is 256×5120; §7.
9. PREDICTED first break: mlp.down_proj. Evidence: §6–7; 62/64 distribution_local_only; n_hold=n_fit=256; rows/dim 0.0147.

EVIDENCE
PACK_REPORT.json:12-25,26-68,77-81
mixed-2p0-v1/PACK_REPORT.json:10-38
/tmp/sub15_composition/{summary,tensors,errors}.json and this-process log
g1-sub15-native-gap.md:56-67,170-184
g1-out-proj-forensics.md:17-33,71-75
ascension_dual_gravity_worker.py:640,1279-1375
residual_compact_codec.py:219-241,522-608

CHANGES
workspace/superwave/g1/g1-sub15-composition.md (this file only)

TESTS
test -s workspace/superwave/g1/g1-sub15-composition.md
wc -l workspace/superwave/g1/g1-sub15-composition.md
git status --porcelain

RISKS
Peak RSS 17111 MB on the lm_head stream exceeded the 15 GB cap. Brief, end of job, no GPU, resident not touched. Down output numbers are fit-set (n_hold=n_fit). Isolated SwiGLU X was rebuilt from BF16 gate/up + captured hidden, not the original pack-time post-SwiGLU buffer; frobenius cosine ≠ header mean-row cosine. Do not read appendix A weight cosine for out_proj as output risk.

UNRESOLVED
Native coherence of packed 1.291. mixer_x still missing. Whether 0.1316 down is survivable under a confound-free generate. Holdout output error for down (needs a new capture, not a rerun on these 256 rows).

NEXT
Serialized GPU lane: C1+C2+C3 then one native greedy with HAWKING_QWEN38_RECON_FUSE=0. Parallel: mixer_x capture. Do not invent a new 2 BPW pack to avoid looking at this artifact.
```
