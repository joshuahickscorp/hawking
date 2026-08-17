# G1 — sparse exact islands on Qwen3.8 attention

STATUS: **FALSIFIED**

Hypothesis: attention tensors fail at low bit because of a small number of
extreme entries; keeping a tiny fraction exact restores Q4-class reconstruction
at almost no bit cost.

Measured on 10 real Qwen3.8 BF16 attention GEMVs (367,001,600 weights).
CPU only. No GPU, no generate, no packed artifact.

## Verdict

KILLS the mechanism as a G1 density path.

A tiny island (≤ 0.1 %) does not restore even the loose MLP organ bar.
Beating that bar needs ~1 % exact on a binary base (~1.367 BPW with rice
index + bf16 values) and still sits at mass-weighted weight cosine 0.830,
hold output cosine 0.894 — versus Q4 incumbent 0.994 / 0.997. Mid-depth
L32 `in_proj_qkv` hold stays 0.862 at 3 % exact, below the MLP hold bar.
Q4-class (weight cosine ≥ 0.993) is not reached at any measured point,
including Q3 + 3 % exact at 3.93 BPW (worse than plain Q4 at 4.25).

Energy is in the bulk, not the tail. Overlay ≈ refit. Outliers at the
fractions that matter are not concentrated in a few rows or channels, so
index cost does not collapse.

REOPEN_IF: a *different* static selector (not `|W|` and not `|W−Q(W)|`)
is shown to capture the bulk residual; or a base that already sits at
≥ 0.99 output cosine (Q3 is 0.983 hold, not enough) plus an island
fraction whose value+index cost still beats 4.25 BPW. Activation-dependent
per-token exact sets are a different mechanism (not a weight encoding).

## Method (measured)

Source (on disk, not re-derived):

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16
```

Identity: `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` BF16 shards, accepted by
`qwen38_accept_config` (`crates/hawking-core/src/model/qwen38_geometry.rs`).
Activations: `activation-capture-v1/hidden/L{00,03,32,63}.f32`, each
5,242,880 bytes = 256 × 5120 f32. Site is
`CAPTURED_REAL_BF16_POST_NORM_HIDDEN` (capture-result.json). Used only when
`W.shape[1] == 5120`. `out_proj` is 5120 × 6144 — weight-space only.

Tensors (10):

| layer | mixer | tensor | shape | elements |
|------:|-------|--------|------:|---------:|
| 0 | DeltaNet | `linear_attn.in_proj_qkv` | 10240×5120 | 52,428,800 |
| 0 | DeltaNet | `linear_attn.out_proj` | 5120×6144 | 31,457,280 |
| 32 | DeltaNet | `linear_attn.in_proj_qkv` | 10240×5120 | 52,428,800 |
| 32 | DeltaNet | `linear_attn.out_proj` | 5120×6144 | 31,457,280 |
| 3 | GQA | `self_attn.q_proj` | 12288×5120 | 62,914,560 |
| 3 | GQA | `self_attn.o_proj` | 5120×6144 | 31,457,280 |
| 3 | GQA | `self_attn.v_proj` | 1024×5120 | 5,242,880 |
| 63 | GQA | `self_attn.q_proj` | 12288×5120 | 62,914,560 |
| 63 | GQA | `self_attn.o_proj` | 5120×6144 | 31,457,280 |
| 63 | GQA | `self_attn.v_proj` | 1024×5120 | 5,242,880 |

Geometry constants: `QWEN38_IN_PROJ_QKV_ROWS = 10240`,
`QWEN38_Q_PROJ_ROWS = 12288`, `QWEN38_O_PROJ_ROWS = 5120`,
`QWEN38_O_PROJ_COLS = 6144`
(`crates/hawking-core/src/model/qwen38_geometry.rs:43,49,51-52`).

Bases:

- `none` — zeros + exact islands (control: “only extremes carry energy”).
- `binary_g128` — sign × group mean-abs, G=128. Nominal 1.125 BPW (HGRAVB01).
- `uniform_q2_g64` / `uniform_q3_g64` — same family as production Q4:
  per-group absmax, `scale = fp16(max_abs / qmax)`, `rint` clamp
  `[−2^(b−1), 2^(b−1)−1]`. Q4 itself is `max_abs/7`, clamp `[-8, 7]`
  (`crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:233,247`;
  `UNIFORM_Q4_GROUP_SIZE = 64` in `uniform_q4.rs:17`;
  `UNIFORM_Q4_NOMINAL_BPW = 4 + 16/64 = 4.25` at `qwen80_uniform_q4.rs:48`).

Modes: **overlay** (quantize all, restore selected); **refit** (exclude
selected from group scale/mean, then restore exact).

Selection: global top-k by `|W|`, and by `|W − Q0(W)|` (Q80 residual
recipe, `receipts/QWEN80_RESIDUAL_ENCODING.json` field `selection`).

Island values costed as bf16 (16 bits). Index encodings measured, not
estimated: dense bitmap; occupied-group bitmap G=64/128; Elias-γ of
sorted-position deltas; Rice of those deltas (k sweep); fixed `ceil(log2 n)`
per index; CSR group-local (`n_groups` × 8-bit count + `k × log2(G)`).

Also measured: fixed-slot-per-group (S ∈ {1,2,4,8}, G=64, local top-S);
row-islands (hottest-RMS output rows stored entirely bf16).

Output scores: `Y = X @ W.T` on the 256 captured rows; fit = 0:192,
hold = 192:256. Matches descent split.

Script: `/tmp/g1_sparse_exact_islands.py`. Raw: `/tmp/g1_sparse_exact_islands.json`
(1,865,374 bytes). Wall 617.0 s. Peak RSS 7375.2 MB.

### Q4 self-check (same family as production)

```
== language_model.model.layers.0.linear_attn.in_proj_qkv.weight ==
   shape=[10240, 5120] kurt=1.478 q4_cos=0.994143
== language_model.model.layers.0.linear_attn.out_proj.weight ==
   shape=[5120, 6144] kurt=149.358 q4_cos=0.993541
```

Probe receipt `receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json`:
L0 in_proj_qkv `weight_cosine` 0.9941425010601682, `excess_kurtosis` 1.47789;
L0 out_proj `weight_cosine` 0.993541491119444, `excess_kurtosis` 149.3577.
Match to 7–8 digits. Quantizer is the production one.

## Bars (cited, not re-derived)

MLP **working BPW** is 0.8480504639008466
(`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/PACK_REPORT.json`
field `mlp_physical_bpw`). Recipe
(`QWEN38_MIXED_2P0_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json` `representation`):

```
gate_proj  = binary_group_128                         1.125023 BPW
up_proj    = binary_plus_rice_q1_rms_residual_2pct    1.287511 BPW
down_proj  = hgravs01_r160_b3_activation_weighted     0.131617 BPW
```

Weight-space organ scores for the two *weight-reconstructed* organs
(`receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json` `summary.by_role_codec`):

| organ | codec | weight_cosine_mean | hold_mean | hold_min |
|-------|-------|-------------------:|----------:|---------:|
| gate_proj | binary_g128 | 0.79683982 | 0.86694807 | 0.83779336 |
| up_proj | rice_q1_rms_2pct | 0.84198590 | 0.87182637 | 0.81262945 |
| **equal-mass mean** | | **0.81941286** | **0.86938722** | |

`down_proj` at working BPW is HGRAVS01 (L@(R@x)), not a weight reconstruction.
No down weight-cosine is claimed here.

MLP bars used as the contract discriminator:

- **MLP-W**: weight cosine 0.8194
- **MLP-H**: hold output cosine 0.8694

Attention-relevant incumbent (this measurement, Q4-g64, 10 tensors):

- mass-weighted weight cosine **0.993639**
- mass-weighted hold output cosine **0.996709** (in-dim=5120 tensors only)
- probe bar for attention: output cosine ≥ 0.99
  (`QWEN_ATTENTION_DENSITY_VERDICT.json` `quality_bound.primary`)

Generation context, **not** a number this lane re-measured (no GPU):
mixed-2p0 (this MLP mix + attention still Q4) is INCOHERENT
(`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json`:
`2.0856_BPW_mixed-2p0-v1` = `INCOHERENT`; `4.2527_BPW_q4_oracle` = `COHERENT`).
Beating MLP-W/MLP-H is therefore a weak bar. Q4-class is the organ-relevant one.

## Curve — reconstruction vs fraction vs scheme BPW

Primary scheme: binary_g128 + `|W|` top-k + **refit** + Rice deltas + bf16 values.

Mass-weighted over the 10 tensors (367,001,600 elements):

| frac | weight_cosine | weight_rel_l2 | hold_out_cos | scheme BPW (rice) |
|-----:|--------------:|--------------:|-------------:|------------------:|
| 0 | 0.790511 | 0.612320 | 0.847040 | 1.1250 |
| 1e-4 | 0.796006 | 0.605123 | 0.862334 | 1.1281 |
| 1e-3 | 0.802789 | 0.596121 | 0.871833 | 1.1527 |
| 1e-2 | 0.830492 | 0.556942 | 0.893734 | 1.3669 |
| 3e-2 | 0.862503 | 0.505990 | 0.914771 | 1.8049 |

Q3-g64 + same islands (rice):

| frac | weight_cosine | hold_out_cos | scheme BPW |
|-----:|--------------:|-------------:|-----------:|
| 0 | 0.967032 | 0.983106 | 3.2500 |
| 3e-2 | 0.982573 | 0.990745 | 3.9299 |

Q4-g64, no islands: weight 0.993639, hold 0.996709, 4.250 BPW.

`none` (islands only, rest 0) at 3 %: weight cosine 0.47–0.55 on every
tensor. Bulk energy. Strong-form “only extremes matter” is dead.

### Per-tensor binary refit `|W|` weight cosine

```
tensor                                0      1e-5   3e-5   1e-4   3e-4   1e-3   3e-3   1e-2   3e-2
0.linear_attn.in_proj_qkv          0.7987 0.7993 0.7998 0.8010 0.8030 0.8069 0.8150 0.8332 0.8641
0.linear_attn.out_proj             0.7851 0.7973 0.8031 0.8066 0.8100 0.8153 0.8236 0.8416 0.8715
32.linear_attn.in_proj_qkv         0.7892 0.7896 0.7900 0.7908 0.7925 0.7966 0.8049 0.8239 0.8567
32.linear_attn.out_proj            0.7954 0.7966 0.7970 0.7978 0.7993 0.8032 0.8110 0.8294 0.8612
3.self_attn.q_proj                 0.7967 0.7970 0.7974 0.7982 0.7999 0.8040 0.8121 0.8305 0.8619
3.self_attn.o_proj                 0.7894 0.8020 0.8063 0.8085 0.8102 0.8138 0.8208 0.8373 0.8663
3.self_attn.v_proj                 0.7940 0.7944 0.7947 0.7955 0.7969 0.8007 0.8085 0.8268 0.8587
63.self_attn.q_proj                0.7905 0.7914 0.7921 0.7935 0.7960 0.8013 0.8104 0.8299 0.8621
63.self_attn.o_proj                0.7719 0.7732 0.7741 0.7758 0.7786 0.7849 0.7965 0.8209 0.8591
63.self_attn.v_proj                0.7653 0.7677 0.7698 0.7741 0.7810 0.7937 0.8101 0.8358 0.8714
```

High-kurtosis early `out_proj` (L0 kurt 149.36, L3 o kurt 132.14) moves
in the first 1e-5 (0.785→0.797, 0.789→0.802). Mid/late platykurtic
tensors barely move until ≥ 0.3 %. Even the kurtotic ones need 0.3–1 %
to clear MLP-W and never approach 0.993.

### Hold output cosine (in-dim 5120 only)

```
tensor                         0      1e-5   3e-5   1e-4   3e-4   1e-3   3e-3   1e-2   3e-2    q4
0.in_proj_qkv               0.8418 0.8460 0.8479 0.8508 0.8546 0.8595 0.8673 0.8815 0.9041  0.9961
32.in_proj_qkv              0.7411 0.7635 0.7702 0.7802 0.7865 0.7957 0.8082 0.8299 0.8620  0.9950
3.q_proj                    0.8478 0.8555 0.8588 0.8640 0.8690 0.8761 0.8848 0.8999 0.9214  0.9969
3.v_proj                    0.8416 0.8421 0.8438 0.8448 0.8466 0.8503 0.8571 0.8722 0.8976  0.9962
63.q_proj                   0.9309 0.9318 0.9323 0.9329 0.9342 0.9364 0.9403 0.9472 0.9579  0.9983
63.v_proj                   0.9486 0.9489 0.9492 0.9496 0.9503 0.9520 0.9552 0.9605 0.9679  0.9985
```

L32 `in_proj_qkv` hold at 3 % = 0.86202708 < MLP-H 0.8694. Worst organ
does not cross the asked bar at any measured fraction.

### Crossing table

| bar | who | fraction needed |
|-----|-----|-----------------|
| MLP-W 0.8194, binary refit | 8/10 tensors | 1e-2 |
| MLP-W 0.8194, binary refit | L0 out, L3 o (kurtotic) | 3e-3 |
| MLP-W 0.8194, Q2 refit | 9/10; L63 o | 1e-2; 3e-2 |
| MLP-W 0.8194, Q3 | all | 0 (Q3@0 is 0.957–0.969) |
| MLP-H 0.8694, binary refit | L0 in | 1e-2 |
| MLP-H 0.8694, binary refit | L3 q | 3e-4 |
| MLP-H 0.8694, binary refit | L3 v | 1e-2 |
| MLP-H 0.8694, binary refit | L63 q/v | 0 |
| MLP-H 0.8694, binary refit | **L32 in** | **> 3e-2 (0.862 at 3 %)** |
| Q4-W 0.993, any base ≤ Q3 | all 10 | **never** (best: Q3+3 % → 0.981–0.984) |
| attn hold 0.990, Q3+islands | L3 q at 3 %; L63 already at 0 | L0/L32/L3v still 0.986–0.989 at 3 % |

### Overlay vs refit vs residual

At 1 % binary, refit − overlay weight cosine = +0.0004 to +0.0042
(largest on L0/L3 `out_proj`). Residual selection vs `|W|` at 1 %:
+0.0002 to +0.0005. Neither axis is the mechanism.

## Index overhead (measured)

L0 `in_proj_qkv`, binary refit `|W|`. `k` = frac × 52,428,800.

| frac | k | val bf16 | dense bmp | occ bmp G64 | γ-delta | **Rice** | fixed log2n | CSR G64 | rice scheme BPW |
|-----:|--:|---------:|----------:|------------:|--------:|---------:|------------:|--------:|----------------:|
| 1e-4 | 5,243 | 0.00160 | 1.000 | 0.01882 | 0.00101 | 0.00152 | 0.00260 | 0.1256 | **1.128** |
| 1e-3 | 52,429 | 0.01600 | 1.000 | 0.03428 | 0.00761 | 0.01186 | 0.02600 | 0.1310 | **1.153** |
| 1e-2 | 524,288 | 0.16000 | 1.000 | 0.33401 | 0.09132 | **0.08243** | 0.26000 | 0.1850 | **1.367** |
| 3e-2 | 1,572,864 | 0.48000 | 1.000 | 0.69670 | 0.22953 | **0.20060** | 0.78000 | 0.3050 | **1.806** |

At 1 % Rice is cheapest index on every tensor (0.0814–0.0826 BPW). Dense
bitmap is a 1.0 BPW tax regardless of k. Occupied-group bitmap loses
because occupancy is high (L0 in: 260,822 / 819,200 groups occupied at
1 % = 31.8 %). Same ranking on all 10 tensors — structure does not flip
the encoding order.

Q80 residual receipt already reported this ranking on expert `up_proj`
at 2 % (`receipts/QWEN80_RESIDUAL_ENCODING.json` `findings.index_only_floor_bits_per_outlier_at_2pct`:
rice_fp16 23.23 bits/outlier, group_local 24.70, bitmap 63.51;
`bitmap_loses_at_these_densities: true`). Confirmed on Qwen3.8 attention.

Value bits dominate index once frac ≥ 1 % (0.16 vs 0.08). Storing the
island as bf16 at the 1 % that clears MLP-W costs more than the index.

## Structure — are outliers concentrated?

At **1 %** (the fraction that clears MLP-W on most tensors):

| tensor | kurt | row occ | hottest-1% rows share | rows to cover 90 % | cols to cover 90 % | row gini |
|--------|-----:|--------:|----------------------:|-------------------:|-------------------:|---------:|
| L0 in_proj_qkv | 1.48 | 0.949 | 0.248 | 5084 / 10240 | 4352 / 5120 | 0.605 |
| L0 out_proj | 149.36 | **1.000** | 0.124 | 3509 / 5120 | 5010 / 6144 | 0.558 |
| L32 in_proj_qkv | 0.49 | 0.999 | 0.047 | 7764 / 10240 | 4131 / 5120 | 0.291 |
| L3 o_proj | 132.14 | **1.000** | 0.064 | 4342 / 5120 | 4904 / 6144 | 0.200 |
| L63 o_proj | 2.60 | 1.000 | 0.051 | 4264 / 5120 | 2406 / 6144 | 0.175 |

L0 out_proj at 1 %, from `/tmp/g1_sparse_exact_islands.json`:

```
rows_occupied 5120/5120
cols_occupied 6143/6144
row_share_hottest_1pct 0.12387
n_rows_to_cover_90pct_outliers 3509
n_cols_to_cover_90pct_outliers 5010
```

Kurtosis 149 is real (a heavy tail exists) but the 1 % tail is *scattered
across essentially every row and column*. Index cannot collapse to a
channel list.

At **0.1 %** L0 in_proj *is* structured (hottest 1 % of rows hold 82.4 %
of that tail; 280 rows cover 90 %). That fraction does not clear MLP-W
(weight cosine 0.8069). The structure that would cheapen the index lives
below the quality floor.

## Fixed-slot-per-group and row-islands

Fixed S slots / group of 64, binary refit, local top-S by `|W|`. Always
pays `S × (log2(64) + 16) / 64` regardless of occupancy.

| S | frac_actual | L0 in wcos | L0 in hold | L32 in hold | scheme BPW |
|--:|------------:|-----------:|-----------:|------------:|-----------:|
| 1 | 1/64=0.01562 | 0.83297 | 0.8798 | 0.8324 | 1.4688 |
| 2 | 0.03125 | 0.85462 | 0.8982 | 0.8582 | 1.8125 |
| 4 | 0.06250 | 0.88423 | 0.9204 | 0.8881 | 2.5000 |
| 8 | 0.12500 | 0.92068 | 0.9465 | 0.9235 | 3.8750 |

S=1 is worse than global-top-1 % + rice (1.367 BPW, L0 in 0.833 / 0.882,
L32 hold 0.830). S=8 at 3.875 BPW is still below plain Q3 at 3.25
(0.969 / 0.979). Dominated.

Row-islands (hottest-RMS output rows stored entirely bf16, rest binary):

| row frac | L0 in wcos / hold | L32 in wcos / hold | scheme BPW |
|---------:|------------------:|-------------------:|-----------:|
| 0.01 | 0.8116 / 0.8586 | 0.7936 / 0.7597 | 1.284 |
| 0.03 | 0.8203 / 0.8673 | 0.8002 / 0.7762 | 1.605 |
| 0.10 | 0.8405 / 0.8813 | 0.8199 / 0.8058 | 2.725 |

Worse than sparse exact at the same value budget. Hottest-RMS rows are
not the residual.

## Complete-model projection (labeled)

Not measured. Arithmetic only, using mixed-2p0 ledger
(`source_weight_elements` 26,895,998,464;
`nonmlp_physical_bpw` 4.250142713483966;
attention GEMV mass ≈ 7.21e9 from geometry: 48×(10240+6144)×5120 +
48×5120×6144 + 16×(12288+1024+1024)×5120 + 16×5120×6144).

If every attention GEMV used binary+1 % rice at 1.367 BPW instead of Q4
4.25: complete BPW ≈ 2.086 − 7.21e9×(4.25−1.367)/26.896e9 ≈ **1.31**.
Quality would be attention at weight cosine ~0.83 / hold ~0.83–0.90.
The 1.29 BPW pack that rice-compressed attention already generated a
degenerate cycle
(`receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json`).
This projection is a quality-fail, not a G1 candidate.

Q3+3 % islands at 3.93 BPW on attention would *raise* complete BPW
versus staying at Q4 4.25, for a worse cosine (0.983 vs 0.994).

## Binding

A production path of “low-bpw + islands → expand to float/Q4 → generic
GEMV” is rejected on two grounds already in force: (1) quality never
reaches the Q4 incumbent this organ needs; (2) even if it did, the
preferred shape is a representation-specific kernel. This lane did not
write a kernel. There is nothing to implement.

Descent already has a strictly better 2.25 BPW cheap point without
islands: `ternary_t0.7_g128` attn_in hold_mean 0.9342
(`QWEN38_BPW_DESCENT.json`). Still fails the 0.99 attention bar
(`QWEN_ATTENTION_DENSITY_VERDICT.json`). Not re-tested here.

## Claim boundary

- Representation / reconstruction screen. Not a kernel. Not TOKEN_NS. Not TPS.
- Not a generation-coherence claim. Generate is forbidden in this lane.
- `out_proj` has no honest X at width 6144; weight-space only.
- In-proj X is post-norm hidden (schema name says post_swiglu; stored
  width is 5120). Same caveat as the attention-density probe.
- Island values costed as bf16. Q1 residual values (Q80 rice_q1_rms)
  would cut value BPW ~16× and cost a further cosine drop
  (`QWEN80_RESIDUAL_ENCODING.json` `value_cosine_delta_vs_fp16` ≈ −0.001
  on *expert up_proj* at 2 % — not re-measured on attention; would not
  close a 0.16 cosine gap).
- 10 tensors / 4 layers, not all 64. Layers chosen to include both
  mixers, early/mid/late, and both high- and low-kurtosis `out_proj`.
- Peak RSS 7.4 GB. No live Genesis interference.

## Evidence index

| claim | pointer |
|-------|---------|
| Q4 cosine match | script stdout; probe `QWEN_ATTENTION_DENSITY_PROBE.json` L0 in/out |
| MLP working 0.848 BPW | `mixed-2p0-v1/PACK_REPORT.json` `mlp_physical_bpw` |
| MLP organ cosines | `QWEN38_BPW_DESCENT.json` `summary.by_role_codec` |
| mixed-2p0 incoherent | `QWEN38_COHERENCE_FLOOR_BRACKETED.json` |
| sub-1.5 rice-attn incoherent | `QWEN38_SUB15_INCOHERENT.json` |
| attention 0.99 bar | `QWEN_ATTENTION_DENSITY_VERDICT.json` `quality_bound` |
| Q4 packer formula | `qwen80_uniform_q4.rs:233,247` |
| all curves + index + structure | `/tmp/g1_sparse_exact_islands.json` |
| runner | `/tmp/g1_sparse_exact_islands.py` ; stdout wall=617.0s rss=7375.2 |

---

```
STATUS
FALSIFIED

CLAIMS
1. Tiny exact islands (≤0.1%) on a low-bit attention base do not restore MLP-W 0.8194, let alone Q4 0.993. Evidence: mass-weighted binary refit at 1e-3 is weight_cosine 0.802789; per-tensor table in this file; /tmp/g1_sparse_exact_islands.json.
2. Worst-organ hold (L32 in_proj_qkv) stays 0.86202708 at 3% exact + binary, below MLP-H 0.8694. Evidence: this file hold table; JSON tensor 32.linear_attn.in_proj_qkv curves.
3. Q4-class weight cosine is never reached at frac≤3% on binary/q2/q3. Best: Q3+3% mass-weighted 0.982573 at 3.93 BPW vs Q4 0.993639 at 4.25. Evidence: this file Q3 table; JSON q4_g64 vs curves.
4. Exact-only (no base) at 3% yields weight cosine 0.47–0.55. Energy is in the bulk. Evidence: this file "none" table.
5. Overlay≈refit (Δwcos ≤0.0042 at 1%) and |W|≈residual (Δwcos ≤0.0005). Extremes are not stretching group scales in a usable way. Evidence: this file overlay/residual deltas.
6. At the 1% that clears MLP-W, outliers occupy ~95–100% of rows; 90% coverage needs thousands of rows/cols. Index does not collapse. Evidence: structure table; L0 out JSON structure block in this file.
7. Rice is the cheapest index (0.082 BPW at 1%); dense bitmap is 1.0; occupied bitmap 0.33–0.46. Scheme at 1% = 1.367 BPW. Evidence: index table; JSON scheme_bpw.delta_rice.
8. Fixed-slot and row-islands are dominated by global-top-k + rice. Evidence: fixed-slot and row-island tables.
9. Q4 self-check matches the attention-density probe to 7+ digits. Evidence: stdout q4_cos vs QWEN_ATTENTION_DENSITY_PROBE.json.

EVIDENCE
- /tmp/g1_sparse_exact_islands.py stdout (617.0 s, rss 7375.2 MB, 10 tensors, q4_cos listed)
- /tmp/g1_sparse_exact_islands.json (1,865,374 bytes)
- receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json summary.by_role_codec
- receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json / QWEN_ATTENTION_DENSITY_VERDICT.json
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json
- receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json
- receipts/QWEN80_RESIDUAL_ENCODING.json findings
- mixed-2p0-v1/PACK_REPORT.json mlp_physical_bpw=0.8480504639008466
- crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:233,247
- crates/hawking-core/src/model/qwen38_geometry.rs:43,49,51-52
- BF16 source + activation-capture-v1 on disk as cited

CHANGES
workspace/superwave/g1/g1-sparse-exact-islands.md (new, this file)

TESTS
test -s workspace/superwave/g1/g1-sparse-exact-islands.md
wc -l workspace/superwave/g1/g1-sparse-exact-islands.md
git status --porcelain

RISKS
- 10/all attention GEMVs; a pathological unsampled layer is possible but L0/L3/L32/L63 already span the kurtosis range the probe documented.
- hold X is post-norm hidden, not confirmed input_layernorm residual (same as the density probe). Wrong residual point, real distribution.
- Generation forbidden; quality bars are reconstruction, not tokens.
- bf16 island values; cheaper residual quantization was not re-fit on attention (would not close a 0.16 cosine gap).

UNRESOLVED
- Exact fraction above 3% where Q3+islands would match Q4 cosine. Projected: value cost 16*f plus ~0.2 index already exceeds the 1.0 BPW Q3-to-Q4 gap before cosine catches up (3% still 0.011 short of 0.993 at +0.68 BPW). Not worth measuring.
- Ternary + islands not run. Descent ternary attn_in hold_mean 0.9342 at 2.25 BPW already beats binary+3% islands and still fails 0.99.
- down_proj HGRAVS01 output cosine at working 0.132 BPW was not in the descent codec catalog; MLP-H uses gate+up only.

NEXT
Do not implement sparse-exact-islands on attention. The density gap remains "attention needs a new codec family or a higher cheap floor (Q3 3.25), not a sparse exception list." Steal Q80 residual science only as a warning: 2% rice residual was an expert-up recipe at the 0.86 bar, not an attention recipe at the 0.99 bar.
```
