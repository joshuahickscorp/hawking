# G1 nonuniform levels — scale rule vs group size at fixed bit width

Date: 2026-08-17
Lane: 08-nonuniform-levels
Base: `2eee9a004` on this worktree
Host work: CPU only. No Metal, no token loop, no pack written.

## Verdict

**MEASURED_WIN on reconstruction. IMPLEMENT_READY as a pack/kernel change.**

Holding codes at 4-bit (G0's width), on every language GEMV of the real Qwen3.8-27B BF16 source:

- **Attention best rule: `learned_shared`** (one nonuniform 16-level set per tensor + one FP16 scale per group).
- **MLP best rule: `learned_shared`** (same).
- At **equal or better** error than G0 absmax / group-64, the switch to learned_shared at group-256 buys **0.1875 complete BPW** (4.2500 → 4.0625) on both classes.
- Drop-in same-kernel runner-up: **`mse_scale`**. Same 0.1875 BPW buy at G=256. Zero Metal change.

Lloyd-Max per-group codebooks win raw error and **lose the complete-BPW curve**. Storing 16 FP16 levels per group is 16× the G0 scale metadata. **KILLS as a BPW-reduction mechanism.** `REOPEN_IF` a decode path can materialise per-group levels without shipping them.

This is a **weight-reconstruction** result plus **captured-activation matvec cosine** on post-norm hiddens. It is **not** a token-level TPS / TOKEN_NS claim.

## 1. G0 identity (independently measured)

G0 language pack on disk:

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json
```

Fields read from that file:

```
schema                  hawking.ascent.qwen38_language_uniform_q4.v1
complete_physical_bpw   4.252735126866492
nominal_codec_bpw       4.25
q4_group_size           64
source_weight_elements  26895998464
tensor_payload_bytes    14297694680
q4_tensors              402
f32_tensors             353
implied 8*bytes/elems   4.252735126866492
```

Class split from the same manifest (8 * bytes / elements):

| class | elements | payload bytes | complete BPW | frac of source elems |
|---|---:|---:|---:|---:|
| mlp | 17112760320 | 9091161600 | 4.250004 | 0.636257 |
| attn | 7239780864 | 3853029248 | 4.257620 | 0.269177 |
| embed_head | 2542796800 | 1350860880 | 4.250000 | 0.094542 |
| other (f32 vectors) | 660480 | 2642952 | 32.012500 | 0.000025 |

`complete_physical_bpw` is `8 * sum(tensor.bytes) / sum(tensor.elements)` (`crates/hawking-core/src/model/qwen38_pack.rs` 673–678). Nominal 4.25 is `4 + 16/64` (`qwen80_uniform_q4.rs:48`, asserted at line 1577). The extra 0.0027 is the f32 small-vector tax, not a different codec.

G0 scale rule is **absmax / 7, FP16 scale, q = rint(w/scale) clipped to [-8, 7]**:

```
233:        let scale = f16::from_f32(max_abs / 7.0);
246:                rint_ties_even(value / reconstructed_scale)
247:                    .clamp(-8.0, 7.0) as i32
```

`crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs`.

Device consume is `float(q) * float(scale)` with `q = nibble - 8`:

```
9://     [-8, 7]; and
10://   * every group has one IEEE FP16 scale, reconstructed as `float(q) * scale`.
```

`crates/hawking-core/shaders/qwen_uniform_q4.metal` lines 9–11.

Manifest `min_q4_cosine = 1.0` is **not a measured reconstruction**. Fold starts at 1.0 over `row.cosine` (`qwen38_pack.rs` 680–684); catalog q4 rows store `cosine: None`. Actual G0 error on `layers.0.mlp.gate_proj` is relative_l2 **0.10873951632127579**, cosine **0.9941447925762601** (decode of the live HQ30UQ4 vs BF16 source). See §7.

The campaign G0 figure `complete BPW ~ 4.2527` is therefore **measured** (this file, this manifest). The companion G0 TPS / TOKEN_NS numbers were **not** re-measured here (forbidden: GPU lane owns timing).

## 2. What was run

Source: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` (census path `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json` `download.path`, 11 BF16 shards, dtype `BF16`).

Population: every language GEMV used in decode.

| class | subclass | n tensors | elements |
|---|---|---:|---:|
| mlp | gate_proj, up_proj, down_proj | 64+64+64 = 192 | 17112760320 |
| attn | DN in_proj_qkv / in_proj_z / out_proj | 48+48+48 = 144 | 5536481280 |
| attn | GQA q / k / v / o | 16+16+16+16 = 64 | 1677721600 |
| **total** | | **400** | **24326963200** |

MLP element count matches the G0 manifest **exactly** (delta 0). Attn GEMV count is 25,577,984 below the G0 attn total: that remainder is `in_proj_a`/`in_proj_b` (48×48×5120 × 2 = 23,592,960) plus `conv1d`. Those are not GEMV-dominant; they stay f32 in the G0 pack. Layers 0..63 all present.

Sweep (CPU, two workers, 4541 s wall, `/tmp/g1-nonuniform-levels/`):

- Bit width held at **4** on all 400 tensors, group sizes **32, 64, 128, 256**.
- Extra bit widths **3 and 5** at G=64 on curve layers `{0,3,15,16,31,32,47,48,63}` (59 tensors: 32 attn + 27 mlp).
- Rules:
  1. `absmax` — G0 formula, generalised: `scale = f16(amax / qmax)`, `qmax = 2^(b-1)-1`, `qmin = -2^(b-1)`.
  2. `percentile` — declared p=99.9. Also recorded p ∈ {95, 98, 99, 99.5, 99.99} at G=64.
  3. `mse_scale` — per-group `s = α · amax / qmax`, α ∈ {0.65, 0.75, 0.82, 0.88, 0.92, 0.96, 1.00}, pick min MSE. α is the only free parameter.
  4. `lloyd_max` — 2 iterations of 1-D Lloyd-Max per group, min/max init, levels stored as FP16. Metadata = `16 · 2^b` bits/group. On tensors larger than 16384 groups this is a uniform random group sample; six full-tensor witnesses bound the sample bias (max |Δ relative_l2| = 3.7e-4, typically 1e-5).
  5. `learned_shared` — k-means (10 iters, 5e5 sample) on values normalised by per-group absmax; one shared 2^b-level codebook per tensor stored as FP16; one FP16 absmax scale per group.

Complete BPW (this lane's definition, scale metadata included, no container magic):

```
groups = ceil(N / G)
absmax / percentile / mse_scale :  (groups * (bits*G + 16)) / N
learned_shared                  :  (groups * (bits*G + 16) + 16*2^bits) / N
lloyd_max                       :  (groups * (bits*G + 16*2^bits)) / N
```

At N ≫ 2^b the learned extra is ~0. For a 89,128,960-element gate, learned G=64 is 4.25000287 vs absmax 4.25000000.

Error: `relative_l2 = ||w − ŵ||_2 / ||w||_2` (same as `UniformQ4PackQuality.relative_l2` in `qwen80_uniform_q4.rs` 312–321). Class numbers are **element-weighted** means. Cosine of `y = X ŵᵀ` vs `X Wᵀ` is also recorded when `W.shape[1]==5120`, using holdout tokens 192:256 of the real captured post-norm hidden `activation-capture-v1/hidden/LXX.f32` (256×5120 f32, `capture-result.json` status `CAPTURED_REAL_BF16_POST_NORM_HIDDEN`). That is a captured-activation matvec, not a generated token.

Absmax 4 / G=64 on `layers.0.mlp.gate_proj` is **bit-identical** to the production HQ30UQ4 decode (`max_abs` mine vs packed = 0). See §7.

## 3. Mechanism: absmax spends the grid on the outlier

Per-tensor |w| census (2e6-sample quantiles, `/tmp/g1-nonuniform-levels/stats.w*.jsonl`, 400 tensors):

| class | n | mean amax/p99 | mean amax/p999 | mean amax/p9999 | mean frac > 8·median |
|---|---:|---:|---:|---:|---:|
| attn | 208 | 12.16 | 8.85 | 6.06 | 1.66e-4 |
| mlp | 192 | 11.48 | 8.73 | 6.81 | 3.04e-5 |

Worst MLP: `layers.2.mlp.down_proj.weight` amax/p99 = **41.75** (`stats.w0.jsonl` line 40: amax=1.1875, p99=0.02844, p50=0.00735). Worst attn: `layers.1.linear_attn.out_proj.weight` amax/p99 = **35.37**.

G0 sets `scale = amax/7`. The bulk of the mass lives at ~p50 ≈ 0.007, i.e. about 0.16 of one Q4 step if the scale is set by a 1.19 outlier. That is the waste.

MSE-optimal α on the same tensors is **0.878** (attn) / **0.879** (mlp), element-weighted, 4-bit G=64. Only a minority of groups keep α=1. Gate-0 histogram (1,392,640 groups):

```
α=0.75:  73351
α=0.82: 403614
α=0.88: 417638
α=0.92: 248478
α=0.96: 175536
α=1.00:  74023   ← 5.3% stay at absmax
```

`/tmp/g1-nonuniform-levels/probe_levels.json`.

Learned shared levels on that tensor, in units of per-group absmax, are **symmetric and denser at 0** (Laplacian/Gaussian optimal-quantizer shape). Uniform absmax places levels at k/7 ∈ {±1, ±0.857, …}:

```
learned: [-0.917, -0.673, -0.502, -0.372, -0.268, -0.182, -0.106, -0.036,
          +0.033, +0.103, +0.179, +0.266, +0.371, +0.501, +0.672, +0.916]
spacing: 0.243, 0.171, 0.130, 0.104, 0.086, 0.076, 0.071, 0.068,
         0.070, 0.076, 0.087, 0.104, 0.130, 0.171, 0.244
```

Center spacing 0.068 vs tail 0.243. Qwen3 GQA `q_proj` layer 3 is the same shape to two digits. The nonuniformity is a tensor-class fact, not a layer-0 accident.

p=99.9 on G≤256 is almost amax (the 99.9-quantile of 64 samples is the top order statistic). That is why declared `percentile` barely moves the number. p=99.5 clips ~1 value per G=64 group and helps; it still loses to mse_scale.

## 4. Reconstruction vs complete BPW, 4-bit, all 400 tensors

Element-weighted relative_l2. Source: `/tmp/g1-nonuniform-levels/report.json` ← `analyze.py` over 12,071 rows.

### Attention (208 tensors, 7,214,202,880 weights)

| G | complete BPW absmax/mse/learned | absmax | percentile p99.9 | p99.5 | mse_scale | learned_shared | lloyd_max (BPW) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4.5000 | 0.099556 | 0.099485 | 0.097809 | 0.089334 | **0.085905** | 0.057850 (12.00) |
| 64 | 4.2500 | 0.111463 | 0.110398 | 0.107376 | 0.098898 | **0.094641** | 0.076824 (8.00) |
| 128 | 4.1250 | 0.122811 | 0.121544 | 0.114798 | 0.106289 | **0.101174** | 0.092844 (6.00) |
| 256 | 4.0625 | 0.134044 | 0.130748 | 0.119638 | 0.111433 | **0.105753** | 0.106265 (5.00) |

G0 point is the absmax / G=64 cell: **0.111463**.

Act-cosine (K=5120 tensors, holdout 64 tokens), same grid, learned_shared at G=64 = **0.997335** vs absmax **0.996387**.

### MLP (192 tensors, 17,112,760,320 weights — the entire G0 MLP mass)

| G | complete BPW absmax/mse/learned | absmax | percentile p99.9 | p99.5 | mse_scale | learned_shared | lloyd_max (BPW) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4.5000 | 0.098450 | 0.098137 | 0.096763 | 0.088308 | **0.085373** | 0.057363 (12.00) |
| 64 | 4.2500 | 0.109778 | 0.108779 | 0.105901 | 0.097397 | **0.093779** | 0.076035 (8.00) |
| 128 | 4.1250 | 0.120383 | 0.118819 | 0.112756 | 0.104252 | **0.099962** | 0.091591 (6.00) |
| 256 | 4.0625 | 0.130494 | 0.127174 | 0.116926 | 0.108739 | **0.104124** | 0.104343 (5.00) |

G0 point: absmax / G=64 = **0.109778**.

Act-cosine (gate + up only): learned_shared G=64 = **0.996832** vs absmax **0.995387**.

Every subclass at 4-bit G=64 ranks the same: learned_shared < mse_scale < p99.5 < p99.9 < absmax. Including GQA k/v (the noisiest, absmax rel_l2 0.123 / 0.186) and DN out_proj.

## 5. How the optimum moves with group size

Winner identity at 4-bit, equal metadata (exclude lloyd):

| G | attn winner | mlp winner | gap learned−mse (attn) | gap learned−mse (mlp) | absmax degradation vs G=32 |
|---:|---|---|---:|---:|---:|
| 32 | learned_shared | learned_shared | 0.00343 | 0.00294 | — |
| 64 | learned_shared | learned_shared | 0.00426 | 0.00362 | +12% / +12% |
| 128 | learned_shared | learned_shared | 0.00511 | 0.00429 | +23% / +22% |
| 256 | learned_shared | learned_shared | 0.00568 | 0.00462 | +35% / +33% |

The **name** of the winner does not move. Two other things do:

1. **Absmax decays fastest as G grows.** One outlier still sets the scale, and a larger group is more likely to contain a worse one. mse_scale's mean α falls with G (k_proj probe: 0.908 at G=32 → 0.771 at G=256) — it is doing more clipping as groups get dirtier. learned_shared's advantage over mse **grows** with G because intra-group shape is no longer uniform and a single scale cannot fix that.
2. **At G=256, learned_shared (4.0625 BPW) matches per-group Lloyd-Max (5.0 BPW) to three digits** (attn 0.1058 vs 0.1063; mlp 0.1041 vs 0.1043). The shared codebook has absorbed the nonuniformity; paying 16 FP16s per group buys nothing.

Largest G whose error is still ≤ G0 absmax G=64 (`report.json` `g_migration`):

| rule | attn largest G | attn rel_l2 | attn BPW | mlp largest G | mlp rel_l2 | mlp BPW |
|---|---:|---:|---:|---:|---:|---:|
| absmax | 64 | 0.111463 | 4.2500 | 64 | 0.109778 | 4.2500 |
| percentile p99.9 | 64 | 0.110398 | 4.2500 | 64 | 0.108779 | 4.2500 |
| percentile p99.5 | 64 | 0.107376 | 4.2500 | 64 | 0.105901 | 4.2500 |
| mse_scale | **256** | 0.111433 | **4.0625** | **256** | 0.108739 | **4.0625** |
| learned_shared | **256** | 0.105753 | **4.0625** | **256** | 0.104124 | **4.0625** |
| lloyd_max | 256 | 0.106265 | 5.0000 | 256 | 0.104343 | 5.0000 |

mse_scale at G=256 is **just** under the G0 absmax error (attn 0.111433 vs 0.111463; mlp 0.108739 vs 0.109778). learned_shared at G=256 is **comfortably** under, with margin to grow G further if someone measures G=512.

## 6. Bit-width curve (curve-layer subset, G=64)

59 tensors (32 attn + 27 mlp), layers 0,3,15,16,31,32,47,48,63. Not the full 400 — label **curve-subset**.

| bits | BPW (uniform/learned) | attn absmax | attn mse | attn learned | mlp absmax | mlp mse | mlp learned |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3.250 | 0.2612 | 0.1993 | **0.1824** | 0.2567 | 0.1960 | **0.1803** |
| 4 | 4.250 | 0.1115 | 0.0989 | **0.0946** | 0.1098 | 0.0974 | **0.0938** |
| 5 | 5.250 | 0.0523 | **0.0494** | 0.0516 | 0.0514 | **0.0485** | 0.0510 |

At 3-bit the shared codebook is worth a lot (learned beats absmax by ~0.08 rel_l2). At 5-bit a uniform grid plus MSE scale is enough and **mse_scale overtakes learned** on this subset (k-means on a 5e5 sample is a slightly weaker 32-level fit than a well-scaled uniform grid). The 4-bit operating point — G0's width — is still learned_shared.

3-bit learned (~0.18) does **not** reach 4-bit absmax (~0.11). The switch does not buy a whole code bit. It buys a **group-size bit of metadata**.

Lloyd-Max at 3-bit is 5.00 BPW (8 FP16 levels / 64) with rel_l2 0.170, worse BPW than 4-bit absmax and worse error than 4-bit learned. Dead.

## 7. Bits bought vs absmax at equal error

G0 reference error E0 = absmax, 4-bit, G=64:

- attn E0 = 0.11146285148748579 at 4.2500 BPW
- mlp  E0 = 0.10977780581333595 at 4.2500 BPW

**Direct (no interpolation), equal-or-better error:**

Both classes: `learned_shared` at G=256 has lower error than E0 at **4.0625 BPW**.

```
Δ complete BPW = 4.2500 − 4.0625 = 0.1875
```

Same number for `mse_scale` at G=256 (error ≤ E0 on both classes, barely on attn).

**Staying at G=64, converting the error drop into an absmax-equivalent BPW** (linear interpolation of the absmax (G, bits) curve to the learned G=64 error):

- attn: absmax would need **4.578** BPW to match learned_shared's 0.09464. Buy = **0.328 BPW** of absmax quality.
- mlp:  absmax would need **4.574** BPW to match 0.09378. Buy = **0.324 BPW**.

G=32 absmax (4.50 BPW) is still worse than learned G=64 (attn 0.0996 > 0.0946), so the interpolant uses the 5-bit G=64 absmax point (5.25 BPW, curve-subset). Treat 0.32 as **estimated** from a mixed full-census / curve-subset absmax curve. Prefer the **measured 0.1875** at the G=256 operating point.

**Projected language-complete BPW** if the 400 GEMVs move to 4.0625 and embed_head / tiny vectors stay at the G0 bytes (label **projected**, not measured):

```
Δelems = 24326963200
Δbpw_on_those = 0.1875
model Δ = 0.1875 * 24326963200 / 26895998464 = 0.16958
projected complete = 4.252735 − 0.16958 = 4.0832
```

Does not touch embed (9.45% of mass) or the 353 f32 vectors. Does not claim 1.5 BPW. It is a **0.17 complete-BPW** cut on the G0 artifact at **better** reconstruction than G0, with a kernel that is still 4-bit grouped.

## 8. Single best rule

**Attention: `learned_shared`.** Wins every G at 4-bit on the full 208-tensor census, wins 3-bit on the curve subset, loses only the 5-bit curve-subset cell to mse_scale. Buys 0.1875 complete BPW at equal-or-better error vs G0.

**MLP: `learned_shared`.** Same ranking, same 0.1875 BPW buy, on the entire 17.11e9-weight MLP mass.

**Drop-in if the kernel cannot change: `mse_scale`.** Same HQ30UQ4 layout, same `q = nibble-8`, same Metal. Only the packer's `amax/7` becomes `α*amax/7` with α chosen per group. Same 0.1875 BPW buy, ~0.004 worse rel_l2 than learned at G=256.

Percentile is not a competitor at these group sizes. Lloyd-Max is a competitor on error and a loser on complete BPW.

## 9. Production path (binding)

Preferred shape is low-BPW consumed by a representation-specific kernel. Two paths, neither expands to float/Q4 then generic GEMV:

1. **mse_scale** — write the same HQ30UQ4. Change one line in `pack_uniform_q4_group64` (`scale = f16(α * amax / 7)`). Existing `qwen_uniform_q4_*` kernels are bit-compatible. **IMPLEMENT_READY, zero new kernel.**
2. **learned_shared** — codes become 4-bit indices into a 16-entry FP16 LUT stored once per tensor; per-group FP16 scale stays. Kernel becomes `float(levels[nibble]) * float(scale)` instead of `float(int(nibble)-8) * float(scale)`. Same DRAM traffic as G0 plus 32 bytes/tensor. **IMPLEMENT_READY, one kernel variant.**

Lloyd-Max per-group LUT is 32 bytes of levels + 32 bytes of codes per group vs G0's 2+32. That is a 5.0 BPW body at G=256. Do not ship it.

This lane did **not** run the new kernel. Token-level proof belongs to the GPU lane.

## 10. What this is not

- Not a TOKEN_NS or TPS number.
- Not a 1.5 BPW codec. 4.06 is still a 4-bit body.
- Not an activation-aware / GPTQ / AWQ fit. Scales and levels are weight-only. Act-cosine on 64 holdout tokens moved in the same direction as weight relative_l2; that is corroboration, not a substitute for a calibration set.
- Embed and lm_head (9.45% of mass) were not swept (5 GB f32 each; not attn/mlp).
- Lloyd on large tensors is a 16384-group sample. Witness |bias| ≤ 3.7e-4 relative_l2. Class ranking is insensitive to that.

## 11. Evidence

### 11.1 Absmax matches production HQ30UQ4

Command: decode `uniform-q4-v1/tensors/c48d8d8932a40c589295c4c5cc9a3803d95ad321f0cf3dfd7b1536d7104db100.hq30uq4` (sha256 of `language_model.model.layers.0.mlp.gate_proj.weight`) and compare to this lane's absmax 4 / G=64. Output `/tmp/g1-nonuniform-levels/validate_pack.json`:

```
mine_vs_packed.max_abs     0.0
mine_vs_packed.relative_l2 0.0
mine_vs_packed.cosine      1.0
packed_vs_source.relative_l2 0.10873951632127579
packed_vs_source.cosine      0.9941447925762601
mine_vs_source  (identical to packed_vs_source)
```

Same numbers land in the sweep row for that tensor (`rows.w0.jsonl` line 132, rule=absmax, G=64, bits=4).

### 11.2 Worker completion

```
/tmp/g1-nonuniform-levels/w0.log
DONE {"seconds": 4541.142598152161, "n_target": 400, "n_chunk": 200, "n_rows_written_this_run": 6054, "worker": 0, ...}

/tmp/g1-nonuniform-levels/w1.log
DONE {"seconds": 4540.428817033768, "n_target": 400, "n_chunk": 200, "n_rows_written_this_run": 6017, "worker": 1, ...}
```

`wc -l` on the JSONL: 6054 + 6017 = 12071 rows; 200 + 200 = 400 tensor-stat lines. 400/400 GEMVs.

### 11.3 One-tensor raw rows (gate-0, 4-bit, G=64)

`/tmp/g1-nonuniform-levels/rows.w0.jsonl`:

```
LINE 132 absmax          relative_l2=0.10873951632127579 cosine=0.9941447925762601 complete_bpw=4.25     act_cosine=0.9966467936497879
LINE 139 mse_scale       relative_l2=0.09649131440078978 cosine=0.9953339143283948 complete_bpw=4.25     act_cosine=0.997283533834066  alpha_mean=0.8793625831604004
LINE 142 learned_shared  relative_l2=0.09297112799416472 cosine=0.9956894348638639 complete_bpw=4.25000287 act_cosine=0.9975784431608078
```

### 11.4 Lloyd sample vs full (4-bit G=64 witnesses)

| tensor | full rel_l2 | sample rel_l2 | Δ |
|---|---:|---:|---:|
| L0 in_proj_qkv | 7.547149e-2 | 7.547331e-2 | +1.8e-6 |
| L0 gate_proj | 7.545963e-2 | 7.547876e-2 | +1.9e-5 |
| L0 down_proj | 7.599535e-2 | 7.562561e-2 | −3.7e-4 |
| L16 up_proj | 7.546318e-2 | 7.550657e-2 | +4.3e-5 |
| L3 q_proj | 7.624039e-2 | 7.628545e-2 | +4.5e-5 |
| L63 o_proj | 8.297516e-2 | 8.307552e-2 | +1.0e-4 |

### 11.5 Artifact paths

| what | path |
|---|---|
| BF16 source | `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` |
| G0 pack | `.../qwen38-27b/uniform-q4-v1/manifest.json` |
| activation capture | `.../qwen38-27b/activation-capture-v1/{capture-result.json,hidden/LXX.f32}` |
| sweep rows | `/tmp/g1-nonuniform-levels/rows.w{0,1}.jsonl` |
| sweep stats | `/tmp/g1-nonuniform-levels/stats.w{0,1}.jsonl` |
| validate | `/tmp/g1-nonuniform-levels/validate_pack.json` |
| aggregate | `/tmp/g1-nonuniform-levels/report.json` |
| learned levels probe | `/tmp/g1-nonuniform-levels/probe_levels.json` |
| worker logs | `/tmp/g1-nonuniform-levels/w{0,1}.log` |
| script | `/tmp/g1-nonuniform-levels/sweep.py` (not a repo write) |

## 12. KILLS / REOPEN_IF

- **Lloyd-Max per-group codebook as a complete-BPW attack: KILLS.** Error win is real (4-bit G=64 rel_l2 0.076 vs absmax 0.110) and the metadata cost (8.0 BPW at G=64, 5.0 at G=256) wipes it. At G=256 a shared codebook matches it at 4.06 BPW. `REOPEN_IF` someone parameterises the per-group levels in ≪ 16·L bits (e.g. 2–3 shape coefficients) **and** a kernel consumes that encoding directly.
- **Percentile clipping at p≥99.9, G≤256: KILLS as a distinct rule.** It is absmax. `REOPEN_IF` group size ≥ 1024, where p99.9 is no longer the group max.
- **3-bit uniform anything as a replacement for G0 4-bit absmax at equal error: KILLS.** Even learned 3-bit sits at ~0.18 rel_l2 vs G0 0.11. `REOPEN_IF` an activation-aware or residual codec recovers the missing 0.07.

---

STATUS
MEASURED_WIN

CLAIMS
1. G0 language complete BPW is 4.252735126866492, which is 4.25 codec + f32-vector tax. Evidence: `uniform-q4-v1/manifest.json` fields `complete_physical_bpw`, `tensor_payload_bytes`, `source_weight_elements`; `qwen38_pack.rs:673-678`.
2. G0 4-bit group-64 codec is absmax/7 + FP16 scale + q∈[-8,7]. Evidence: `qwen80_uniform_q4.rs:233-247`; `qwen_uniform_q4.metal:9-11`.
3. This lane's absmax 4/G=64 is bit-identical to the production HQ30UQ4 body of `layers.0.mlp.gate_proj`. Evidence: `/tmp/g1-nonuniform-levels/validate_pack.json` `mine_vs_packed.max_abs=0`.
4. Real Qwen3.8 GEMVs are heavy-tailed: class-mean amax/p99 = 12.16 (attn) / 11.48 (mlp). Evidence: `/tmp/g1-nonuniform-levels/report.json` `outlier_stats`; `stats.w0.jsonl:40` (down_proj L2 amax/p99=41.75).
5. On all 208 attn GEMVs, 4-bit, the lowest equal-metadata relative_l2 at every G∈{32,64,128,256} is `learned_shared`. Evidence: `report.json` `winners_4bit.attn`.
6. On all 192 MLP GEMVs (the entire 17.11e9-weight G0 MLP mass), same ranking. Evidence: `report.json` `winners_4bit.mlp`.
7. Versus G0 absmax G=64 error, `learned_shared` at G=256 is better on both classes at 4.0625 complete BPW. The switch buys 0.1875 BPW at equal-or-better error. Evidence: `report.json` `g_migration.{attn,mlp}.learned_shared`.
8. `mse_scale` buys the same 0.1875 BPW at G=256 with no kernel change. Evidence: `g_migration.*.mse_scale`; pack formula is a scale-only edit to `qwen80_uniform_q4.rs:233`.
9. Lloyd-Max per-group codebooks do not buy complete BPW (8.0 at G=64, 5.0 at G=256) and at G=256 tie learned_shared which costs 4.06. Evidence: §4 table; `g_migration.*.lloyd_max.bpw_saved = -0.75`.
10. No token-level number was produced. Evidence: this lane's permitted-scope list; no Metal, no `ascension_qwen38_*` binary invoked.

EVIDENCE
- `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json` (complete_physical_bpw=4.252735126866492)
- `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/model.safetensors.index.json` (1184 tensors; 400 language GEMVs measured)
- `crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:48,233-247,312-321,1577`
- `crates/hawking-core/src/model/qwen38_pack.rs:673-684`
- `crates/hawking-core/shaders/qwen_uniform_q4.metal:1-11`
- `/tmp/g1-nonuniform-levels/validate_pack.json`
- `/tmp/g1-nonuniform-levels/w0.log` `DONE` 4541.14 s, 200/200
- `/tmp/g1-nonuniform-levels/w1.log` `DONE` 4540.43 s, 200/200
- `/tmp/g1-nonuniform-levels/rows.w0.jsonl` + `rows.w1.jsonl` (12071 rows)
- `/tmp/g1-nonuniform-levels/report.json` (class aggregates, g_migration, bits_bought)
- `/tmp/g1-nonuniform-levels/probe_levels.json` (shared codebook + α histogram)
- `/tmp/g1-nonuniform-levels/rows.w0.jsonl:132,139,142` (gate-0 4/64 raw)

CHANGES
- Created `workspace/superwave/g1/g1-nonuniform-levels.md` (this file).
- No tracked file modified. No artifact packed. No process other than two CPU python workers (now exited).

TESTS
- `test -s workspace/superwave/g1/g1-nonuniform-levels.md` — see final-message TESTS
- `wc -l workspace/superwave/g1/g1-nonuniform-levels.md` — see final-message TESTS
- `git status --porcelain` — see final-message TESTS

RISKS
- mse α grid is 7 points; a denser search can only improve mse_scale, not reverse the ranking against learned at 4-bit.
- learned k-means uses a 5e5 sample and 10 iters; 5-bit already shows this underfits vs mse. A better shared-level fit can only help 3–4 bit.
- Act-cosine covers only K=5120 tensors and 64 holdout tokens. down_proj / out_proj unweighted by activations.
- Projected 4.083 language-complete BPW assumes embed stays Q4-absmax. Unmeasured.
- No kernel was written or timed. Reconstruction-is-free (prior Qwen3.8 receipt) is **not** re-verified here; the GPU lane owns that.

UNRESOLVED
- G=512 / G=1024 for learned_shared: G=256 still has margin under E0 (mlp 0.1041 vs 0.1098). Next cheapest experiment: rerun learned+absmax+mse at G=512 on the 9 curve layers (~10 min CPU).
- Embed / lm_head scale-rule (1.27e9 weights each). Same script, one tensor at a time, ~5 GB peak.
- mse_scale pack of one existing HQ30UQ4 tensor vs this script, then a Metal parity test — GPU lane.
- Activation-aware scale (AWQ-style) is out of scope; would need the capture's per-layer X and a different objective.

NEXT
1. Pack-only G1 candidate: mse_scale, 4-bit, G=256, same HQ30UQ4 magic, no new kernel. Expected complete GEMV BPW 4.0625, language-complete ~4.083 projected, reconstruction strictly better than G0 on both classes.
2. Kernel variant for learned_shared 4-bit G=256 (16-entry LUT). Same DRAM as (1), ~0.004 lower rel_l2.
3. Hand both to the GPU lane for token-level TOKEN_NS. Do not promote on reconstruction alone.
