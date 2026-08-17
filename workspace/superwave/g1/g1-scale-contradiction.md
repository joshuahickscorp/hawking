# G1 — activation-scale contradiction, reconciled

Date: 2026-08-17
Lane: 32-scale-contradiction
Write scope: this file only. CPU/numpy. No GPU, no generate, no pack, no resident process.

Measurement JSON: `/tmp/qwen38_scale_contradiction.json`
sha256 `16d649478e6bec60b64fa7d0c37d4e312774a3303e93899642957a4a4ce8f386`
Script: `/tmp/qwen38_scale_contradiction.py` sha256 `b3ddbce69d91faa18f38fcb25f5a1a07b3134d2e1f14d916a33be796bd0d9bb3`
Wall: 145.7 s. Peak RSS: 2.144 GB (under 20 GB).
Codec helpers reused from `/tmp/qwen38_out_proj_forensics.py` (wave 1 lane 05).

Tensors: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16`
Activations: `.../activation-capture-v1` (`sha256_self=fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512`, file sha256 `01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe`, L00.f32 sha256 `edc47c2ac99bf5446c775179dbcf9850b73320f25df5deea2df1d23d518a0243`)

Every number below is **measured** on those tensors unless marked **claimed** (prior receipt) or **projected**.

---

## Reconciled statement

Both wave 1 numbers are right. They are not the same experiment.

| lane | operation | bits | baseline | claimed | reproduced (this lane) |
|---|---|---:|---|---:|---:|
| 05 out_proj forensics | exact-preserve top-42 \|X\|-energy columns, Q3 the rest | 3 | Q3 absmax g=64 mixer-output cosine 0.95310 | 0.97616 | **0.9762145081598549** |
| 04 doctor (probe receipt) | AWQ fold: quantize `W * rms(X_fit)`, unscale | 4 | Q4 absmax g=64 mixer-output cosine 0.99224 | 0.91865 | **0.9186496062432181** |

Under **one metric** (mean row-cosine of mixer output `Y = X_hold @ W.T`, odd rows of the 256-token capture) and **one baseline** (HGRAVU01 absmax g=64, float32 group scale):

L0 `linear_attn.out_proj` `[5120, 6144]`, X = DeltaNet mixer-site proxy `v * silu(z)`:

| op | Q3 out | Δ vs Q3 absmax | Q4 out | Δ vs Q4 absmax | Q3 residual-proxy | Q4 residual-proxy |
|---|---:|---:|---:|---:|---:|---:|
| absmax (baseline) | **0.953103** | 0 | **0.992249** | 0 | 0.997449 | 0.999636 |
| exact top-42 \|X\|, rest q | **0.976215** | **+0.02311** | 0.995409 | +0.00316 | 0.999551 | 0.999916 |
| exact top-42 \|W\|, rest q | 0.953268 | +0.00016 | 0.992280 | +0.00003 | 0.997457 | 0.999637 |
| act-colscale α=1 (doctor op) | **0.842589** | **−0.11051** | **0.918653** | **−0.07360** | 0.997691 | 0.998765 |
| group-MSE scale (fit) | 0.974600 | +0.02150 | 0.993152 | +0.00090 | 0.999410 | 0.999871 |
| act-colscale α=0.25 | 0.974987 | +0.02188 | 0.995094 | +0.00285 | 0.999380 | 0.999880 |

Exact-preservation of important columns and rescaling by importance are opposite operations. The first spends bits. The second spends the group's dynamic range. On this tensor the second destroys the 63 X-cold columns that share a g=64 block with an X-hot column.

**What the 256-token capture can be used for**

- Rank the L0 mixer-site spike (42 / 6144 columns, 90% of X energy) for exact-island / extra bits. Top-42 even∩odd = 42/42. Leave-one-prompt-out overlap = 42/42. First-64 and last-64 also recover the same 42. Honest fit-only selection equals the leaked all-token selection (both 0.976215).
- Fit a **per-group** scale that minimises mixer-output MSE against that X. Same BPW as absmax. Wins on L0, L32, L63.
- Show that absmax protects the wrong columns: L0 and L32 top-42 \|W\| ∩ top-42 \|X\| = **0 / 42**. L63 = 3 / 42.

**What it cannot be used for**

- Raw AWQ column-RMS fold (`W * s`, quantize, `/ s`, α=1). KILLS on L0 (0.992249 → 0.918653), L32 (0.99380 → 0.98451), L63 (0.99247 → 0.97849). Not a thin-capture artifact: every 64–256 token s-source still kills L0 Q4 into 0.912–0.929.
- Estimating a 99%-energy column set. L0 n99 = 3555 / 6144; even∩odd top-3555 Jaccard 0.76. Ranking of the long tail is not stable.
- Treating post-norm hidden (width 5120) as if it were out_proj X (width 6144). The working X is the derived mixer proxy, same as the density probe (`x_site=real_derived_v_silu_z_from_fused_in_proj_qkvz`).
- A generate claim. Residual-proxy on L0 Q3 is already 0.99745 because write/R = 0.353; that bar hides both the exact-preserve win and the AWQ kill.

**Single scale rule that survives**

Per-group scale `s = argmin_s ||X_g (w − q(w,s))||` on the even-row fit split. Consume with the existing uniform-Qn kernel. Do not fold raw column RMS into W. Do not protect \|W\| outliers.

On this capture the search (8 multipliers around absmax) moved 89.6% of L0 Q3 groups *down* from absmax. Absmax is too large because the max element is X-cold. Shrinking the scale clips that unused weight and returns codes to the body. AWQ α=1 does the opposite: it inflates the X-hot column until it *becomes* the group absmax and clips the body.

The capture does **not** have to be thickened before this rule can be stated. Thickening is the next experiment for a production scale tensor and for generate, not for the sign of the rule.

---

## Common protocol (this lane)

```
tensor     L0 / L32 linear_attn.out_proj, L63 self_attn.o_proj   [5120, 6144]
X          DeltaNet v*silu(z) or GQA repeat(v)*sigmoid(q_gate), from captured post-norm hidden
fit        even rows (128)     hold: odd rows (128)
metric     mean row-cosine of mixer output Y_hold vs Yq_hold     PRIMARY
also       weight cosine, residual-proxy cosine (post-norm hidden + write), min-row, rel-L2
codec      HGRAVU01 absmax g=64, float32 group scale (forensics). f16-scale twin also scored.
```

f16 vs float32 group scale is a 7e-5 effect (L0 Q3 0.953103 vs 0.953171). It is not the contradiction. The doctor/probe Q4 and Q4-act-colscale numbers match the f16-scale twin to all reported digits.

---

## Candidate explanations

### 1. Different error metrics — KILLS as the source of the sign flip. PARTIAL as a reporting trap.

Both lanes scored mixer-output mean-row-cosine. Evidence:

- Probe receipt `receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json` tensor `qwen38.L0.linear_attn.out_proj` field `output_cosine` for `HGRAVU01_q4_g64` = 0.9922374383267348 and for `HGRAVU01_q4_g64_act_colscale` = 0.9186496062432181. This lane's f16-scale twins are bit-identical to those two fields.
- Forensics follow-up `/tmp/qwen38_out_proj_forensics_followup.json` `layers.0.out_proj.baseline_q3.output_cosine` = 0.9531034548050097 and `exact_top90e_rest_q3.output_cosine` = 0.9762145081598549. This lane matches both.

Weight cosine moves the **same direction** as mixer-output (L0 Q4 act-colscale weight 0.85310 vs absmax 0.99355). Residual-proxy on L0 **hides** the damage (act-colscale Q4 residual 0.998765 vs absmax 0.999636) because write/R = 0.353. If a later reader scored residual-proxy they would miss the kill. That is not what the doctor published. The doctor number 0.992→0.919 is mixer-output.

### 2. Different baselines — SUPPORTS as a reporting confound. Does not create the sign flip.

Forensics published a Q3 pair (0.95310 → 0.97616). Doctor published a Q4 pair (0.99224 → 0.91865). Those two headline numbers are not comparable as a single delta.

The sign flip survives under either common baseline (table above). Baseline mismatch explains why the headlines looked incommensurable. It does not explain why one op helps and the other hurts.

### 3. Exact-preservation and importance-rescaling are different operations — SUPPORTS. This is the reconciliation.

Exact-preserve: `Wq = absmax_q(W); Wq[:, hot] = W[:, hot]`. The 42 columns stay exact. The other 6102 stay at baseline group quality. Extra storage if shipped as f16 columns: 16 × 42 × 5120 / (5120 × 6144) = **0.109 BPW**.

AWQ fold (`uniform_act_scaled` in `lab/operators/qwen_attention_density_probe.py`):

```
s = rms(X_fit, axis=0)          # per input column
Wh = absmax_q(W * s) / s
```

Groups are 64 consecutive input columns of one output row (C-order `[out, in]`, 6144/64 = 96 blocks). L0 90% X-energy sits in 42 columns that land in **41 / 96** blocks. Folding s inflates those columns (L0 col-RMS max/median = 83.42) until they own the group absmax.

Measured pollution, L0, Q4, hold X:

| | absmax Q4 | act-colscale Q4 |
|---|---:|---:|
| share of column-output error on the 42 hot cols | 0.9048 | **0.0011** |
| share on cold cols inside a hot block | 0.0373 | **0.8724** |
| mean weight rel-L2 of hot cols | 0.1727 | **0.0043** |
| mean weight rel-L2 of cold-in-polluted | 0.1142 | **0.7402** |
| mean weight rel-L2 of cold-clean blocks | 0.1139 | 0.2471 |

AWQ buys the 42 columns (weight error ×40 smaller) by spending the group's codes. 87% of the remaining output error is the wreckage of their 63 neighbours. Exact-preserve buys the same 42 columns by spending bits and does not touch the neighbours.

α sweep on the same fold, L0, same metric:

| α | Q3 out | Q4 out | note |
|---:|---:|---:|---|
| 0 (absmax) | 0.95310 | 0.99225 | no fold |
| 0.25 | 0.97499 | 0.99509 | 83^0.25 ≈ 3.0; mild |
| 0.5 | 0.95875 | 0.99158 | Q4 already under baseline |
| 0.75 | 0.90817 | 0.96672 | kill starts |
| 1.0 | 0.84259 | 0.91865 | doctor op |

Clipped fold `clip(s/median, 1, cap)` never beats α=0.25 and at cap=2 is a wash on Q4 (0.99215 vs 0.99225).

### 4. 256 tokens too thin for magnitude, thick enough for top-k — PARTIAL. Does not explain the kill.

L0 mixer X energy is a spike. Ranking of that spike is robust on this capture. Magnitude of the spike is also stable. The fold still kills.

| test | top-16 | top-42 | top-100 | top-3555 |
|---|---:|---:|---:|---:|
| even ∩ odd | 16 | **42** | 87 | 3078 |
| Jaccard even vs odd | 1.00 | **1.00** | 0.77 | 0.76 |
| first-128 ∩ last-128 | 16 | 42 | 90 | 3269 |
| LOO ∩ all (min over 5 prompts) | 16 | **42** | 97 | 3459 |
| even/odd RMS ratio on the even-top set, median | 1.148 | 1.135 | 1.132 | 1.062 |
| even/odd RMS ratio on the even-top set, p90 | 1.166 | 1.166 | 1.210 | 1.272 |
| even/odd RMS ratio on the even-top set, max | 1.170 | 1.185 | **7.28** | 7.28 |

Spearman even vs odd column energy: **0.864**. First-128 vs last-128: 0.927. LOO vs all: 0.979–0.997.

Exact-preserve Q3 output cosine is **0.976215** for every ranking source tested (all, fit, hold, first64, last64, each LOO). Selection of the L0 island does not need a thicker capture.

Act-colscale Q4 output cosine by s-source, all scored on the same hold split:

| s source | Q4 out |
|---|---:|
| fit even (original) | 0.91865 |
| hold odd | 0.92775 |
| all 256 | 0.92252 |
| first 64 | 0.91206 |
| last 64 | 0.92884 |
| LOO prompts 0..4 | 0.9199–0.9286 |

Every magnitude estimate kills. The range 0.912–0.929 is thin-sample jitter around a bad operation, not a sign flip. Cold-half-column even/odd RMS ratio p90 = 1.19, max = 2.21 — the long tail is noisier, which is why a 99% set cannot be ranked here.

L32 is the contrast: 50% energy needs 506 columns, 90% needs 2770. Top-42 even∩odd = 32/42. Exact-preserve of 42 columns only moves Q3 0.96789 → 0.97187. Group-MSE, which uses all columns, moves it to 0.98132. On a non-spike mixer, selection of 42 is the weak use and scale-reaim is the strong use.

---

## Cross-layer (same protocol)

| | L0 out_proj | L32 out_proj | L63 o_proj |
|---|---:|---:|---:|
| mixer | DeltaNet proxy | DeltaNet proxy | GQA proxy |
| n50 / n90 / n99 X-energy | 16 / 42 / 3555 | 506 / 2770 / 5356 | 327 / 2727 / 4704 |
| col-RMS max/median | 83.42 | 19.24 | (less spiked) |
| top-42 \|X\| ∩ \|W\| | **0** | **0** | 3 |
| Q3 absmax out | 0.95310 | 0.96789 | 0.96040 |
| Q3 exact-42-X out | 0.97621 | 0.97187 | 0.96657 |
| Q3 act-colscale α=1 | 0.84259 | 0.93641 | 0.92286 |
| Q3 act-colscale α=0.25 | 0.97499 | 0.97340 | 0.96785 |
| Q3 group-MSE | **0.97460** | **0.98132** | **0.97834** |
| Q3 residual-proxy absmax | 0.99745 | 0.98738 | 0.97749 |
| Q4 absmax out | 0.99225 | 0.99380 | 0.99247 |
| Q4 act-colscale α=1 | 0.91865 | 0.98451 | 0.97849 |
| Q4 group-MSE | 0.99315 | 0.99547 | 0.99467 |
| Q4 α=0.25 | 0.99509 | 0.99492 | 0.99389 |

Group-MSE is the only same-BPW rule that wins on all three layers. α=0.25 is a cheap proxy that also beats absmax everywhere. α=1 loses everywhere. Exact-42-X is the L0-spike tool; it is weaker than group-MSE once X energy spreads (L32, L63).

None of the Q3 same-BPW rules clear the 0.99 **mixer-output** bar. L0 Q3 already clears residual-proxy 0.99 (0.99745) because write/R = 0.353. L63 Q3 residual-proxy is 0.97749 (write/R claimed 1.845 in forensics); group-MSE lifts that to 0.98831. Still not 0.99. Q3 attention is not licensed by this lane.

---

## Group-MSE search detail (the surviving rule, as run)

Per output row, per 64-col block: Gram `G = X_fit[:, block].T @ X_fit[:, block]` (96 Grams, shared across 5120 rows). For each group, try `s = absmax(w)/bound * m` for `m ∈ {0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00}`, pick `argmin e^T G e`, `e = w − q(w,s)`.

L0 Q3 picks (491,520 groups):

```
m        0.50     0.70     0.85     1.00     1.15     1.30     1.50     2.00
n     115544   153380   118068    51045    22866    13750     9239     7628
frac    0.235    0.312    0.240    0.104    0.047    0.028    0.019    0.016
```

89.61% of groups leave absmax. 78.7% move to a *smaller* scale. That is the geometry: the absmax element is X-cold (overlap 0/42), so the incumbent scale is wasted on a weight the mixer does not read.

This is an 8-point search around absmax, not a full discrete code-boundary search. It is enough to show the direction. A production fit would run the same objective with a finer 1-D search on a thicker X.

---

## Wave 1 claims, restated with this lane's numbers

Forensics named root cause: absmax tracks \|W\|; mixer-output error tracks \|X\|; those column sets are disjoint. **Held.** Overlap 0/42 reproduced. Exact-42-X 0.976215 reproduced. Exact-42-W 0.953268 reproduced.

Doctor: naive AWQ from this capture is a KILL on L0 out_proj, 0.992 → 0.919. **Held.** Reproduced 0.992237 → 0.918650, bit-identical to `QWEN_ATTENTION_DENSITY_PROBE.json`.

Doctor's inference that "wrong-site scales are a KILL" is **too wide**. The site used for out_proj in the probe is the mixer proxy, not the unconfirmed post-norm hidden. The kill is the fold, not the site. Using the same X to *select* columns or to *re-aim group scale* helps.

Forensics' "do not fold column RMS into W (already failed)" is **held** for α=1 and **refined**: α=0.25 of the same fold is a win. The failure is the raw RMS, not every X-aware scale.

---

## Command output (this lane)

```
$ /opt/homebrew/bin/python3 /tmp/qwen38_scale_contradiction.py
[11:23:39] rss_max=0.033G ===== L0 out_proj mixer=delta_net full=True =====
[11:23:40] rss_max=1.076G loaded W(5120, 6144) X(256, 6144)
[11:23:40] rss_max=1.865G   absmax_q3_g64                                    out=0.95310 w=0.96680 res=0.99745
[11:23:40] rss_max=1.870G   absmax_f16scale_q3_g64                           out=0.95317 w=0.96674 res=0.99746
[11:23:40] rss_max=1.997G   act_colscale_fit_q3                              out=0.84259 w=0.73481 res=0.99769
[11:23:41] rss_max=2.104G   act_colscale_fit_f16scale_q3                     out=0.84258 w=0.73481 res=0.99769
[11:23:41] rss_max=2.104G   exact_top42_X_all_rest_q3                        out=0.97621 w=0.96701 res=0.99955
[11:23:41] rss_max=2.109G   exact_top42_X_fit_rest_q3                        out=0.97621 w=0.96701 res=0.99955
[11:23:41] rss_max=2.109G   exact_top16_X_all_rest_q3                        out=0.96800 w=0.96688 res=0.99889
[11:23:41] rss_max=2.109G   exact_top16_X_fit_rest_q3                        out=0.96800 w=0.96688 res=0.99889
[11:23:42] rss_max=2.109G   exact_top42_W_rest_q3                            out=0.95327 w=0.96701 res=0.99746
[11:23:42] rss_max=2.109G   act_colscale_fit_alpha0.25_q3                    out=0.97499 w=0.96242 res=0.99938
[11:23:42] rss_max=2.109G   act_colscale_fit_alpha0.5_q3                     out=0.95875 w=0.93008 res=0.99925
[11:23:42] rss_max=2.109G   act_colscale_fit_alpha0.75_q3                    out=0.90817 w=0.83947 res=0.99855
[11:23:42] rss_max=2.110G   absmax_q4_g64                                    out=0.99225 w=0.99355 res=0.99964
[11:23:43] rss_max=2.116G   absmax_f16scale_q4_g64                           out=0.99224 w=0.99354 res=0.99964
[11:23:43] rss_max=2.116G   act_colscale_fit_q4                              out=0.91865 w=0.85310 res=0.99877
[11:23:44] rss_max=2.117G   act_colscale_fit_f16scale_q4                     out=0.91865 w=0.85310 res=0.99877
[11:23:44] rss_max=2.117G   exact_top42_X_all_rest_q4                        out=0.99541 w=0.99359 res=0.99992
[11:23:44] rss_max=2.117G   exact_top42_X_fit_rest_q4                        out=0.99541 w=0.99359 res=0.99992
[11:23:44] rss_max=2.117G   exact_top16_X_all_rest_q4                        out=0.99378 w=0.99356 res=0.99978
[11:23:44] rss_max=2.117G   exact_top16_X_fit_rest_q4                        out=0.99378 w=0.99356 res=0.99978
[11:23:44] rss_max=2.117G   exact_top42_W_rest_q4                            out=0.99228 w=0.99359 res=0.99964
[11:23:45] rss_max=2.117G   act_colscale_fit_alpha0.25_q4                    out=0.99509 w=0.99255 res=0.99988
[11:23:45] rss_max=2.117G   act_colscale_fit_alpha0.5_q4                     out=0.99158 w=0.98409 res=0.99986
[11:23:45] rss_max=2.117G   act_colscale_fit_alpha0.75_q4                    out=0.96672 w=0.94237 res=0.99946
[11:23:45] rss_max=2.117G   act_colscale_hold_odd_q4                         out=0.92775 w=0.86320 res=0.99892
[11:23:45] rss_max=2.117G   exact_top42_hold_odd_rest_q3                     out=0.97621 w=0.96701 res=0.99955
[11:23:45] rss_max=2.117G   act_colscale_all256_q4                           out=0.92252 w=0.85892 res=0.99883
[11:23:45] rss_max=2.117G   exact_top42_all256_rest_q3                       out=0.97621 w=0.96701 res=0.99955
[11:23:46] rss_max=2.117G   act_colscale_first64_q4                          out=0.91206 w=0.84266 res=0.99866
[11:23:46] rss_max=2.117G   exact_top42_first64_rest_q3                      out=0.97621 w=0.96701 res=0.99955
[11:23:46] rss_max=2.117G   act_colscale_last64_q4                           out=0.92884 w=0.86611 res=0.99891
[11:23:46] rss_max=2.117G   exact_top42_last64_rest_q3                       out=0.97621 w=0.96701 res=0.99955
[11:23:46] rss_max=2.117G   loo prompt 0 act4=0.92194 exact3=0.97621 ov=42
[11:23:46] rss_max=2.117G   loo prompt 1 act4=0.92212 exact3=0.97621 ov=42
[11:23:46] rss_max=2.117G   loo prompt 2 act4=0.91990 exact3=0.97621 ov=42
[11:23:46] rss_max=2.117G   loo prompt 3 act4=0.92162 exact3=0.97621 ov=42
[11:23:47] rss_max=2.117G   loo prompt 4 act4=0.92861 exact3=0.97621 ov=42
[11:24:07] rss_max=2.120G   group_mse_search_q3                              out=0.97460 w=0.96426 res=0.99941
[11:24:27] rss_max=2.120G   group_mse_search_q4                              out=0.99315 w=0.98950 res=0.99987
[11:24:27] rss_max=2.120G   act_colscale_clip2x_q4                           out=0.99215 w=0.99015 res=0.99973
[11:24:27] rss_max=2.120G   act_colscale_clip2x_q3                           out=0.96123 w=0.95132 res=0.99867
[11:24:28] rss_max=2.120G   q4_on_fit42_q3_rest                              out=0.97307 w=0.96697 res=0.99927
[11:24:28] rss_max=2.120G checkpoint L0
[11:24:29] rss_max=2.120G   absmax_q3_g64                                    out=0.96789   # L32
[11:24:29] rss_max=2.120G   act_colscale_fit_q3                              out=0.93641
[11:24:30] rss_max=2.144G   exact_top42_X_all_rest_q3                        out=0.97187
[11:24:30] rss_max=2.144G   act_colscale_fit_alpha0.25_q3                    out=0.97340
[11:24:31] rss_max=2.144G   absmax_q4_g64                                    out=0.99380
[11:24:32] rss_max=2.144G   act_colscale_fit_q4                              out=0.98451
[11:24:56] rss_max=2.144G   group_mse_search_q3                              out=0.98132
[11:25:16] rss_max=2.144G   group_mse_search_q4                              out=0.99547
[11:25:17] rss_max=2.144G checkpoint L32
[11:25:17] rss_max=2.144G   absmax_q3_g64                                    out=0.96040   # L63 o_proj
[11:25:18] rss_max=2.144G   act_colscale_fit_q3                              out=0.92286
[11:25:18] rss_max=2.144G   exact_top42_X_all_rest_q3                        out=0.96657
[11:25:19] rss_max=2.144G   act_colscale_fit_alpha0.25_q3                    out=0.96785
[11:25:19] rss_max=2.144G   absmax_q4_g64                                    out=0.99247
[11:25:20] rss_max=2.144G   act_colscale_fit_q4                              out=0.97849
[11:25:44] rss_max=2.144G   group_mse_search_q3                              out=0.97834
[11:25:44] rss_max=2.144G   group_mse_search_q4                              out=0.99467
[11:26:05] rss_max=2.144G DONE wall_s=145.7 rss_max_gb=2.144
```

```
$ shasum -a 256 /tmp/qwen38_scale_contradiction.json /tmp/qwen38_scale_contradiction.py
16d649478e6bec60b64fa7d0c37d4e312774a3303e93899642957a4a4ce8f386  /tmp/qwen38_scale_contradiction.json
b3ddbce69d91faa18f38fcb25f5a1a07b3134d2e1f14d916a33be796bd0d9bb3  /tmp/qwen38_scale_contradiction.py
```

JSON excerpts (`/tmp/qwen38_scale_contradiction.json`):

```
# layers.0.out / ops  (role=out_proj)
ops.absmax_q3_g64.output_cosine                         0.9531034548050097
ops.absmax_f16scale_q4_g64.output_cosine                0.9922374383267348
ops.act_colscale_fit_f16scale_q4.output_cosine          0.9186496062432181
ops.exact_top42_X_all_rest_q3.output_cosine             0.9762145081598549
ops.exact_top42_X_fit_rest_q3.output_cosine             0.9762145081598549
ops.exact_top42_W_rest_q3.output_cosine                 0.953267548386614
ops.group_mse_search_q3.output_cosine                   0.974599872428159
ops.group_mse_search_q3.frac_groups_not_absmax          0.896148681640625
ops.act_colscale_fit_alpha0.25_q3.output_cosine         0.9749873430790348
x_energy.overlap_top42_X_vs_W                           0
x_energy.n50_all / n90_all / n99_all                    16 / 42 / 3555
thinness.selection.top42.overlap                        42
thinness.selection.top42.jaccard                        1.0
pollution_act_colscale_q4.n_blocks_containing_hot       41
pollution_act_colscale_q4.share_col_out_err_hot         0.0010850845470620572
pollution_act_colscale_q4.share_col_out_err_cold_in_polluted_block  0.8723619993675678
pollution_act_colscale_q4.mean_weight_rel_l2_hot_cols   0.004315657075494528
pollution_act_colscale_q4.mean_weight_rel_l2_cold_in_polluted  0.740205705165863
```

Bit-identity against the probe receipt (claimed, re-read this lane via `git show HEAD:receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json`):

```
HGRAVU01_q4_g64.output_cosine              0.9922374383267348
HGRAVU01_q4_g64_act_colscale.output_cosine 0.9186496062432181
HGRAVU01_q3_g64.output_cosine              0.9531713530055139
HGRAVU01_q3_g64_act_colscale.output_cosine 0.8425799898189834
x_site  real_derived_v_silu_z_from_fused_in_proj_qkvz
n_x_rows  256
```

This lane's `absmax_f16scale_q4_g64` and `act_colscale_fit_f16scale_q4` match the first two fields exactly. `absmax_f16scale_q3_g64` = 0.9531713530055139 matches the third. `act_colscale_fit_f16scale_q3` = 0.8425799898189834 matches the fourth.

Wave 1 source quotes:

```31:33:workspace/superwave/g1/g1-out-proj-forensics.md
Protecting the 42 fattest **weight** columns and Q3-ing the rest changes L0 out_proj output cosine **0.9531 → 0.9533** (+0.0002).
Protecting the 42 hottest **activation-energy** columns changes it **0.9531 → 0.9762** (+0.0231).
Those two 42-column sets are disjoint.
```

```73:73:workspace/superwave/g1/g1-doctor-tensor-map.md
Naive AWQ column-scale on the captured X is **not** the correction. Probe `HGRAVU01_q4_g64_act_colscale` on L0 `out_proj` drops output cosine 0.99224 → **0.91865**. The capture site is `UNCONFIRMED_POST_NORM` for in-proj and a derived proxy for out-proj. Wrong-site scales are a KILL.
```

Probe implementation of the killing op:

```
# lab/operators/qwen_attention_density_probe.py  uniform_act_scaled
s = np.sqrt(np.mean(np.square(X, dtype=np.float64), axis=0)).astype(np.float32)
s = np.maximum(s, 1e-8)
packed = _uniform_codec(W * s[None, :], bits=bits, group_size=GROUP_UNIFORM)
what = packed.reconstruction.reshape(W.shape) / s[None, :]
```

---

## What this lane did not measure

- Generate / greedy-id / token identity. Serialized GPU lane owns that. Residual-proxy ≥ 0.99 is not generate. Mixer-output 0.9746 is not generate.
- True DeltaNet recurrent mix or softmax GQA mix as out_proj X. Site proxy only, same as both source lanes.
- Pre-norm residual (RMSNorm not inverted).
- Layers other than 0, 32, 63. MLP down_proj. embed / lm_head.
- A packed artifact or a Metal kernel. Group-MSE writes a different f16 scale plane into the existing HQ30UQ4 container; that pack was not built.
- Full discrete per-group scale search (code-boundary set). 8 multipliers only.
- A thicker capture.

Cheapest experiment that turns the surviving rule into a generate-facing claim: pack L0/L32 `out_proj` and L3/L63 `o_proj` with the group-MSE scales (keep HGRAVU01 codes, existing uniform-Qn kernel), leave everything else at G0 Q4, hand to the GPU lane vs the Q4 oracle. Do not AWQ-fold. Do not expand to float.

Cheapest experiment that would *thicken* the capture: 2k–8k tokens of the same BF16 parent, same mixer-proxy construction, re-fit group-MSE, report hold cosine vs the 256-token fit. Needed before shipping a scale plane. Not needed to know α=1 is dead or that group-MSE beats absmax.

---

## Completion report

```
STATUS
SUPPORTED

CLAIMS
1. Both wave 1 numbers reproduce on the real tensors under one metric (mixer-output mean-row-cosine, hold odd). Forensics exact-42-X Q3 = 0.9762145081598549 vs claimed 0.97616. Doctor/probe AWQ-Q4 = 0.9186496062432181 vs claimed 0.91865, bit-identical to QWEN_ATTENTION_DENSITY_PROBE.json. Evidence: /tmp/qwen38_scale_contradiction.json layers.0 ops.exact_top42_X_all_rest_q3 and ops.act_colscale_fit_f16scale_q4.
2. The contradiction is two operations, not two facts. Exact-preserve spends bits and raises L0 Q3 0.953103 → 0.976215. AWQ α=1 spends group dynamic range and drops L0 Q4 0.992249 → 0.918653 and L0 Q3 0.953103 → 0.842589. Same metric, either common baseline, opposite signs. Evidence: common-protocol table in this file; pollution_act_colscale_q4.share_col_out_err_cold_in_polluted_block = 0.8724.
3. Different-metric and different-baseline hypotheses do not create the sign flip. Both lanes used mixer-output cosine. Baseline was Q3 vs Q4 (a reporting confound). Residual-proxy on L0 hides both effects (write/R=0.353) but was not the published doctor number. Evidence: probe receipt fields cited above; residual_proxy_cosine on the same ops.
4. 256 tokens is thick enough to rank the L0 42-column spike (even∩odd=42/42, LOO=42/42, every ranking source gives exact-preserve 0.976215) and too thin to rank a 99% set (n99=3555, Jaccard 0.76). Thinness does not explain the AWQ kill: every s-source still lands in 0.912–0.929. Evidence: layers.0.thinness and act_colscale_*_q4 ops.
5. Surviving scale rule: per-group s = argmin ||X_g (w−q(w,s))|| on the fit split. Same BPW as absmax. L0/L32/L63 Q3: 0.97460 / 0.98132 / 0.97834 vs absmax 0.95310 / 0.96789 / 0.96040. α=0.25 is a cheap proxy. α=1 is dead. \|W\|-exact-42 is dead (0.95327, overlap 0/42). Evidence: ops.group_mse_search_q3 and x_energy.overlap_top42_X_vs_W.
6. Capture does not have to be thickened to state that rule. Thickening is required before a production scale plane or a generate claim.

EVIDENCE
- /tmp/qwen38_scale_contradiction.json sha256 16d649478e6bec60b64fa7d0c37d4e312774a3303e93899642957a4a4ce8f386
- command log in this file, wall 145.7 s, rss_max 2.144 GB
- /tmp/qwen38_out_proj_forensics_followup.json sha256 4110044edbe580a5ef143225da72934306d8bd8b7b75f08f184b7d803f70e89c (wave 1, matched)
- receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json tensor qwen38.L0.linear_attn.out_proj (bit-identical Q4 pair)
- workspace/superwave/g1/g1-out-proj-forensics.md:31-33
- workspace/superwave/g1/g1-doctor-tensor-map.md:73
- lab/operators/qwen_attention_density_probe.py uniform_act_scaled
- .../activation-capture-v1/capture-result.json sha256_self fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512

CHANGES
workspace/superwave/g1/g1-scale-contradiction.md (this file). No other path touched.

TESTS
see end of lane message (test -s, wc -l, git status --porcelain)

RISKS
- out_proj X is a mixer-site proxy, not the recurrent / softmax mix. REOPEN_IF a captured mixer-output X (width 6144) exists and α=1 stops killing or \|W\|∩\|X\| is no longer ~0.
- residual-proxy uses post-norm hidden, not pre-norm residual.
- group-MSE used 8 multipliers, 128 fit rows, 3 layers. Direction is consistent. Magnitude of the shipped scale plane is not.
- Mixer-output 0.9746 / residual-proxy 0.9994 is not generate.

UNRESOLVED
- Whether group-MSE Q3 on the four write tensors is token-safe vs the Q4 oracle.
- Whether a finer scale search or a thicker X moves L32/L63 Q3 residual-proxy across 0.99.
- Whether the L0 42-column island is the same set under a true recurrent mix.

NEXT
Pack L0/L32 out_proj and L3/L63 o_proj with group-MSE scales into the existing HQ30UQ4 container; consume with qwen_uniform_qn_*; GPU lane scores native generate vs the Q4 oracle. Do not AWQ-fold. Do not extract \|W\| outliers. Optional cheap add-on on L0 only: exact the 42 ranked X columns. Thicken the capture before freezing a scale plane.
```
