# G1 group-partition geometry

Wave-1 left one tensor (L0 `out_proj`) at Q3 g=48 cosine 0.9602 beating head-aligned g=64 at 0.9531. This lane re-measured that cell, then swept group size, head-relative phase, and the other axis on real BF16 weights and the 256-token capture, across 11 layers and both attention and MLP. Complete BPW includes the f16 scale plane. The production kernel's 64-thread × 8-wide K walk is the cheap-mapping filter.

**Verdict: MEASURED_NEGATIVE.** The 0.9602 number is real. Head alignment is not why. Smaller g is better cosine on 60/60 tensors because it buys more scales. That purchase does not pay: at matched complete BPW, K-axis g=64 is on the Pareto front of the cheap family. No group geometry makes Q3 match Q4-g64 at complete BPW ≤ 4.26 (0/60). A geometry the kernel cannot address cheaply is not a win; the geometries that *are* cheap are not a win on the error-vs-BPW curve.

Not a token-level claim. No GPU. No generate.

---

## 1. Scope (MEASURED this lane)

| item | value | pointer |
|---|---|---|
| weights | BF16 shards | `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` |
| activations | 256 × 5120 × 64, real BF16 post-norm hidden | `.../activation-capture-v1/hidden/LXX.f32` |
| G0 catalog | 402 HQ30UQ4 + 353 f32v2 | `.../uniform-q4-v1/manifest.json` |
| layers | 0,3,8,15,16,31,32,47,48,60,63 | 6 DeltaNet + 5 GQA |
| tensors | 60 | gate, up, down on all 11; `in_proj_qkvz`+`out_proj` on DN; `q_proj`+`v_proj`+`o_proj` on GQA |
| holdout | odd rows of 256 (n=128) | same rule as `g1-out-proj-forensics.md` |
| codec | uniform absmax RTN, bound=`2^(b-1)-1`, per-row groups along K unless named | matches `/tmp/qwen38_out_proj_forensics.py:170-179` |
| group sizes | 8,12,16,20,24,32,40,48,56,64,80,96,112,128,160,192,256 | includes non-p2 |
| bits | 3 and 4 on every tensor; 2 only in the 402-shape BPW recipes | |
| wall | 1089.96 s | `/tmp/g1_group_partition_geometry_sweep.json` `wall_s` |
| rss_max | 9.019 GB | same, `rss_max_gb` |
| configs scored | 2850 | sum of `tensors[*].configs` |
| script | `/tmp/g1_group_partition_geometry_sweep.py` | CPU/numpy only |

X construction (same as wave-1 forensics, not a new site claim):

- gate / up / q / v / `in_proj_qkvz`: captured post-norm hidden
- down: `silu(H @ Wg.T) * (H @ Wu.T)`
- DeltaNet `out_proj`: mixer proxy `v * silu(z)` from fused `in_proj_qkvz`
- GQA `o_proj`: mixer proxy `(repeat(v) * σ(q_gate))`

`out_proj` / `o_proj` X is a mixer-site proxy, not a recurrent or softmax mix. Labeled as such in the JSON `claim_boundary`.

---

## 2. Sanity vs wave 1 (MEASURED)

L0 `linear_attn.out_proj` Q3, odd-row holdout, K-axis phase 0:

| g | wave-1 claimed | this lane | weight cosine |
|---:|---:|---:|---:|
| 48 | 0.9602 | **0.9601527843826347** | 0.969404847339141 |
| 64 | 0.9531 | **0.9531034548050097** | 0.9667973524806457 |
| 96 | 0.9434 | 0.9434075218950777 | 0.9628406441307725 |
| 128 | 0.9443 | 0.9443471992882664 | 0.9603143351631379 |

Wave-1 L0 slot ratio 1.014 reproduced: `slot_max_over_min = 1.0140965599336427`, slots `[1.72456334840352e-05, 1.7488737590036536e-05]`.

Incumbent complete BPW reconstructed from the 402 Q4 shapes + measured f32 bytes:

```
8 * (14287109840 + 10584840) / 26895998464 = 4.252735126866492
```

Matches the audited G0 number exactly. `tensors_exact_div=402` at g=64.

Evidence: sweep stdout 11:27:28–11:27:29 (section 12); JSON `sanity`, `ground_truth.reconstructed_incumbent`.

---

## 3. Complete BPW (DERIVED from 402 shapes, formula MEASURED)

Per-tensor payload, HQ30 family, rank-2 header, per-group full-slot codes (the layout `geo_tpr64` indexes):

```
bytes = 40 + n_groups * (2 + ceil(bits * g / 8))
n_groups_K = rows * ceil(cols / g)     # per-row; phase-0
complete_bpw_model = 8 * (sum_402 bytes + 10584840) / 26895998464
```

When `K % g == 0` for every GEMV, body BPW is `bits + 16/g` and the scale plane is `16/g`. That 16/g term is why small groups stop paying.

Codec-formula check on the live artifact (wave-1 inventory, not re-derived): 402/402 files match `32+4*rank + ceil(E/64)*2 + ceil(E/64)*32`. This lane's g=64 Q4 reconstruction hits those same 14_287_109_840 Q4 bytes.

Unique K on the 402 GEMVs: `{5120, 6144, 17408}`. `gcd = 1024`. Group sizes that divide every K with no short last group are the divisors of 1024: **8, 16, 32, 64, 128, 256** (and 512, 1024, unused).

| g | divides all K? | 5120 rem | 6144 rem | 17408 rem |
|---:|---|---:|---:|---:|
| 8 | yes | 0 | 0 | 0 |
| 12 | no | 8 | 0 | 8 |
| 16 | yes | 0 | 0 | 0 |
| 20 | no | 0 | 4 | 8 |
| 24 | no | 8 | 0 | 8 |
| 32 | yes | 0 | 0 | 0 |
| 40 | no | 0 | 24 | 8 |
| **48** | **no** | **32** | **0** | **32** |
| 56 | no | 24 | 40 | 48 |
| 64 | yes | 0 | 0 | 0 |
| 80 | no | 0 | 64 | 48 |
| 96 | no | 32 | 0 | 32 |
| 128 | yes | 0 | 0 | 0 |
| 256 | yes | 0 | 0 | 0 |

g=48 is exact only on the 64 tensors with K=6144 (48 `out_proj` + 16 `o_proj`). The other 338 GEMVs need a short last group.

Model-complete BPW if the named geometry is applied to all 402 GEMVs (f32v2 held fixed):

| recipe | complete BPW | vs G0 | exact / short | kernel class |
|---|---:|---:|---|---|
| Q4 g=64 K (G0) | **4.252735126866492** | 0 | 402 / 0 | CHEAP |
| Q4 g=128 K | 4.127747421929657 | −0.1250 | 402 / 0 | CHEAP |
| Q4 g=48 K | 4.346559665240767 | +0.0938 | 64 / 338 | CHEAP |
| Q4 g=32 K | 4.502710536740161 | +0.2500 | 402 / 0 | CHEAP |
| Q4 2D 2×32 | 4.252735126866492 | 0 | 402 / 0 | CHEAP_REWRITE |
| Q4 2D 2×64 | 4.127747421929657 | −0.1250 | 402 / 0 | CHEAP_REWRITE |
| hybrid Q4 g=48 on K=6144 else 64 | 4.258972938049615 | +0.0062 | — | CHEAP |
| Q3 g=8 K | 5.002661356487502 | +0.7499 | 402 / 0 | CHEAP_UNPACK_REWRITE |
| Q3 g=16 K | 4.002759716992822 | −0.2500 | 402 / 0 | CHEAP_UNPACK_REWRITE |
| Q3 g=32 K | 3.502808897245481 | −0.7499 | 402 / 0 | CHEAP_UNPACK_REWRITE |
| Q3 g=48 K | 3.344235086880766 | −0.9085 | 64 / 338 | CHEAP_UNPACK_REWRITE |
| Q3 g=64 K | 3.252833487371811 | −0.9999 | 402 / 0 | CHEAP_UNPACK_REWRITE |

Awkward g can cost *more* complete BPW than a smaller clean g: on K=5120, Q3 g=96 bills 3.206 body vs g=80 at 3.200, because `ceil(5120/96)=54` groups vs 64 clean groups at g=80. Same inversion at Q4 g=192 vs g=160 on the 60-tensor mean (`mean_bpw` 4.118 vs 4.113). That is the scale-and-pad term, not a coding bug.

Evidence: JSON `divisibility.table`, `model_complete_bpw_recipes`; `g1-artifact-inventory.md:125-149`; `g1-bit-budget-accounting.md:154-186`.

---

## 4. Output cosine vs complete BPW (MEASURED, 60 tensors)

K-axis, phase 0. Mean and min over the 60 real-X holdout scores. `mean_bpw` is the mean *tensor* complete BPW (header included; not the 402-wide model number).

| cfg | mean out | min out | mean tensor BPW |
|---|---:|---:|---:|
| Q3 g=8 | 0.989375 | 0.977282 | 5.000 |
| Q3 g=16 | 0.985014 | 0.967918 | 4.000 |
| Q3 g=32 | 0.980487 | 0.958128 | 3.500 |
| Q3 g=48 | 0.977746 | 0.952311 | 3.341 |
| Q3 g=64 | 0.975758 | 0.948312 | 3.250 |
| Q3 g=128 | 0.970960 | 0.938269 | 3.125 |
| Q4 g=8 | 0.997982 | 0.995674 | 6.000 |
| Q4 g=32 | 0.996243 | 0.991849 | 4.500 |
| Q4 g=48 | 0.995706 | 0.990681 | 4.343 |
| Q4 g=64 | 0.995324 | 0.989797 | 4.250 |
| Q4 g=128 | 0.994355 | 0.987709 | 4.125 |

Monotone: smaller g → higher cosine, higher BPW. No inversion on the 60-tensor mean.

0.99 mixer-output bar, same 60 cells:

| cfg | n ≥ 0.99 | worst cell |
|---|---:|---|
| Q4 g=32 | 60/60 | L47 `o_proj` 0.9918488138 |
| Q4 g=64 | 59/60 | L47 `o_proj` **0.9897967539** |
| Q4 g=128 | 58/60 | L47 `o_proj` 0.9877089928 |
| Q3 g=8 | 24/60 | L47 `o_proj` 0.977282 |
| Q3 g=32 | 7/60 | L47 `o_proj` 0.958128 |
| Q3 g=64 | 5/60 | L47 `o_proj` 0.948312 |

Q3 at complete BPW ≤ 4.26 never matches Q4 g=64 on the same tensor: **0/60**. Q3 g=16 (model 4.003 BPW) mean 0.985014 / min 0.967918; Q4 g=64 mean 0.995324 / min 0.989797; Q4 g=64 > Q3 g=16 on all 60.

**KILLS:** “shrink the group until Q3 is as good as incumbent Q4 at similar complete BPW.” The scale plane eats the bit budget before the cosine catches up. Q3 g=8 is 5.003 model BPW — *worse* than Q4 g=64 — and still loses on 36/60 against the 0.99 bar.

Q4 g=128 vs g=64: mean Δ cosine −0.000969 (min −0.002157, max −0.000200). A 0.125 complete-BPW cut at a ~0.001 cosine cost. L47 `o_proj` is already under 0.99 at g=64 and falls further. Not generate-scored.

---

## 5. Is g=48 a real win? Is it head alignment?

### 5.1 Real, and not one tensor (MEASURED)

Q3 g=48 − g=64 output cosine, K-axis phase 0, all 60 tensors: **60/60 positive**.

| organ / role | n | mean Δ | min Δ | max Δ |
|---|---:|---:|---:|---:|
| mlp | 33 | +0.001876 | +0.000394 | +0.003007 |
| delta_net | 12 | +0.002504 | +0.001211 | +0.007049 |
| gqa | 15 | +0.001822 | +0.000305 | +0.003999 |
| `out_proj` | 6 | +0.003449 | +0.002547 | **+0.007049** |
| `o_proj` | 5 | +0.002561 | +0.000305 | +0.003999 |
| `down_proj` | 11 | +0.001999 | +0.000782 | +0.003007 |
| `gate_proj` | 11 | +0.001672 | +0.000610 | +0.002066 |
| `in_proj_qkvz` | 6 | +0.001559 | +0.001211 | +0.001735 |
| `q_proj` | 5 | +0.001215 | +0.000823 | +0.001532 |
| `v_proj` | 5 | +0.001691 | +0.000900 | +0.002447 |

Q4 g=48 − g=64: mean **+0.000382** (min +0.000075, max +0.000884). Same sign, an order of magnitude smaller — Q4 already has more levels, so extra scales buy less.

Δ > 0.003 only on: L0 `out_proj` **+0.007049**, L31 `o_proj` +0.003941, L47 `o_proj` +0.003999, L60 `out_proj` +0.003158, L63 `down_proj` +0.003007, L63 `o_proj` +0.003056.

Typical MLP / in-proj / q / v cell is +0.0015 to +0.0025. That is the ordinary absmax group-size tradeoff (more f16 scales). L0 `out_proj` is a 3–5× outlier on top of that baseline, not the typical cell.

### 5.2 Head alignment is not the driver (MEASURED) — FALSIFIED as a general rule

Geometry facts (source, not re-derived): DeltaNet value head 128, GQA head 256, `out_proj`/`o_proj` K=6144. `128 % 64 == 0`, `256 % 64 == 0`, `6144 % 64 == 0`. g=64 cannot straddle a head or an output row. `g1-out-proj-forensics.md:103-106`, `qwen38_geometry.rs:31-52`.

Intra-head slot error ratio at Q3 g=64, every head-structured tensor in the 60 (16 cells):

| | ratio |
|---|---:|
| min | 1.00143 L32 `out_proj` |
| mean | 1.02171 |
| max | 1.13608 L63 `v_proj` |

L0 `out_proj` 1.01410 (wave-1 1.014). L3 `o_proj` 1.0186 (wave-1 1.019). L63 `o_proj` 1.0608 (wave-1 1.061). Slots are flat. Head-aligned groups do not dump error into one intra-head slot.

MLP tensors have no heads and still show the same g=48 > g=64 sign and the same ~0.002 magnitude. Head structure is not required for the effect.

Gaussian-X discriminator (iid N(0, rms(X_hold)²), same hold_n), Q3:

| tensor | real Δ(48−64) | gauss Δ(48−64) |
|---|---:|---:|
| L0 `out_proj` | **+0.007049** | +0.003049 |
| L0 gate | +0.001464 | +0.002408 |
| L0 down | +0.000782 | +0.002529 |
| L3 `o_proj` | +0.000305 | +0.002748 |
| L32 `out_proj` | +0.002547 | +0.002485 |
| L63 `o_proj` | +0.003056 | +0.003443 |

On ordinary tensors the real-X delta ≈ the gaussian-X / weight-cosine delta (~0.0025). On L0 `out_proj` real-X is ~2× the gaussian-X delta. Half of the wave-1 “straddle win” is the ordinary scale-count effect; the other half is this layer’s spiked mixer X (wave-1: 50% energy in 16/6144 columns). It is not “48 straddles 128-d heads.”

**KILLS:** “g=64 is the wrong choice because it is head-aligned.”
**KILLS:** “g=48 wins because it straddles heads.”
**REOPEN_IF:** a hold-out capture in which intra-head slot ratio at g=64 exceeds ~1.2 *and* a head-aligned g loses to a straddling g at *matched* complete BPW on more than the L0 out_proj cell.

---

## 6. Alignment offset relative to head boundaries (MEASURED)

Phase = short first group of `phase` columns, then groups of g. Q3. Deep write tensors:

L0 `out_proj` (the spiked-X cell) — phase moves cosine by several 10⁻³:

| g | phase | out cosine | vs phase 0 |
|---:|---:|---:|---:|
| 64 | 0 | 0.953103 | 0 |
| 64 | 16 | 0.957550 | +0.004446 |
| 64 | 32 | 0.958027 | +0.004923 |
| 64 | 48 | 0.959083 | **+0.005979** |
| 96 | 64 | 0.954224 | +0.010817 |
| 128 | 64 | 0.936835 | −0.007512 |

Shifting g=64 by 48 columns almost recovers g=48's 0.96015. That is group-phase vs hot K-columns, not head membership. `phase=48` at g=64 *breaks* head alignment (groups now cross 128-d boundaries) and helps this one cell.

L3 / L32 / L63 `out`/`o` — phase is noise, |Δ| typically < 0.001, often negative:

- L3 `o_proj` g=64 ph=32: −0.001541
- L32 `out_proj` all listed phases: |Δ| ≤ 0.000394
- L63 `o_proj` g=64 all phases: |Δ| ≤ 0.000294

**KILLS as a general lever:** “offset groups relative to head boundaries.” L0-specific, and even there g=48 phase-0 already captures most of the gain at a known BPW cost. Not generate-scored.

---

## 7. Other axis (MEASURED)

### 7.1 M-axis (groups of g along the output dim, per column)

Same complete body BPW as K-axis at the same g (`bits + 16/g` when the dim divides). Kernel class **NOT_CHEAP**: the production 8-wide K unpack would need 8 scales.

Q3 g=64, M − K, deep tensors:

| tensor | Δ(M−K) |
|---|---:|
| L0 `out_proj` | **+0.013955** (0.967058 vs 0.953103) |
| L63 down | +0.007350 |
| L32 up | +0.005504 |
| L63 `o_proj` | +0.003192 |
| L0 `in_proj_qkvz` | −0.007254 |
| L0 gate | −0.002576 |
| L3 `q_proj` | −0.002799 |
| L32 `out_proj` | −0.002030 |

M-axis isolates a hot *input* column (each K column has its own scale groups along M). That is why L0 `out_proj` jumps: wave-1's 42 hot mixer columns no longer share a scale with 63 neighbors. On tensors whose error is not a few hot K-columns, M is equal or worse.

**NOT a win:** NOT_CHEAP, not general, and on L0 `out_proj` Q3 M g=64 is still 0.967 — fails 0.99.

### 7.2 2D tiles, including gm=2 (the TG's 2 rows)

`gm=2, gk=32` has the same scale budget as K g=64 (one f16 per 64 weights). `gm=2, gk=64` matches K g=128. Production TG already owns 2 consecutive rows. Class **CHEAP_REWRITE** (scale index becomes `(row/gm, col/gk)`; no shuffle, no second pass).

Q3, matched BPW, deep set (phase 0):

| tensor | K g=64 | 2D 2×32 | Δ |
|---|---:|---:|---:|
| L0 `out_proj` | 0.953103 | 0.958525 | +0.005422 |
| L0 gate | 0.981029 | 0.980763 | −0.000266 |
| L3 `o_proj` | 0.983429 | 0.982856 | −0.000573 |
| L32 `out_proj` | 0.967893 | 0.966501 | −0.001392 |
| L63 `o_proj` | 0.960402 | 0.960706 | +0.000304 |
| L63 down | 0.972786 | 0.973926 | +0.001140 |

2D 2×32 is a wash to a small loss on the typical cell; a L0-`out_proj`-only gain. 2D 2×64 vs K g=128 is the same story (typical Δ ≈ −0.0003).

Flat C-order groups (the incumbent *storage* flatten) coincide with per-row iff `K % g == 0`. When they differ (g=48 on K=5120/17408), flat is **NOT_CHEAP** (a scale straddles two output rows; `row*groups_per_row + col/g` is the wrong index) and the cosine change vs per-row is < 0.0001 (L0 gate Q3 g=48: 0.982499 flat vs 0.982493 per-row).

**KILLS:** “group along M, or 2D-tile, as a general quality upgrade at matched BPW.”
**REOPEN_IF:** a pack that is allowed to be NOT_CHEAP (new thread mapping) and a generate score shows L0-style hot-K isolation is worth the kernel rewrite on the write tensors. This lane does not grant that.

---

## 8. Production kernel thread mapping (SOURCE + DERIVED)

`qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` (`qwen_uniform_q4.metal:181-221`):

```
TG=128, 4 simdgroups, 2 rows/TG, 64 threads/row
lane_in_row = split*32 + simd_lane          // 0..63
col         = lane_in_row * 8 + t * 512
scale       = scales[row * groups_per_row + col / 64]
packed      = *(uint*)(codes + group*32 + local/2)   // 8 Q4 values
```

Constants: `GROUP_SIZE=64`, `CODE_BYTES_PER_GROUP=32` (`qwen_uniform_q4.metal:19-20`).

Cheap = that walk, one scale per 8-pack, no cross-lane shuffle, no second pass.

| condition | class | why |
|---|---|---|
| K-axis, Q4, `g % 8 == 0`, `phase % 8 == 0` | **CHEAP** | change the two constants; uint load stays aligned |
| K-axis, Q3/Q2, same alignment | **CHEAP_UNPACK_REWRITE** | same ownership; nibble-uint unpack is Q4-specific (`unpack8` at :166-178). Qn is a bit-stream (`uniform_qn.rs:149-152`) |
| K-axis, `g % 8 != 0` or `phase % 8 != 0` | **NOT_CHEAP** | 8-pack straddles two groups |
| K-flat when `K % g != 0` | **NOT_CHEAP** | scale shared across two rows |
| M-axis | **NOT_CHEAP** | 8 distinct scales in one unpack |
| 2D, `gk % 8 == 0` | **CHEAP_REWRITE** | one scale per 8-pack; `gm=2` matches 2 rows/TG |

So g=48 *can* be consumed by this mapping (48 % 8 == 0). The reason it is not a win is complete BPW (+0.094 global, +0.006 hybrid), not addressability.

g=12 and g=20 are quality-plausible non-p2 sizes and are NOT_CHEAP.

---

## 9. Promising geometries, after the kernel filter

Only CHEAP / CHEAP_REWRITE / CHEAP_UNPACK_REWRITE.

| geometry | complete BPW | quality vs G0 Q4 g=64 | kernel | decision |
|---|---:|---|---|---|
| Q4 g=64 K | 4.252735126866492 | control | CHEAP | keep |
| Q4 g=128 K | 4.127747421929657 | mean −0.00097; L47 o 0.98980→0.98771 | CHEAP | not a quality win; 0.125 BPW cut is real on paper, unmeasured at generate |
| Q4 g=32 K | 4.502710536740161 | 60/60 ≥0.99, pays +0.25 BPW | CHEAP | **KILLS** (pays) |
| Q4 g=48 K | 4.346559665240767 | mean +0.00038, pays +0.094, 338 short groups | CHEAP | **KILLS** (pays) |
| hybrid Q4 g=48 on K=6144 | 4.258972938049615 | writes get ~+0.0004 at Q4 | CHEAP | **KILLS** (noise-sized) |
| Q4 2D 2×32 | 4.252735126866492 | typical −0.0003, L0 out +0.005 at Q3 | CHEAP_REWRITE | **KILLS** as general; L0-only |
| Q4 2D 2×64 | 4.127747421929657 | ≈ Q4 g=128, slightly worse | CHEAP_REWRITE | no better than 1D g=128 |
| Q3 any g | 3.25–5.00 | 0/60 match Q4 g=64 at BPW≤4.26 | UNPACK_REWRITE | **KILLS** as a grouping play |
| M-axis any g | same as K at same g | L0 out +0.014 at Q3 g=64; not general | NOT_CHEAP | **KILLS** under this mapping |
| phase-shifted g=64 | +tiny (extra prefix group) | L0 out +0.006; other layers ~0 | CHEAP if phase%8==0 | **KILLS** as general |

No CHEAP geometry is Pareto-better than incumbent Q4 g=64 on output cosine vs complete BPW. The only cheap BPW reduction inside this family is **larger** groups (g=128), which spends quality, not grouping cleverness.

---

## 10. Named KILLS / REOPEN_IF

1. **Head-aligned g=64 is a defect.** KILLS. Slot ratios 1.001–1.136 (mean 1.022). MLP, which has no heads, shows the same g=48>g=64 sign. REOPEN_IF slot ratio at g=64 exceeds ~1.2 *and* a matched-BPW straddling g wins on more than L0 `out_proj`.

2. **Straddling g=48 is an exploitable geometry win.** KILLS as a G1 lever. Real cosine gain, paid for by `16/g` (+0.094 complete BPW globally; +0.006 hybrid). At Q4 the gain is +0.00038 mean. At Q3 the gain does not reach 0.99. Kernel *can* address it. REOPEN_IF a generate-facing Q3 (or mixed) pack at g=48 beats Q4 g=64 on complete-token *and* oracle match. Binding: native reader, no expand-to-Q4.

3. **Q3 + small groups ≈ Q4 at similar complete BPW.** KILLS. 0/60. Q3 g=16 at 4.003 BPW loses to Q4 g=64 on every tensor. Q3 g=8 overshoots to 5.003 BPW and still fails the 0.99 bar on 36/60. REOPEN_IF a different scale (not absmax) or a different code family is under test — that is not this lane.

4. **M-axis / 2D as a general upgrade.** KILLS under `geo_tpr64`. M is NOT_CHEAP. 2D at matched BPW is a wash except L0 `out_proj`. REOPEN_IF a new thread mapping is authorized and generate confirms the L0 hot-K isolation.

5. **Phase offset vs heads as a general upgrade.** KILLS. L0-only. REOPEN_IF a later capture shows systematic phase wins on late GQA `o_proj` at matched BPW.

6. **Group geometry as the G1 BPW lever.** KILLS for this codec family. To move complete BPW you change *bits* or drop tensors from Q4, not g. g=128 is the only cheap same-family BPW cut and it is a quality-for-bytes trade, not a free lunch. REOPEN_IF a new codec's scale metadata is not `16/g`.

---

## 11. What this lane did not measure

- Generate, oracle-32, TOKEN_NS, TPS. GPU lane owns those. A component cosine is not a token claim.
- Bits other than 3/4 on the 60-tensor quality sweep (bit=2 is in the 402-shape BPW table only).
- Embed / lm_head output cosine (no matching 248320-d X in the capture).
- Non-absmax scales (output-MSE scale, activation-weighted). Wave-1 already killed naive act-colscale on L0 `out_proj` (0.992→0.919). This lane did not retry it.
- A native g≠64 pack on disk. All scores are in-register recon.

Cheapest experiment that would turn Q4 g=128 from a paper −0.125 BPW into a G1 candidate: pack the 402 GEMVs at HQ30UQ4 g=128 (same kernel, `GROUP_SIZE=128`, `CODE_BYTES=64`), native generate vs the Q4 g=64 oracle. Do not expand.

---

## 12. Evidence

### 12.1 Sweep stdout (actual)

```
[11:27:28] rss_max=0.034G manifest q4 tensors=402 elems=26893352960
[11:27:28] rss_max=0.034G incumbent reconstructed complete_bpw=4.252735126866492 q4_bytes=14287109840 (manifest q4=14287109840)
[11:27:28] rss_max=0.034G SANITY L0 out_proj Q3 g48/g64
[11:27:29] rss_max=1.855G SANITY {"q3_g48": {"output_cosine": 0.9601527843826347, "weight_cosine": 0.969404847339141}, "q3_g64": {"output_cosine": 0.9531034548050097, "weight_cosine": 0.9667973524806457}, "q3_g96": {"output_cosine": 0.9434075218950777, "weight_cosine": 0.9628406441307725}, "q3_g128": {"output_cosine": 0.9443471992882664, "weight_cosine": 0.9603143351631379}}
...
[11:45:38] rss_max=9.019G DONE tensors=60 wall_s=1090.0 rss=9.019G
```

Full log: grok session `call-5362e365-88a9-440a-b7a0-75cff8a89130-25.log`.
JSON: `/tmp/g1_group_partition_geometry_sweep.json` (3_388_432 bytes, schema `hawking.g1.qwen38_group_partition_geometry.v1`).
Script: `/tmp/g1_group_partition_geometry_sweep.py`.

### 12.2 JSON fields

`ground_truth.reconstructed_incumbent.complete_bpw` = `4.252735126866492`
`sanity.measured_this_lane.q3_g48.output_cosine` = `0.9601527843826347`
`sanity.measured_this_lane.q3_g64.output_cosine` = `0.9531034548050097`
L0 `out_proj` head_slots g=64: `slot_max_over_min = 1.0140965599336427`
L0 `out_proj` Q3 M g=64: `0.9670584246418459` (`axis=M_per_col`, `kernel.class=NOT_CHEAP`)
L0 `out_proj` Q3 2D 2×32: `0.9585250366004359` (`kernel.class=CHEAP_REWRITE`)
L47 `o_proj` Q4 g=64: `0.9897967539`; Q4 g=128: `0.9877089928`

### 12.3 Production kernel (repo)

`crates/hawking-core/shaders/qwen_uniform_q4.metal:19-20`:

```
constant uint QWEN_UNIFORM_Q4_GROUP_SIZE = 64u;
constant uint QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP = 32u;
```

`crates/hawking-core/shaders/qwen_uniform_q4.metal:199-210`:

```
const uint lane_in_row = split * 32u + simd_lane;
const uint row = group_id * 2u + team;
...
for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
    const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
    const float scale = float(scales[rgb]);
    const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
    acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
}
```

`crates/hawking-core/src/model/qwen38_geometry.rs:31-52`: `LINEAR_VALUE_HEAD_DIM=128`, `GQA_HEAD_DIM=256`, `O_PROJ_COLS=6144`.

`crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs:3-6,15-18`: HQ30UQ4, f16 scale per 64, 32 code bytes/group.

`g1-out-proj-forensics.md:48`: “g=64 is already head-aligned … g=48 … is *better* (0.9602)”
`g1-out-proj-forensics.md:165-174`: slot ratio 1.014; g48=0.9602, g64=0.9531.
`g1-artifact-inventory.md:127-135`: complete BPW 4.252735126866492.
`g1-bit-budget-accounting.md:159`: `4.0 + 16.0 / GROUP_SIZE` = 4.25.

### 12.4 Wave-1 number this lane was asked not to re-derive

G0 complete BPW 4.252735126866492, 26_895_998_464 params, 402 HQ30UQ4 g=64, kernel `geo_tpr64_tg128`. Independently recomputed here from the 402 shapes; same number.

---

## Completion report

```
STATUS
MEASURED_NEGATIVE

CLAIMS
C1. L0 out_proj Q3 g=48 output cosine 0.96015278 and g=64 0.95310345. Wave-1 0.9602/0.9531 is real. Evidence: §2, JSON sanity, sweep stdout 11:27:29.
C2. g=48 > g=64 on 60/60 tensors. Typical Δ ≈ +0.002 (Q3) / +0.00038 (Q4). L0 out_proj +0.007049 is an outlier, not the mean. Evidence: §5.1.
C3. Head alignment is not the cause. g=64 slot ratio mean 1.0217 (L0 1.01410). MLP has no heads and the same sign. Gaussian X cuts L0's extra Δ in half. Evidence: §5.2, JSON head_slots / gaussian_x_q3.
C4. Complete BPW = bits + 16/g plus headers and short-group pad. G0 g=64 Q4 reconstructs to 4.252735126866492 exactly. g=48 global is 4.34656 (338/402 short). Evidence: §3, JSON model_complete_bpw_recipes.
C5. Q3 cannot match Q4 g=64 at complete BPW ≤ 4.26 (0/60). Small groups stop paying: Q3 g=8 is 5.003 BPW and still fails 0.99 on 36/60. Evidence: §4.
C6. Phase offset and M-axis help L0 out_proj only (phase 48 at g=64 → 0.95908; M g=64 → 0.96706). Other layers ~0. 2D at matched BPW is a wash. Evidence: §6–7.
C7. geo_tpr64 can cheaply consume K-axis Q4 with g%8==0 (including 32,48,64,128). M-axis and g%8≠0 cannot. g=48 is addressable; it is not a BPW win. Evidence: §8, metal:181-221.
C8. No cheap geometry is Pareto-better than Q4 g=64 on output cosine vs complete BPW. Evidence: §9.

EVIDENCE
/tmp/g1_group_partition_geometry_sweep.json
/tmp/g1_group_partition_geometry_sweep.py
sweep stdout §12.1
crates/hawking-core/shaders/qwen_uniform_q4.metal:19-20,166-221
crates/hawking-core/src/model/qwen38_geometry.rs:31-52
crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs:3-18
workspace/superwave/g1/g1-out-proj-forensics.md:48,165-174
workspace/superwave/g1/g1-artifact-inventory.md:125-149
workspace/superwave/g1/g1-bit-budget-accounting.md:154-186

CHANGES
created workspace/superwave/g1/g1-group-partition-geometry.md only

TESTS
test -s / wc -l / git status --porcelain: run at lane end, pasted in the chat completion report

RISKS
- Mixer-proxy X for out/o is not a recurrent/softmax mix (same as wave-1). Labeled.
- 256-token / 5-prompt capture can invent persistent channels. L0's extra Δ shrinks under gaussian X, which is the check.
- Q4 g=64 already sits at 0.98980 on L47 o_proj; a 0.99 bar on that cell is inside capture noise. Not a token claim.
- rss_max 9.019 GB. Live organism not touched. No GPU.

UNRESOLVED
- Generate of Q4 g=128 (the only cheap same-family BPW cut).
- Whether an output-MSE (not absmax) scale at g=64 beats absmax enough to matter; that is a scale question, not a geometry question.
- Embed/lm_head grouping (no 248320-d X in the capture).

NEXT
Do not spend a pack on g=48, phase offsets, M-axis, or 2D tiles. If a same-family BPW cut is wanted, the GPU lane can score native HQ30UQ4 g=128 vs the g=64 oracle. Bits, not grouping, is the remaining uniform-Qn lever.
```
