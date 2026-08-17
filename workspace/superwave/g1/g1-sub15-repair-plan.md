# G1 mixed-sub15-v1 Doctor repair plan

Lane: `61-sub15-repair-plan`. One new file. No GPU. No generate. No pack.
No resident touch. No tracked-file edit.

Every number is tagged **MEASURED** (this process, or an on-disk/receipt
integer), **CITED** (wave-1/2 report, not re-derived), **DERIVED** (exact
arithmetic on those integers), **PROXY** (wrong site / interpolated /
underdetermined), or **HYPOTHESIS** (ranked guess, not a finding).

A reconstruction cosine is not a generate claim. A confounded INCOHERENT
is not a floor. Binding: no repair whose production path expands to float
or Q4 before a generic GEMV.

---

## 0. What this file is for

Assume native load of mixed-sub15-v1 (catalog + codec-4 + K-complete bind)
returns INCOHERENT. Do not start a new research cycle. Apply the matching
row of the decision tree. Spend the smallest billed exception that the
signature selects.

Target shape: `1.2910781930062503` plus a protected correction. Not a
global raise to 2.0.

| letter | complete BPW | meaning |
|---|---|---|
| A | 1.291 coherent native | ideal |
| B | ≤ 1.5 coherent native | strong G1 |
| C | (1.5, 2.0] | acceptable bootstrap |
| D | > 2.0 | not the target |

1.5 slack vs the packed ledger, **DERIVED**:

```
1.5 * 26895998464 / 8 − 4340604637 = 702395075 bytes
1.5 − 1.2910781930062503          = 0.2089218069937497 BPW
```

Any single repair with `ΔBPW ≤ 0.2089218069937497` stays in B (strict
`< 1.5` needs a byte strictly under that slack).

---

## 1. Pack ledger (CITED, independently checked)

Artifact: `…/qwen38-27b/mixed-sub15-v1`.
Authority: `PACK_REPORT.json` via `g1-sub15-native-gap.md:57–67,376–386`.

```
complete_physical_bpw = 8 * 4340604637 / 26895998464
                      = 1.2910781930062503     MEASURED identity
```

This process: `8 * 4340604637 / 26895998464 == 1.2910781930062503` is
True. Class bytes sum to `4340604637` (delta 0).

| class | n | elements | bytes | physical BPW | codec |
|---|---:|---:|---:|---:|---|
| mlp.gate_proj | 64 | 5704253440 | 802177344 | 1.1250234267290902 | HGRAVB01 g128 |
| mlp.up_proj | 64 | 5704253440 | 918036000 | 1.2875108157887178 | HGRAVR02 rice_q1_rms_2pct |
| mlp.down_proj | 64 | 5704253440 | 93847197 | 0.13161714918473189 | HGRAVS01 r160_b3 |
| attention GEMV | 304 | 7237795840 | 1165098376 | 1.2877935788805008 | HGRAVR02 rice_q1_rms_2pct |
| embed | 1 | 1271398400 | 675430440 | 4.250000251691366 | HQ30UQ4 g64 |
| lm_head | 1 | 1271398400 | 675430440 | 4.250000251691366 | HQ30UQ4 g64 |
| small | 353 | 2645504 | 10584840 | 32.00853977162764 | f32v2 |
| **complete** | **851** | **26895998464** | **4340604637** | **1.2910781930062503** | |

`N`, class elements, Qn/binary formulas: `g1-bit-budget-accounting.md:90–144,154–232`
and `qwen38_geometry.rs:20–52`. Embed and lm_head are the G0 oracle
inodes (`g1-sub15-native-gap.md:190–191`). They are not a first-break
organ.

MLP organs are byte-identical to mixed-2p0-v1 (`g1-sub15-native-gap.md:170–184`).
Attention is the composition delta.

Both prior INCOHERENT verdicts are **confounded** (contract; native-gap
§8.5). They are signature *priors*, not floors.

- expand-to-Q4 sub15: period-2 `{220, 264}` = space / ` a`
  (`QWEN38_SUB15_INCOHERENT.json`).
- native mixed-2p0 (also `*_tg256` K-truncated on gate/up): `{198, 8}` =
  newline / `)` (`QWEN38_NATIVE_MIXED_2P0_GENERATE.json`).
- G0 Q4 oracle, same harness: first id `248068` (`<think>`),
  `QWEN38_COHERENCE_SEAL.json`.

`Qwen3.8` EOS ids: `248046` (`im_end`), `248044` (`end_of_text`)
(`qwen38_geometry.rs:57–58`).

---

## 2. Precondition — H0 kernel K-truncation (not a codec)

Production HGRAVB01 / HGRAVR02 tiles cover 2048 columns.
Every such GEMV in this pack has `K ∈ {5120, 6144}`
(`g1-sub15-native-gap.md:216–241`; `q80_mixed_decode.metal:722–728`).
0 fallbacks does not mean the tile consumed K.

If C3 (or `HAWKING_QWEN38_RECON_FUSE=0`) is missing, the generate is
not a codec verdict. Repair: 0 bytes, 0 BPW. Re-run. Do not spend bits.

Discriminator: same garbage on every prompt, 0 fallbacks, 0 dense-W,
and the binary is still bound to `*_tg256`. After C3/fuse-off the
signature is allowed to change; re-classify from scratch.

---

## 3. Exception representation (what “protected correction” means)

Two legal vehicles. Both stay in-register. Neither expands the host
tensor to float/Q4.

### 3.1 Sidecar `H38EX01` — overlay exact f16 rows or columns

Billed in complete BPW (payload is a loaded weight artifact).
Catalog `.hq38m20` is **not** billed (`qwen38_pack.rs:673–679`).
mixed-2p0 catalog tax was 184307 B = 0.00005482 BPW if someone bills
artifact-complete (`g1-bit-budget-accounting.md:250–256`). HQ38M20 at
851 × 128 = 108928 B = **DERIVED** 0.0000323997638967 BPW. Ignore
unless using artifact-complete.

```
header   32 B   magic 8 + u32 ver + u32 nrec + u32 reserved
record   12 B   u32 tensor_index + u16 kind + u16 n_idx + u32 n_val_bytes
index     2 B   per u16 (row id or column id)
values    2 B   per f16
pad       0–15  to 16
```

`kind=0` exact output row: 1-row GEMV + add into residual at that row.
`kind=1` exact input columns: skinny `[out, k]` GEMV + add into Y.
Existing Qn / f32 row kernels. No new shader family.

Punch-hole credit on rice out/o for one row of 6144 is
`64 * 6144 * 1165098376 / 7237795840 ≈ 63298` B = **DERIVED**
−0.000018827 BPW. HGRAVS01 cannot punch a row (factors are not a
row store). **Bill overlay. Do not take the punch-hole discount.**

f16 vs f32: f32 doubles the value plane and is not earned. f16 matches
the island costing in `g1-sparse-exact-islands.md:87` and the Qn scale
type.

### 3.2 Whole-tensor codec swap to HQ30UQ4 / HGRAVB01

No sidecar. Catalog codec 3 or 0. Header 40 B/tensor is inside the
Qn formula. Formula **MEASURED** equal to G0 down class bytes
(`3030387200`) and to sub15 gate bytes (`802177344`):

```
qn_bytes(n, bits, T) = 40*T + n*bits/8 + 2*(n/64)     n % 64 == 0
bin_bytes(n, T)      = n/8 + 2*(n/128) + 261*T
```

(`uniform_q4.rs:15–18`; `g1-bit-budget-accounting.md:154–199`).

Subset rice replacement old-bytes, **DERIVED** (per-file rice sizes
are not on this worktree):

```
old = floor(e_subset * 1165098376 / 7237795840)
```

Rounding vs the exact share is < 1 B. `8/N ≈ 3×10^{-10}` BPW. Floor
used below so every `Δbytes` is an integer.

Subset HGRAVS replacement old-bytes:

```
old = floor(k * 93847197 / 64)
```

HGRAVS payloads are not equal per layer (activation-weighted). The
29-byte remainder across 64 tensors is not allocated. Label DERIVED.

`ΔBPW = 8 * Δbytes / 26895998464`
`complete' = 8 * (4340604637 + Δbytes) / 26895998464`

---

## 4. Ranked hypotheses (what breaks first)

Ranking is **HYPOTHESIS**. It is conditioned on C3/fuse-off already
being in the vehicle. It uses pack composition, reconstruction, write
gain, and the two confounded signatures as priors. It does not claim
a generate floor.

Two mechanisms that must not be collapsed (`g1-out-proj-forensics.md:17–33`;
contract):

- `|X|` / residual energy is a signal for **what to protect**.
- `|X|` as a **rescale** of W (act_colscale) **KILLS**: L0 out_proj
  Q4 cosine 0.9922374383267348 → 0.9186496062432181
  (`QWEN_ATTENTION_DENSITY_PROBE.json`; `g1-doctor-recovery.md:619–630`).

`|W|` and `|X|` hottest-42 columns on L0 out_proj overlap **0 / 42**.
Exact `|W|` 42 cols: 0.95331 (no move). Exact `|X|` 42 cols: 0.97616
(`g1-out-proj-forensics.md:31–33,370–377`).

Doctor 256-token cube: `rows_per_dim` 0.0500 / 0.0417 / 0.0147. Worse
than NS-014 0.0449 for any fit of those dims. 1-D channel RMS is
determined (`fit_dim=1`, rpd=256). Channel 3994 as a *statistic* is
legal. AWQ / act_colscale / 58-layer interp is not
(`g1-doctor-recovery.md:275–288,429–448`).

1-bit and 2-bit doctor rungs are the same RTN operator (`qmax`
clamped to 1). Six `sensitivity_bits=1` rows are mislabeled 2-level
holds (`g1-doctor-recovery.md:161–173`). Do not spend or withhold
bits on a “1-bit floor”.

### H1 — attention write rice (first codec break)

| field | value |
|---|---|
| organ | residual write |
| tensor class | `linear_attn.out_proj` (48) + `self_attn.o_proj` (16) |
| layer band | all 64; early kurtotic L0/L1/L3/L4/L7/L8/L11 and late write-gain L47–63 first |
| channel | output row **3994** (top-5 on all 128 writes); input-side a different axis (H5) |
| scale rule | rice `value_scale=rms` on a 2% residual over binary g128. Absmax of W is the wrong axis. |
| error mode | residual-write MSE, amplified by write/R (L0 0.353, L63 o 1.845) |

Why first: rice weight cosine 0.834–0.847 vs BF16 is **MEASURED** pack
metadata on all 304 (`g1-sub15-native-gap.md:147–168`). Write tensors
are the only large rice class that injects into the residual. L0 out
Q3 mixer-output is already 0.9531; rice sits with Q2 0.7063
(`g1-out-proj-forensics.md:222,246`). Attention is not higher entropy
than MLP (7.5915 vs 7.6162). The failure is the metric and the write,
not entropy (`g1-out-proj-forensics.md:138–155`). This class is also
the only large composition delta vs mixed-2p0.

**Minimal correction (try in this order):**

| step | vehicle | Δbytes | ΔBPW DERIVED | complete' | target |
|---|---|---:|---:|---:|---|
| H1a | sidecar f16 row 3994 on all 128 writes | 3016480 | 0.0008972278918108 | 1.2919754208980609 | A/B |
| H1b | + HQ30UQ4 on 4 probed writes (L0/L32 out, L3/L63 o) | 49608119 | 0.0147555389152479 | 1.3058337319214981 | A/B |
| H1c | + HQ30UQ4 on 7 10× `o`/`lin_o` (L0,1,3,4,7,8,11) | 84551848 | 0.0251492721084653 | 1.3162274651147154 | A/B |
| H1d | HQ30UQ4 all 16 GQA `o` | 186366554 | 0.0554332434988646 | 1.3465114365051147 | B |
| H1e | HQ30UQ4 all 64 writes | 745466215 | 0.2217329736980163 | 1.5128111667042665 | C |

H1a arithmetic (overlay, no punch-hole):

```
values = 64*17408*2 + 64*6144*2 = 2228224 + 786432 = 3014656
index  = 128*2 = 256
meta   = 128*12 = 1536
header = 32
pad    = 0
Δbytes = 3016480
ΔBPW   = 8*3016480 / 26895998464 = 0.0008972278918108
```

H1b incremental Q4 (4 writes):

```
e      = 4*31457280 = 125829120
qn4    = 40*4 + 125829120*4/8 + 2*(125829120/64) = 66846880
old    = floor(125829120*1165098376/7237795840) = 20255241
Δbytes = 46591639
```

H1a **earns**: 3994 is activation-hot ≥10× in 54/64 layers (mean RMS
14.19) and the top-5 output row of all 128 write tensors; L0 lin_o
kurtosis 149.36, out-row 3994 = 20.70×
(`g1-doctor-tensor-map.md:53–71`). Mass 0.008% of params if every
down row is kept (`g1-doctor-tensor-map.md:154`). This is the
residual-stream island, not the killed `|W|`-column patch.

H1b **earns**: Q4 is the last cheap codec that clears 0.99
mixer-output on every probed write (`g1-out-proj-forensics.md:222–237`).
4 tensors / 64 is the probed worst set.

H1e does **not** earn as a first spend: it crosses 1.5 and recreates
the write half of mixed-2p0’s Q4 attention. Hold for a signature that
survives H1a–d.

Sparse-exact-islands as a *bulk* attention path is **KILLED**
(`g1-sparse-exact-islands.md` STATUS FALSIFIED). H1a is one output
row, not a 1% `|W|` island. Do not reopen the 1% path.

### H2 — down_proj HGRAVS01 r160_b3

| field | value |
|---|---|
| organ | MLP write |
| tensor class | `mlp.down_proj` × 64 |
| layer band | late L47–63 first (L63 binary hold 0.7297; write/R 2.792). L0 is the easy end (Q3 out 0.9915) and the only 10× down (11.47× on 3994). |
| channel | output row 3994 (L0 11.47×; top-5 all 64 downs) |
| scale rule | `activation_weighted_svd_low_rank_q` rank 160 bits 3 g64. Fit on the 256-token cube, `rpd = 256/17408 = 0.0147`. NS-014 class. |
| error mode | low-rank factor miss on the residual write; capability, not just cosine |

Why second: same bytes as mixed-2p0. Isolated hold_output_rel_l2 was
**never scored** (`g1-sparse-exact-islands.md:433`; hetero.md:184).
0.1316 BPW is 4.044% of the matrix as 3-bit factors
(`g1-bit-budget-accounting.md:216–224`). Late down is the hard organ
on every screen. The generate prior `{198,8}` is shared with
mixed-2p0 — but that run is K-confounded on gate/up, so it does not
prove H2.

**Minimal correction:**

| step | vehicle | Δbytes | ΔBPW | complete' | target |
|---|---|---:|---:|---:|---|
| H2a | H1a island (covers L0 down 3994) | 3016480 | 0.0008972278918108 | 1.2919754208980609 | A/B |
| H2b | L63 down → HQ30UQ4 | 45883438 | 0.0136476622904078 | 1.3047258552966581 | A/B |
| H2c | 16 late downs (L48–63) → Q3 | 555877081 | 0.1653411995078853 | 1.4564193925141355 | B |
| H2d | 20 late downs → Q3 (1.5 max) | 694846351 | 0.2066764993104961 | 1.4977546923167462 | B |
| H2e | all 64 downs → binary g128 | 708330147 | 0.2106871467733290 | 1.5017653397795792 | C |
| H2f | all 64 downs → Q3 | 2223508323 | 0.6613647977340991 | 1.9524429907403493 | C |
| H2g | all 64 downs → Q4 | 2936540003 | 0.8734503779602834 | 2.1645285709665334 | D |

H2c arithmetic:

```
qn3(16) = 40*16 + 16*89128960*3/8 + 2*(16*89128960/64) = 579338880
old     = floor(16*93847197/64) = 23461799
Δbytes  = 555877081
```

H2e is 0.0018 BPW over 1.5. Do not take it to “sit on 1.5”. H2c
spends 79% of the 1.5 slack and stays in B.

H2g does **not** earn: it is the mixed-q4down down-half at a worse
complete than mixed-2p0 because attention stays rice
(`g1-doctor-tensor-map.md:219` mixed-q4down = 2.959, Q4 attn).

H2-scale-only (re-fit r160_b3 **weight-only**, same rank/bits):
Δbytes ≈ 0. Legal as a 0-BPW experiment. Not an allocator input
until `n_fit ≥ 160` on real `post_swiglu` X (`g1-doctor-recovery.md:312–315`).
Current 256-row act-weighted fit is UNDERDETERMINED. **REOPEN_IF** a
determined weight-only or output-MSE r160 is packed and a native
generate of *only* that change is scored.

### H3 — DeltaNet in_proj rice (qkv / z)

| field | value |
|---|---|
| organ | DeltaNet recurrence input |
| tensor class | `in_proj_qkv` 10240×5120 (48), `in_proj_z` 6144×5120 (48) |
| layer band | mid L32 worst (binary hold 0.7411; 3% exact still 0.8620) |
| channel | none. Platykurtic. Bulk energy (`g1-sparse-exact-islands.md:186–217`). |
| scale rule | rice_q1_rms_2pct, g128 |
| error mode | recurrence state / `v*silu(z)` collapse → cycle |

Why third: same 0.83 cosine, but mixed-2p0 kept these at Q4 and still
emitted `{198,8}`. If native sub15 after C3 matches mixed-2p0, H3 is
not the first break. If it matches expand-sub15 `{220,264}`, H3 and
H1 move together (both rice, both new vs mixed-2p0).

**Minimal correction:** do not lift all 96. First H1a. Then L32
`in_proj_qkv` → Q4:

```
e      = 52428800
qn4    = 40 + 52428800*4/8 + 2*(52428800/64) = 27852840
old    = floor(52428800*1165098376/7237795840) = 8439683
Δbytes = 19413157
ΔBPW   = 0.0057742885510599
complete' = 1.2968524815573101
```

All 48 qkv → Q4: Δbytes 931831489, ΔBPW 0.2771658364711007,
complete' 1.5682440294773510 (**C**). All qkv+z → Q4: ΔBPW 0.4435,
complete' 1.7345 (**C**). Those do not earn as a first spend.

### H4 — mlp.gate HGRAVB01 + mlp.up HGRAVR02

| field | value |
|---|---|
| organ | SwiGLU |
| tensor class | gate 64 + up 64 |
| layer band | mid L15–47. L31 up binary hold 0.7639. L63 gate/up binary 0.954/0.960 is a last-layer exception, do not average (`g1-doctor-tensor-map.md:133–143`). |
| channel | none. Kurtosis 0.07–0.51. Zero 4× output rows (`g1-doctor-tensor-map.md:79`). |
| scale rule | gate: sign × group mean-abs g128. up: rice_q1_rms_2pct. |
| error mode | SwiGLU bulk. Semantic / reasoning, or cycle if the residual is destroyed. |

Why fourth: shared with mixed-2p0. Mid-depth binary is not
quality-intact on the descent screen. L0 “1-bit hold” is a
space/newline id hold on a one-unit ablation
(`g1-doctor-recovery.md:175–184,595–601`). mixed-2p0 is the joint
counterexample, confounded.

**Minimal correction:** 3 mid ups (L15, L31, L47) → Q3.

```
qn3(3) = 40*3 + 3*89128960*3/8 + 2*(3*89128960/64) = 108626040
old    = floor(3*918036000/64) = 43032937
Δbytes = 65593103
ΔBPW   = 0.0195101447786876
complete' = 1.3105883377849379
```

16 mid ups → Q3: Δbytes 349829880, ΔBPW 0.1040541046931553,
complete' 1.3951322976994054 (B). All 64 ups → Q3: ΔBPW 0.4162,
complete' 1.7073 (C). All 64 gates → Q3: ΔBPW 0.4507, complete' 1.7418
(C). Do not lift all gates first: they are the cheaper-looking organ
and the vacuous L0 hold.

### H5 — `|X|`-hot *input* columns on out_proj / o_proj

| field | value |
|---|---|
| organ | mixer write, input axis |
| tensor class | same 64 writes as H1 |
| layer band | L0 DeltaNet first (50% energy in 16/6144 cols; 90% in 42). L3 GQA is not a spike (50% in 29.2% of cols). |
| channel | the 42 mixer-site columns, **disjoint** from the 42 fattest `|W|` columns |
| scale rule | protect, do not rescale |
| error mode | mixer-output MSE on a near-one-hot write |

Distinct from H1a (output *row* 3994). Both can be true.

**PROXY.** mixer_x was never captured. The 42-set is
`v*silu(z)` / `repeat(v)*sigmoid(q_gate)`
(`g1-out-proj-forensics.md:265–284`; `g1-doctor-recovery.md:354`).
REOPEN_IF a real 6144-wide mixer_x moves the set.

| step | vehicle | Δbytes | ΔBPW | complete' | target |
|---|---|---:|---:|---:|---|
| H5a | f16 42 cols on 4 probed writes | 1720736 | 0.0005118191844941 | 1.2915900121907442 | A/B |
| H5b | f16 42 cols on all 64 writes | 27531296 | 0.0081889641797386 | 1.2992671571859888 | A/B |
| H5c | H1a + H5a | 4737216 | 0.0014090470763049 | 1.2924872400825551 | A/B |
| H5d | H1a + H5b | 30547776 | 0.0090861920715493 | 1.3001643850777995 | A/B |

H5a arithmetic:

```
values = 4*42*5120*2 = 1720320
index  = 4*42*2 = 336
meta   = 4*12 = 48
header = 32
Δbytes = 1720736
```

H5 **earns a local cosine move** (+0.0231 on L0 out Q3) and **does
not earn Q4-class mixer-output** (needs 3555/6144 cols for 0.9950,
+9.26 tensor-BPW — dead). Residual-proxy on the same Ŵ is already
0.99745 (`g1-out-proj-forensics.md:241–261`). Spend H5 only if the
signature is DeltaNet-early mixer collapse after H1a failed.

### H6 — late write-gain band (compound H1+H2, late only)

| field | value |
|---|---|
| organ | residual, both MLP and attention |
| tensor class | late `down_proj` + late `o_proj` / `out_proj` |
| layer band | L47–63. L63 o write/R=1.845 residual-proxy Q3=0.97749. L63 down write/R=2.792 residual-proxy Q3=0.97324. |
| channel | 3994 still; no extra island |
| scale rule | absmax / HGRAVS, both blind to write gain |
| error mode | residual compounding. Reasoning collapse (deep residual), not token-0 cycle. |

H1e+H2c without the island:

```
late16 down Q3 + 16 GQA o Q4
Δbytes = 555877081 + 186366554 = 742243635
ΔBPW   = 0.2207744430067499
complete' = 1.5118526360130000   C, 0.0119 over 1.5
```

Prefer the island-first combo that stays in B:

```
H1a + 4 write Q4 + 16 late down Q3
Δbytes = 3016480 + 46591639 + 555877081 = 605485200
ΔBPW   = 0.1800967384231332
complete' = 1.4711749314293834   B
```

H1a + L63 down Q4 + 4 write Q4: Δbytes 95491557, ΔBPW 0.0284032012056557,
complete' 1.3194813942119059 (B). Cheaper probe of the same band.

### H7 — DeltaNet `in_proj_a` / `in_proj_b` rice

| field | value |
|---|---|
| organ | DeltaNet decay / β |
| tensor class | a 48×5120 (48), b 48×5120 (48). Mass 0.088%. |
| layer band | all 48 DN. Kurtosis a ≤ 11.12, b ≤ 9.44 (`g1-doctor-tensor-map.md:108–109`). |
| channel | none worth a list |
| scale rule | rice_q1_rms on a 48-row matrix |
| error mode | decay → 0 ⇒ early EOS; decay → 1 ⇒ runaway / cycle |

`A_log` / `dt_bias` are already f32. a/b are the only crushed decay
path. Cheap enough to fire on an EOS signature without a research
cycle.

```
e      = 23592960
qn4    = 40*96 + 23592960*4/8 + 2*(23592960/64) = 12537600
old    = floor(23592960*1165098376/7237795840) = 3797857
Δbytes = 8739743
ΔBPW   = 0.0025995667754660
complete' = 1.2936777597817162
```

H1a+H7: Δbytes 11756223, ΔBPW 0.0034967946672768, complete' 1.2945749876735269.

### H8 — GQA q-gate / k / v rice

| field | value |
|---|---|
| organ | GQA (16 layers, `(layer+1)%4==0`) |
| tensor class | `q_proj` 12288×5120 includes `attn_output_gate` (`qwen38_geometry.rs` + census). `k`,`v` 1024×5120. |
| layer band | GQA only. L63 q/v Q3 already clear 0.99 (0.9909 / 0.9925); L63 k Q3 0.9633 does not (`g1-doctor-tensor-map.md:186–189`). |
| channel | none 10×. Late k/v kurtosis 6.06 / 6.52 is richer-levels, not an island. |
| scale rule | rice_q1_rms |
| error mode | GQA-only: code collapse, tool syntax, gated output |

| step | Δbytes | ΔBPW | complete' |
|---|---:|---:|---:|
| 16 `o` → Q4 (also H1d) | 186366554 | 0.0554332434988646 | 1.3465114365051147 |
| 16 `q` → Q4 | 372732468 | 0.1108662966348391 | 1.4019444896410893 |
| 16 `k` → Q4 | 31061626 | 0.0092390326513665 | 1.3003172256576168 |
| 16 `v` → Q4 | 31061626 | 0.0092390326513665 | 1.3003172256576168 |
| H1a + o + q | 562115502 | 0.1671967680255144 | 1.4582749610317647 |

k and v are cheap. Lift k before q if the signature is GQA-only and
o already held. q is 12× k in mass because of the gate half.

### H9 — lm_head / embed calibration

Already HQ30UQ4, same inodes as the coherent G0 oracle
(`g1-sub15-native-gap.md:186–191`). lm_head is 4.96% of active bytes
and 2.66% of token time (`g1-lm-head-and-tails.md:11–22`). It is a
BPW lever only if someone tries to crush it; it is not a first-break
organ at 1.291.

Doctor lm_head “floor 8” is a mid-prompt id flip at 4 bits
(last-agree 1.0, all-agree 0.9, logit_rel 0.0459)
(`g1-doctor-recovery.md:603–609`). Not a generate floor.

**Do not lift lm_head or embed.** If the signature is calibration
collapse, the residual *into* lm_head moved (H1/H2/H6). Repair the
write path.

Q3 lm_head would *save* `1271398400 * (4.25−3.25) / 26895998464 =
0.04727` BPW and is untested at generate. Out of scope for a repair
that assumes incoherence of the 1.291 recipe.

### H10 — group geometry g=128 rice

g=64 is head-aligned on 6144. Intra-head slot error ratio 1.014.
Straddling g=48 was *better* (Q3 out 0.9602 vs 0.9531)
(`g1-out-proj-forensics.md:157–174`). Rice is tied to binary g128.
Regrouping is a repack, not a protected correction. **KILLS as a
first repair.** REOPEN_IF a native generate of identical bits at g=64
or g=48 is the only remaining delta.

### H11 — hetero 2.0 / 1.5 / 1.2 tables as the repair

**KILLS.** Interpolates 58/64 layers. Scores `attn_out` in weight
space. Uses Gravity GPT-OSS priors. 1-bit rung is the 2-bit operator
(`g1-doctor-recovery.md:260–288`; `g1-heterogeneous-allocation.md:68–102`).
REOPEN_IF the §5 capture in `g1-doctor-recovery.md` passes the §6
adequacy gate and a native generate of that table is scored.

---

## 5. Signature taxonomy

Operational, from one `ascension_qwen38_hybrid_greedy` battery.
Use the six mixed-2p0 prompts plus the G0 seal prompts. Judge against
the G006 harvest bar, **not** id-identity to the Q4 seal
(`HARVEST_NOTE_G006.json`): well-formed text, no cycle, France→Paris,
0 fallbacks, 0 dense-W. Report greedy-id drift vs
`QWEN38_COHERENCE_SEAL.json` as a signal, never as pass/fail.

| signature | operational test | measured prior |
|---|---|---|
| **degenerate cycle** | period ≤ 4 over ≥ 8 new tokens; or ≥ 50% of new ids in a 2-id set | expand-sub15 `{220,264}`; mixed-2p0 `{198}`, `{8}`, `{1076,8}`, `{578}` |
| **early EOS** | a new token in `{248046, 248044}` in the first 4, or length < 3 before EOS | not observed on the two confounded runs |
| **semantic collapse** | well-formed English; France ≠ Paris (or equivalent fact fail) | unmeasured at 1.291 native |
| **reasoning collapse** | first id ≠ `248068` on the seal prompts, or empty `<think>…</think>` then salad | G0 first id is `248068` on all 3 seal prompts. mixed-2p0/sub15 never emit it |
| **code collapse** | prose/fact OK; `reverse string` / `def fibonacci` is salad or unmatched parens | mixed-2p0 code prompt was `......)...)` — not isolated, whole-model cycle |
| **tool syntax collapse** | prose OK; JSON / tool-call / unmatched `()` `{}` | mixed-2p0 emitted `)` as a cycle id. Not a clean tool fail |
| **calibration collapse** | fluent, facts close, no cycle; greedy-id drift vs seal; logit scale off | doctor L31/L47: cosine 0.999941, last-agree 0.8. Not generate |

A cycle that includes `248068` then a 2-id loop is still a cycle.
Reasoning collapse without a cycle is the more specific label.

---

## 6. Signature → organ classes

Plausible, not proven. One generate can hit more than one row.
Apply the **first matching** row in §7.

| signature | plausible organs (most → least) | implausible |
|---|---|---|
| degenerate cycle `{220,264}` | H1 write rice; H3 DN in_proj rice; H0 if C3 missing | H9 (lm_head already oracle); H10 |
| degenerate cycle `{198,8}` | H2 down HGRAVS; H4 gate/up; H0 on gate/up tiles | H1-only (mixed-2p0 had Q4 attention) |
| degenerate cycle other 2-id | H0; H2 rank collapse; H7 decay=1 | H9 |
| early EOS | H7 a/b; H1a 3994 (residual death); H3 z | H9 lift; H4 (bulk SwiGLU rarely EOS) |
| semantic collapse | H1a 3994; H2 late down; H6 write-gain; H5 if DN-only facts die | H7 (would EOS or run away) |
| reasoning collapse | H6 late write-gain; H2c late down; H1d late o; H8 GQA | H7; H5 (L0-specific) |
| code collapse | H8 GQA q/o/k/v; H4 mid up; H1 GQA o | H5 L0-only; H9 |
| tool syntax collapse | H8 q-gate; H1d GQA o; H7 if `)`-cycle | H9 lift |
| calibration collapse | H1/H2 residual into an already-Q4 lm_head; final-norm is f32 | lifting lm_head |

`conv1d`, `A_log`, `dt_bias`, all RMSNorms are already f32
(`g1-doctor-tensor-map.md:203–207`). They are not in this tree.

---

## 7. Decision tree (one generate → one repair)

Preconditions, in order. Stop at the first fire.

```
0  fallbacks>0 or dense_w>0
     → not a codec verdict. Fix loader. +0 BPW

1  C3 missing and HAWKING_QWEN38_RECON_FUSE≠0
     → H0. Bind K-complete or fuse-off. Re-run. +0 BPW

2  early EOS
     → H7 a/b → Q4
       Δbytes=8739743  ΔBPW=0.0025995667754660  complete'=1.2936777597817162
     still EOS → + H1a 3994
       Δbytes=11756223 ΔBPW=0.0034967946672768  complete'=1.2945749876735269

3  degenerate cycle, id-set ∩ {220,264} nonempty
     → H1a 3994 f16
       Δbytes=3016480  ΔBPW=0.0008972278918108  complete'=1.2919754208980609
     still cycle → H1b + 4 write Q4
       Δbytes=49608119 ΔBPW=0.0147555389152479  complete'=1.3058337319214981
     still cycle → H1c + 7 10× o Q4
       Δbytes=84551848 ΔBPW=0.0251492721084653  complete'=1.3162274651147154
     still, and cycle is DN-early only → H5a 42-col on 4 writes
       (with H1a) Δbytes=4737216 ΔBPW=0.0014090470763049
       complete'=1.2924872400825551   [cheaper than H1b; use if H1b not yet applied]
     still → H1d 16 GQA o Q4
       Δbytes=186366554 ΔBPW=0.0554332434988646  complete'=1.3465114365051147
     still → STOP and treat as compound H1+H3.
       Next legal spend is H3 L32 qkv Q4
       (Δbytes=19413157 ΔBPW=0.0057742885510599 complete'=1.2968524815573101)
       then H2c, not H1e.

4  degenerate cycle, id-set ∩ {198,8} nonempty, {220,264} empty
     → H1a 3994 (covers L0 down 11.47×)
       Δbytes=3016480  ΔBPW=0.0008972278918108  complete'=1.2919754208980609
     still → H2b L63 down Q4
       Δbytes=45883438 ΔBPW=0.0136476622904078  complete'=1.3047258552966581
     still → H2c 16 late down Q3
       Δbytes=555877081 ΔBPW=0.1653411995078853  complete'=1.4564193925141355
     still → H4 3 mid up Q3
       Δbytes=65593103 ΔBPW=0.0195101447786876  complete'=1.3105883377849379
       (apply on top of H1a, not instead of H2c if H2c already fired)
     still → STOP. Do not take H2e (1.5018, C) or H2f (1.952, C)
       until a second generate with H2c applied is still {198,8}.

5  reasoning collapse (no cycle, no 248068 / empty think)
     → H1a + L63 down Q4 + 4 write Q4
       Δbytes=95491557 ΔBPW=0.0284032012056557  complete'=1.3194813942119059
     still → H1a + 4 write Q4 + 16 late down Q3
       Δbytes=605485200 ΔBPW=0.1800967384231332  complete'=1.4711749314293834
     still → H1d GQA o Q4 on top if not already
       (H1a+o+q = 1.45827 if q also needed)

6  semantic collapse (think present or English present; France wrong)
     → H1a 3994
       complete'=1.2919754208980609
     still, DN prompts fail more than GQA → H5a
       complete'=1.2924872400825551
     still → H2b L63 down Q4
       complete'=1.3047258552966581
     still → H2c
       complete'=1.4564193925141355

7  code collapse only
     → 16 GQA k Q4 + 16 GQA v Q4
       Δbytes=62123252 ΔBPW=0.0184780653027331  complete'=1.3095562583089833
     still → + 16 GQA o Q4
       Δbytes=248489806 ΔBPW=0.0739113088015976  complete'=1.3649895018078477
     still → + 16 GQA q Q4  (H1a optional)
       H1a+o+q complete'=1.4582749610317647

8  tool syntax collapse only
     → H7 a/b Q4 if `()`-shaped
       complete'=1.2936777597817162
     else 16 GQA q Q4 (gate half)
       complete'=1.4019444896410893
     do not touch lm_head

9  calibration collapse (fluent, facts close, id drift, no cycle)
     → H1a only
       complete'=1.2919754208980609
     still → H6 cheap (H1a+L63 down Q4+4 write Q4)
       complete'=1.3194813942119059
     do not lift lm_head or embed

10 two or more of {cycle, reason, code} after a first repair
     → compound B-legal stack, stop at 1.5:
         H1a + 4 write Q4 + 16 late down Q3
         complete'=1.4711749314293834
     if that fails, the 1.291 body is not repairable by a small
     exception. Next honest artifact is not “raise everything to 2.0”.
     It is a determined 64-layer eval (capture N=23216, doctor-recovery
     §5) or mixed-q3mlp native generate (3.614, already packed, no
     generate). Both are outside this tree’s first apply.
```

Do not jump to H1e (all writes Q4, 1.5128) or all-attn Q4 (2.0882,
mixed-2p0) from a single failed H1a.

---

## 8. Ladder vs A/B/C/D

Cheapest executable artifacts this plan is willing to emit, in
increasing complete BPW. Each line is one apply, native generate,
then stop if coherent.

| # | apply | complete' DERIVED | letter |
|---:|---|---:|---|
| 0 | C3 / fuse-off only | 1.2910781930062503 | A |
| 1 | H1a 3994 f16 overlay | 1.2919754208980609 | A/B |
| 2 | + H5a 42-col on 4 writes | 1.2924872400825551 | A/B |
| 3 | H7 a/b Q4 | 1.2936777597817162 | A/B |
| 4 | H1a + H7 | 1.2945749876735269 | A/B |
| 5 | H1a + H5b 42-col × 64 | 1.3001643850777995 | A/B |
| 6 | H2b L63 down Q4 | 1.3047258552966581 | A/B |
| 7 | H1b H1a+4 write Q4 | 1.3058337319214981 | A/B |
| 8 | H8 k+v Q4 | 1.3095562583089833 | A/B |
| 9 | H4 3 mid up Q3 | 1.3105883377849379 | A/B |
| 10 | H1c 3994+7 10× o Q4 | 1.3162274651147154 | A/B |
| 11 | H6 cheap 3994+L63 down Q4+4 write Q4 | 1.3194813942119059 | A/B |
| 12 | H1d 16 GQA o Q4 | 1.3465114365051147 | B |
| 13 | H2c 16 late down Q3 | 1.4564193925141355 | B |
| 14 | H1a+4 write Q4+16 late down Q3 | 1.4711749314293834 | B |
| 15 | 20 late down Q3 (H2d) | 1.4977546923167462 | B |
| — | 1.5 line | 1.5000000000000000 | — |
| 16 | all 64 down binary (H2e) | 1.5017653397795792 | C |
| 17 | all 64 writes Q4 (H1e) | 1.5128111667042665 | C |
| 18 | all attn Q4 (mixed-2p0 attn) | 2.0882206608977891 | D |
| 19 | all down Q4, attn stays rice | 2.1645285709665334 | D |

#0–15 are the plan. #16–19 are listed so nobody “rounds up to 2.0”
and calls it the repair.

How many organs fit in the 702395075-byte slack, **DERIVED**:

| swap | bytes each | max n in slack |
|---|---:|---:|
| one down → Q3 | 34742318 | 20 |
| one down → Q4 | 45883438 | 15 |
| one down → binary | 11067659 | 63 (all 64 overflows by 5.9 MB) |
| one write → Q4 | 11647910 | 60 |
| one write → Q3 | 7715750 | 91 |

---

## 9. KILLS and REOPEN_IF

| item | status | REOPEN_IF |
|---|---|---|
| Global raise 1.291 → 2.0 | KILLS as this repair | a native generate of #1–15 all fail and mixed-q3mlp (3.614) is also incoherent |
| act_colscale / AWQ from the 256-token cube | KILLS | mixer_x captured, `n_fit ≥ 6144`, and a hold score beats unscaled Qn |
| `|W|` outlier extract as the write patch | KILLS | a new selector not `|W|` and not `|W−Q(W)|` (`g1-sparse-exact-islands.md` REOPEN) |
| 1% sparse-exact islands on attention | KILLS | see that file |
| 1-bit doctor floors as allocator input | KILLS | a real 1-bit quantizer, `qmax` not clamped onto 2-bit |
| hetero 2.0/1.5/1.2 tables | KILLS | doctor-recovery §5–§6 |
| generator+residual, VQ, entropy coding | KILLS | contract dead-family rules; unchanged premise |
| expand-to-Q4 generate vehicle | KILLS | complete-token measurement shows a net physical win |
| Conclude a floor from expand-sub15 or native-2p0 | KILLS | C3/fuse-off native of *this* pack |
| g=64 regroup as first repair | KILLS | H10 |
| Lift embed/lm_head | KILLS as a first repair | isolated Q3 table generate vs G0 seal, GPU lane |
| HGRAVS01 transfer onto attention | KILLS | density probe SVD/HGRAVS on attn 0.66–0.91 (`g1-doctor-tensor-map.md:191`) |
| Punch-hole discount on HGRAVS rows | KILLS | a row-factor encoding exists |

Negative results stay first-class. A family disproved under one
construction is not every mechanism sharing its name.

---

## 10. What is unavailable, and the cheapest experiment

Unavailable here (sparse checkout + GPU lock):

- Per-file rice byte sizes (class total only).
- Isolated hold of HGRAVS01 down.
- True mixer_x (6144) and post_swiglu (17408).
- Native generate of this pack.
- mixed-q3mlp / mixed-q4down generate (packed, never run).

Cheapest experiment that turns this tree from a plan into a
measurement, **GPU lane only**:

1. Land C1+C2+C3 (or C1+C2+fuse-off). Estimated 250–500 lines,
   no new shader (`g1-sub15-native-gap.md:337–341`).
2. One greedy battery, 6 mixed-2p0 prompts + 3 seal prompts,
   `max_new_tokens ≥ 16`, 0 fallbacks required.
3. Classify with §5. Apply **one** §7 row. Do not stack.
4. One more generate. Stop if coherent. Else the next row.

Do not pause the resident. Do not run this lane’s arithmetic again.

---

## 11. Evidence

### 11.1 Complete-BPW identity (this process)

```
>>> 8 * 4340604637 / 26895998464 == 1.2910781930062503
True
>>> 802177344+918036000+93847197+1165098376+675430440+675430440+10584840
4340604637
>>> 40*64 + 5704253440*4//8 + 2*(5704253440//64)
3030387200          # G0 down class, formula match
>>> 5704253440//8 + 2*(5704253440//128) + 261*64
802177344           # sub15 gate, formula match
>>> int(1.5 * 26895998464 / 8) - 4340604637
702395075
```

### 11.2 Receipts (`git show HEAD:…`)

`receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json`:

```
RESULT: INCOHERENT. FAILS criterion 1 … degenerate cycle.
prompt_1 token_ids: 220,264,220,220,220,264,…   # space / " a"
control on 4.2527 oracle: "<think>"
fallbacks: 0
generate vehicle: HQ30UQ4 of reconstructed mixed/rice   # confound
```

`receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json`:

```
fallbacks_total: 0  dense_w_materialized_total: 0
"Say hi."            ids all 198
"What is the capital of France?"  198×15 + 8
"Write a function that reverses a string."  1076/8 mix
```

`receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json`:

```
"What is the capital of France?": [248068, 198, 760, 1156, 369, 9859, …]
"Say hi.":                        [248068, 198, 760, 1156, 4777, …]
```

`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json`:

```
4.2527_BPW_q4_oracle     COHERENT
2.0856_BPW_mixed-2p0-v1  INCOHERENT (native, verified twice)
1.2910_BPW_mixed-sub15-v1 INCOHERENT
```

Both low points confounded (contract; native-gap §8.5). Floor is
**not** located. This file does not move that bracket.

`receipts/ascent-2026-08-16/HARVEST_NOTE_G006.json`:

```
do NOT judge a lower-BPW artifact against id-identity to the Q4 seal
bar: well-formed English or code; no degenerate cycles; France→Paris;
     0 fallbacks; 0 dense_w; report id-drift, do not fail on it
```

### 11.3 Wave reports (CITED)

```
g1-sub15-native-gap.md:57-67,147-168,170-184,216-241,376-447
g1-doctor-recovery.md:161-184,249-288,348-356,429-448,595-630
g1-doctor-tensor-map.md:53-71,79-83,133-155,186-223
g1-out-proj-forensics.md:17-55,138-174,185-237,241-284,370-386
g1-sparse-exact-islands.md:12-26,159-217,388-407
g1-bit-budget-accounting.md:90-144,154-232,250-256,422-428
g1-heterogeneous-allocation.md:68-102,164-184
g1-lm-head-and-tails.md:11-22,96-98
g1-artifact-inventory.md:355-357
```

### 11.4 Source (this tree)

```
crates/hawking-core/src/model/qwen38_pack.rs:673-679
crates/hawking-core/src/model/qwen38_geometry.rs:20-52,57-58
crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs:15-18
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:661
  "unknown mixed codec {other}; refusing silent fallback"
```

`lab/operators/doctor6/rungs.py` `quant_outlier_channel` /
`l3_outlier_residual` exist on HEAD (not materialized). Steal the
*one-row* idea, not the Q80 5% column cut
(`g1-doctor-tensor-map.md:155`).

---

## 12. What this lane did not do

- No Metal, no generate, no pack, no resident RPC.
- No numpy/torch on the 27B tensors.
- Did not re-derive G0 TOKEN_NS / TPS / roofs.
- Did not emit `catalog.hq38m20`. That is the native-gap implement lane.

```
STATUS
IMPLEMENT_READY

CLAIMS
1. mixed-sub15-v1 complete physical BPW is 8*4340604637/26895998464 = 1.2910781930062503. Class bytes sum to that integer. DERIVED/MEASURED. Evidence: §11.1; g1-sub15-native-gap.md:23,57-67.
2. After C3, the first codec hypothesis is attention write rice (out_proj/o_proj), then HGRAVS01 down, then DN in_proj rice, then gate/up. HYPOTHESIS. Evidence: pack composition delta §1; rice cosine 0.834-0.847; write/R table; mixed-2p0 vs sub15 signatures §11.2.
3. H0 K-truncation is a 0-BPW precondition. A native generate without C3/fuse-off is not a codec verdict. CITED. Evidence: g1-sub15-native-gap.md:216-241,498-499.
4. Minimal billed exception is f16 output-row 3994 on 128 write tensors: Δbytes=3016480, ΔBPW=0.0008972278918108, complete'=1.2919754208980609. DERIVED. Evidence: §3.1, §4 H1a arithmetic; g1-doctor-tensor-map.md:53-71.
5. Protect-by-|X| and rescale-by-|X| are different operators. Act_colscale KILLS. |W|∩|X| = 0/42 on L0 out_proj. CITED. Evidence: g1-out-proj-forensics.md:17-33,370-377; QWEN_ATTENTION_DENSITY_PROBE.json.
6. 1.5 slack is 702395075 bytes = 0.2089218069937497 BPW. All of §8 #1-15 fit in B except none; #16 all-down-binary is the first C at 1.5017653397795792. DERIVED. Evidence: §8, §11.1.
7. All-attn Q4 recreates mixed-2p0 attention at complete' 2.0882206608977891 (D) and is not this repair. DERIVED. Evidence: §4 H1 / §8 #18.
8. lm_head and embed are already the G0 Q4 oracle. Do not lift them. CITED. Evidence: g1-sub15-native-gap.md:186-191; §4 H9.
9. Hetero 2.0/1.5 tables, 1-bit doctor floors, VQ, generator+residual, entropy coding, expand-to-Q4 vehicles are not repairs. KILLS. Evidence: §9; contract dead families; g1-doctor-recovery.md:161-173,260-288.
10. One generate classifies the hypothesis via §5-§7. Signatures {220,264} vs {198,8} vs missing 248068 vs EOS vs code-only split H1 / H2+H4 / H6 / H7 / H8. HYPOTHESIS map, MEASURED priors. Evidence: §5-§7; §11.2.

EVIDENCE
§11.1-11.4
workspace/superwave/g1/g1-sub15-native-gap.md
workspace/superwave/g1/g1-doctor-recovery.md
workspace/superwave/g1/g1-doctor-tensor-map.md
workspace/superwave/g1/g1-out-proj-forensics.md
workspace/superwave/g1/g1-bit-budget-accounting.md
git show HEAD:receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json
git show HEAD:receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json
git show HEAD:receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json
crates/hawking-core/src/model/qwen38_pack.rs:673-679
crates/hawking-core/src/model/qwen38_geometry.rs:20-52,57-58

CHANGES
workspace/superwave/g1/g1-sub15-repair-plan.md (this file only)

TESTS
see lane message

RISKS
- H1 vs H2 ranking is a prior, not a generate ablation. Applying the wrong first row costs one generate, not a floor.
- 3994 and the 42-col set are 256-token / proxy-X statistics. mixer_x may move H5. H1a as an output-row island is a weight-space fact (851/851) and does not depend on X.
- Subset rice old-bytes are class-average floors. Per-file variance UNMEASURED.
- HGRAVS per-layer byte variance UNMEASURED. 16-late old-bytes use k/64 of the class total.
- Sidecar needs a 1-row / skinny GEMV bind. That is not a new family; it is still an implement-lane item. Until it exists, H1a can be stored as 128 HQ30UQ4 1-row tensors (40 B header each, +5120 B). Header tax 128*40=5120 B = 1.52e-6 BPW. Negligible vs 3016480.

UNRESOLVED
- Native coherence of 1.291. GPU lane, after C1-C3.
- Isolated HGRAVS01 down hold. Cheapest: one-layer output cosine on determined post_swiglu X, not a generate.
- Whether {220,264} vs {198,8} survives C3. If both become a third signature, start at §7 row 10.

NEXT
Native-gap implement C1-C3. GPU lane runs the battery. Apply §7 once. Do not invent a 2.0 artifact.
```
