# G1 — residual channel 3994 island

Lane: `41-channel-3994-island`. One new file. No GPU. No generate. No pack. No live-organism contact.

Measurement: `/tmp/g1_channel_3994_island.py` → `/tmp/g1_channel_3994_island.json`
sha256 `eafcdfdaf1c0d42cc2214ef6ef1a00bec87fd631d5e077ec0ad933b385321d14` (856,760 bytes).
Wall **98.4 s**. Peak RSS **6.796 GB**. CPU/numpy only.

Every number below is **MEASURED** on the real BF16 tensors / 256-token capture unless marked **PROJECTED** (arithmetic on geometry) or **claimed** (prior lane, not re-run).

---

## Verdict

**MEASURED_NEGATIVE** as a G1 density mechanism. **SUPPORTED** as a real, weight-native, compile-time-constant residual island.

KILLS: “hold the body at aggressive low bit-width, exact residual 3994 (or a tiny fixed set) in both directions, recover Q4-class output cosine at nearly-zero index cost.”

- Q2 + exact island never reaches 0.99 on any of 35 scored GEMVs. Unweighted mean hold cosine at k=32 is 0.8668 (read) / 0.8056 (write) versus Q4 0.9958 / 0.9947.
- Q3 + exact island moves the tensors that fail 0.99 by a few thousandths, not to the bar. L0 `out` 0.95310 → 0.96195. L32 `out` 0.96789 → 0.96813. L47 `out` 0.94831 → 0.94858. L32 `gate` 0.97528 → 0.98181.
- The curve flattens at **k=1 (channel 3994)** on every write tensor and on every read tensor once 3994 is activation-hot. k=3 ({3994, 3456, 310}) is the early-layer knee. k=8..32 is bulk.
- Index cost is actually zero. Value cost at k=1 is 5,252,608 bf16 weights = 0.01953 % of params = **+0.00249 BPW** versus a Q3 body. Cheap, and not enough.

REOPEN_IF: a native generate (not expand-to-Q4) shows that zeroing write-error on residual 3994, or read-error on {3994, 310, 3456}, fixes a named coherence failure that Q3/Q2 otherwise produces. Mixer-site |X|-hot *columns* of `out_proj` (width 6144, disjoint from residual 3994) are a different mechanism; forensics already scored them (42 cols → 0.9762, still under 0.99).

---

## 1. Is the island real, and is it one channel?

**A small set, not one channel. 3994 is the unique member that is both the fattest write row and the hottest mid/late activation.**

Activation, 256 × 5120 × 64, site `CAPTURED_REAL_BF16_POST_NORM_HIDDEN`, `sha256_self=fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512`:

| ch | n_hot4 | n_hot10 | mean RMS | mean energy frac | role |
|---:|---:|---:|---:|---:|---|
| **3994** | **54** | **54** | **14.192** | **0.0976** | mid/late singleton |
| 3456 | 63 | 24 | 5.503 | 0.0137 | persistent moderate |
| 310 | 52 | 32 | 4.068 | 0.0198 | early singleton (L0–L5, L7) |
| 3842 | 37 | 1 | 2.497 | 0.0032 | secondary |
| next 8 | ≤36 | 0 | ≤2.68 | ≤0.003 | not an island |

Only **3** channels are ≥10× median RMS in ≥4 layers. Mean-RMS rank[:8] = `[3994, 3456, 310, 2519, 4303, 3842, 1726, 4042]`.

3994 is **not** hot at the start of the net:

| layer | rank | xmed | energy | tokens top-1 | tokens ≥10× |
|---:|---:|---:|---:|---:|---:|
| 0 | 54 | 3.48 | 0.0015 | 0/256 | 22/256 |
| 3 | 41 | 2.02 | 0.0006 | 0/256 | 0/256 |
| 5 | 151 | 1.70 | 0.0005 | 0/256 | 0/256 |
| **6** | **1** | **46.93** | **0.2476** | **256/256** | **256/256** |
| **7** | **5120** | **0** | **0** | **0/256** | **0/256** |
| 8–10 | 358–830 | 1.21–1.35 | ≤0.0003 | 0 | 0 |
| 11 | 1 | 22.07 | 0.0715 | 217/256 | 256/256 |
| 14 | 2 | 11.13 | 0.0196 | 0/256 | 256/256 |
| 15–63 | 1 | 15–50 | 0.035–0.261 | 131–251 | 162–256 |

L7 hidden column 3994 is **identically 0** in the raw file (`n_nonzero=0` over 256 tokens). Independently, `L7.post_attention_layernorm.weight[3994] = 0.0` exactly (the only zero among 64 post-attn norms; L0–L5/L8 sit at 0.00390625 = 2⁻⁸). Capture site is still `UNCONFIRMED_POST_NORM` (wave 1). The pair is a hard layer-7 gate, not a token fluke.

So: one channel is the right *compile-time default* for mid/late depth. The honest protected set that is the same at every layer is **{3994, 3456, 310}**. A singleton {3994} is dead on L0–L5 and L7–L10, where 310 is the activation island.

---

## 2. Weights independently of the capture

**Yes. The island is planted in the write weights. It is not a 256-token artifact.**

This lane, 7 layers × {down, out} = 14 write tensors, all 3994 **rank 1** of output-row RMS:

| tensor | kurtosis | kurtosis drop **only** row 3994 | row 3994 xmed | F-frac in row 3994 | n10 |
|---|---:|---:|---:|---:|---:|
| L0 down | 15.52 | **0.20** | 11.47 | 0.02465 | 1 |
| L0 out | **149.36** | **1.91** | **20.70** | **0.06878** | 1 |
| L3 out | **132.14** | **0.44** | 17.76 | 0.05670 | 1 |
| L6 out | 20.41 | 1.31 | 9.27 | 0.01608 | 0 |
| L15 out | 2.71 | 0.37 | 5.72 | 0.00620 | 0 |
| L32 out | 2.07 | 0.57 | 4.73 | 0.00431 | 0 |
| L63 out | 2.60 | 2.32 | 3.32 | 0.00215 | 0 |
| other 7 downs | 0.13–1.90 | lower | 2.01–3.26 | 0.0008–0.0021 | 0 |

L0 `lin_o` kurtosis 149.36 is **that one residual-side row**. Doctor lane already scored 3994 as top-5 on all 128 write tensors; this lane’s 14/14 are all rank-1, matching doctor `Counter({1: 128})` on the same `weight_stats.json`.

Read-side |W| is the opposite: on gate/up, column 3994 is typically **rank 5120 / 5120** (the weakest input column), xmed 0.25–0.93. `in_proj` is 25–45 at three early layers, still only 1.16–1.36×. The island is **not** a fat weight column on the tensors that *read* the residual.

Two exceptions, both weight-native:

- `lm_head` input column 3994 is the **unique** 4× column: rank 1, xmed **4.795**, the only `in_n_hot4=1`. Top-5 in: 3994, 4316, 505, 56, 220.
- `embed` column 3994 is rank 5120, xmed 0.929. Not an embed island.

RMSNorm at 3994 is **small**, not large. `final_norm.weight[3994]=0.7148` is rank 5120 / 5120. Post-attn 3994 is the smallest (or tied-smallest) gamma on every sampled layer. Input-norm 3994 is ordinary (L0 rank 709 / 5120). The activation spike is not a fat gamma.

Not a capture artifact:

- Write-row rank-1 at L0, before 3994 is activation-hot.
- `first128` top-1 == `last128` top-1 on **64 / 64** layers.
- All 5 prompts agree on the L6 / L11 / L32 / L63 top-1 (3994) and on the L0 / L3 / L7 top-1 (310).
- Shared 45-token system prefix has the same ranks as the full 256.
- L6: 3994 is top-1 on **256 / 256** tokens.

---

## 3. Tensor-class split

| class | direction vs residual 3994 | |W| of that axis | |X| of that axis | Qn error that matters |
|---|---|---|---|---|
| down, lin_o, o | **output row** | fat (rank 1 on every write) | write lands in residual | channel-3994 rel-err 0.09–0.69 at Q2 |
| gate, up, qkv/q/k/v, z | **input column** | thin (often weakest) | fat after L6 (10–50×) | col-3994 MSE share 0.03–0.28 of Q3 error when hot |
| lm_head | input column | **unique 4.8×** | L63 30× after final RMSNorm | cosine already 0.990 at Q3; k=1 adds +0.00013 |
| embed | input column | thin | n/a (lookup) | not scored as GEMV |
| norms, A_log, dt, conv | already f32 in G0 | L7 post γ=0 | — | not a Qn island |

This is why |W|-selected sparse-exact (wave 1 `g1-sparse-exact-islands`, FALSIFIED) and this island disagree. |W| top-k is scattered across rows/cols. Residual 3994 is one **fixed index** that is fat on the write axis and hot on the activation axis, and ordinary on the read-weight axis. The read-side payoff is |X|-amplification of an ordinary column, the same axis out-proj forensics named, but on width-5120 residual, not width-6144 mixer.

Wave-1 contradiction, resolved as two different operators:

- Residual-column *scale* of `out_proj` (in-dim 6144) from this capture: claimed kill, 0.992 → 0.919. Wrong site.
- Residual-**row** exact of `out_proj` (out-dim 5120, this lane): L0 Q3 0.95310 → 0.96195. Does not destroy. Does not reach 0.99. Mixer-column exact of the 42 |X|-hot *inputs* remains the stronger out-proj patch (forensics 0.9762).

---

## 4. Payoff

Codec = production absmax group-64 (same family as HQ30UQ4). Overlay: quantize the whole tensor, restore the protected residual channels (input columns on read tensors, output rows on write tensors). Hold = odd rows of the 256-token capture. Q4 self-check on L0 `out`: weight cosine **0.993546**, kurtosis **149.3577** (probe 0.993541 / 149.3577). L0 `out` Q3 hold **0.953103** / residual-proxy **0.997449** match forensics 0.95310 / 0.99745.

Protected-set ranking = mean activation RMS (same set at every layer). k=1 is {3994}. k=3 is {3994, 3456, 310}.

### 4.1 Unweighted means, 21 read + 14 write GEMVs, layers {0,3,6,15,32,47,63}

| body | dir | k=0 | k=1 | k=3 | k=8 | k=32 | Q4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Q2 | read in-col | 0.84068 | 0.85223 | 0.85982 | 0.86220 | 0.86676 | 0.99584 |
| Q2 | write out-row | 0.79149 | 0.80276 | 0.80318 | 0.80394 | 0.80563 | 0.99474 |
| Q3 | read in-col | 0.97875 | 0.98093 | 0.98223 | 0.98254 | 0.98311 | 0.99584 |
| Q3 | write out-row | 0.97206 | 0.97302 | 0.97305 | 0.97308 | 0.97329 | 0.99474 |

Q2 + any k in 1..32 does not approach Q4. Q3 + island closes ~20 % of the Q3-to-Q4 read gap at k=1 and ~35 % at k=32; the write gap barely moves after k=1.

### 4.2 Where k=1 actually pays (hold cosine)

Read, Q2, Δ(k=1 − k=0). Only layers where 3994 is activation-hot move:

| tensor | Q2 k=0 | Q2 k=1 | Δ | Q3 k=0 | Q3 k=1 | Q3 col-3994 MSE share | X energy in 3994 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0.gate | 0.84797 | 0.84831 | +0.0003 | 0.98103 | 0.98107 | 0.0015 | 0.0015 |
| L6.gate | 0.80129 | 0.82597 | **+0.0247** | 0.96607 | 0.97719 | **0.283** | 0.248 |
| L32.gate | 0.78092 | 0.85958 | **+0.0787** | 0.97528 | 0.98181 | **0.269** | 0.254 |
| L47.gate | 0.81038 | 0.83810 | +0.0277 | 0.97614 | 0.97839 | 0.080 | 0.071 |
| L63.gate | 0.93557 | 0.94702 | +0.0114 | 0.99365 | 0.99458 | 0.134 | 0.133 |
| L32.up | 0.78345 | 0.79538 | +0.0119 | 0.96635 | 0.97198 | 0.155 | 0.254 |
| L32.in | 0.83521 | 0.84811 | +0.0129 | 0.97555 | 0.97928 | 0.140 | 0.254 |

Write, the only large Q2 moves are the kurtotic early rows:

| tensor | Q2 k=0 | Q2 k=1 | Δ | Q3 k=0 | Q3 k=1 | Q3 ch3994 rel-err k=0 → k=1 | F-frac |
|---|---:|---:|---:|---:|---:|---|---:|
| L0.down | 0.88660 | **0.93065** | **+0.0440** | 0.99147 | 0.99157 | 0.029 → 0 | 0.0247 |
| L0.out | 0.70633 | 0.72390 | +0.0176 | 0.95310 | **0.96195** | **0.562 → 0** | 0.0688 |
| L3.out | 0.81100 | **0.87188** | **+0.0609** | 0.98343 | 0.98445 | 0.080 → 0 | 0.0567 |
| L3.down | 0.83655 | 0.84527 | +0.0087 | 0.98066 | 0.98090 | 0.069 → 0 | 0.0019 |
| L32.out | 0.76857 | 0.76920 | +0.0006 | 0.96789 | 0.96813 | 0.375 → 0 | 0.0043 |
| L47.out | 0.68752 | 0.69058 | +0.0031 | 0.94831 | 0.94858 | 0.119 → 0 | 0.0017 |
| L63.out | 0.72334 | 0.72887 | +0.0055 | 0.96040 | 0.96064 | 0.109 → 0 | 0.0022 |

L0 `out` Q3: 30.3 % of mixer-output MSE lives in residual channel 3994 (`ch3994_mse_share=0.303`). Exacting that one row zeros it and lifts hold cosine 0.95310 → 0.96195. The other 70 % is the mixer-column problem. Residual-proxy on the same Ŵ is already 0.99745 at k=0 (write/R dilution, forensics).

Only one Q3 cell *crosses* 0.99 because of the island: L15 `q_proj` 0.98932 → 0.99015 at k=2. L0.down / L63.gate / L63.up / L63.q already clear 0.99 at k=0.

`lm_head` (chunked, X = L63 hidden then final RMSNorm `(1+w)`; site **not** a confirmed lm_head capture): Q4 0.998188, Q3 k=0 0.990334, k=1 0.990464, k=3 0.9905-class. Already at the bar; the 4.8× column is real and cheap to exact and does not move greedy-id (not measured).

### 4.3 Controls

Q3, Δ hold cosine versus k=0:

- **Random k channels** (seed 3994): |Δ| ≤ 0.00007 at k=8 on every tensor. The set is specific.
- **|W|-rank** on write tensors: k=1 is 3994 (same as act-rank). Same Δ as act-rank. Expected: 3994 is rank-1 of |W| rows.
- **|W|-rank** on read tensors: k=1 is some other column (3212, 1689, 2090, …), never 3994. Δ ≈ 0. The fat |W| input column is not the residual island.
- **Refit vs overlay** (zero the columns, requantize the group, restore): max |Δ| = 3.5e-5 (L3/L15 `in` k=3). Overlay is the production shape. Group-scale pollution is not the mechanism.

### 4.4 Flattening

Write direction: **k=1 is the knee**. L0.down Q2: +0.0440 of a total +0.0450 to k=32. L3.out Q2: +0.0609 of +0.0638. L0.out Q3: +0.00885 of +0.00945.

Read direction, 3994-hot layers: **k=1 is the first step, k=3 finishes the small set**. L32.gate Q2: +0.0787 at k=1, +0.0870 at k=3, +0.0978 at k=32. L0 (3994 cold): k=1 is nothing; k=3 ({…, 310}) is the step (+0.0063 Q2 gate).

k=8 / 16 / 32 is ordinary bulk exacting. Not an island.

---

## 5. Complete BPW including the exception

Index bits = **0** (compile-time constant). Island values costed as bf16 (16 bits). Body charged at nominal `bits + 16/64`. Small f32 class charged 32 bits (G0 already stores them f32). **PROJECTED** arithmetic, not a packed artifact. Official G0 complete BPW remains 4.252735126866492.

Island mass per protected residual channel, both directions, all residual-touching GEMVs:

```
down rows 64×17408     1,114,112
gate cols 64×17408     1,114,112
up   cols 64×17408     1,114,112
o/lin_o rows 64×6144     393,216
qkv cols 48×10240        491,520
z    cols 48×6144        294,912
q    cols 16×12288       196,608
k+v  cols 16×1024×2       32,768
a+b  cols 48×48×2          4,608
lm_head cols 248320      248,320
embed   cols 248320      248,320
                       ---------
per k                  5,252,608   = 0.019529 % of 26,895,998,464
```

| k | island elems | +BPW vs Q3 body | all-Q2+island | all-Q3+island | MLP-Q3 / attn-Q4 / emb-Q4 +island |
|--:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2.252926 | 3.252828 | 3.616473 |
| 1 | 5,252,608 | **0.002490** | 2.255612 | 3.255318 | 3.618892 |
| 3 | 15,757,824 | 0.007470 | 2.260982 | 3.260298 | 3.623730 |
| 8 | 42,020,864 | 0.019920 | 2.274408 | 3.272748 | 3.635824 |
| 32 | 168,083,456 | 0.079680 | 2.338855 | 3.332508 | 3.693880 |

The exception is nearly free. The body bit-width is the whole story, and the island does not make Q2 or Q3 a Q4-class body.

---

## 6. Kernel consequence (64 threads / row)

Production kernel `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
(`crates/hawking-core/shaders/qwen_uniform_q4.metal:183-221`):

```
TG=128, 4 simdgroups, 2 rows/TG, 64 threads/row
lane_in_row = split*32 + simd_lane          // 0..63
row         = group_id*2 + team
col         = lane_in_row*8 ; stride 512
```

Mapping of the compile-time set:

| index | where | mapping |
|---|---|---|
| output row 3994 | write organs (down/o, M=5120) | TG `1997`, team `0`, partner row 3995. That TG still has to exist for 3995. |
| input col 3994 | read organs (K=5120) | group 62, local 26, 8-unpack at col 3992, `lane_in_row=51`, split 1, simd_lane 19. Middle of a packed uint. |
| input col 3456 | same | group 54, local 0 (group-aligned). |
| input col 310 | same | group 4, local 54. |

**Do not** put `if (row == 3994)` or `if (col == 3994)` inside the TPR64 loop. Every row would pay a divergent branch; the 8-unpack of group 62 would break the word-load law (`g1-direct-gemv-geometry.md` constraint 2).

**Do** this:

1. Pack-time: write 0 into the protected columns (read tensors) and skip / zero the protected rows (write tensors) in the Qn body. Same HQ30UQ4 / Qn stream. `geo_tpr64_tg128` is unchanged.
2. After the GEMV, an epilogue that knows the compile-time set:
   - read organs: `y += x[c] * W_exact[:, c]` for each protected c. One saxpy of length `rows`.
   - write organs: `y[r] = dot(W_exact[r, :], x)` for each protected r. One dot of length `cols`.
3. `W_exact` is `k` bf16 vectors, addressed as a tiny side allocation, not an index stream.

**Common-case cost inside the 401 geo_tpr64 GEMVs: zero.** No extra branch, no per-weight indirection, no change to the 32-uint word walk.

Epilogue cost is **PROJECTED**, not timed (GPU lane owns measurement): k=1 gate saxpy is 17,408 × 2 B = 34,816 B; k=1 down row-dot is the same; k=3 is 3×. At 639 GB/s that is tens of nanoseconds per GEMV, not a token-level claim.

A specialized *in-kernel* path that skips TG 1997 on 5120-row organs does not even save a TG (partner row 3995). Discard it.

Binding: this is a representation-specific consume (Qn body + k saxpy/dot), not expand-to-float-then-generic-GEMV.

---

## Evidence

Runner stdout (full log `/tmp/g1_channel_3994_island.log`):

```
[11:29:19] rss=1.489G Q4 self-check weight_cosine=0.99354645 kurt=149.3577
[11:29:19] rss=1.535G act L00 rms=0.0998 3994 rank=54 xmed=3.48 efrac=0.0015 tok_top1=0/256
[11:29:19] rss=1.545G act L06 rms=0.2352 3994 rank=1 xmed=46.93 efrac=0.2476 tok_top1=256/256
[11:29:19] rss=1.546G act L07 rms=0.2108 3994 rank=5120 xmed=0.00 efrac=0.0000 tok_top1=0/256
[11:29:20] rss=1.554G act L32 rms=0.6788 3994 rank=1 xmed=46.17 efrac=0.2544 tok_top1=251/256
[11:29:20] rss=1.554G act L63 rms=1.1668 3994 rank=1 xmed=30.35 efrac=0.1329 tok_top1=229/256
[11:29:20] act persist hot10>=4: 3 channels; mean-rms rank[:8]=[3994, 3456, 310, 2519, 4303, 3842, 1726, 4042]
[11:29:22] Wwrite L0.out kurt=149.36 drop3994=1.91 row3994 rank=1 xmed=20.701 frob=0.06878
[11:29:24] Wread  L0.gate col3994 rank=5120 xmed=0.541
[11:29:35] lm_head col3994 rank=1 xmed=4.795 max_ch=3994
[11:29:36] L0.gate Q2 k=0 hold=0.847972 k=1 hold=0.848313 k=3 hold=0.854282 col_mse_share=0.000777
[11:29:45] L0.down Q2 k=0 hold=0.886599 k=1 hold=0.930647 k=3 hold=0.930707 ch3994_rel k0=0.318
[11:29:46] L0.out  Q3 k=0 hold=0.953103 k=1 hold=0.961951 k=3 hold=0.962014 ch3994_rel k0=0.562
[11:29:59] L6.gate Q3 k=0 hold=0.966072 k=1 hold=0.977192 col_mse_share=0.283
[11:30:20] L32.gate Q2 k=0 hold=0.780918 k=1 hold=0.859585 col_mse_share=0.277
[11:30:57] lm_head Q4 hold=0.998188 Q3 k0=0.990334 k1=0.990464
[11:30:57] WROTE /tmp/g1_channel_3994_island.json wall=98.4s rss_max=6.796G
```

L7 raw capture file:

```
$ python3 -c "import numpy as np; x=np.fromfile('.../hidden/L07.f32', dtype='<f4').reshape(256,5120); \
print(x[:,3994].min(), x[:,3994].max(), int(np.count_nonzero(x[:,3994])))"
0.0 0.0 0
```

JSON pointers (same file, sha256 above):

- `q4_selfcheck.weight_cosine = 0.993546445453284`
- `activation.cross_layer.ch3994_n_hot10 = 54`
- `activation.cross_layer.act_rank_by_mean_rms[:3] = [3994, 3456, 310]`
- `activation.cross_layer.n_layers_first128_last128_top1_agree = 64`
- `activation.layers[7].ch3994.rms = 0.0`
- `weight_sample.writes[*].row3994_rank` all 1
- `weight_sample.lm_head.col3994_rank = 1`, `col3994_xmed = 4.795221499361983`
- `weight_sample.norms` L7 post `w3994 = 0.0`
- `payoff.tensors[L32.gate].curves.2` k0=0.780918 → k1=0.859585
- `payoff.tensors[L0.out].curves.3` k0=0.953103 → k1=0.961951, `ch3994_mse_share=0.30334`
- `complete_bpw.1.island_elems = 5252608`, `island_added_bpw_vs_q3_body = 0.002490`
- `kernel.row_3994.threadgroup_id = 1997`, `kernel.col_3994.lane_in_row = 51`

Prior-lane facts used, not re-derived: doctor 128/128 write top-5 (`/tmp/g1-doctor/weight_stats.json`, 851/851, 365.4 s, 2.116 GB); forensics L0 out Q3 0.95310 / residual-proxy 0.99745 / |W|∩|X| = 0 (`/tmp/qwen38_out_proj_forensics_followup.json`); G0 complete BPW 4.252735126866492; production kernel identity.

```183:221:crates/hawking-core/shaders/qwen_uniform_q4.metal
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(
    ...
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
```

```20:52:crates/hawking-core/src/model/qwen38_geometry.rs
pub const QWEN38_LAYERS: usize = 64;
pub const QWEN38_HIDDEN: usize = 5_120;
pub const QWEN38_INTERMEDIATE: usize = 17_408;
pub const QWEN38_O_PROJ_ROWS: usize = 5_120;
pub const QWEN38_O_PROJ_COLS: usize = 6_144;
```

```1:8:/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json
"schema": "hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1",
"status": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN",
"n_tokens": 256, "n_layers": 64, "hidden": 5120
```

---

## What this lane did not measure

- Generate / greedy-id / TOKEN_NS. Forbidden.
- True DeltaNet recurrent mix or softmax GQA mix as `out_proj` X. Same mixer-site proxy as forensics.
- Pre-norm residual (RMSNorm not inverted). Residual-proxy uses captured post-norm hidden.
- Layers other than {0,3,6,15,32,47,63} for the GEMV sweep. Activation and write-row rank used all 64 / a 14-tensor sample + doctor 128.
- A packed artifact or a Metal epilogue. Kernel section is source geometry + PROJECTED bytes.

Cheapest next experiment that would turn the island into a generate-facing claim: pack Q3 MLP + Q4 attention with pack-time-zero of residual {3994} (or {3994,3456,310}) plus a bf16 side plane, consume with unchanged `geo_tpr64` plus the saxpy/dot epilogue, hand to the GPU lane for native generate vs the Q4 oracle. Do not expand to float. Do not expect the 0.99 mixer-output bar on L0 `out` to move past ~0.962.

---

```
STATUS
MEASURED_NEGATIVE

CLAIMS
1. Residual 3994 is a real island, not a 256-token artifact. Weight-native: rank-1 output row on 14/14 write tensors this lane (L0 lin_o xmed 20.701, kurtosis 149.358 → 1.91 after dropping only that row). Activation-native: hot10 in 54/64 layers, mean RMS 14.192, energy 0.098. first128==last128 top-1 on 64/64 layers. Evidence: /tmp/g1_channel_3994_island.json weight_sample.writes, activation.cross_layer; stdout Q4 self-check + act L00/L06/L32/L63.
2. It is a small set, not one channel. Only 3 channels are hot10 in ≥4 layers: 3994 (54), 310 (32), 3456 (24). 3994 is cold on L0–L5 and L8–L10 and identically zero on L7 (raw L07.f32 n_nonzero=0; L7 post_attn_norm[3994]=0). Honest compile-time set is {3994, 3456, 310}. Evidence: activation.cross_layer.top_persist_hot4; layers[7]; L07.f32.
3. Read-side |W| does not see the island (gate/up col3994 usually rank 5120, xmed 0.25–0.93) except lm_head, where 3994 is the unique 4.795× input column. Evidence: weight_sample.reads, weight_sample.lm_head.
4. Q2 + exact island does not recover Q4-class output cosine. Unweighted mean hold at k=32: read 0.86676 / write 0.80563 vs Q4 0.99584 / 0.99474. No tensor crosses 0.99. Evidence: payoff.tensors curves.2.
5. Q3 + exact island flattens at k=1 on writes and on hot reads. L0.out 0.953103 → 0.961951 (30.3 % of MSE was ch3994). L32.gate Q2 0.78092 → 0.85958. Neither reaches Q4. Only new 0.99-cross is L15.q at k=2 (0.98932 → 0.99015). Evidence: payoff.tensors L0.out / L32.gate / L15.in.
6. Random channels do nothing (|Δ|≤7e-5 at k=8). |W|-rank on reads does nothing (never picks 3994). Overlay≈refit (max +3.5e-5). Evidence: random_control, weight_rank_control, refit.
7. Exception cost at k=1 is 5,252,608 bf16 weights, index bits 0, +0.00249 BPW vs Q3 body. PROJECTED. Evidence: complete_bpw.1.
8. Kernel branch is pack-time zero + post-GEMV saxpy/dot. geo_tpr64_tg128 is unchanged. Common-case cost inside the 401 GEMVs is zero. Row 3994 is TG 1997 team 0; col 3994 is group 62 local 26 lane 51. In-kernel if(row==3994) is the wrong shape. Evidence: qwen_uniform_q4.metal:183-221; kernel object in the JSON.

EVIDENCE
- /tmp/g1_channel_3994_island.py
- /tmp/g1_channel_3994_island.json sha256 eafcdfdaf1c0d42cc2214ef6ef1a00bec87fd631d5e077ec0ad933b385321d14 (98.4 s, 6.796 GB)
- /tmp/g1_channel_3994_island.log
- crates/hawking-core/shaders/qwen_uniform_q4.metal:183-221
- crates/hawking-core/src/model/qwen38_geometry.rs:20-52
- .../activation-capture-v1/capture-result.json sha256_self fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512
- .../activation-capture-v1/hidden/L07.f32 (ch 3994 identically 0)
- /tmp/g1-doctor/weight_stats.json (claimed 128/128 rank-1; this lane reconfirmed 14/14)
- /tmp/qwen38_out_proj_forensics.json + _followup.json (Q3 0.95310 / residual-proxy 0.99745 / |X| 42-col 0.97616)
- workspace/superwave/g1/g1-doctor-tensor-map.md §3.1
- workspace/superwave/g1/g1-out-proj-forensics.md
- workspace/superwave/g1/g1-sparse-exact-islands.md (different selector, already FALSIFIED)

CHANGES
workspace/superwave/g1/g1-channel-3994-island.md (this file). No other path touched.

TESTS
test -s workspace/superwave/g1/g1-channel-3994-island.md
wc -l workspace/superwave/g1/g1-channel-3994-island.md
git status --porcelain

RISKS
- out_proj X is the mixer-site proxy, not the recurrent / softmax mix. A true-mix X could move the *mixer-column* set; it cannot un-plant write-row 3994 in the weights.
- Capture site UNCONFIRMED_POST_NORM. L7 exact-zero is in the file; the gamma=0 fact is in the weights. Causal link between them is not proven.
- 7-layer GEMV sweep, not 64. Activation census and doctor write census cover 64.
- Residual-proxy ≥ 0.99 is not generate. Q3 L0.out residual-proxy is already 0.99745 without the island.
- complete_bpw is geometry arithmetic, not a packed manifest.

UNRESOLVED
- Why L7 hidden 3994 is identically 0. Weight fact (post_attn γ=0) is real; whether the capture is post-attn or the residual was already 0 is open.
- Whether exacting {3994} changes greedy tokens. GPU lane.
- Combined mixer-column island (forensics, 16–42 cols, L0-specific) + residual-row 3994. Not this mechanism.

NEXT
Do not implement a TPR64 in-kernel branch on row/col 3994. If a later generate lane wants the free patch, pack Q3-MLP / Q4-attn with pack-time-zero of {3994} (or {3994,3456,310}) plus a bf16 side plane and an epilogue saxpy/dot. Do not sell it as a bit-width drop. The density gap remains the unmeasured Qwen3.8 coherence floor between 2.0856 and 4.2527, not an un-exacted residual channel.
```
