# G1 — attention stack: MSE scale + residual island + geometry + bitwidth

Lane write scope: this file only. CPU/numpy. No GPU, no generate, no pack, no resident contact.

Measurement: `/tmp/g1_attention_2bpw_stack.py` sha256 `70640a8226e67c0238044a6bdcf477f5407d93bd048b3ee3327d55e4a10b7647` (965 lines)
JSON: `/tmp/g1_attention_2bpw_stack.json` sha256 `47b790ad5bf00d2f488c1e479407241ae668908d4ad226bebf02a7458ffaa46e` (10,359,514 bytes, 10,672 rows)
Log: `/tmp/g1_attention_2bpw_stack.log` sha256 `d40b3c25a256f68d0f4ae1e631ed1b812c584c61f04fadb4dacdb57008217407` (853 lines)
Wall **2149.4 s**. Peak RSS **5.775 GB**. 29 real attention GEMVs × 368 configs.

Every number is **MEASURED** on the BF16 tensors / 256-token capture unless marked **DERIVED** (integer arithmetic on those) or **CITED** (prior lane, not re-run).

---

## Verdict

**MEASURED_NEGATIVE.** No tested attention encoding holds output cosine ≥ 0.99 at or under ~2.07 complete attention BPW.

Best ≤ ~2.07: **uniform Q2, g=256, MSE-optimal group scale, island k=1 {3994}**.
- attention physical BPW **2.065634196169866** DERIVED
- mean hold output cosine **0.9187436180818551** MEASURED (29/29)
- min hold output cosine **0.818094759283152** MEASURED (`L0.out_proj`)
- cells ≥ 0.99: **0 / 29**

Strict 1.5-inversion (attention ≤ 2.064157091228481, CITED contract, uses mixed-2p0 `mlp_physical_bpw=0.8480504639008466`): **same family at k=0**, b_attn **2.062509196169866**, min **0.8096739548** (`L0.out_proj`). Still 0/29.

**0.99 floor** (all 29 cells, this family): **uniform Q4, g=128, MSE scale, k=0**.
- b_attn **4.125009196169866** DERIVED
- min **0.9900882450327616** MEASURED (`L47.o_proj`)
- mean **0.9960865443097543**
- 29/29 ≥ 0.99

Nothing between 2.07 and 4.125 cleared the min-cell bar. Best Q3 (g=16 MSE k=32, 4.100 BPW) still has min **0.974296** on `L47.o_proj`.

Incumbent Q4 g=64 **absmax** does **not** clear: `L47.o_proj` **0.9897967539** MEASURED (matches CITED `g1-group-partition-geometry.md` 0.9897967539). MSE scale is what takes g=64 over the bar (0.991123) and is the only reason g=128 clears.

Weight-space cosine is **not** the bar. Residual-proxy is **not** the bar. Both hide misses (below).

---

## 1. Threshold (DERIVED)

```
complete = (E_mlp*b_mlp + E_attn*b_attn + 8*TAB_BYTES_Q4 + 8*SMALL_BYTES) / N
N=26895998464  E_mlp=17112760320  E_attn=7237795840
TAB_BYTES_Q4=1350860880  SMALL_BYTES=10584840   # G0 MEASURED
```

Solve `complete=1.5` for `b_attn`:

| b_mlp | max b_attn | source |
|---:|---:|---|
| 0.848 exactly | **2.064276406094372** | this lane, integer bytes |
| **0.8480504639008466** | **2.064157091228481** | CITED contract; mixed-2p0 `mlp_physical_bpw` |
| 0.989 | 1.730901541914728 | this lane |

Delta of the two 0.848-band figures is `E_mlp * 0.0000504639008466 / E_attn ≈ 0.000119`. The ~2.07 band is that inversion plus slack. Q2 g=256 body (2.062509) sits under both; Q2 g=256 + k=1 island (2.065634) sits between them and 2.07; Q2 g=64 (2.250) is already over.

`b_attn` bills HQ30-family payload on the 208 fused attention GEMVs: 40 B header + f16 scale + `ceil(bits*g/8)` code bytes per per-row group (last group padded to g). Island: 0 index bits, bf16 values, **1,413,632 elems/k** on attention only (write rows of `out`/`o` + read cols of `qkvz`/`ba`/`q`/`k`/`v`). Overlay costing: body still paid in full.

```
island_attn(k=1) = 1413632
+BPW = 16*1413632 / 7237795840 = 0.003125
```

Full-model residual island including MLP+tables is CITED 5,252,608 elems/k = +0.002490 vs N (`g1-channel-3994-island.md` §5). Not added to `b_attn` here.

---

## 2. Method (MEASURED)

Weights: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16`
Activations: `.../activation-capture-v1` status `CAPTURED_REAL_BF16_POST_NORM_HIDDEN`, `sha256_self=fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512`, 256 × 5120 × 64.

Fit = even rows (128). Hold = odd rows (128). Primary metric = mean row-cosine of `Y_hold = X_hold @ W.T`. Weight cosine and residual-proxy (`post-norm hidden + write`) reported separately and **never substituted**.

Layers `{0,3,6,7,15,32,47,63}` = 3 DeltaNet + 5 GQA. 29 GEMVs: `in_proj_qkv`, `in_proj_z`, `out_proj` / `q,k,v,o_proj`. `in_proj_a/b` skipped for cosine (tiny; billed in `b_attn` via fused `ba`).

X:

| tensors | X | label |
|---|---|---|
| in-dim 5120 reads | captured post-norm hidden | `CAPTURED_REAL_BF16_POST_NORM_HIDDEN` |
| DeltaNet `out_proj` | `v * silu(z)` from fused `in_proj_qkvz` | `DERIVED_MIXER_PROXY` — **mixer_x never captured** |
| GQA `o_proj` | `repeat(v) * sigmoid(q_gate)` | `DERIVED_MIXER_PROXY` — **mixer_x never captured** |

256 tokens vs in-dim 6144 is rows-per-dim 0.0417, worse than NS-014. Magnitude fits are underdetermined. The compile-time island set is not: weight-native, L7 col 3994 identically 0 in the raw file.

Grid: bits {2,3,4} × g {16,32,48,64,128,256,512} × scale {absmax, mse_mult8, α=0.25} × k {0,1,3,8,32}, plus binary {meanabs, mse-closed-form} at g {16,32,64,128,256}, plus rice_q1_rms_2pct (existing 1.291 attention body). Groups are per-row along K (equals flat C-order iff `K % g == 0`).

MSE rule (CITED surviving rule, this lane vectorized): per group `s = argmin_m ||X_g (w − q(w, s0·m))||`, `m ∈ {0.50,0.70,0.85,1.00,1.15,1.30,1.50,2.00}`, `s0 = absmax/bound`. Bound `2^(b-1)−1`. 1-bit RTN is not a distinct operator (`qmax` clamps to 1); binary is sign×scale.

Island: compile-time ranked `[3994, 3456, 310, 2519, 4303, 3842, 1726, 4042, …]` from 64-layer mean RMS, seed `{3994,3456,310}` first. Overlay exact bf16 on write-rows / read-cols. L7 `hidden[:,3994]` n_nonzero = **0** MEASURED.

---

## 3. Sanity vs prior lanes (MEASURED)

L0 `linear_attn.out_proj` `[5120,6144]`, odd-row hold, same codec family:

| cell | this lane | prior | pointer |
|---|---:|---:|---|
| Q3 g=64 absmax out | **0.9531034548050097** | 0.9531034548050097 | forensics / group-geom / scale |
| Q3 g=64 absmax k=1 | **0.961951184333634** | 0.961951 | island |
| Q3 g=48 absmax | **0.9601527843826347** | 0.9601527843826347 | group-geom |
| Q3 g=64 MSE | **0.9746023673644195** | 0.97460 | scale-contradiction |
| Q3 g=64 MSE+k=1 | **0.9761885255871752** | (not previously stacked) | this lane |
| Q4 g=64 absmax out | **0.9922487465196606** | 0.992249 | scale-contradiction |
| Q4 g=64 absmax weight | **0.993546445453284** | 0.993546 | island / probe |
| kurtosis / drop row 3994 | **149.3577 / 1.9095** | 149.36 / 1.91 | island |
| L47.o Q4 g=64 absmax | **0.9897967539** | 0.9897967539 | group-geom |
| island rank[:8] | 3994,3456,310,2519,4303,3842,1726,4042 | same | island |
| L7 ch3994 nonzero | 0 | 0 | island |

L0 Q3 MSE moved **89.61%** of 491,520 groups off absmax (picks `[115544,153380,118068,51045,22866,13750,9239,7628]` for the 8 multipliers). Absmax is too large. Matches scale-contradiction.

Log L1–L9.

---

## 4. Best ≤ 2.07 (does not clear 0.99)

**Config:** uniform Q2, g=256 (divides 5120 and 6144), MSE scale, island k=1.

| quantity | value | tag |
|---|---:|---|
| body BPW `2 + 16/256` + header | 2.062509196 | DERIVED |
| + k=1 island | **2.065634196** | DERIVED |
| complete @ mlp 0.848 + tables Q4 | 1.500365 | DERIVED (>1.5) |
| complete @ mlp 0.989 + tables Q4 | 1.590078 | DERIVED |
| mean output cosine | 0.918744 | MEASURED |
| min output cosine | **0.818095** | MEASURED `L0.out_proj` |
| n ≥ 0.99 | **0 / 29** | MEASURED |
| mean weight cosine | 0.861296 | MEASURED, **not the bar** |

Second place under 2.07, and the best **strictly under 2.064157**: same minus island (k=0), min **0.809674**, mean 0.916181, complete @ mlp 0.848 = **1.499524** (the only Qn stack that keeps complete < 1.5 in this band).

Per-tensor hold output cosine, Q2 g=256 MSE k=1 (MEASURED):

| tensor | out | X site |
|---|---:|---|
| L0.out_proj | **0.818095** | derived mixer |
| L47.o_proj | 0.825020 | derived mixer |
| L63.o_proj | 0.858491 | derived mixer |
| L63.k_proj | 0.874952 | captured hidden |
| L32.out_proj | 0.887364 | derived mixer |
| L6.out_proj | 0.889615 | derived mixer |
| L47.v_proj | 0.903266 | captured hidden |
| L0.in_proj_z | 0.907297 | captured hidden |
| L47.k_proj | 0.913041 | captured hidden |
| 20 others | 0.920–0.975 | mixed |
| L63.v_proj | 0.975112 | captured hidden |

Write tensors (derived mixer_x) are the floor. Read tensors at mid/late depth are better because 3994 is activation-hot there; that is the island's only real payment, and it is not enough.

L0.out Q2 g=256 residual-proxy k=1 = **0.994786** at write/R = **0.3528**. That is write-gain dilution, not 0.99 output. Do not cite it as a pass.

Other families under 2.07, all fail harder:

| family | b_attn | min out | worst |
|---|---:|---:|---|
| Q2 g=256 MSE k=1 | 2.065634 | 0.818095 | L0.out |
| Q2 g=256 MSE k=0 | 2.062509 | 0.809674 | L0.out |
| Q2 g=512 MSE k=8 | 2.056259 | 0.804707 | L0.out |
| rice_q1_rms_2pct k=3 | 1.297169 | 0.772611 | L47.o |
| rice_q1_rms_2pct k=0 (sub15 body) | **1.287794** CITED packed | 0.759815 | L47.o |
| binary g=16 MSE k=8 | 2.025009 | 0.751569 | L47.o |

Existing 1.291 attention (rice_q1_rms_2pct, MEASURED this lane on the same 29 GEMVs): unweighted mean **0.861478**, min **0.759815** (`L47.o_proj`), L0.out **0.806323**. CITED packed artifact range 0.834–0.847 was a different aggregate; the encoding is the same operator. Affine Q2 g=64 absmax is worse than rice at higher BPW (L0.out 0.70633 vs rice 0.80632), as CITED.

---

## 5. Where 0.99 is actually cleared

MSE k=0, min and n/29 ≥ 0.99, 29 GEMVs (MEASURED). `b_attn` includes headers. `complete` is mlp=0.848 + tables Q4 + this attention (DERIVED).

| cfg | b_attn | mean out | min out | n≥0.99 | complete@0.848 |
|---|---:|---:|---:|---:|---:|
| Q2 g=256 MSE | 2.062509 | 0.916181 | 0.809674 | 0 | 1.499524 |
| Q2 g=16 MSE | 3.000009 | 0.947470 | 0.870119 | 0 | 1.751809 |
| Q3 g=64 MSE | 3.250009 | 0.986232 | 0.964399 | 5 | 1.819084 |
| Q3 g=48 MSE | 3.340862 | 0.986983 | 0.965867 | 7 | 1.843533 |
| Q3 g=16 MSE | 4.000009 | 0.990396 | 0.974011 | 19 | 2.020912 |
| Q4 g=512 MSE | 4.031259 | 0.995287 | 0.988755 | 28 | 2.029321 |
| Q4 g=256 MSE | 4.062509 | 0.995668 | 0.989347 | 28 | 2.037731 |
| **Q4 g=128 MSE** | **4.125009** | **0.996087** | **0.990088** | **29** | **2.054550** |
| Q4 g=64 MSE | 4.250009 | 0.996628 | 0.991123 | 29 | 2.088187 |
| Q4 g=64 absmax | 4.250009 | 0.995652 | **0.989797** | **28** | 2.088187 |
| Q4 g=48 absmax | 4.343117 | — | 0.990681 | 29 | — |
| Q4 g=32 MSE | 4.500009 | 0.997238 | 0.992548 | 29 | 2.155463 |

Q4 g=256 MSE k=32 still fails `L47.o` at **0.989505**. Island cannot buy the last 5e-4 at g=256.

Q4 g=128 **absmax** min 0.987709 (27/29). Q4 g=128 **α=0.25** min 0.988026 (28/29). Only MSE clears g=128.

Production tile is `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` (g=64). On that tile the 0.99 floor is **Q4 g=64 MSE k=0 at 4.250009**, pack-time scale only, same kernel. g=128 is a cheap-mapping divisor of every K (CITED group-geom) but is not the shipping HQ30UQ4 group.

`L47.o_proj` Q4 g=128 MSE k=0 residual-proxy = 0.998326 at write/R = 0.466. Output is 0.990088. Residual-proxy would have licensed Q3 g=16 MSE (residual 0.995609, output **0.974011**). That is the reporting trap named in scale-contradiction.

---

## 6. Stack is real and subadditive

Q3 g=64, hold output cosine (MEASURED):

| tensor | absmax | MSE | island k=1 | MSE+k=1 | ΔMSE | Δisl | Δstack | ΔMSE+Δisl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0.out | 0.953103 | 0.974602 | 0.961951 | 0.976189 | +0.02150 | +0.00885 | +0.02309 | +0.03035 |
| L32.out | 0.967893 | 0.981316 | 0.968135 | 0.981491 | +0.01342 | +0.00024 | +0.01360 | +0.01366 |
| L47.o | 0.948312 | 0.964399 | 0.948582 | 0.964824 | +0.01609 | +0.00027 | +0.01651 | +0.01636 |
| L63.o | 0.960402 | 0.978346 | 0.960644 | 0.978509 | +0.01794 | +0.00024 | +0.01811 | +0.01818 |

MSE is the large same-BPW move everywhere. Island k=1 is an L0-write tool (row 3994). Together they overlap on L0 (stack < sum) and barely add on late GQA. Neither combination, at any tested g, puts Q3 over 0.99 on `L47.o` or `L0.out`.

α=0.25 tracks MSE on L0 Q3 (0.97499 vs 0.97460) and beats absmax everywhere it was scored; it is **not** a substitute at the floor (fails Q4 g=128). α=1 was not re-run; CITED kill 0.99225→0.91865 stands.

g=48 still beats g=64 at matched bits (L0.out Q3 absmax 0.96015 vs 0.95310; Q3 MSE 0.97701 vs 0.97460). It does not change the 2.07 or 0.99 conclusions. g=48 does not divide K=5120 (short last group on 338/402 GEMVs, CITED).

---

## 7. Sites and what this is not

- `mixer_x` was never captured. Every `out_proj` / `o_proj` number is a **derived mixer-site proxy**, same construction as forensics / scale / group-geom / density probe. A real recurrent DeltaNet mix or softmax GQA mix can move those cells. It is the cheapest experiment that would change the write-tensor ranking.
- 256-token capture is underdetermined for scale magnitudes (rows/dim 0.0417 on 6144, 0.0147 on 17408). Island **set** `{3994,3456,310}` is compile-time and weight-native; L7 3994 = 0 is in the raw `L07.f32`.
- Not a generate claim. Not a complete-token claim. Not a consume-path claim. Scoring reconstructs `Ŵ` in float32 and does `X @ Ŵ.T`. Binding: a shipping path must consume packed Qn + saxpy, not expand-to-Q4/float then generic GEMV.
- 1-bit RTN was not scored as a third rung (identical to Q2 under this operator). Binary (sign×scale) **was** scored; it loses to Q2 MSE at every g under 2.07.

---

## 8. KILLS / REOPEN_IF

**KILLS** (this construction, 29 real GEMVs, hold output cosine):

- Any Q2 / binary / rice attention encoding at ≤2.07 as a 0.99 path. Best min 0.818. REOPEN_IF a different scale search or a captured (not derived) mixer_x lifts `L0.out` and `L47.o` Q2+MSE+island to ≥0.99 at the same BPW.
- Q3 + MSE + island + any tested g, including straddling g=48 and g=16 (4.0 BPW), as a 0.99 path. Best min 0.9743. REOPEN_IF Q3 g≤8 plus MSE clears `L47.o` at complete attention BPW still under the 2.07 inversion — g=8 is 5.0 BPW, already worse than Q4 g=64.
- Residual-proxy ≥0.99 as a license for attention Q3/Q2. L0.out Q2 g=256 MSE k=1 residual 0.9948 / output 0.8181. L47.o Q3 g=16 MSE residual 0.9956 / output 0.9740.
- Weight-space cosine as the bar. Several Q2 MSE cells have weight cosine ~0.86–0.88 with output 0.82–0.92; Q4 g=128 MSE L0.out weight 0.9874 < output 0.9913.
- Island-alone or MSE-alone as sufficient for 1.5-complete. CITED separately; this stack is their sum and still misses by ~0.17 cosine at 2.07.
- The 1.291 rice attention body as a coherence-quality candidate under this metric (min 0.760). Native-load of mixed-sub15-v1 is a different question (consume, not this encoding's cosine).

**Does not kill:** MSE-optimal pack-time scale on the Q4 g=64 production tile. That is a same-kernel, same-BPW-as-G0 win that takes `L47.o` 0.98980 → 0.99112 and is the production-compatible 0.99 floor. Not a 2.07 win.

**REOPEN_IF** a thicker mixer-site capture (the other lane) flips `L47.o` Q4 g=256 MSE (0.98935) above 0.99 — that would drop the family floor from 4.125 to 4.063. Unlikely to drop it to 2.07; the Q2 gap is 0.17 cosine.

---

## 9. Evidence pointers

Sanity + inversion + island rank: `/tmp/g1_attention_2bpw_stack.log:1-9`.
Sweep completion: same file `:851-853`.
JSON keys: `sanity.*`, `summary.best_under_2p07`, `summary.lowest_bpw_all_cells_ge_0p99`, `summary.configs[*]`, `rows[tensor_id=…]`.
Prior: `g1-scale-contradiction.md` (MSE rule, 0.97460), `g1-channel-3994-island.md` (set, 0.96195, +0.00249), `g1-group-partition-geometry.md` (g=48, L47.o 0.989797), `g1-bit-budget-accounting.md` (N, inversion), `g1-out-proj-forensics.md` (mixer proxy, |W|∩|X|=0), mixed-sub15 `PACK_REPORT.json` (`attention_gemv_rice` 1.2877935788805008).

```
$ python3 -c "..."   # inversion
max_b_attn 0.848 2.064276406094372
max_b_attn @ mixed-2p0 0.8480504639008466 = 2.064157091228481
Q2 g=256 body+hdr = 2.062509196
Q2 g=256 +k1     = 2.065634196
Q4 g=128 body+hdr = 4.125009196
```

Capture: `activation-capture-v1/capture-result.json` `status=CAPTURED_REAL_BF16_POST_NORM_HIDDEN`, `n_tokens=256`, `hidden=5120`.

---

## Completion report

```
STATUS
MEASURED_NEGATIVE

CLAIMS
1. No tested attention encoding holds hold-output cosine ≥ 0.99 at attention physical BPW ≤ ~2.07. Best ≤2.07 is Q2 g=256 MSE k=1, min 0.818095 on L0.out_proj, 0/29 cells clear. EVIDENCE: /tmp/g1_attention_2bpw_stack.log:851; JSON summary.best_under_2p07.
2. Strict 1.5-inversion max b_attn is 2.064157091228481 at mlp=0.8480504639008466 (CITED) / 2.064276406094372 at mlp=0.848 exactly (DERIVED this lane). Best under that cut is Q2 g=256 MSE k=0, min 0.809674, complete@mlp0.848=1.499524. EVIDENCE: JSON inversion + summary.configs q2 g=256 mse k=0; bit-budget §7; mixed-2p0 PACK_REPORT mlp_physical_bpw.
3. 0.99 floor on this family is Q4 g=128 MSE k=0, b_attn=4.125009196169866, min=0.9900882450327616 (L47.o_proj), 29/29. Q4 g=256 MSE never clears (min 0.989347). EVIDENCE: log:852; JSON summary.lowest_bpw_all_cells_ge_0p99 and L47.o_proj q4 g=128/256 mse k=0.
4. Incumbent Q4 g=64 absmax does not clear (L47.o 0.9897967539). Q4 g=64 MSE does (0.991123). MSE is a same-tile pack-time win, not a 2.07 win. EVIDENCE: JSON L47.o_proj q4 g=64 absmax/mse k=0; g1-group-partition-geometry.md:147-148.
5. Stack is real and subadditive. L0.out Q3 g=64: absmax 0.953103, MSE 0.974602, k=1 0.961951, MSE+k=1 0.976189. Island dies after L0 writes. EVIDENCE: sanity log:4-8; JSON additivity on L0/L32/L47/L63.
6. Existing 1.291 rice attention body, rescored on these 29 GEMVs: mean 0.861478, min 0.759815 (L47.o). Not 0.99. EVIDENCE: JSON family=rice_q1_rms_2pct k=0; mixed-sub15 PACK_REPORT attention_gemv_rice 1.2877935788805008.
7. mixer_x was never captured. All out/o numbers are derived mixer-site proxies. Residual-proxy and weight cosine are not output cosine. EVIDENCE: JSON tensors[*].X_site; L0.out q2 g=256 mse k=1 residual 0.994786 vs output 0.818095; L47.o q3 g=16 mse residual 0.995609 vs output 0.974011.
8. Sanity matches sealed prior cells to all reported digits. Peak RSS 5.775 GB. EVIDENCE: log:1-9,853; JSON sanity, wall_s=2149.4, rss_max_gb=5.775.

EVIDENCE
/tmp/g1_attention_2bpw_stack.py sha256 70640a8226e67c0238044a6bdcf477f5407d93bd048b3ee3327d55e4a10b7647
/tmp/g1_attention_2bpw_stack.json sha256 47b790ad5bf00d2f488c1e479407241ae668908d4ad226bebf02a7458ffaa46e
/tmp/g1_attention_2bpw_stack.log sha256 d40b3c25a256f68d0f4ae1e631ed1b812c584c61f04fadb4dacdb57008217407
BF16 /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16
capture sha256_self fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512

CHANGES
workspace/superwave/g1/g1-attention-2bpw-stack.md (new)

TESTS
see runner message

RISKS
256-token capture underdetermined for scale magnitudes. Write-tensor X is a proxy. g=128 is not the shipping HQ30UQ4 group; production-tile 0.99 floor is Q4 g=64 MSE at 4.250. Floor min 0.990088 is 8.8e-5 above the bar — a thicker mixer capture could flip g=128 vs g=64, not 2.07.

UNRESOLVED
Real mixer_x. Whether Q4 g=64 MSE generate-coheres (quality screen only). Native consume of mixed-sub15-v1 (other lane).

NEXT
Do not pack a 2 BPW attention artifact from this family. If a 0.99 attention screen is required, the cheap next measurement is Q4 g=64 MSE scale on the production tile (pack-time only). A 1.5-complete plan cannot put attention at Qn≤2 under this bar.
```
