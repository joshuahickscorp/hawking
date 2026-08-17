# G1 — group-MSE scale rule, production recipe

Date: 2026-08-17
Lane: 74-mse-scale-rule
Write scope: this file only. CPU/numpy. No GPU, no generate, no pack artifact, no resident touch.

Measurement JSON: `/tmp/qwen38_mse_scale_rule.json`
sha256 `bcdf0a4595cc620ad31aca66c3420a8e692d9988df8d42b58a05e4513cb0f3cc`
Script: `/tmp/qwen38_mse_scale_rule.py` sha256 `fa84927b28a8a41375b10ee3a70e9aa43c42acde5f99b2e5bb0db4ec664e7c35`
Wall: 2539.6 s. Peak RSS: 32.057 GB (lm_head only; 34-layer GEMV sweep peaked 9.373 GB).
Codec / X helpers reused from `/tmp/qwen38_out_proj_forensics.py`.

Tensors: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16`
Activations: `.../activation-capture-v1` (`sha256_self=fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512`, file sha256 `01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe`, L00.f32 sha256 `edc47c2ac99bf5446c775179dbcf9850b73320f25df5deea2df1d23d518a0243`)

Every number below is **MEASURED** on those tensors unless marked **CITED** (prior receipt / sibling report) or **ESTIMATED** (arithmetic on measured costs). Mixer-output cosine is not generate.

---

## Recipe (the thing that can be packed)

At pack time, after the existing qkvz / ba fuse, for every HQ30UQ4 GEMV:

```
X_fit  = site-correct activations, prompt-held-out split
G_b    = X_fit[:, 64b:64b+64].T @ X_fit[:, 64b:64b+64]     # 64x64 Gram, shared across rows
s0     = f16(absmax(w) / bound)                            # bound = 7 for Q4
for m in {0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00}:
    s  = f16(s0 * m)
    e  = w - clip(rint(w/s), -bound, bound) * s
    pick argmin e^T G_b e
write HQ30UQ4 with those f16 scales and RTN codes against them
```

- Complete physical BPW = absmax at the same bits. MEASURED L0 `out_proj` Q4 payloads both 16,711,720 bytes.
- Decode path is `float(q) * float(scales[group])` with `q = nibble - 8`. The kernel does not know how the scale was chosen. MEASURED kernel-faithful unpack max abs err vs numpy recon = 0.
- Kernel: `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`. No shader change. No extra buffer. The number stored in the existing f16 scale plane changes; the layout does not.

Do **not** fold column RMS into W (α=1). Do **not** exact-preserve `|W|` outliers. Do **not** expand to float.

Do **not** freeze this 256-token scale plane into a pack that will be generated from. The sign of the rule is determined. The numbers on the plane are not.

---

## Protocol

```
layers     34 of 64: DN {0,1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,62}
           GQA {3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63}  (all 16 GQA)
classes    gate, up, down, in_proj_qkv, in_proj_z, in_proj_qkvz_fused,
           out_proj, q, k, v, o; a/b on {0,32,62}; lm_head; embed gather
bits       2, 3, 4 on the 34-layer grid. Q8 sanity on 3 write tensors.
           Q1 not run (bound=(1<<(b-1))-1 is 0; CITED doctor-recovery 1-bit==2-bit).
fit/hold   even 128 / odd 128 of the 256-token capture
metric     mean row-cosine of Y = X_hold @ W.T                 PRIMARY
also       weight cosine, residual-proxy on write tensors (post-norm hidden + write)
baseline   HGRAVU01 absmax g=64, f32 group scale
rule       per-group s = argmin_s ||X_g (w − q(w,s))|| via e^T G e
multipliers {0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 2.00}   sibling 8-point
WIN        Δ output_cosine > +5e-4 vs absmax
NEUTRAL    |Δ| ≤ 5e-4
LOSE       Δ < −5e-4
X sites    post_norm_hidden UNCONFIRMED as in-proj / mlp-in
           out/o : mixer proxy v*silu(z) or repeat(v)*sigmoid(q_gate)   (not recurrent / softmax mix)
           down  : silu(X@Wg.T)*(X@Wu.T) reconstructed, not captured SwiGLU
           lm_head: L63 post-norm, NOT confirmed final-norm
           embed : token gather, 63/248320 vocab rows observed
```

Nominal codec BPW = bits + 16/64 (Q2 2.25, Q3 3.25, Q4 4.25). Same as absmax at that width.

Production Q2/Q3 kernel (`uniform_qn.rs`) is **g=128**, not g=64. The Q2/Q3 numbers here are the forensics g=64 family (same as the sibling). The drop-in production win is Q4 g=64.

---

## Sibling rule, reproduced

L0 `out_proj` Q3, even-fit / odd-hold, f32 search, no f16 snap:

| | this lane | sibling (`g1-scale-contradiction.md`) |
|---|---:|---:|
| absmax output cosine | **0.9531034548050097** | 0.9531034548050097 |
| group-MSE output cosine | **0.9745938398738019** | 0.974599872428159 |
| abs err vs sibling MSE | 6.033e-6 | — |
| frac groups not absmax | 0.896132 | 0.896149 |
| frac groups smaller than absmax | 0.78737 | 0.787 |

Absmax is bit-identical. MSE differs by 6e-6 because this search rints in f32 (production pack is f32); the sibling rinted in f64. Direction and magnitude hold.

L32 `out_proj` Q3: 0.96789 → **0.98131** (sibling 0.98132).
L63 `o_proj` Q3: 0.9604024629560054 → **0.9783455154737788** (sibling 0.97834). Residual-proxy 0.97749 → **0.98831**.

f16 snap of the same search, L0 out: Q3 +2.66e-6, Q4 −7.98e-6. Production may snap; it does not move the metric.

Finer grid m ∈ {0.40, 0.45, …, 2.00} (33 points), L0 out Q3: **0.976006** vs 8-point 0.974594 (+0.00141). Mode of the fine grid is m=0.75. The 8-point recipe is the cheap form; the leftover is real and small.

---

## Generalization: 732 cells, 0 LOSEs vs absmax

34 layers × every GEMV class × {Q2,Q3,Q4} = 732 scored cells.

| class | n (per bits) | Q2 Δ mean / med / min | Q3 Δ mean / med / min | Q4 Δ mean / med / min | Q2 W/N/L | Q3 W/N/L | Q4 W/N/L |
|---|---:|---:|---:|---:|---|---|---|
| out_proj | 18 | +0.140 / +0.137 / +0.101 | +0.0154 / +0.0144 / +0.0099 | +0.00197 / +0.00171 / +0.00090 | 18/0/0 | 18/0/0 | 18/0/0 |
| o (GQA) | 16 | +0.138 / +0.143 / +0.085 | +0.0133 / +0.0143 / +0.0054 | +0.00130 / +0.00122 / +0.00057 | 16/0/0 | 16/0/0 | 16/0/0 |
| down | 34 | +0.114 / +0.112 / +0.084 | +0.0098 / +0.0091 / +0.0039 | +0.00108 / +0.00084 / +0.00051 | 34/0/0 | 34/0/0 | 34/0/0 |
| up | 34 | +0.117 / +0.123 / +0.035 | +0.0114 / +0.0116 / +0.0022 | +0.00152 / +0.00145 / +0.00030 | 34/0/0 | 34/0/0 | 33/1/0 |
| gate | 34 | +0.112 / +0.115 / +0.045 | +0.0092 / +0.0093 / +0.0029 | +0.00116 / +0.00113 / +0.00036 | 34/0/0 | 34/0/0 | 32/2/0 |
| in_proj_qkv | 18 | +0.089 / +0.089 / +0.076 | +0.0079 / +0.0077 / +0.0056 | +0.00097 / +0.00089 / +0.00056 | 18/0/0 | 18/0/0 | 18/0/0 |
| in_proj_z | 18 | +0.084 / +0.080 / +0.063 | +0.0074 / +0.0074 / +0.0053 | +0.00088 / +0.00081 / +0.00059 | 18/0/0 | 18/0/0 | 18/0/0 |
| in_proj_qkvz_fused | 18 | +0.087 / +0.087 / +0.072 | +0.0077 / +0.0075 / +0.0055 | +0.00092 / +0.00086 / +0.00057 | 18/0/0 | 18/0/0 | 18/0/0 |
| q | 16 | +0.081 / +0.084 / +0.051 | +0.0067 / +0.0073 / +0.0040 | +0.00083 / +0.00082 / +0.00045 | 16/0/0 | 16/0/0 | 12/4/0 |
| k | 16 | +0.092 / +0.092 / +0.074 | +0.0086 / +0.0086 / +0.0050 | +0.00116 / +0.00101 / +0.00044 | 16/0/0 | 16/0/0 | 15/1/0 |
| v | 16 | +0.096 / +0.102 / +0.041 | +0.0088 / +0.0096 / +0.0031 | +0.00107 / +0.00097 / +0.00039 | 16/0/0 | 16/0/0 | 15/1/0 |
| in_proj_a | 3 | +0.093 / +0.091 / +0.083 | +0.0148 / +0.0134 / +0.0095 | +0.00231 / +0.00201 / +0.00081 | 3/0/0 | 3/0/0 | 3/0/0 |
| in_proj_b | 3 | +0.070 / +0.060 / +0.049 | +0.0093 / +0.0090 / +0.0055 | +0.00100 / +0.00090 / +0.00018 | 3/0/0 | 3/0/0 | 2/1/0 |

**LOSE vs absmax: 0 / 732.** Every NEUTRAL is Q4 with absmax already ≥ 0.997 and a still-positive Δ ∈ [+0.00018, +0.00050]. Ceiling, not a reversal.

Largest Q3 wins are the write tensors the sibling found, plus late depth:

| tensor | Q3 absmax | Q3 MSE | Δ |
|---|---:|---:|---:|
| L62 out_proj | 0.96002 | 0.98541 | +0.02538 |
| L60 out_proj | 0.96312 | 0.98655 | +0.02343 |
| L0 out_proj | 0.95310 | 0.97459 | +0.02149 |
| L43 o | 0.94164 | 0.96146 | +0.01982 |
| L63 down | 0.97279 | 0.99060 | +0.01781 |
| L63 o | 0.96040 | 0.97835 | +0.01794 |

Q2 is where the rule earns its keep: out_proj / o mean Δ ≈ +0.14, several late writes +0.16 to +0.19. Q4 is a small same-BPW polish (+0.001 to +0.002 typical). Q8 is saturated (L0 out 0.99998 → 0.99997, Δ −6.8e-6 NEUTRAL; L0 down and L63 o Δ ≈ 0).

Fused `in_proj_qkvz` (the tensor `qwen38_pack` actually writes) tracks the split qkv/z pair. Production must search **after** fuse.

### Residual-proxy crossings at Q3 (write tensors only; not generate)

Absmax residual-proxy < 0.99 and MSE ≥ 0.99, same hold:

- `out_proj` L20, L24, L28, L32, L36, L40, L44, L48
- `o` L51
- `down` L62, L63 (L63: 0.97324 → **0.99086**)

Late GQA `o` still misses residual 0.99 after MSE: L63 0.97749 → 0.98831; L59 0.96997 → 0.98031. Sibling already said L63 Q3 residual 0.98831 does not license Q3 attention.

### lm_head and embed

- `lm_head` [248320, 5120], X = L63 post-norm (NOT confirmed final-norm). Q3 0.99420 → **0.99636** WIN (+0.00216). Q4 0.99892 → 0.99914 NEUTRAL. Search 44 s / bits. Peak RSS **32.057 GB** (over the 15 GB cap; only this tensor).
- `embed_tokens` is a gather. 256 tokens hit **63 / 248320** vocab rows (max count 27). Group-MSE collapses to per-row weight-MSE on observed rows, absmax on the rest. Observed-row gathered cosine Q3 0.96982 → 0.97626. Not an activation-scale rule. Not a production plane.

### Classes that are not scale-rule consumers

conv1d (G0 f32, in-dim 4), A_log, dt_bias, all RMSNorm / q_norm / k_norm / ΔNet norm, vision. No group-64 GEMV decode. Left as-is.

30 DN layers were not swept. Sign is consistent on the 18 that were; the scale plane of an unswept layer is **UNMEASURED**, not interpolated.

---

## α=1 is still dead. α=0.25 is a cheap proxy, not the rule.

α=1 (doctor AWQ fold) on L0/L31/L32/L63, same metric:

| | L0 out Q4 | L32 out Q4 | L63 o Q4 | L0 down Q4 | L0 gate Q4 |
|---|---:|---:|---:|---:|---:|
| α=1 Δ | **−0.07360** | −0.00929 | −0.01398 | −0.01286 | −0.00322 |

L0 out Q4 0.99225 → **0.91865** reproduces the doctor kill bit-identically (CITED probe 0.9186496062432181). α=1 loses on Q3/Q4 of every class scored except a handful of Q2 WIN (L31/L32/L63 gate/up, L32 z, L63 q/k/down) that are smaller than group-MSE on the same cell. α=1 on L0 is a LOSE on every class at every bit.

α=0.25 vs group-MSE, 732 cells: MSE bigger on **728**, α=0.25 bigger on **4**. α=0.25 never loses to absmax on this grid. It is the cheap proxy the sibling named. It is not a substitute for the Gram search.

---

## Production form

### Search, pack time

1. Fuse qkv+z and a+b first (`qwen38_pack.rs`). Groups are 64 consecutive **input** columns of one output row in the packed layout.
2. Load site-correct `X_fit` for that organ (see capture).
3. Build 64×64 Grams, one per column-block (`n_in/64` of them). Shared across all output rows.
4. 8-point search around absmax, **snap each candidate to f16 before rint** (matches `pack_uniform_q4_group64`: stored f16 is authority).
5. Emit HQ30UQ4: same magic, version 1, group_size 64, f16 scale plane, nibble codes. Replace only `scale = f16(max_abs/7)` with the chosen s.

Optional upgrade, same container: 33-point grid 0.40–2.00 step 0.05. L0 out Q3 +0.00141 extra, 4.34 s vs 1.11 s.

A discrete code-boundary search (every threshold where some `w_i` crosses a bin) was **not** run. 8-point is the specified recipe. REOPEN_IF a boundary search moves hold cosine by more than the 8-point→33-point leftover on a thick capture.

### Cost per tensor (MEASURED, 4 Accelerate threads, 8-point, one bit width)

| class | shape | n_groups | search wall (mean) |
|---|---|---:|---:|
| out / o | 5120 × 6144 | 491,520 | 1.2 s |
| in_proj_z | 6144 × 5120 | 491,520 | 1.2 s |
| k / v | 1024 × 5120 | 81,920 | 0.20 s |
| in_proj_qkv | 10240 × 5120 | 819,200 | 2.0 s |
| q | 12288 × 5120 | 983,040 | 2.4 s |
| in_proj_qkvz_fused | 16384 × 5120 | 1,310,720 | 3.2 s |
| gate / up | 17408 × 5120 | 1,392,640 | 3.4 s |
| down | 5120 × 17408 | 1,392,640 | 3.4 s |
| a / b | 48 × 5120 | 3,840 | 0.01 s |
| lm_head | 248320 × 5120 | 19,865,600 | 43 s |

ESTIMATED full-model Q4-only pack-time search, language GEMVs after fuse, 8-point, 4 threads:

```
64*(gate+up+down) + 48*(fused+out) + 16*(q+k+v+o) + lm_head
= 64*10.2 + 48*(3.2+1.2) + 16*(2.4+0.2+0.2+1.2) + 43
= 653 + 211 + 64 + 43 ≈ 970 s
```

Call it **20 min** ESTIMATED, plus Gram build and I/O. Q2+Q3+Q4 if someone is choosing a bit: ~1 hour ESTIMATED. Peak working set per tensor is the BF16→f32 weight (357 MB gate/up/down; 5.08 GB lm_head). The 34-layer sweep stayed at 9.373 GB; lm_head search in this script peaked 32 GB (chunking was not tight enough — a packer must stream lm_head rows).

Same BPW as absmax: Q4 nominal 4.25 (`UNIFORM_Q4_NOMINAL_BPW = 4 + 16/64`). MEASURED L0 out payload 16,711,720 bytes either rule. `physical_payload_bpw = 4.250010172526042` (header).

Q3 codes **can** live in the HQ30UQ4 nibble container (`q ∈ [-3,3] ⊂ [-8,7]`). MEASURED same 16,711,720 bytes, unpack err 0. That is **not** a 3.25 BPW artifact; it is a 4.25 BPW container with unused code range. A true 3.25 g=64 bitstream does not exist in the shipping kernel (Qn is g=128). The free win is Q4 g=64 MSE scales.

### Kernel consumption — no decode change

```10:11:crates/hawking-core/shaders/qwen_uniform_q4.metal
//   * each nibble is offset-binary signed Q4: `q = nibble - 8`, so q is in
//     [-8, 7]; and
//   * every group has one IEEE FP16 scale, reconstructed as `float(q) * scale`.
```

```183:209:crates/hawking-core/shaders/qwen_uniform_q4.metal
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(
    ...
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
```

```233:247:crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs
        let scale = f16::from_f32(max_abs / 7.0);
        ...
        let reconstructed_scale = scale.to_f32();
        ...
                rint_ties_even(value / reconstructed_scale)
                    .clamp(-8.0, 7.0) as i32
```

The packer today *computes* `max_abs/7`. The kernel *consumes* whatever f16 is stored. Layout proof on L0 out Q4:

```
magic HQ30UQ4 v1, group_size 64
n_elements 31457280  n_groups 491520
scale_bytes 983040   code_bytes 15728640
payload_bytes_mse == payload_bytes_absmax == 16711720
kernel_decode_max_abs_err_vs_numpy 0.0
unpacked_scale_max_abs_err_vs_f16  0.0
decode_path_changed False
```

A scale rule that changed the decode path would be a new shader family. This one is a different number in an existing half.

### Capture: what the rule needs before a pack is generated from it

Standing caveat, confirmed:

| use of the 256-token capture | verdict | evidence |
|---|---|---|
| Rank a sharp column spike | legal | CITED sibling: L0 top-42 even∩odd = 42/42 |
| State the **sign** of group-MSE vs absmax | legal | 722/732 WIN, 10/732 NEUTRAL, 0 LOSE; even/odd/first/last splits all same sign |
| Freeze a production scale plane | **not legal** | even-vs-odd scale spearman 0.75–0.88; same-multiplier agreement 0.72–0.83 on the *same five prompts* |
| AWQ / length-`in_dim` fold | KILLS | α=1 L0 out Q4 0.99225 → 0.91865; rpd 256/6144 = 0.0417 < Q80 NS-014 0.0449 |

Stability, Q3, complementary-split hold (MEASURED):

| tensor | spearman even/odd | same-m even/odd | spearman first/last | fit-even hold-odd Δ | fit-odd hold-even Δ |
|---|---:|---:|---:|---:|---:|
| L0 out_proj | 0.876 | 0.827 | 0.922 | +0.02149 | +0.01851 |
| L0 gate | 0.776 | 0.757 | 0.875 | (WIN) | (WIN) |
| L0 down | 0.750 | 0.748 | 0.844 | (WIN) | (WIN) |
| L0 in_proj_qkv | 0.844 | 0.759 | 0.912 | +0.00791 | +0.00800 |
| L32 out_proj | 0.828 | 0.724 | 0.843 | (WIN) | (WIN) |
| L63 o | 0.877 | 0.798 | 0.895 | (WIN) | (WIN) |
| L63 down | 0.842 | 0.832 | 0.861 | +0.01781 | +0.01805 |

12–25 % of groups pick a different 8-point multiplier across a row-split of the same 5 prompts. Scale rel-L2 even vs odd is 0.15–0.18. The hold *cosine* is stable (L0 out 0.97459 vs 0.97442); the *plane* is not.

`n_fit` sweep on L0 out Q3, always hold odd: 32 → 0.97382, 64 → 0.97370, 96 → 0.97355, 128 → 0.97459. Extra tokens from the **same 5 prompts** barely move the Gram. That is redundancy, not adequacy. It does not license N=32.

fit_dim of this rule is **64** (one Gram per group), not 6144. rpd on 128 fit rows = 2.0. Doctor-recovery §5.2 is the authority for what N makes that determined in a *production* sense:

| purpose | min N (CITED doctor-recovery §5.2) |
|---|---:|
| group-64 output-MSE scale, `n_fit ≥ 64` | 86 |
| 2048-token census (`n_fit=1536`) | determines group-64 scales **and** gives 64-layer eval |
| site-correct mixer X, `n_fit ≥ 6144` | 8192 |
| site-correct down X, `n_fit ≥ 17408` | 23216 |

Cheapest pack-from capture for **this** rule: **N=2048**, ≥32 prompts, **sequence** holdout (not row-shuffle), parent BF16 only, sites written at the in-dim that consumes them:

| site | width | consumed by | today |
|---|---:|---|---|
| `post_input_norm` | 5120 | q/k/v, qkv/z/a/b, gate, up | YES (this dump) |
| `post_swiglu` | 17408 | down | NO (reconstructed here) |
| `mixer_x` | 6144 | out / o. True recurrent mix / softmax GQA, not the proxy | NO (proxy used here) |
| `final_norm` | 5120 | lm_head | NO |

CITED `g1-doctor-recovery.md` §5.4. A sibling lane is building that capture. Until it exists, group-MSE may be **stated** and **prototyped** on 256 tokens; it may not be the scale plane of a generate-facing pack.

Q4-vehicle X is still ABSENT. If parent-X and vehicle-X floors diverge, the live genome ranks and the parent teaches the codec.

---

## Uniform application

Applying this rule to every GEMV because it helped attention is **not** the 2.0856 failure mode. That failure was bit-allocation (attention left at 4.250, `down_proj` crushed to 0.1316). This rule does not change BPW.

On this metric, uniform MSE-vs-absmax is safe: 0 LOSEs. Uniform **low bits** is a different claim and is not licensed here. Q3 mixer-output still misses 0.99 on every `out`/`o` scored (L0 0.97459, L63 0.97835). Residual-proxy 0.99 is crossed on some mid-depth `out` and on L63 `down`; that is not generate.

---

## KILLS / REOPEN_IF

| claim | status | REOPEN_IF |
|---|---|---|
| group-MSE vs absmax, same g=64, same bits, this metric | **MEASURED_WIN** on 34 layers × all GEMV classes × {2,3,4}. 0 LOSEs. | a site-correct thick capture flips the sign on a write tensor |
| α=1 AWQ fold from this capture | **KILLS** (L0 out Q4 0.99225→0.91865; L0 every class Q3/Q4 LOSE) | mixer_x / confirmed in-proj X exists and α=1 stops killing |
| α=0.25 as a proxy | **SUPPORTED** as cheap same-BPW helper; **dominated** by Gram search (728/732) | — |
| `\|W\|` exact-42 | **KILLS** (CITED sibling 0.95327, overlap 0/42). Not re-run. | — |
| 256-token plane as a pack source | **KILLS** | N≥2048, sequence holdout, site-correct X, scale even/odd spearman ≥0.95 **and** a native generate vs the Q4 oracle |
| Q3 attention licensed by residual 0.99 | **not licensed**. L63 o residual 0.98831. Mid-depth out residual crosses; late GQA o does not. | GPU-lane native generate of MSE-scaled Q3 write tensors vs Q4 oracle |
| 1-bit rung | **not measured**. bound=0. | a real 1-bit quantizer |
| Q3 g=64 as a 3.25 BPW shipping codec | **not available**. Qn kernel is g=128. Q3-in-Q4-nibble is still 4.25 physical. | a g=64 Q3 bitstream + kernel, or accept 4.25 physical |

---

## Command output (this lane)

```
$ /opt/homebrew/bin/python3 /tmp/qwen38_mse_scale_rule.py
[11:47:38] rss_max=0.038G ===== REPRO L0 out_proj sibling protocol =====
[11:47:40] rss_max=1.737G   repro mse=0.974593839874 abs=0.953103454805 err_mse=6.033e-06
[11:47:40] rss_max=1.737G ===== F16 SNAP vs F32 SEARCH L0 out Q3/Q4 =====
[11:47:42] rss_max=1.747G   q3 f32=0.974594 f16=0.974596 d=+2.66e-06
[11:47:45] rss_max=1.747G   q4 f32=0.993150 f16=0.993142 d=-7.98e-06
[11:47:45] rss_max=1.748G ===== FINER MULTIPLIER GRID L0 out Q3 =====
[11:47:49] rss_max=1.748G   fine mse=0.976006 d_vs_8pt=+0.001406 wall=4.34
[11:47:49] rss_max=1.748G ===== LAYOUT / KERNEL CONSUMPTION PROOF L0 out Q4 =====
[11:47:51] rss_max=2.119G   payload mse=16711720 abs=16711720 equal=True decode_err=0.000e+00
[11:47:52] ... L0 in_proj_qkv q3 abs=0.97947 mse=0.98738 d=+0.00791 WIN
[11:48:17] ... L0 out_proj    q3 abs=0.95310 mse=0.97459 d=+0.02149 WIN
[11:48:54] ... L0 down        q3 abs=0.99147 mse=0.99535 d=+0.00387 WIN
[12:07:16] ... L32 out_proj   q3 abs=0.96789 mse=0.98131 d=+0.01341 WIN
[12:25:17] ... L63 o          q3 abs=0.96040 mse=0.97835 d=+0.01794 WIN
[12:25:53] ... L63 down       q3 abs=0.97279 mse=0.99060 d=+0.01781 WIN
[12:26:13]   spearman even/odd=0.8764 first/last=0.9219 same_m even/odd=0.827   # L0 out
[12:27:03]   spearman even/odd=0.7499 first/last=0.8444 same_m even/odd=0.748   # L0 down
[12:28:02] ===== L63 lm_head ... rss=14.855G =====
[12:29:01]     q3 abs=0.99420 mse=0.99636 d=+0.00216 WIN search_s=43.78
[12:29:58]   embed observed=63 mse=0.97626 abs=0.96982
[12:29:58] DONE wall_s=2539.6 rss_max_gb=32.057 n_rows=732
```

```
$ shasum -a 256 /tmp/qwen38_mse_scale_rule.json /tmp/qwen38_mse_scale_rule.py
bcdf0a4595cc620ad31aca66c3420a8e692d9988df8d42b58a05e4513cb0f3cc  /tmp/qwen38_mse_scale_rule.json
fa84927b28a8a41375b10ee3a70e9aa43c42acde5f99b2e5bb0db4ec664e7c35  /tmp/qwen38_mse_scale_rule.py
```

JSON excerpts (`summary.rollup`, `reproduce_sibling`, `layout_proof`, `stability.L0_out_proj_q3`, `lm_head.ops`).

---

## What this lane did not measure

- Generate / greedy-id / token identity. Serialized GPU lane owns that.
- True DeltaNet recurrent mix or softmax GQA mix as out/o X. Proxy only.
- Pre-norm residual (RMSNorm not inverted).
- The 30 unswept DN layers.
- Discrete per-group code-boundary search.
- Q2/Q3 at production g=128 (`qwen_uniform_qn_*`).
- A packed artifact.

Cheapest next experiment that is generate-facing: pack L0/L32 `out_proj` and L3/L63 `o_proj` (and, if bytes allow, L63 `down`) with group-MSE Q4 scales into the existing HQ30UQ4 container; leave everything else at G0 Q4; GPU lane scores native generate vs the Q4 oracle. Do not AWQ-fold. Do not expand to float. Do not treat that pack as a frozen production plane — it is a generate probe of the sign.

Cheapest next experiment that licenses a production plane: the doctor-recovery §5 capture (N=2048 census minimum; site-correct mixer_x and post_swiglu; sequence holdout), then re-fit and report hold cosine vs the 256-token plane.

---

## Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
1. Sibling surviving rule reproduces. L0 out Q3 absmax 0.9531034548050097 bit-identical; group-MSE 0.9745938398738019 vs claimed 0.974599872428159 (abs err 6.033e-6, f32 vs f64 rint). L32 out Q3 0.96789→0.98131. L63 o Q3 0.96040→0.97835. Evidence: /tmp/qwen38_mse_scale_rule.json reproduce_sibling and layers.32/63.
2. The rule generalizes. 34 layers, every language GEMV class, bits {2,3,4}: 732 cells, 722 WIN, 10 NEUTRAL, 0 LOSE vs absmax. Neutrals are Q4 ceiling (absmax≥0.997, Δ still >0). Q2 mean Δ is +0.08 to +0.14. Q3 +0.007 to +0.015. Q4 +0.0008 to +0.002. Evidence: summary.rollup.
3. Uniform MSE-vs-absmax is not the 2.0856 misallocation. Same complete BPW. Applying it to MLP and in-proj does not lose on this metric. Applying a low bit width uniformly is a different, unlicensed claim. Evidence: rollup; L63 o Q3 mixer-output 0.97835 still <0.99.
4. α=1 is dead on every organ scored at Q3/Q4, not just out_proj. L0 out Q4 0.99225→0.91865. α=0.25 wins everywhere MSE wins and loses the head-to-head 728/732. Evidence: ops.alpha1_* on L0/L31/L32/L63; a025 vs mse count.
5. Production form is a pack-time 8-point Gram search writing f16 scales into HQ30UQ4. Cost MEASURED 1–4 s/tensor (lm_head 43 s). Full-model Q4-only ESTIMATED ~20 min at 4 threads. Capture needed before a pack is generated from: N≥2048, sequence holdout, site-correct mixer_x and post_swiglu. Evidence: search wall_s; doctor-recovery §5.2; stability spearman 0.75–0.88.
6. Scales are consumable by qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 with no kernel change. L0 out Q4 payload 16,711,720 bytes either rule; kernel-faithful unpack max abs err 0. Evidence: layout_proof; shaders/qwen_uniform_q4.metal:10-11,183-209; qwen80_uniform_q4.rs:233-247.
7. 256 tokens is thick enough to rank and to sign the rule, not thick enough to estimate a production scale plane. Evidence: stability.*; CITED sibling thinness; rpd 256/6144=0.0417.

EVIDENCE
- /tmp/qwen38_mse_scale_rule.json sha256 bcdf0a4595cc620ad31aca66c3420a8e692d9988df8d42b58a05e4513cb0f3cc
- command log in this file, wall 2539.6 s, rss_max 32.057 GB (lm_head); sweep rss 9.373 GB
- workspace/superwave/g1/g1-scale-contradiction.md (sibling rule)
- workspace/superwave/g1/g1-doctor-recovery.md §5.2 / §5.4 (capture N)
- crates/hawking-core/shaders/qwen_uniform_q4.metal
- crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs
- crates/hawking-core/src/model/qwen_complete_binary/uniform_qn.rs:10 (Qn g=128)
- .../activation-capture-v1/capture-result.json sha256_self fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512

CHANGES
workspace/superwave/g1/g1-mse-scale-rule.md (this file). No other path touched.

TESTS
see end of lane message (test -s, wc -l, git status --porcelain)

RISKS
- out/o X is a mixer-site proxy, not the recurrent / softmax mix.
- down X is reconstructed SwiGLU, not a captured intermediate.
- lm_head X is L63 post-norm, not confirmed final-norm.
- residual-proxy uses post-norm hidden, not pre-norm residual.
- 8-point search, 128 fit rows, 34/64 layers. Sign is consistent. Shipped scale values are not.
- Mixer-output 0.9746 / residual-proxy 0.9994 is not generate.
- lm_head search exceeded the 15 GB RSS cap (32.057 GB). A packer must stream that tensor.
- 30 DN layers unswept; do not interpolate a scale plane.

UNRESOLVED
- Whether MSE-scaled Q4 write tensors are token-safe vs the Q4 oracle.
- Whether a thicker, site-correct X moves L63 o Q3 residual across 0.99.
- Whether a code-boundary search beats the 33-point leftover.
- Q2/Q3 at production g=128.

NEXT
Do not pack a production plane from this capture. GPU lane: HQ30UQ4 of L0/L32 out and L3/L63 o (optional L63 down) with group-MSE Q4 scales, native generate vs Q4 oracle. Capture lane: doctor-recovery §5, N=2048 minimum, then re-fit.
```
