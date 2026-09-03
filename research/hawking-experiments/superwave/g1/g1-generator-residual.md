# G1 generator + incoherent residual — Qwen3.8

**STATUS: FALSIFIED** as a complete-model storage mechanism.

Weight-space SVD on the on-disk BF16 Qwen3.8-27B tensors does not produce a cheap generator whose residual then quantizes “far better”. Per-class spectra are above an iid-Gaussian control but operationally flat for any rank that fits a G1 bit budget. Residual codecs improve relative Frobenius error by ~3–9% on the mass-dominant classes and ~13–33% only on the tiny GQA `k_proj`/`v_proj` matrices. No tested `(r, residual codec)` pair hits incumbent Q4 reconstruction error at a lower *total* BPW than Q4 of the original (factors counted at f16). No pair hits `rel_l2 ≤ 0.15` at `total_bpw < 2.0`.

This is a weight-space / storage measurement. It is not a token-level claim and not a Metal timing claim.

---

## 1. What was measured

| item | value | label |
|---|---|---|
| artifact | `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` | on-disk BF16 safetensors |
| identity | Qwen3.8-27B, `qwen3_5`, 64 layers, dense SwiGLU, hybrid DeltaNet/GQA | from `QWEN38_ARCH_CENSUS.json` + `qwen38_geometry.rs` |
| tensors | 48 language GEMV weights | MEASURED |
| layers | MLP `{0,3,15,31,47,63}`; GQA `{3,15,31,63}`; DeltaNet `{0,16,32,48}`; plus embed + lm_head | MEASURED |
| SVD | exact if `min(m,n)≤1536`, else rSVD `k=256 p=16 q=2` (`k=128` embed/lm_head) | MEASURED |
| error | weight-space `‖W−Ŵ‖_F / ‖W‖_F` | MEASURED |
| codecs | `binary_g128`, `uniform_q{2,3,4}_g64` (symmetric maxabs for uniform; mean-abs for binary; f16 scale/group) | MEASURED |
| factors | `A=U[:,:r]·S[:r]`, `B=Vh[:r]`, stored f16, S folded; `factor_bytes = r(m+n)·2` | MEASURED |
| GPU / generate / pack | not run | binding + lane contract |
| activation-weighted SVD | not run | see REOPEN_IF |
| vision tower | excluded | 333 tensors, text path skips them |

Codec cross-check against prior descent (same L0 `gate_proj`): this run `binary_g128` cosine `0.79836135` and `uniform_q4_g64` `rel_l2=0.10873971` match `receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json` (`0.7983613876884476`, `0.10873951632127646`). Same affine/sign-scale family.

Raw machine result: `/tmp/g1_gen_residual.json` (48 tensors, `errors: []`, `rss_max_gb: 10.542`). Scripts: `/tmp/g1_gen_residual.py`, `/tmp/g1_gen_residual_embed.py`. First embed attempt allocated extra full copies inside `stats()` and hit `ru_maxrss=23.621G`; that process was killed. Embed/lm_head were redone streamed. Successful-run peak is 10.542 GB.

Geometry used to name classes (`crates/hawking-core/src/model/qwen38_geometry.rs`):

```21:52:crates/hawking-core/src/model/qwen38_geometry.rs
pub const QWEN38_DELTANET_LAYERS: usize = 48;
pub const QWEN38_GQA_LAYERS: usize = 16;
pub const QWEN38_FULL_ATTENTION_INTERVAL: usize = 4;
pub const QWEN38_HIDDEN: usize = 5_120;
pub const QWEN38_INTERMEDIATE: usize = 17_408;
pub const QWEN38_VOCAB: usize = 248_320;
...
pub const QWEN38_Q_PROJ_ROWS: usize = 12_288;
pub const QWEN38_KV_PROJ_ROWS: usize = 1_024;
pub const QWEN38_O_PROJ_ROWS: usize = 5_120;
pub const QWEN38_O_PROJ_COLS: usize = 6_144;
```

```82:90:crates/hawking-core/src/model/qwen38_geometry.rs
/// Source rule: GQA iff `(layer + 1) % 4 == 0` (layers 3, 7, …, 63).
pub fn qwen38_mixer_kind(layer: usize) -> Result<Qwen38MixerKind> {
    ...
    if (layer + 1) % QWEN38_FULL_ATTENTION_INTERVAL == 0 {
        Ok(Qwen38MixerKind::Gqa)
```

`q_proj` is 12288×5120 (output-gated), not 24×256.

---

## 2. Verdict in one line per class

Never average these into one number.

| class | n | energy@64 mean [min,max] | vs Gaussian@64 | r=64 binary rel improvement | r=64 Q4 rel improvement | Q4-eq error at lower total BPW? |
|---|---:|---:|---:|---:|---:|---|
| `mlp.gate_proj` | 6 | 0.0808 [0.0625, 0.1131] | 3.19× | 1.044 | 1.047 | no |
| `mlp.up_proj` | 6 | 0.0631 [0.0350, 0.1063] | 2.49× | 1.034 | 1.036 | no |
| `mlp.down_proj` | 6 | 0.0667 [0.0481, 0.0849] | 2.63× | 1.035 | 1.037 | no |
| `full.q_proj` | 4 | 0.1455 [0.1104, 0.1623] | 5.74× | 1.086 | 1.100 | no |
| `full.k_proj` | 4 | 0.2769 [0.2301, 0.3612] | 10.92× | 1.210 | 1.279 | no |
| `full.v_proj` | 4 | 0.2136 [0.1782, 0.2578] | 8.43× | 1.132 | 1.136 | no |
| `full.o_proj` | 4 | 0.1482 [0.1188, 0.1828] | 5.85× | 1.088 | 1.095 | no |
| `lin.in_proj_qkv` | 4 | 0.1258 [0.0799, 0.2218] | 4.96× | 1.073 | 1.080 | no |
| `lin.in_proj_z` | 4 | 0.1350 [0.1019, 0.1551] | 5.33× | 1.076 | 1.082 | no |
| `lin.out_proj` | 4 | 0.1585 [0.1364, 0.2077] | 6.26× | 1.092 | 1.090 | no |
| `embed` | 1 | 0.0672 | 2.65× | 1.036 | 1.037 | no |
| `lm_head` | 1 | 0.1009 | 3.98× | 1.062 | 1.093 | no |

Source: `/tmp/g1_gen_residual.json` `summary_by_class` lines 36023–36167.

Gaussian control, same MLP shape 17408×5120, iid `N(0,0.02²)`:

```36001:36020:/tmp/g1_gen_residual.json
  "gaussian_control": {
    "kind": "iid_gaussian_control",
    "shape": [17408, 5120],
    "energy_frac": {
      "64": 0.02534171874841687,
      "128": 0.04895399716312801,
      "256": 0.09179178793146126
    },
    "s1_over_s64": 1.044512210656373
  },
```

Spectra are **not** Gaussian-flat. They are **too flat to pay for**. Rank-64 leaves 88–96% of MLP / embed Frobenius mass in the residual. That is the kill.

---

## 3. Singular-value energy sweep (MEASURED)

`energy_frac[r] = Σ_{i<r} σ_i² / ‖W‖_F²`. For rSVD tensors this is the captured mass of the computed `k` values, not a full spectrum. `approx_captured_energy_of_computed_S` at k=256 is ~0.15–0.24 on MLP, so the missing tail is the majority.

Selected rows (full 48-tensor table in `/tmp/g1_gen_residual.json`):

| class | L | shape | e1 | e8 | e32 | e64 | e128 | e256 | σ1/σ64 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| gate | 0 | 17408×5120 | 0.0048 | 0.0187 | 0.0512 | 0.0874 | 0.1491 | 0.2427 | 2.11 |
| up | 0 | 17408×5120 | 0.0028 | 0.0137 | 0.0422 | 0.0757 | 0.1337 | 0.2227 | 1.67 |
| **down** | **0** | 5120×17408 | **0.0254** | 0.0329 | 0.0514 | 0.0702 | 0.1018 | 0.1540 | **6.84** |
| gate | 15 | 17408×5120 | 0.0037 | 0.0137 | 0.0400 | 0.0675 | 0.1111 | 0.1746 | 2.16 |
| up | 15 | 17408×5120 | 0.0015 | 0.0098 | 0.0324 | 0.0571 | 0.0977 | 0.1593 | 1.45 |
| down | 15 | 5120×17408 | 0.0027 | 0.0169 | 0.0507 | 0.0849 | 0.1365 | 0.2064 | 1.68 |
| gate | 63 | 17408×5120 | 0.0210 | 0.0481 | 0.0834 | 0.1131 | 0.1496 | 0.1984 | 5.37 |
| up | 63 | 17408×5120 | 0.0223 | 0.0487 | 0.0816 | 0.1063 | 0.1400 | 0.1888 | 5.87 |
| down | 63 | 5120×17408 | 0.0026 | 0.0137 | 0.0349 | 0.0566 | 0.0911 | 0.1459 | 2.06 |
| q | 15 | 12288×5120 | 0.0194 | 0.0543 | 0.1129 | 0.1623 | 0.2311 | 0.3178 | 3.82 |
| k | 15 | 1024×5120 | 0.0052 | 0.0359 | 0.1278 | 0.2301 | 0.3813 | 0.5896 | 1.35 |
| **k** | **63** | 1024×5120 | 0.0133 | 0.0712 | 0.2237 | **0.3612** | **0.5227** | **0.7031** | 1.97 |
| v | 63 | 1024×5120 | 0.0433 | 0.0924 | 0.1718 | 0.2578 | 0.3976 | 0.6056 | 4.15 |
| o | 15 | 5120×6144 | 0.0072 | 0.0391 | 0.1141 | 0.1828 | 0.2760 | 0.3884 | 1.98 |
| **in_qkv** | **0** | 10240×5120 | **0.0768** | **0.1287** | 0.1832 | **0.2218** | 0.2708 | 0.3310 | **8.87** |
| in_qkv | 16 | 10240×5120 | 0.0035 | 0.0212 | 0.0615 | 0.1014 | 0.1619 | 0.2451 | 1.77 |
| out | 0 | 5120×6144 | 0.0694 | 0.1057 | 0.1692 | 0.2077 | 0.2518 | 0.3168 | 8.79 |
| embed | — | 248320×5120 | 0.0101 | 0.0260 | 0.0479 | 0.0672 | 0.0911 | — | — |
| lm_head | — | 248320×5120 | 0.0610 | 0.0710 | 0.0871 | 0.1009 | 0.1199 | — | — |

L0 `down_proj` spectrum (rank-1 spike, then flat) from the result file:

```1568:1588:/tmp/g1_gen_residual.json
      "energy_frac": {
        "1": 0.02535176542452853,
        "8": 0.03293428106979786,
        "64": 0.07015253794854669,
        "256": 0.15396710593416377
      },
      "spectrum": {
        "0": 15.98422622680664,
        "1": 3.591336727142334,
        "63": 2.337270498275757,
        "255": 1.9046088457107544
      },
```

σ1/σ2 ≈ 4.45 on that tensor; σ2…σ256 decay like the Gaussian control. One cheap scale mode, not a useful subspace.

### PROJECTED energy-weighted r=64 (class-mean × class param count)

Language GEMV only, vision excluded. Class means from the 48 measured tensors, applied to full 64/16/48 copies.

| class | params | e64_mean | σ² captured (projected) |
|---|---:|---:|---:|
| mlp.{gate,up,down} | 3 × 5.704e9 | 0.0808 / 0.0631 / 0.0667 | 1.201e9 |
| GQA q/k/v/o | 1.678e9 | 0.146 / 0.277 / 0.214 / 0.148 | 0.262e9 |
| DeltaNet qkv/z/out | 5.536e9 | 0.126 / 0.135 / 0.159 | 0.760e9 |
| embed+lm_head | 2.543e9 | 0.067 / 0.101 | 0.214e9 |
| **total** | **26.870e9** | | **2.436e9** |

**PROJECTED** energy-weighted explained fraction at shared r=64: **0.0907**.

Prior unverified/refuted shared-r64 claim on the *Q4* artifact was 0.0405 energy-weighted / 0.141 unweighted mean (`receipts/ascent-2026-08-16/GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json`). Direction matches: r=64 is a small slice of mass. Their number is on already-quantized Q4, so it should sit below this BF16 figure.

---

## 4. Residual distribution after removing top-r (MEASURED)

Hypothesis part 1: residual is closer to incoherent noise.

On most MLP gate/up tensors, W is already near-Gaussian (excess kurtosis 0.05–0.40). Removing r=64 barely moves kurtosis or peak/rms. Residual top-32 energy of R is 1.5–3.2% — flatter than W, because the little structure that existed was the top of W.

Where W *does* have a heavy tail, residual flattening is real and large:

| tensor | kurt(W) | kurt(R@64) | peak/rms W | peak/rms R |
|---|---:|---:|---:|---:|
| L0 `down_proj` | **15.52** | **0.17** | **92.2** | **24.1** |
| L3 `o_proj` | **132.1** | **0.31** | 90.1 | 31.4 |
| L0 `lin.out_proj` | **149.4** | **1.68** | 85.9 | 31.7 |
| L3 `down_proj` | 0.70 | 0.09 | 61.3 | 26.4 |
| L0 `gate_proj` | 0.31 | 0.24 | 41.5 | 32.5 |
| L63 `k_proj` | 6.06 | 1.54 | 19.8 | 20.8 |

L0 `down_proj` weight_stats (the tail that SVD removes):

```1549:1557:/tmp/g1_gen_residual.json
        "max_abs": 0.98046875,
        "peak_over_rms": 92.20525327259023,
        "excess_kurtosis": 15.521897493396448,
```

So: a *dynamic-range* generator exists on a few organs (especially early `down_proj` / `out_proj`). It is a rank-1 / low-rank spike, not a 64-dimensional subspace, and it is a few percent of Frobenius mass. Flattening the tail does not move group-uniform or binary error enough to matter (next section).

f16 vs f32 factors: extra `rel_l2` ≤ 4.6e-4 on every tested `(tensor, r)`. **MEASURED: f16 factors are enough.**

---

## 5. Residual bits vs target reconstruction error (MEASURED)

Targets used:

- incumbent Q4 `rel_l2` on the *same* tensor (≈ 0.108–0.123 except late k/v)
- 0.05, 0.10, 0.15

Storage: `total_bpw = 8 · (r(m+n)·2 + residual_payload) / (m·n)`.

### 5.1 Incumbent (no generator)

Class-mean original `rel_l2`:

| class | binary 1.125 bpw | q2 2.25 | q3 3.25 | q4 4.25 |
|---|---:|---:|---:|---:|
| mlp.gate | 0.604 | 0.721 | 0.256 | 0.110 |
| mlp.up | 0.604 | 0.720 | 0.255 | 0.109 |
| mlp.down | 0.607 | 0.723 | 0.259 | 0.111 |
| full.k | 0.627 | 0.733 | 0.286 | 0.124 |
| embed | 0.601 | 0.718 | 0.252 | 0.108 |

Binary of W is ~0.60. Q4 of W is ~0.11. Those are the two poles.

### 5.2 After removing r=64 (f16 factors)

L0 `gate_proj` (typical MLP, 89.1e6 weights):

| store | rel_l2 | total_bpw |
|---|---:|---:|
| Q4(W) | 0.1087 | 4.250 |
| binary(W) | 0.6022 | 1.125 |
| r64 only (f16) | 0.9553 | 0.259 |
| r64 + binary(R) | 0.5751 | **1.384** |
| r64 + q2(R) | ~0.68 | 2.509 |
| r64 + q3(R) | ~0.24 | 3.509 |
| r64 + q4(R) | 0.1037 | **4.509** |

Binary residual is 4.7% better than binary(W) and **worse storage than binary(W)**. Q4 residual is 4.9% better than Q4(W) and **0.26 BPW more expensive**.

L0 `down_proj` (the tail-flattening poster child):

| store | rel_l2 | total_bpw | notes |
|---|---:|---:|---|
| Q4(W) | 0.1100 | 4.250 | |
| binary(W) | 0.6022 | 1.125 | |
| r8 + binary(R) | 0.5938 | 1.157 | kurtosis already collapsing; +1.4% |
| r64 + binary(R) | 0.5822 | 1.384 | +3.4% vs binary(W) |
| r64 + q4(R) | 0.1055 | 4.509 | +4.3% vs Q4(W) |

L63 `k_proj` (best spectrum, 5.24e6 weights — 0.31% of language GEMV):

| store | rel_l2 | total_bpw |
|---|---:|---:|
| Q4(W) | 0.1461 | 4.250 |
| binary(W) | 0.6667 | 1.125 |
| r64 + binary(R) | 0.5011 | **2.325** (factor_bpw=1.20) |
| r64 + q4(R) | 0.0986 | **5.450** |
| r256 + q4(R) | 0.0644 | 9.05 |

Best relative improvement in the whole sweep (1.33× binary, 1.48× Q4) and it still **loses on total BPW**. Skinny 1024×5120 pays `8·r·(m+n)·2/(mn) = 1.20` BPW at r=64 just for factors.

L0 `lin.in_proj_qkv` (best *large* matrix, 52.4e6 weights):

r=64 explains 0.222; binary residual 0.532 vs 0.602 (`imp=1.130`); total_bpw 1.425. Still ~5× the Q4 error.

### 5.3 Bits-to-target, whole sweep

Across all 48 tensors and all tested r ∈ {8,16,32,64,128,256} (embed/lm_head subset {8,32,64,128}):

| question | count |
|---|---:|
| any `(r,codec)` with `rel_l2 ≤ Q4(W)` **and** `total_bpw < Q4 bpw` | **0** |
| any `(r,codec)` with `rel_l2 ≤ 0.15` **and** `total_bpw < 2.0` | **0** |
| any residual **binary / q2 / q3** hitting Q4(W) error | **0** |
| `rel_l2 ≤ 0.10` at all | 69, every one is `uniform_q4_g64` residual **plus** factors, total_bpw 4.40–9.05 |

To reach Q4-equivalent weight error the residual still wants 4 bits. Adding the generator then *increases* bytes. That is the same arithmetic the prior adjudication already applied to a dishonest 1.125-BPW residual headline:

> “The HQ30UQ4 residual is exactly as large as the original Q4 tensors; adding the generator increases storage by 726,151,136 bytes.”
> — `receipts/ascent-2026-08-16/GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json` `adjudication[1]`

This run reproduces that conclusion on BF16, per class, with honest f16 factor cost.

---

## 6. `down_proj` does not invert the ranking on Qwen3.8 weight-space

Q80 prior: expert `down_proj` (2048×512) preferred activation-weighted low-rank (`hgravs01`) while gate/up preferred binary. That is **not** a Qwen3.8 dense weight-space fact.

| | gate e64 | up e64 | down e64 |
|---|---:|---:|---:|
| mean | 0.0808 | 0.0631 | **0.0667** |
| min | 0.0625 | 0.0350 | 0.0481 |
| max (L63 gate/up, L15 down) | 0.1131 | 0.1063 | 0.0849 |
| r64 binary improvement | 1.044 | 1.034 | 1.035 |

`down_proj` is *less* low-rank than `gate_proj`. L0 `down` has a rank-1 spike (e1=0.025) that flattens kurtosis 15.5→0.17; that is a scale mode, not Q80’s “rank 160 of 512 rows = 31%”.

Qwen3.8 descent already recorded a *different* inversion, in **output-space quality vs bits**, not SVD energy (`QWEN38_BPW_DESCENT.json` `findings`):

- L63 gate/up: binary hold cosine 0.95+ (easy)
- mid-depth up/down: binary hold 0.73–0.79 (fails moderate bar)
- L47/L63 `down_proj`: cheapest codec that cleared moderate was `uniform_q3_g64` at 3.25 BPW, not binary

This run’s weight-space spectra agree that L63 gate/up are the most structured MLP matrices (e64 0.113 / 0.106) and L63 down is not (0.057). The Q80 low-rank-for-down preference does not transfer to these 5120×17408 matrices under weight-space SVD.

`QWEN38_BPW_DESCENT.json` `findings.lowrank_is_different_on_dense` noted that r=160 on 5120×17408 is 3.1% of rows = 0.13 BPW if consumed as `L@(R@x)`. That algebra is cheap. This measurement says the *quality* of that 0.13 BPW generator is a 96% residual, so the residual still needs ~Q4. Net loss.

---

## 7. Relation to G0 / prior generator residual

| claim | source | this run |
|---|---|---|
| shared-r64 explains 4.05% energy-weighted on Q4 | adjudication `measured_facts_retained` (Q4 artifact, not independently re-run here) | PROJECTED 9.07% on BF16 language GEMV at per-tensor r=64 |
| 1.125 residual BPW + net reduction | **REFUTED** by adjudication (they counted lossy binary residual, not the admissible HQ30UQ4 residual) | CONFIRMED REFUTATION: residual that matches Q4 error is still ~4.25 BPW; factors add 0.05–1.20 BPW |
| G0 complete BPW 4.2527 / 26.4 TPS / 37.9 ms | campaign claim, **not remeasured** (GPU lane owns timing) | unused |
| sub-1.5 pack incoherent | `QWEN38_SUB15_INCOHERENT.json` (binary gate + rice up + r160 down) | consistent: binary residual of a ~7% generator is still a ~0.58 rel_l2 matrix |

G1 targets (complete BPW < 1.5, TOKEN_NS ≤ 10 ms, TPS ≥ 100) are not addressed by this mechanism. A 1.38 BPW r64+binary pack would be below 1.5 on paper and would be the same class of incoherent store as `mixed-sub15-v1`.

Binding: even a winning residual store would be rejected if the production path is “decode to float/Q4 then generic GEMV” without a complete-token win. No kernel was written. Irrelevant: storage already loses.

---

## 8. KILLS / REOPEN_IF

**KILLS** generator + incoherent-residual as a G1 *weight-space* path on Qwen3.8:

1. MLP / embed / lm_head spectra are operationally flat (r=64 captures 6–11%; r=256 captures 16–24%).
2. Residual of those classes quantizes ~3–6% better, not “far better”.
3. Honest factor+residual BPW is strictly worse than Q4 at Q4 error, and strictly worse than binary at binary error.
4. The only structured class (`k_proj`, then `v_proj`) is 0.6% of language GEMV and pays a large factor_bpw because it is skinny.

**REOPEN_IF** (cheapest next experiment, not done here):

1. **Activation-weighted SVD on `down_proj`** using the existing capture at `.../qwen38-27b/activation-capture-v1` (256 real post-norm hidden tokens; reconstruct post-SwiGLU X = `silu(H W_gᵀ) ⊙ (H W_uᵀ)`). Score `‖X W − X Ŵ‖`, not `‖W−Ŵ‖`. This is the Q80 `hgravs01` question on dense geometry. Weight-space being flat does not kill output-space rank on a 256-token sample — but 256 tokens also cannot certify rank > 256.
2. **Rank-1 / rank-8 *scale* generator only**, stored as a vector of f16 row scales plus a shared direction, not `r(m+n)`. Targets the L0 down / early out_proj kurtosis-15–150 tails. Measure whether binary/q3 of that residual moves hold cosine on the descent activation set. Storage would be `O(m+n)` not `O(r(m+n))`.
3. **Per-class mixed, not shared r**. Only `k_proj`/`v_proj`/`L0 in_proj_qkv` have enough spectrum to discuss. Mass is too small to move complete BPW unless (2) also hits MLP.
4. A residual *entropy* coder on already-flat R, consumed by a representation-specific Metal kernel, **and** a protected-generate coherence run. Do not reopen for another weight-space rSVD.

Do **not** reopen for: another shared-r64 on Q4; reconstruct-to-Q4-then-GEMV; averaging classes.

---

## 9. What this does *not* say

- Not a complete-token measurement. No generate, no Metal, no TPS, no TOKEN_NS.
- Not an output-space / activation-weighted measurement except by reference to `QWEN38_BPW_DESCENT.json`.
- Vision tower unmeasured.
- `in_proj_a` / `in_proj_b` (48×5120) unmeasured; they are already tiny.
- rSVD `k=256` does not give the exact tail of the spectrum; energy@r for r≤128 is the quantity used for decisions and is stable (power iteration q=2, and exact SVD on k/v agrees with the same story).
- G0 4.2527 BPW / 26.4 TPS remains an unverified campaign claim.

---

## 10. Evidence appendix (command output)

### 10.1 Artifact + codec identity

```
$ python3 -c '... header of model-00001 ...'
language_model.model.embed_tokens.weight {'dtype': 'BF16', 'shape': [248320, 5120]}
language_model.model.layers.0.mlp.down_proj.weight {'dtype': 'BF16', 'shape': [5120, 17408]}
language_model.model.layers.0.mlp.gate_proj.weight {'dtype': 'BF16', 'shape': [17408, 5120]}
language_model.model.layers.15.self_attn.q_proj.weight {'dtype': 'BF16', 'shape': [12288, 5120]}
language_model.lm_head.weight {'dtype': 'BF16', 'shape': [248320, 5120]}
```

L0 `gate_proj` smoke (codec parity with descent):

```
shape=(17408,5120) ||W||_F=95.9871
energy {'1': 0.004755..., '64': 0.087377..., '256': 0.242716...}
q4 0.10873971018218612
orig_binary {'rel_l2': 0.6021785858525733, 'cosine': 0.7983613497334838, 'bpw': 1.125}
r64 binary {'rel_l2': 0.575139..., 'total_bpw': 1.38382..., 'rel_improvement': 1.04701...}
```

### 10.2 Full sweep log (excerpt)

```
[11:00:01] rss_max=0.187G start threads=8 bf16=.../qwen38-27b/bf16
[11:00:01] rss_max=0.187G GAUSSIAN CONTROL 17408x5120
[11:00:07] DONE mlp.gate_proj energy@64=0.08737742476868851 q4_rel=0.1087
[11:00:17] DONE mlp.down_proj energy@64=0.07015253794854669 q4_rel=0.1100
[11:01:24] DONE mlp.down_proj L63 energy@64=0.056613780961659164 q4_rel=0.1168
[11:02:25] DONE full.k_proj L63 energy@64=0.3611875872906905 q4_rel=0.1461
[11:02:37] DONE lin.in_proj_qkv L0 energy@64=0.22182586217703099 q4_rel=0.1088
[11:03:17] rss_max=23.621G LOAD embed   # killed; extra copies in stats()
[11:05:07] resume embed+lm_head streamed
[11:05:36] DONE embed energy@64=0.06717105705139495 q4_rel=0.1081 rss_max=9.741G
[11:06:03] DONE lm_head energy@64=0.10093692658756094 q4_rel=0.1123 rss_max=10.542G
WROTE /tmp/g1_gen_residual.json tensors=48
```

### 10.3 Hits that would have saved the mechanism

```
===== HITS Q4 REL AT LOWER TOTAL BPW THAN INCUMBENT Q4 =====
n_hits 0
===== HITS REL<=0.15 with total_bpw<2.0 =====
n 0
```

### 10.4 Prior adjudication (git, not re-derived)

`git show HEAD:receipts/ascent-2026-08-16/GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json`:

```
"classification": "NEGATIVE_SCIENCE",
"headline_status": "REFUTED",
"shared_r64_energy_weighted_explained_fraction": 0.0404921380578193,
"corrected_net_reduction": false
```

### 10.5 Result file pointer

- `/tmp/g1_gen_residual.json` lines 1–19 method; 36001–36020 gaussian; 36023–36167 `summary_by_class`; 36169 `rss_max_gb=10.542221069335938`; 1540–1578 L0 down_proj energy + tail stats.
- 48 tensors, 0 errors.

---

## Completion

Lane write-scope is this file only.
