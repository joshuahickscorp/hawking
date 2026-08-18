# G1 shared basis — Qwen3.8-27B

STATUS: **FALSIFIED** for this parent.

Family under test: one shared hidden-side basis / shared row codebook fitted across all
layers of a tensor class, or a within-layer shared head basis, as a route to a
genuinely sub-0.5 complete BPW. This is the Qwen3.8 analogue of the Q80
cross-expert shared-basis claim.

Every number below is **MEASURED** on the real BF16 language tensors unless
labelled NULL (Monte Carlo) or PRIOR (another receipt). No GPU, no Metal, no
token generate, no pack. Peak RSS of the measurement process: **9.22 GiB**.

---

## Verdict

Same-class tensors in different layers are mutually near-orthogonal as flattened
vectors. Adjacent-layer cosine sits at **1e-5 .. 8e-3**. That is the same regime
as the Q80 expert result (gate pairwise cosine **0.00414**). A single shared
rank-256 hidden basis leaves **69–93 %** relative Frobenius residual; the
per-layer basis of the same rank leaves **38–87 %**. Adjacent residual
`||W_{l+1}−W_l||_F / ||W_{l+1}||_F` is **1.40–1.42 ≈ √2**, i.e. the next layer
is not a cheap delta of the previous one. Within-layer heads are also near
orthogonal (DeltaNet q/k/v mean cosine **~0**; GQA q-heads **0.022–0.048**).

KILLS the family on this parent.

REOPEN_IF:

1. A *different* parent whose same-class adjacent flattened cosine **mean ≥ 0.05**
   (the NS-010 bar) **and** whose shared rank-256 relative error is within
   **+0.05** of the per-layer error on the same tensors.
2. Adjacent `rel_delta_fro` mean **< 0.5** (delta actually smaller than the
   tensor, not √2).
3. Within-layer head cosine mean **≥ 0.10** **and** a shared-head codebook beats
   per-head at matched rate. Not observed here.
4. A *trained* shareable parent (learned generator / compressibility training).
   That is different weights. This measurement does not bind it.

A minority of **distance-16** pairs have moderate cosine (max **0.46** on GQA
`k_proj` L27–L43). That is not adjacent-layer sharing, does not make a
single all-layer basis work, and even those pairs still have
`rel_delta_fro ≈ 0.98–1.28`. It is not a reopen.

This is a **weight-space** measurement. It is not a token-level claim and not
an activation-aware claim.

---

## Prior result (does not transfer; quoted, not re-run)

Q80 layer-10, 96 of 512 routed experts, `receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json`:

```
11:      "pairwise_cosine_mean": 0.004142791032791138,
12:      "pairwise_cosine_p95": 0.007685263641178608,
13:      "subspace_overlap_top32": 0.02506876550614834
21:      "pairwise_cosine_mean": -5.968913319520652e-05,
23:      "subspace_overlap_top32": 0.020372524857521057
```

`receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json` NS-010
(lines 300–325) records that finding as REFUTED for Q80, `retry_when`:
"never on Q80"; reopen bar on a new model is pairwise cosine mean
"materially above ~0.05". Qwen3.8 is dense, not MoE
(`QWEN38_ARCH_CENSUS.json` line 33 `"DENSE_NOT_MOE"`; config
`text_config.num_experts` is null). The Q80 number was not transferred.
It was re-measured.

---

## What was loaded

**MEASURED.** Path
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16`
(the on-disk copy of `workspace/campaign/records/runs/qwen38-27b/bf16` from
census `download.path`). 11 BF16 safetensor shards, mlx header, 1184 tensors.

```
identity ( /tmp/qwen38_shared_basis.json lines 43401–43436 ):
  config_model_type: qwen3_5
  text_model_type:   qwen3_5_text
  num_hidden_layers: 64
  hidden_size:       5120
  intermediate_size: 17408
  num_experts:       null
  attn_output_gate:  true
  l0_gate_shape:     [17408, 5120]
  l0_gate_dtype:     BF16
  l0_gate_std:       0.010167245753109455
  l0_gate_finite:    true
```

Stdout `/tmp/qwen38_shared_basis_run/stdout.log:2`:

```
identity ok: {"n_header_tensors": 1184, "config_model_type": "qwen3_5",
"num_hidden_layers": 64, "hidden_size": 5120, "l0_gate_shape": [17408, 5120],
"l0_gate_std": 0.010167245753109455, "num_experts": null}
```

Geometry authority `crates/hawking-core/src/model/qwen38_geometry.rs:20-40`:
64 layers, 48 DeltaNet + 16 GQA, hidden 5120, intermediate 17408, GQA 24:4
head_dim 256, linear 16 key / 48 value heads dim 128. Mixer rule
`(layer+1)%4==0` is GQA.

Head splits used (source layout, not guessed):

- `in_proj_qkv` rows = `[Q 16×128 | K 16×128 | V 48×128]`
  (`qwen38_geometry.rs` `fuse_in_proj_qkvz`, q_src / k_src / v_src).
- GQA `q_proj` rows per head = `[q 256 | gate 256]`
  (`qwen38_device_activations.metal:133` `head * (2u * head_dim)`).

---

## Method (so the numbers are interpretable)

Script `/tmp/qwen38_shared_basis.py`. CPU only. BLAS threads 8.

- BF16 shards parsed without `safetensors` (8-byte header + JSON + payload).
  `u16 << 16` viewed as f32.
- Per tensor class, hidden-side Gram `G = Σ tiles` (right if `cols==5120`,
  left if `rows==5120`). Top-384 eigenpairs via Halko range finder + one
  power iteration (`eig_method: randomized_halko_power1`). Labelled
  approximate. Rank-k relative error uses `1 − Σ_{i≤k} λ_i / ||W||_F^2`
  (local) or a second-pass `||W V_shared[:,:k]||_F^2` (shared).
- Flattened cosine and `rel_delta_fro` are **exact** (tiled f32 accumulators
  in f64). Adjacent pairs plus distances {4,8,16,32}. CountSketch s=8192 is
  diagnostic only; exact-vs-sketch abs error mean **0.008–0.009**, max
  **0.03–0.04**. Sketch means are **not** used as claims.
- Shared codebook: k-means, 192 hidden-dim vectors sampled per layer,
  k ∈ {32,128}, 8 Lloyd iters. Metric = relative MSE of those samples.
  Not a production VQ fit.
- Head cosine: flatten each head matrix; top-32 hidden-subspace overlap
  as `||V_a^T V_b||_F^2 / 32`.
- NULL: 24 pairs of random orthonormal k-frames in R^{5120}.

```
/tmp/qwen38_shared_basis.json  null_subspace:
  k=8   mean 0.001564   theory k/n 0.0015625
  k=32  mean 0.006361   theory k/n 0.00625
  k=128 mean 0.024920   theory k/n 0.025
```

Stdout `:3` `null k32 mean=0.00636 theory=0.00625`.

Elapsed **3990.35 s**. RSS max **9.216 GiB**.
`/tmp/qwen38_shared_basis.json` lines 44332–44333.

---

## 1. Cross-layer flattened cosine — MEASURED, near zero

Exact adjacent cosine, all layers of the class:

| class | n | adj cos mean | adj min | adj max | adj rel_delta mean | √2 |
|---|---:|---:|---:|---:|---:|---:|
| `self_attn.k_proj` | 16 | +0.000399 | −0.001876 | +0.003055 | 1.4214 | 1.4142 |
| `self_attn.v_proj` | 16 | +0.000293 | −0.001711 | +0.001962 | 1.3955 | 1.4142 |
| `self_attn.q_proj` | 16 | +0.007632 | +0.003259 | +0.012517 | 1.4181 | 1.4142 |
| `self_attn.o_proj` | 16 | −0.000010 | −0.000909 | +0.000624 | 1.4097 | 1.4142 |
| `linear_attn.in_proj_z` | 48 | +0.002857 | +0.000025 | +0.009066 | 1.4128 | 1.4142 |
| `linear_attn.out_proj` | 48 | +0.000216 | −0.000285 | +0.001091 | 1.4135 | 1.4142 |
| `linear_attn.in_proj_qkv` | 48 | +0.000021 | −0.000416 | +0.000366 | 1.4134 | 1.4142 |
| `mlp.gate_proj` | 64 | +0.004354 | +0.000880 | +0.011759 | 1.4080 | 1.4142 |
| `mlp.up_proj` | 64 | +0.000014 | −0.000291 | +0.000303 | 1.4110 | 1.4142 |
| `mlp.down_proj` | 64 | +0.000061 | −0.000158 | +0.000339 | 1.4132 | 1.4142 |

Source: `/tmp/qwen38_shared_basis.json` `classes[*].adjacent`.

Highest adjacent class-mean is GQA `q_proj` **0.00763**. MLP `gate_proj`
**0.00435** is numerically the Q80 expert number. Both are an order of
magnitude below the 0.05 reopen bar.

Exact cosine by layer-index distance, MLP `gate_proj`
(`classes["mlp.gate_proj.weight"].exact_cosine_by_distance`):

```
d=1  n=63  mean 0.004354  min 0.000880  max 0.011759
d=4  n=60  mean 0.003455  min 0.000603  max 0.008744
d=8  n=56  mean 0.002388  min 0.000198  max 0.005833
d=16 n=48  mean 0.101285  min 0.000232  max 0.200028
d=32 n=32  mean 0.046825  min -0.000136 max 0.106698
```

`up_proj` / `down_proj` d=1 means are **1.4e-5** / **6.1e-5**. d=16 means
rise to **0.085** / **0.099**. Same pattern on every class.

Hottest exact pair per class (still not a shared-all-layers object):

```
k_proj     L27–L43 d=16  cos 0.462183  rel_delta 1.0788
v_proj     L27–L43 d=16  cos 0.484345  rel_delta 0.9767
q_proj     L27–L43 d=16  cos 0.233355  rel_delta 1.2855
in_proj_qkv L29–L45 d=16 cos 0.222931  rel_delta 1.2588
gate_proj  L29–L45 d=16  cos 0.200028  rel_delta 1.2658
```

One pair (`v_proj` L27–L43) has `rel_delta < 1`. It is 0.98, not 0.5.
All others stay ≥ 1.07. Period-16 echo exists; it is not a delta code.

---

## 2. Hidden-subspace overlap — MEASURED, above null, not usable

Top-32 right/left singular subspace overlap
`||V_i^T V_j||_F^2 / 32`, hidden dim 5120.

| class | pairwise mean | adj mean | far (Δlayer≥16) mean | NULL k=32 |
|---|---:|---:|---:|---:|
| `k_proj` | 0.1611 | 0.3476 | 0.1003 | 0.00636 |
| `v_proj` | 0.0424 | 0.1007 | 0.0250 | 0.00636 |
| `q_proj` | 0.0897 | 0.2345 | 0.0498 | 0.00636 |
| `o_proj` | 0.0958 | 0.2282 | 0.0569 | 0.00636 |
| `in_proj_z` | 0.0727 | 0.2954 | 0.0317 | 0.00636 |
| `out_proj` | 0.0852 | 0.3243 | 0.0533 | 0.00636 |
| `in_proj_qkv` | 0.0866 | 0.3375 | 0.0390 | 0.00636 |
| `gate_proj` | 0.0923 | 0.4400 | 0.0423 | 0.00636 |
| `up_proj` | 0.0474 | 0.2522 | 0.0209 | 0.00636 |
| `down_proj` | 0.0625 | 0.3009 | 0.0348 | 0.00636 |

Source: `classes[*].subspace_overlap["32"]`.

Adjacent layers share more of their **top-32** hidden directions than a
random 32-frame (null 0.006). That is a real spectral-head correlation.
It does not move bytes: see §3. Q80's expert top-32 overlap was 0.020–0.025
on a different matrix size; not numerically comparable, but the same
conclusion (overlap of the head ≪ a shared code).

Local spectra are not low-rank. Top-384 energy fraction mean on `gate_proj`
is **0.20-class** (k50/k90 not inside the 384-head; `k90` is None). Shared
head-384 energy fraction on `gate_proj` is **0.110**. The mass is not in a
shared low-dimensional subspace.

---

## 3. Shared basis vs per-layer — MEASURED, shared loses

Relative Frobenius residual after rank-k projection onto the hidden-side
basis. Shared = one V fitted on `Σ_l G_l`. Per-layer = one V per layer.

| class | k=32 local | k=32 shared | k=256 local | k=256 shared | shared/local @256 |
|---|---:|---:|---:|---:|---:|
| `k_proj` | 0.8430 | 0.9166 | 0.3784 | 0.6922 | 1.83 |
| `v_proj` | 0.8947 | 0.9661 | 0.4584 | 0.7979 | 1.74 |
| `q_proj` | 0.8994 | 0.9539 | 0.7000 | 0.8407 | 1.20 |
| `o_proj` | 0.9196 | 0.9609 | 0.7039 | 0.8558 | 1.22 |
| `in_proj_z` | 0.9208 | 0.9691 | 0.7459 | 0.8854 | 1.19 |
| `out_proj` | 0.9308 | 0.9714 | 0.7610 | 0.8917 | 1.17 |
| `in_proj_qkv` | 0.9459 | 0.9772 | 0.7944 | 0.9015 | 1.13 |
| `gate_proj` | 0.9561 | 0.9808 | 0.8347 | 0.9177 | 1.10 |
| `up_proj` | 0.9734 | 0.9892 | 0.8679 | 0.9348 | 1.08 |
| `down_proj` | 0.9685 | 0.9873 | 0.8542 | 0.9305 | 1.09 |

Source: `classes[*].recon_compare`. `gate_proj` excerpt
(`/tmp/qwen38_shared_basis.json` class body):

```
"256": {
  "per_layer_rel_err_mean": 0.8346703941040466,
  "shared_rel_err_mean": 0.917677517508702,
  "shared_minus_local": 0.08300712340465533,
  "shared_over_local": 1.0994489848819389
}
```

A shared basis is **never** nearly as good as per-layer. The closest ratio
is 1.08 (`up_proj` k=256), and both residuals are still > 0.86 — rank-256
does not reconstruct these matrices even *per layer*. There is no
low-rank shared object to amortize.

Stdout samples:

```
mlp.gate_proj.weight SHARED head_frac=0.1100
mlp.gate_proj.weight shared-recon L0  k256=0.9420
mlp.gate_proj.weight shared-recon L32 k256=0.9142
mlp.gate_proj.weight shared-recon L63 k256=0.9121
mlp.gate_proj.weight pair L0-L1 cos=0.000880 rel_delta=1.3817
```

---

## 4. Shared codebook vs per-layer — MEASURED, shared collapses

k-means relative MSE on 192 sampled hidden-vectors per layer.

| class | n vec | k=32 shared-on-layer | k=32 per-layer | k=128 shared-on-layer | k=128 per-layer |
|---|---:|---:|---:|---:|---:|
| `k_proj` | 3072 | 0.9854 | 0.8215 | 0.9524 | 0.3194 |
| `v_proj` | 3072 | 0.9884 | 0.8215 | 0.9542 | 0.3243 |
| `q_proj` | 3072 | 0.9851 | 0.8140 | 0.9453 | 0.3163 |
| `o_proj` | 3072 | 0.9865 | 0.8195 | 0.9524 | 0.3175 |
| `in_proj_z` | 9216 | 0.9950 | 0.8193 | 0.9816 | 0.3245 |
| `out_proj` | 9216 | 0.9915 | 0.8221 | 0.9826 | 0.3234 |
| `in_proj_qkv` | 9216 | 0.9961 | 0.8265 | 0.9838 | 0.3257 |
| `gate_proj` | 12288 | 0.9945 | 0.8238 | 0.9883 | 0.3233 |
| `up_proj` | 12288 | 0.9966 | 0.8285 | 0.9890 | 0.3273 |
| `down_proj` | 12288 | 0.9959 | 0.8279 | 0.9887 | 0.3262 |

Source: `classes[*].codebook`. k=128 per-layer uses 128 centroids on 192
vectors and is therefore optimistic for the per-layer side; even then the
shared book stays at **0.95–0.99**. The clean comparison is k=32: shared
**~0.99** vs per-layer **~0.82**. A pooled codebook does not describe
another layer's rows.

This is a sample-row proxy, not a packed VQ artifact. Directionally it
agrees with the shared-basis residual.

---

## 5. Adjacent delta vs independent encode — MEASURED, no win

Independent unit-norm matrices have `||A−B||/||B|| = √2 ≈ 1.414`.
Every class's adjacent mean is **1.395–1.421**.

Rank-k residual of the *delta* `W_{l+1}−W_l` vs rank-k residual of
`W_{l+1}` (hidden-side, same k). `gate_proj` k=256:

```
delta_rel_err_mean   0.844237
indep_rel_err_mean   0.835975
delta_minus_indep   +0.008263
n = 63
```

Delta is **slightly harder** to reconstruct than the tensor itself, at
every k in {8,32,64,128,256} on `gate_proj`. Same sign on `down_proj`
(+0.0044 @256) and `up_proj` (+0.0028 @256). Encoding the increment
does not beat encoding the layer.

`classes["mlp.gate_proj.weight"].adjacent`:

```
n 63
cosine_mean          0.0043541495509902255
rel_delta_fro_mean   1.408005157627894
rel_delta_fro_min    1.3755216408077071
rel_delta_fro_max    1.4338722126870858
independent_unit_rel_delta 1.4142135623730951
```

---

## 6. Within-layer heads — MEASURED, near orthogonal

Layers probed: DeltaNet 0,16,32,62; GQA 3,31,63.
`/tmp/qwen38_shared_basis.json` `heads`.

DeltaNet `in_proj_qkv` L0 (stdout + JSON `heads["linear_attn.in_proj_qkv.weight"]["0"]`):

```
q  n=16  (128,5120)  pairwise cos mean -0.000307  p95 0.00666  ov32 mean 0.0884
k  n=16  (128,5120)  pairwise cos mean -0.000054  p95 0.00537  ov32 mean 0.0682
v  n=48  (128,5120)  pairwise cos mean  0.003276  p95 0.01895  ov32 mean 0.0623
                                                     max 0.393
```

L16/L32/L62 q/k means stay inside ±4e-4; v means 0.0014–0.0019.
`in_proj_z` L0..L62 z-head means 0.0033–0.0128. `out_proj` o-head means
0.0019–0.0098.

GQA:

```
q_proj L3   q    cos mean 0.02160  p95 0.156  max 0.568  ov32 0.0335
q_proj L3   gate cos mean 0.02875  p95 0.055  max 0.269  ov32 0.0443
q_proj L31  q    cos mean 0.04046  p95 0.273  max 0.433  ov32 0.0445
q_proj L63  q    cos mean 0.04850  p95 0.282  max 0.567  ov32 0.0643
k_proj L3/31/63  n=4  cos mean -0.0038 / -0.0036 / -0.0030
v_proj L3/31/63  n=4  cos mean  0.0001 / -0.0008 /  0.0012
o_proj L3        cos mean  0.0172  max  0.469
o_proj L31       cos mean -0.0176  min -0.537  max 0.440
```

GQA query heads are the least orthogonal group in the model (mean 0.02–0.05,
still below the 0.05–0.10 reopen region as a *mean*; a few pairs hit 0.57).
Four KV heads is a thin sample. None of this is a shared-head codebook.

Compare Q80 experts: gate cos 0.00414, overlap 0.025. Qwen3.8 heads are in
the same near-orthogonal band except GQA-q, which is only modestly above it.

---

## 7. What this is not

- Not a complete-token measurement. No TPS, no TOKEN_NS, no BPW of a packed
  artifact. Binding: even if a shared object had existed, the production
  path would have to consume it in a representation-specific kernel, not
  expand to float/Q4 then generic GEMV. That path was not built because
  the object does not exist.
- Not an activation-weighted SVD. Weight-space MSE only.
- Not a vision-tower measurement. Language tensors only.
- G0 complete BPW 4.2527 / 26.4 TPS / 37.9 ms are **unverified claims**
  from the campaign brief. Unused here.

---

## Evidence index

| claim | pointer |
|---|---|
| Weights are real Qwen3.8-27B BF16, dense, 64×5120 | `/tmp/qwen38_shared_basis.json` identity; stdout `:2`; `config.json` `model_type=qwen3_5`; shards listed above; `qwen38_geometry.rs:12-40`; census `DENSE_NOT_MOE` line 33 |
| Q80 prior cosine 0.00414 | `receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json:11`; NS-010 lines 308–325 |
| Adjacent cosine ~0 | JSON `classes[*].adjacent.cosine_mean`; stdout pair lines e.g. `mlp.gate_proj.weight pair L0-L1 cos=0.000880` |
| Adjacent delta ≈ √2 | JSON `adjacent.rel_delta_fro_mean`; `independent_unit_rel_delta` 1.41421 |
| Shared basis worse than per-layer | JSON `recon_compare` |
| Shared codebook ~0.99 vs per-layer ~0.82 | JSON `codebook` |
| Head cosine ~0 (DN) / 0.02–0.05 (GQA-q) | JSON `heads` |
| Null overlap 0.00636 | JSON `null_subspace.k32`; stdout `:3` |
| RSS 9.22 GiB, 3990 s, CPU | JSON `rss_max_gib`, `elapsed_s`; stdout last line |
| Distance-16 cosine minority | JSON `exact_pairs` top; `exact_cosine_by_distance["16"]` |

Raw dump: `/tmp/qwen38_shared_basis.json` (1 385 001 bytes, 44 335 lines).
Run log: `/tmp/qwen38_shared_basis_run/stdout.log` (738 lines).
Script: `/tmp/qwen38_shared_basis.py`.

---

## Completion report

See the agent final message. This file is the lane deliverable.
