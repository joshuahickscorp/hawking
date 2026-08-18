# G1 mixed-2p0 organ bisect

STATUS: FALSIFIED

Pre-registered story: mixed-2p0-v1 died because down_proj was crushed to 0.1316 BPW while attention sat at 4.25. CPU dequant vs the BF16 parent says the opposite ranking. Isolated residual-stream drift names `mlp.gate_proj` (binary 1.125) as the worst organ on 56/64 layers. Packed HGRAVS01 down is the *best* MLP organ on the 256-token fit set (mean residual rel-L2 0.302 vs gate 0.545). Attention Q4 in-proj output cosine stays ≥ 0.9935. Weight-space ranking inverts this (down weight cosine 0.173) and is the number the packer's 0.907 mean cosine reported.

A ≤2.0856 reallocation from attention to down exists (rice attention + Q3 down = **1.949744** payload BPW). Residual ranking wants those bits on gate, not down (rice attention + Q3 gate + keep HGRAVS down = **1.739057**). Whether 2.0856 is below the coherence floor is not a generate result and is not claimed.

No GPU, no generate, no live-organism control. Decode authority: `tools/qwen38_sub15_pack.py:144-156` (`HGRAVB01` / `HGRAVR02` / `HGRAVS01` / `HGRAVU01`).

---

## 1. Method (MEASURED)

Artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1`
Parent: `.../qwen38-27b/bf16` (safetensors BF16 → f32)
Capture: `.../activation-capture-v1/hidden/Lxx.f32` (256 × 5120 f32, `CAPTURED_REAL_BF16_POST_NORM_HIDDEN`)
Oracle: `/tmp/g1-mixed2p0-organ-bisect/oracle.py` → `per_tensor.jsonl` (851), `mlp_residual_isolation.jsonl` (64), `alt_codec.jsonl` (102), `summary.json`
Holdout: `/tmp/g1-mixed2p0-organ-bisect/hgravs_holdout_thin.py` → `hgravs_holdout.jsonl` (6)

Per tensor: dequantize packed payload, compare to BF16 parent.

- Weight: flat cosine, rel L2, abs L2.
- Output, when in-dim is 5120: `Y = X @ W.T` on the captured post-norm hidden. Site labeled `UNCONFIRMED_POST_NORM` (wave-1 doctor).
- down_proj output: `X_down = silu(X @ Wg.T) * (X @ Wu.T)` from **BF16** gate/up, then `Y = X_down @ Wd.T`. Isolates the down organ. Matches packer `post_swiglu` in `lab/operators/q80_mixed_representation_pack.py:307-317`.
- Residual isolation, per layer: swap one of {gate, up, down} for its reconstruction, keep the other two BF16, write `y = down(swiglu(gate(X), up(X)))`. Rank by `sum_ℓ ||Δy_ℓ||_2`, not weight error.
- out_proj in-dim 6144: weight only. embed: weight only. lm_head output is PROXY on L63 hidden, not final-norm.

HGRAVS01 header `n_fit_tokens=256` on every down (packed on the whole capture). Fit-set output for down is optimistic. Holdout re-fits r160 + factor-q3 on tokens 0:192 with the thin-SVD equivalent of `G = XᵀX/n + λI` and scores 192:256.

Command (layers stayed ≤9.28 GB; embed/lm_head peaked 32.36 GB — over the 20 GB cap, see Risks):

```
python3 /tmp/g1-mixed2p0-organ-bisect/oracle.py
# /tmp/g1-mixed2p0-organ-bisect/oracle.log:141-143
[11:41:42] rss_max=32.36G done tensors=851 isol=64 alts=102
RANK residual [('only_gate', 16161.423269514487), ('only_up', 12226.241693421707), ('only_down', 8021.437171712401)]
```

Decode self-check: HGRAVS two-stage `y = (X Rᵀ) Lᵀ` vs dense `L@R` max rel-L2 **3.05e-6** over 64 layers.

---

## 2. What the pack actually is (MEASURED, not re-derived)

`mixed-2p0-v1/PACK_REPORT.json:10-39`

| field | value |
|---|---|
| complete_physical_bpw | 2.0855934079220506 |
| mlp_physical_bpw | 0.8480504639008466 |
| nonmlp_physical_bpw | 4.250142713483966 |
| tensor_payload_bytes | 7011580330 |
| gate | HGRAVB01 1.1250234267290902 / 802177344 B |
| up | HGRAVR02 1.2875108157887178 / 918036000 B |
| down | HGRAVS01 0.13161714918473189 / 93847197 B |
| attention+embed+norm | HGRAVU01 4.250142713483966 / 5197519789 B |

Catalog parse (851 rows) splits the 4.25 bucket into attention GEMV **3845162320 B** / 7237795840 elems and embed+lm_head+small **1352357469 B**. Native generate of this pack is INCOHERENT, 0 fallbacks, 0 dense-W (`receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json:2-12`; `QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-10`). Not re-run.

---

## 3. Rankings

Primary rank = isolated residual-stream `sum_ℓ ||Δy||_2` (`summary.json:1949-1961`):

| organ swap | sum \|\|Δy\|\|_2 | mean rel-L2 | mean cosine | worst-organ count |
|---|---:|---:|---:|---:|
| only_gate | **16161.42** | 0.54519 | 0.85233 | **56 / 64** |
| only_up | 12226.24 | 0.45549 | 0.89226 | 8 / 64 (L27–L34) |
| only_down | 8021.44 | 0.30230 | 0.95595 | **0 / 64** |
| gate+up | 19966.33 | 0.69301 | — | — |
| all three | 17220.77 | 0.57908 | 0.91402 | — |

`only_down` abs-L2 equals down's output abs-L2 (isolation identity, L0 delta 0.0).

Direct GEMV output `sum ||ΔY||` (`summary.json:1963-1974`) still puts gate > up > down > any attention in-proj. That ranking is polluted by shape: gate/up write 17408-wide intermediates, down writes 5120-wide residual. Residual isolation is the one to use.

Weight-space `sum ||ΔW||` **inverts** the residual rank:

| organ | mean weight cosine | mean weight rel-L2 | mean output cosine | mean output rel-L2 |
|---|---:|---:|---:|---:|
| mlp.down_proj | **0.1731** | **0.9856** | **0.9560** | 0.3023 |
| mlp.gate_proj | 0.7966 | 0.6044 | 0.8616 | 0.5211 |
| mlp.up_proj | 0.8416 | 0.5414 | 0.8430 | 0.5422 |
| dn.in_proj_qkv | 0.9938 | 0.1115 | 0.9959 | 0.0904 |
| gqa.q_proj | 0.9938 | 0.1116 | 0.9971 | 0.0760 |
| embed | 0.9942 | 0.1081 | n/a | n/a |
| lm_head | 0.9938 | 0.1123 | 0.99916 PROXY | 0.0411 PROXY |

Pack-time `mean_component_cosine` 0.90697 (`PACK_REPORT.json:41`) is the weight-space mix: 659×~0.994 + 64×0.797 + 64×0.842 + 64×0.173 ≈ 0.907. It is not an output number.

L0 per_tensor.jsonl:1-3,10 (exact):

```
gate  wcos=0.7983613877 wL2=0.6021786235 ocos=0.8625563307 oL2=0.5214607384
up    wcos=0.8424793159 wL2=0.5399726608 ocos=0.8956412580 oL2=0.4485489806
down  wcos=0.2111878518 wL2=0.9786166191 ocos=0.9741969096 oL2=0.2327164828
qkv   wcos=0.9941425011                   ocos=0.9961549397 oL2=0.0880283635
```

L0 gate output rel-L2 0.52146 matches wave-1 hetero L0-gate-binary 0.5221 (`g1-heterogeneous-allocation.md:77`). L0 down Q4 alt 0.0532 matches hetero 0.0538. Oracle is on the same scale as the descent screen.

---

## 4. Per-organ reconstruction (MEASURED, all 64 layers / all 851 tensors)

`/tmp/g1-mixed2p0-organ-bisect/summary.json` `organ_summary`. Payload BPW is 8 × catalog bytes / elements.

| organ | n | BPW | w-cos mean | w-relL2 | o-cos mean | o-relL2 | o-absL2 sum | site |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp.gate_proj | 64 | 1.1250 | 0.7966 | 0.6044 | 0.8616 | 0.5211 | 52566.6 | post-norm hidden PROXY |
| mlp.up_proj | 64 | 1.2875 | 0.8416 | 0.5414 | 0.8430 | 0.5422 | 42058.6 | same |
| mlp.down_proj | 64 | 0.1316 | 0.1731 | 0.9856 | 0.9560 | 0.3023 | 8021.4 | post-SwiGLU from BF16 gate/up; HGRAVS **fit-set** |
| dn.in_proj_qkv | 48 | 4.2500 | 0.9938 | 0.1115 | 0.9959 | 0.0904 | 6495.7 | post-norm UNCONFIRMED |
| dn.in_proj_z | 48 | 4.2501 | 0.9940 | 0.1102 | 0.9968 | 0.0795 | 4673.2 | same |
| dn.in_proj_a | 48 | 4.2587 | 0.9919 | 0.1278 | 0.9962 | 0.0846 | 640.9 | same |
| dn.in_proj_b | 48 | 4.2587 | 0.9910 | 0.1349 | 0.9972 | 0.0746 | 431.7 | same |
| gqa.q_proj | 16 | 4.2500 | 0.9938 | 0.1116 | 0.9971 | 0.0760 | 2245.6 | same |
| gqa.k_proj | 16 | 4.2504 | 0.9924 | 0.1234 | 0.9957 | 0.0927 | 732.5 | same |
| gqa.v_proj | 16 | 4.2504 | 0.9930 | 0.1186 | 0.9950 | 0.0990 | 952.1 | same |
| dn.out_proj | 48 | 4.2501 | 0.9939 | 0.1112 | — | — | — | no X (6144) |
| gqa.o_proj | 16 | 4.2501 | 0.9938 | 0.1122 | — | — | — | no X (6144) |
| embed | 1 | 4.2500 | 0.9942 | 0.1081 | — | — | — | gather |
| lm_head | 1 | 4.2500 | 0.9938 | 0.1123 | 0.99916 | 0.0411 | 971.0 | L63 hidden PROXY |
| input_layernorm | 64 | 4.6516 | 0.9988 | 0.0498 | — | — | — | vector |
| post_attention_layernorm | 64 | 4.6516 | 0.9989 | 0.0489 | — | — | — | vector |
| final_norm | 1 | 4.6516 | 0.9989 | 0.0488 | — | — | — | vector |
| small (A_log, dt_bias, norms, conv1d) | 353 | 4.52–47.2 | ≥0.9916 | ≤0.1303 | — | — | — | not residual writers |

Attention in-proj max output rel-L2 across every scored tensor is 0.1407 (`dn.in_proj_a`). Min output cosine 0.9902. Q4 attention is not the collapse.

down weight cosine range [0.1525, 0.2112]; output cosine range [0.9315, 0.9895]. One organ, two different stories.

---

## 5. Per-layer residual isolation (MEASURED)

`mlp_residual_isolation.jsonl` (64 lines). `y_ref` RMS grows from 0.037 (L0) to 3.194 (L63). Abs drift is late-layer dominated: L63 alone is 9.60% of gate's sum abs-L2.

```
layer gate_rel  up_rel  down_rel all3_rel  gate_abs  down_abs  ref_rms
0     0.405671  0.266550 0.232716 0.558774    17.27     9.91  0.037183
3     0.471746  0.372100 0.316176 0.585079    12.37     8.29  0.022903
8     0.444221  0.332088 0.373270 0.576681    27.22    22.87  0.053528
15    0.521914  0.438484 0.301860 0.567403    38.87    22.48  0.065052
31    0.559771  0.583487 0.310648 0.565198   138.87    77.07  0.216688
47    0.602160  0.486590 0.282345 0.580107   282.34   132.39  0.409548
63    0.424435  0.349231 0.311291 0.554232  1552.13  1138.37  3.194192
```

Block means of residual rel-L2:

| block | gate | up | down | all3 |
|---|---:|---:|---:|---:|
| L0–15 | 0.4904 | 0.3860 | 0.3133 | 0.5785 |
| L16–31 | 0.5319 | 0.4963 | 0.3190 | 0.5754 |
| L32–47 | 0.5712 | 0.5173 | 0.3034 | 0.5702 |
| L48–63 | 0.5873 | 0.4222 | 0.2736 | 0.5923 |

Full 64-row CSV is the jsonl. all3 residual cosine min 0.8241 at L30. Recipe-level MLP write is ~0.58 rel-L2 on every layer of this capture.

L0 isolation (`mlp_residual_isolation.jsonl:1`):

```
only_gate  cos=0.988232 rel=0.405671 abs=17.269
only_up    cos=0.994795 rel=0.266550 abs=11.347
only_down  cos=0.974197 rel=0.232716 abs= 9.907
all_three  cos=0.970446 rel=0.558774 abs=23.786
```

L63 (`summary.json:1935-1946`):

```
only_gate  cos=0.909878 rel=0.424435 abs=1552.13
only_up    cos=0.955519 rel=0.349231 abs=1277.11
only_down  cos=0.951854 rel=0.311291 abs=1138.37
all_three  cos=0.890118 rel=0.554232 abs=2026.79
```

---

## 6. HGRAVS01 is fit-set optimistic (MEASURED)

Packed downs carry `fit.n_fit_tokens = 256` (whole capture). `hgravs_holdout.log` re-fits r160+q3 on 192 tokens.

| layer | packed all256 oL2 | refit q3 fit192 | refit q3 hold64 | refit float hold64 | packed binary down oL2 (alt) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2327 | 0.2324 | 0.2666 | 0.1176 | 0.4645 |
| 3 | 0.3162 | 0.3174 | 0.4762 | 0.3685 | 0.5418 |
| 15 | 0.3019 | 0.3035 | **0.5633** | 0.4886 | 0.5654 |
| 31 | 0.3106 | 0.3109 | 0.4588 | 0.3573 | 0.5796 |
| 47 | 0.2823 | 0.2799 | 0.3744 | 0.2741 | 0.6265 |
| 63 | 0.3113 | 0.3167 | 0.3516 | 0.1703 | 0.6995 |
| mean | 0.2925 | 0.2935 | **0.4151** | 0.2961 | 0.5796 |

`hgravs_holdout.log:8` L15 hold 0.5633 ≈ binary 0.5654: mid-layer HGRAVS is worthless off the fit tokens. L63 hold 0.3516 still beats binary 0.6995 and loses to Q3 0.2365 (`alt_codec.jsonl` L63 q3).

Packed-per-prompt slices (still in-sample) do not move: L0 prompts sit at 0.24 except the 10-token "Explain gravity" slice at 0.139. In-sample prompt splits are not a holdout.

Even after the holdout correction, mean down OOD rel-L2 0.415 is still below gate's honest residual 0.545. Gate remains the worse organ. Down is no longer "fine" off-capture.

256 tokens vs down in-dim 17408 is eval_thin (wave-1). Generate tokens are a further domain shift. This oracle cannot score that.

---

## 7. Alternative codecs on {0,3,15,31,47,63} (MEASURED encode)

Same gravity codecs the pack uses (`_binary_codec`, `encode_residual_compact`, `_uniform_codec`, `_ternary_codec`).

down vs packed HGRAVS (mean over 6 layers):

| codec | mean BPW | mean o-relL2 | vs packed 0.2925 |
|---|---:|---:|---|
| HGRAVS01 packed | 0.1316 | 0.2925 | fit-set |
| binary | 1.1250 | 0.5796 | worse |
| rice 2% | 1.2875 | 0.5040 | worse |
| ternary t0.7 | 2.2500 | 0.4303 | worse |
| q3 g64 | 3.2500 | **0.2099** | better, honest |
| q4 g64 | 4.2500 | **0.0898** | better, honest |

HGRAVS at 0.13 BPW beats every integer-bit codec below Q3 **on the fit set**. Q3 is the first honest codec that beats it. L63: HGRAVS 0.311 / binary 0.700 / q3 0.237 / q4 0.101.

gate (packed binary oL2 L0 = 0.5215): q3 mean 0.1949, q4 0.0833, rice 0.4329.
up (packed rice): binary 0.5558, q3 0.2114, q4 0.0919.
attn qkv L0 packed Q4 0.0880: binary 0.5466, rice 0.4594, q3 0.2060.
gqa.q mean packed Q4 0.0785: binary 0.5265, rice 0.4139, q3 0.1674.

Rice-on-attention costs ~0.42–0.46 output rel-L2 vs Q4's 0.08. That is a real quality tax, not free budget.

---

## 8. Reallocation ledger (MEASURED bytes + DERIVED sums)

N = 26895998464. Payload BPW = 8 × bytes / N. Catalog tax on mixed-2p0 is 184307 B = 0.0000548 BPW (`PACK_REPORT.json:13-14`); not re-applied below.

Fixed from mixed-2p0 catalog (MEASURED):
gate 802177344, up 918036000, embed 675430686, lm_head 675430686, small 1496097.
attn GEMV Q4 3845162320. down HGRAVS 93847197. Sum 7011580330 = `PACK_REPORT.json:14`.

Replacement bytes:
- attn rice 1165098376 MEASURED `mixed-sub15-v1/PACK_REPORT.json:45-49`
- attn binary 1017836235 DERIVED = gate_bpw × 7237795840 / 8 (same HGRAVB01, same group 128)
- down Q3 / gate Q3 2317370880 MEASURED `mixed-q3mlp-v1/PACK_REPORT.json:33-49`
- down Q4 3030402560 MEASURED `mixed-q4down-v1/PACK_REPORT.json:43-45`

| assignment | attn | down | gate | payload bytes | payload BPW | ≤ 2.085539 |
|---|---|---|---|---:|---:|---|
| current mixed-2p0 | Q4 | HGRAVS | binary | 7011580330 | 2.085539 | = |
| attn_rice + down_q3 | rice | Q3 | binary | 6555040069 | **1.949744** | YES |
| attn_binary + down_q3 | binary | Q3 | binary | 6407777928 | **1.905942** | YES |
| attn_rice + down_q4 | rice | Q4 | binary | 7268071749 | 2.161830 | no |
| attn_binary + down_q4 | binary | Q4 | binary | 7120809608 | 2.118028 | no |
| attn_binary + up_binary + down_q4 | binary | Q4 | binary | 7004950952 | **2.083567** | YES |
| attn_q3 + down_q3 | Q3 | Q3 | binary | 8330334769 | 2.477792 | no |
| attn_q4 + down_q3 | Q4 | Q3 | binary | 9235104013 | 2.746908 | no |
| attn_rice + down_binary | rice | binary | binary | 5039846533 | 1.499062 | YES, but down binary is worse than HGRAVS |
| **attn_rice + gate_q3 + down_HGRAVS** | rice | HGRAVS | **Q3** | 5846709922 | **1.739057** | YES |
| attn_rice + gate_q3 + down_q3 | rice | Q3 | Q3 | 8070233605 | 2.400427 | no |

Integer check for the residual-supported row:
`918036000 + 675430686 + 675430686 + 1496097 + 1165098376 + 2317370880 + 93847197 = 5846709922`
`5846709922 * 8 / 26895998464 = 1.7390571849788756`

Cannot lift **both** gate and down to Q3 and stay ≤ 2.0856 if embed/lm_head stay Q4. Attention rice + both-Q3 lands at 2.400.

`mixed-q4down-v1` already exists at MEASURED 2.959043 (`PACK_REPORT.json:27`) — Q4 down, **same binary gate**. `mixed-q3mlp-v1` exists at 3.613865 — all MLP Q3, Q4 attention. Neither has a generate.

---

## 9. Decisive question

Is 2p0 dead because 2.0856 is below the coherence floor, or because the budget was allocated backwards?

1. Attention is over-served. MEASURED. Q4 in-proj output cosine ≥ 0.9935, rel-L2 ~0.08–0.10. Not the collapse.
2. Budget is allocated backwards relative to residual drift, but **not** toward down. Gate at 1.125 is the residual-starved organ. Down at 0.1316 is the BPW-starved organ and, on the fit set, the most bit-efficient MLP write.
3. An assignment that moves attention bits onto down and stays under 2.0856 exists: rice attention + Q3 down = **1.949744** payload BPW. That is the question as asked, answered YES.
4. The assignment residual ranking actually wants is rice attention + Q3 gate + keep HGRAVS down = **1.739057**. Down-lift without gate-lift is the wrong inversion.
5. HGRAVS down does not hold its fit-set 0.30 off those tokens (hold mean 0.415, L15 0.563). Generate is a further shift. This is a real risk, not a residual-rank winner.
6. Whether any ≤2.0856 assignment is token-coherent is **unmeasured**. This lane did not generate. The floor remains (2.0856, 4.2527] for *this recipe*. A different recipe at 1.74–1.95 is a different point.

Prior claim this lane falsifies: `QWEN38_COHERENCE_FLOOR_BRACKETED.json:15` "The MLP compresses to 0.848 BPW fine; it is attention at 4.250 ... that dominates." MLP-as-recipe residual rel-L2 is 0.579. Attention is the part that is fine.

---

## 10. KILLS / REOPEN_IF

- **KILLS** "down_proj is the organ that destroys mixed-2p0-v1." Isolated residual names gate. Weight-space down disaster (cosine 0.173) is a low-rank artefact, not residual drift. REOPEN_IF a native generate of `mixed-q4down-v1` (Q4 down, **same binary gate**, 2.959 BPW, already packed) is coherent. That is the cheapest falsifier of this kill: if it suddenly speaks English, OOD down was the generate killer and the capture residual lied.
- **KILLS** "move attention bits onto down and the 2p0 recipe is saved." Arithmetically possible (1.95 BPW). Residual ranking says those bits belong on gate. REOPEN_IF `mixed-q4down-v1` generate passes or a new rice-attn+Q3-down pack generate passes.
- **KILLS** using pack-time mean cosine 0.907, or down weight cosine, as a quality signal on this model. Wave-1 "weight ranking misleads" is reproduced, violently, on HGRAVS down.
- **Does not kill** "MLP at 0.848 BPW is below a token floor." all3 residual 0.579 on the capture is a large write error. The floor statement is about generate, which this lane did not run.
- **Does not locate** the coherence floor inside (2.0856, 4.2527].

Cheapest next measurement (GPU lane, artifacts already on disk, no new pack):

1. Native generate `mixed-q4down-v1` (2.959, down Q4, gate still binary).
2. Native generate `mixed-q3mlp-v1` (3.614, all MLP Q3, attention Q4).

If (1) fails and (2) passes, gate/up were the token killers. If (1) passes, down OOD was the token killer and this capture residual is incomplete. If both fail, 2.0856-class MLP recipes are not the only problem.

---

## 11. Labels

| number | label |
|---|---|
| all 851 weight / output / residual figures | MEASURED vs BF16 parent, this lane |
| HGRAVS down output on 256 tokens | MEASURED, **fit-set** (n_fit=256) |
| HGRAVS hold 192/64 | MEASURED, thin-SVD equivalent of the packer eigh, not the packed bytes |
| sibling organ bytes / BPW | MEASURED from those PACK_REPORT.json files |
| assignment payload BPW | DERIVED integer sums of MEASURED bytes |
| attn binary bytes | DERIVED from measured gate HGRAVB01 BPW × attn elements |
| lm_head output | PROXY (L63 post-norm, not final-norm) |
| captured X as MLP / attn input | PROXY, site UNCONFIRMED_POST_NORM |
| token coherence of any new assignment | UNMEASURED |
| live G0 4.2527 / 25.43 TPS / 39.3 ms | established, not re-measured |

---

## 12. Command output (required)

Oracle:

```
# /tmp/g1-mixed2p0-organ-bisect/oracle.log:1-5,141-143
[11:28:02] rss_max=0.04G open catalog + weight map
[11:28:02] rss_max=0.04G catalog tensors 851
[11:28:02] rss_max=0.04G layer 0 start
[11:41:42] rss_max=32.36G language_model.lm_head.weight done in 83.8s
[11:41:42] rss_max=32.36G done tensors=851 isol=64 alts=102
RANK residual [('only_gate', 16161.423269514487), ('only_up', 12226.241693421707), ('only_down', 8021.437171712401)]
```

Holdout L15 (worst OOD) and L63:

```
# /tmp/g1-mixed2p0-organ-bisect/hgravs_holdout.log:8
{"layer": 15, "packed_all256_rel_l2": 0.3018604514839271,
 "refit_q3_fit192_rel_l2": 0.30348313940773347,
 "refit_q3_hold64_rel_l2": 0.5632849448106798, ...}

# hgravs_holdout.log:17
{"layer": 63, "packed_all256_rel_l2": 0.31129091717779606,
 "refit_q3_hold64_rel_l2": 0.35159048952095995, ...}
```

Native generate (not re-run):

```
# receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json:2-12
"dense_w_materialized_total": 0,
"fallbacks_total": 0,
"generated_text": "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
```

```
# receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-10
"4.2527_BPW_q4_oracle": "COHERENT",
"2.0856_BPW_mixed-2p0-v1": "INCOHERENT (native, verified twice - lane and controller)",
```

---

```
STATUS
FALSIFIED

CLAIMS
1. down_proj is not the organ that destroys mixed-2p0-v1 on residual-stream drift. only_gate sum||Δy|| 16161 > only_up 12226 > only_down 8021. Gate is worst on 56/64 layers, down on 0/64. Evidence: /tmp/g1-mixed2p0-organ-bisect/oracle.log:142-143; summary.json:1949-1961; mlp_residual_isolation.jsonl (64 lines).
2. Weight-space ranking inverts the residual ranking. down mean weight cosine 0.1731 / rel-L2 0.9856 vs output cosine 0.9560 / rel-L2 0.3023 (fit-set). Evidence: per_tensor.jsonl:3; summary organ_summary mlp.down_proj; PACK_REPORT.json:41 (0.90697 is the weight mix).
3. Attention Q4 is over-served, not the collapse. in-proj output cosine ≥ 0.9935, mean rel-L2 0.076–0.099. Evidence: per_tensor.jsonl:10 (L0 qkv oL2 0.08803); organ_summary dn.in_proj_* / gqa.*_proj.
4. Packed HGRAVS01 down is fit on all 256 capture tokens. Re-fit hold64 mean oL2 0.415 vs fit 0.293; L15 hold 0.563 ≈ binary. Evidence: hgravs_holdout.log:8,17; hgravs_holdout.jsonl.
5. A ≤2.0856 assignment that moves attention bytes onto down exists: rice attn + Q3 down = 1.949744 payload BPW (6555040069 B). Evidence: mixed-sub15-v1/PACK_REPORT.json:45-49; mixed-q3mlp-v1/PACK_REPORT.json:45-49; integer sum in §8.
6. Residual-supported assignment at still-lower BPW: rice attn + Q3 gate + keep HGRAVS down = 1.739057 payload BPW (5846709922 B). Evidence: same sibling reports + mixed-2p0 PACK_REPORT.json:16-20.
7. Cannot lift both gate and down to Q3 at ≤2.0856 with embed/lm_head at Q4 (2.400 BPW). Evidence: §8 last row.
8. Token coherence of any reallocation is UNMEASURED. Floor for *this* recipe remains (2.0856, 4.2527]. Evidence: QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-10; no generate in this lane.

EVIDENCE
- /tmp/g1-mixed2p0-organ-bisect/oracle.log
- /tmp/g1-mixed2p0-organ-bisect/per_tensor.jsonl (851)
- /tmp/g1-mixed2p0-organ-bisect/mlp_residual_isolation.jsonl (64)
- /tmp/g1-mixed2p0-organ-bisect/alt_codec.jsonl (102)
- /tmp/g1-mixed2p0-organ-bisect/summary.json:1949-1983
- /tmp/g1-mixed2p0-organ-bisect/hgravs_holdout.log
- mixed-2p0-v1/PACK_REPORT.json:10-41
- mixed-sub15-v1/PACK_REPORT.json:45-49
- mixed-q3mlp-v1/PACK_REPORT.json:27-49
- mixed-q4down-v1/PACK_REPORT.json:27-45
- receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json:2-12
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json:6-15
- tools/qwen38_sub15_pack.py:144-156

CHANGES
workspace/superwave/g1/g1-mixed2p0-organ-bisect.md (this file). No other tracked path.

TESTS
see final message

RISKS
- Peak RSS 32.36 GB on embed/lm_head decode (oracle.log:138-140). Layer loop stayed 9.28 GB. 20 GB cap breached for ~3 min.
- Capture X is post-norm hidden used as MLP input (PROXY).
- HGRAVS hold is a thin-SVD equivalent, not a byte-identical re-encode of the packed eigh path.
- Residual ranking is not a generate. Token collapse could still be OOD-down even though capture residual names gate.

UNRESOLVED
- Native generate of mixed-q4down-v1 and mixed-q3mlp-v1 (already packed, never generated).
- True generate-token HGRAVS error (no capture of those tokens).
- out_proj output-space (no 6144-wide X).
- Whether 1.74 or 1.95 BPW reallocations are coherent.

NEXT
GPU lane: generate mixed-q4down-v1, then mixed-q3mlp-v1. Do not pack a new 2p0-reallocation until those two existing artifacts have a native generate.
```
