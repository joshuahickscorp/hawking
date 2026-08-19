I'll read the frontier packet first so the base-correction proposals match its schema and guardrails.The packet is loaded. Next I’ll pull Hawking’s prior results and codec machinery so the proposals cite real patients, not generic quantization lore.I’ll pull the prior Hawking numbers, residual/codec machinery, and Odyssey notes so the proposals stay on real patients.I have the Q80 mixed-rep ledger. Next I’ll read GLM/Qwen sub-bit receipts and Odyssey rulebase so the proposals don’t re-invent killed levers.I’ll pull GLM 0.167-BPW evidence, residual-codec rates, and patient organs so each proposal can name real numbers instead of guesses.I have the measured Q80 residual win and the arithmetic floors. Next I’ll write harvestable proposals that stay inside those constraints.Census math is in: stored sub-0.2 is blocked by the 8-bit non-expert floor. I’ll write the proposals against that constraint.Writing the harvestable packet: arithmetic floors first, then eight mechanisms with complete byte accounting.Writing the harvestable packet with floors, the 2-bit beat condition, and eight mechanisms.I have the harvest keys. Writing the packet with both the frontier schema and the novelty-lane fields.Composing the packet as structured JSON with every required field and the arithmetic floors attached.Writing the packet to `evidence/sub1` and the Odyssey receipts drop.I have the measured Q80 residual win and the O005 mass split. The proposals below stay inside those floors.

# BASE+CORRECTION LANE — sub-1 / sub-0.2

Lane question: when does an extreme low-bit **base** plus a **small conditional repair** beat a naked 2-BPW codec at **lower `complete_bpw`**, and can that reach stored sub-1 or sub-0.2 **honestly**?

Doctor is authority. Cosine is a screen. Numbers without a tag are not claims.

---

## 0. Accounting floors (do not violate)

**O005 census MEASURED** (`patients/O005/census.json`): 30.532B params; expert 28.991B (94.985%); always-on embed+attn+router+norm+lm_head 1.541B (5.015%). Active 3.353B = 8/128 of experts (1.812B) + all always-on (1.541B) → expert is **54.0% of active**, always-on **46.0% of active**.

| Quantity | Formula | 8-bit always-on | 4-bit always-on | 2-bit always-on | 1-bit always-on |
|---|---|---:|---:|---:|---:|
| O005 **stored** complete | `0.94985·e + 0.05015·n` | `0.94985e + 0.401` | `0.94985e + 0.201` | `0.94985e + 0.100` | `0.94985e + 0.050` |
| O005 **active** complete | `0.5404·e + 0.4596·n` | `0.5404e + 3.677` | `0.5404e + 1.838` | `0.5404e + 0.919` | `0.5404e + 0.460` |
| Q80 **stored** complete | `0.97032·e + 0.02968·n` MEASURED identity | `0.97032e + 0.237` | `0.97032e + 0.119` | `0.97032e + 0.059` | `0.97032e + 0.030` |

**Impossible-as-stated (DERIVED, not a quality claim):**

| Target | O005 | Q80 |
|---|---|---|
| Stored complete **< 0.2** with always-on ≥ 4-bit | **impossible** (floor 0.201 even at e=0) | **impossible** at 8-bit always-on (floor 0.237) |
| Stored complete **< 1.0** with always-on = 8-bit | needs **e < 0.630** | needs **e < 0.786** |
| Active complete **< 1.0** with always-on = 8-bit | **impossible** (floor 3.677) | **impossible** (attn+DeltaNet dominate active) |
| Active complete **< 1.0** with always-on = 4-bit | **impossible** (floor 1.838) | same class |
| Active complete **< 1.0** | needs always-on ≤ ~2-bit **and** e ≲ 0.15, **or** always-on 1-bit and e ≲ 1.0 | crush attn/embed/head or give up active-sub-1 |
| Active complete **< 0.2** | **impossible** even at 1-bit always-on (floor 0.460) | **impossible** |

**Law:** stored sub-0.2 is an **expert-body** (or always-on-also-crushed) target, not a complete-artifact target while embed/attn/lm_head stay 8-bit. Active sub-1 is an **attention/embed/head** problem, not an expert-quant problem. 8/128 = 0.0625 is a **selection** lever, not a 16× cost cut (policy `selected_full_is_selection_not_cost`; bible §29).

O006 vision tower **raises** always-on mass vs O005 (active 12.5% vs 11.0%) — active floor **worse**. O003 has 2 **shared** experts always-on plus vision. O001/O004: stored = active for weights; SSM state on O001 is extra active, not a weight-BPW discount.

---

## 1. What “naked 2-BPW” actually costs

| Codec | nominal_bits | complete expert_bpw | tag | vs bar 0.8604 (Q80, REAL x) |
|---|---:|---:|---|---|
| `uniform_b2` g128 scale+bias | 2 | **2.252** | MEASURED `QWEN80_REPRESENTATION_FRONTIER_SWEEP.json` | gate 0.862 PASS-ish; **up 0.801 FAIL**; down 0.816 FAIL |
| affine q2-g32 (mlx-style) | 2 | **3.0** | DERIVED: 2 + (16+16)/32 | O005 analog: q3-g32 experts byte-ratio **4.000** MEASURED (`O005_GRAVITY_q3-g32-experts.json` expert 14.496 GB / 28.991B params) |
| affine q2-g64 | 2 | **2.5** | DERIVED | UNKNOWN Doctor |
| affine q2-g128 | 2 | **2.25** | DERIVED | matches uniform_b2 |
| `binary_g` g128 scale, **no** residual | 1 | **1.127** | MEASURED | gate **0.893 PASS**; up **0.828 FAIL**; down **0.81–0.83 FAIL** |
| `binary+resid_2pct` **legacy u32+fp16** (48 b/outlier) | 1+corr | **2.088** | MEASURED | up 0.865 PASS — **over-budget**, ≈ naked 2-bit **complete**, fake “sparse” win |
| `binary + rice_q1_rms @ 2%` | 1+corr | **1.292** | MEASURED mixed pack | up **0.864–0.865 PASS** at **8.24 b/outlier** |

**Existence proof (screen, not Doctor):** rice-coded 1-bit residual at 2% **already beats** `uniform_b2` on up_proj (1.29 vs 2.25 complete; 0.865 vs 0.801 cosine). The 2% residual with **legacy indexing** does **not** (2.088 ≈ 2.252). **Encoding of the correction, not the idea of a correction, is the complete-BPW gate.**

Q80 mixed (gate `binary_g` 1.127 / up rice_q1 1.292 / down `hgravs01_r160_b3` 1.27) → mixed expert **1.230**, complete **1.431** at 8-bit non-expert. Claim boundary: **not packed, not generated, no decode kernel** (`QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json`).

**0.3-BPW base law:** grouped 1-bit + fp16 scale **floors at ~1.13 complete**. Ternary packed 5 trits/8 bits = **1.6** bpw (BitNet b1.58) — **worse density** than binary. A 0.3 complete **base** requires **multi-weight codes** (PQ / sign-block VQ / implicit lattice) or **structural zeros whose pattern is free**. You cannot get 0.3 from 1-bit signs.

---

## 2. Kills this lane must not rediscover

| ID | Implication for base+corr |
|---|---|
| **NS-raw-weight-pq-vq-at-one-bit** | Raw-weight PQ/VQ @ ~1 BPW collapsed 6/6 on qwen3-235b real forward. **Does not kill** activation-aware PQ (changes the source). **Does kill** “just make the codebook bigger.” |
| **Qwen3.8 gaussian-proxy sub-bit** | All 0.5–0.8 BPW “wins” collapsed; output-div ~0.69. Eval on **REAL routed activations only**. |
| **GLM-5.2 sub-bit MoE expert path** | 0.116–0.157 cos @ 0.75 BPW, none beat null 0.898. Type-1. Distinct from GLM **activation-aware 0.755 cos @ 0.167 BPW** (the one real sub-1/5). |
| **NS-uniform-subbit-allocation** | Organs do not degrade together. Gate/up ≠ down. |
| **NS-entropy-coded-pq-indices** | Lloyd indices ~uniform; 0.0–0.7% savings. Do not budget 10–25% from entropy of k-means indices. |
| **NS-posthoc-scalar-gain** | Scalar gain on k-means is algebraically pinned at 1. **Does not kill additive / sparse residual.** |
| **NS-ternary-factorization** | Ternary **factors** lose to VQ at matched rate. Does not kill ternary **weights** as a BitNet-style base. |
| **NS-inter-expert-redundancy / QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE** | Pairwise expert cosine ~0–0.004. Shared expert-template + delta is **structural-lane**, not this lane. A **local** codebook of d-wide blocks is not a shared expert template. |
| **Learned codebook LUT** (`dead_levers.md`) | Random LUT gather Type-1 dead on Apple GPU. Resurrection: **lookup-free** (QTIP/QuIP# class) **or** a codebook small enough to live in registers (k=256 × 32-bit is 1 KB — **not** the punished case). Mark as experiment, not assumed native. |
| **QTIP Metal trellis decode** | Type-1 dead for TPS (serial state, more ALU than dead Q3). Quality Type-2 leaning NO-GO. Resurrection: lane-independent layout. |
| **Low-rank residual codec** | Type-1 for **weight-space SVD / ASVD**. Does **not** kill Q80 **activation-weighted** `hgravs01_r160_b3` on down_proj (LIVE at 1.27, cosine 0.886–0.898). |
| **FFN block-256 sparsity** | Type-1: skip 0.2% @ 99% recall; scattered. Structured N:M only if we **force** the pattern and **correct** violations. |
| **Cold-expert compression** | 0 cold on O005/O003/O006. R-uniform-routing. Per-expert skip-repair has **no popularity target**. O005 **single-expert zero**: Doctor Δhits = 0. Damage is **diffuse** → repair inside each expert (channel/organ), not “fix expert 49.” |
| **Calibration-88-tokens** | Route-conditioned anything needs ≥ ~1000 tokens disjoint from holdout. |

**Not this lane:** student dense map @ **0.0104 BPW** (function-space, not weight-space) — latent/generated lane.

---

## 3. When 0.3-base + small corr beats naked 2-BPW

| Condition | Why | Status |
|---|---|---|
| Correction **index** is rice / group-local, **not** u32+fp16 | 48 b/outlier → 2% costs +0.96 bpw (legacy 2.088 ≈ uniform_b2 2.252). rice_q1 → 8.24–12.1 b/outlier, 2% costs **+0.16–0.24** | MEASURED Q80 |
| Residual **values** are 1–4 bit, not fp16 | fp16 vs q1 cosine delta **−2e-4** at 0.25% on gate | MEASURED; 2% q1 value-quant **UNKNOWN** |
| Selection is **activation/Hessian**, not `\|W−binary(W)\|` | Weight-top-k needed 2% to clear up_proj bar; 0.25% moves cosine **+0.005** | 0.25% MEASURED; act-select HYPOTHESIS |
| Topology is **organ-conditioned** (gate binary-ok, up needs residual, down wants act-LR) | NS-uniform-subbit; Q80 mixed screen | MEASURED screen |
| Base payload **< 1 bit/weight** if the target is 0.3 complete base | 1-bit + g128 scale = 1.13 floor | DERIVED |
| Native path is **fused decode-matvec**, not expand-to-fp16 | Fake density otherwise. Q80 mixed `decode_kernel_exists: false` | MEASURED gap |
| LUT, if any, is **tiny** (≤ ~1–25 KB/codebook) or **implicit** | Large learned LUT Type-1 dead | INFERRED |
| Eval on **real routed x**, held-out Doctor | Proxy trap; cosine ≠ capability | MEASURED history |
| Always-on organs **not** “paid for” by expert sub-bit | 5% at 8-bit = 0.40 stored BPW floor on O005 | DERIVED |

---

## 4. Proposals

### BC-RICE-ACT — activation-Hessian Rice-q1 residual on grouped-binary base

One-line: keep the **MEASURED** rice residual encoder; change **selection** from `|W−Wbin|` to real-x Hessian / activation-magnitude; do not change the 1-bit base yet.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Payload** 1 bit/W packed. **Scale** fp16 per g=128 → 0.125 bpw. **Bias** 0 (signs). **Container/align** → MEASURED binary_g **1.127**. **Correction** rice-coded positions + 1-bit sign × rms (or mean-abs) + 1 scale; MEASURED 8.24 b/outlier @ 2% → **+0.165** (up 1.292). **Tables/codebooks** 0. **Calibration x** not stored. **nominal_bits** ≈ 1.02; **complete expert** 1.16–1.35. |
| **stored_bpw** | Expert **1.13–1.35** (MEASURED range binary → rice 2%). O005 stored @ n=8: **1.47–1.68** DERIVED. **Not sub-1** at 8-bit always-on. |
| **active_bpw** | Packed into expert body → active expert_bpw = stored expert_bpw. O005 active @ n=8: **4.29–4.41** DERIVED (always-on 8-bit floor 3.68). @ n=4: **2.45–2.57**. Active sub-1: **no** unless always-on crushed. |
| **expected_reachable_bpw** | Expert **1.16–1.40** HYPOTHESIS if act-select lets residual **frac drop below 2%** at same cosine. Floor **1.13** unless base changes (BC-SIGNPQ/ACTPQ). Sub-0.2: **no**. |
| **beats naked 2?** | **Yes on complete** vs 2.25/3.0. **Yes on cosine screen** for up_proj (0.865 vs 0.801). Doctor: **UNKNOWN**. |
| **quality_risk** | High on **up_proj** (binary 0.828 < 0.860; residual is the only reason it passes). Down_proj residual at 1.5% only passed under **legacy** 1.85 bpw — wrong codec for down (see BC-OUT-LR / ORGAN-HETERO). **Router / attn / embed / lm_head** must stay out of this codec (Q80 sensitive 3% untouched; GLM R0 embed collapse). |
| **cheapest_falsifier** | One organ (O005 `gate_proj` + `up_proj`, one expert, **real routed x**, ≥1k tokens disjoint). Compare `|W−Wbin|` top-2% rice_q1 vs **activation-weighted** top-k' at **matched complete_bpw**. Kill if act-select does not beat weight-select on **held-out** `‖X(W−Ŵ)‖` **and** fast-Doctor organ ablation. Do **not** raise frac before a topology miss. |
| **execution_path** | Native: **no** (`decode_kernel_exists: false`). Credible path: bitpacked XNOR-popcount GEMV (BitNet-class) **fused** with scatter-add of rice residuals into the accumulator. **Not** decode-to-fp16. Decode cost/token: <REDACTED> µs; extra residual stream is an index+value gather — on kernel-bound decode this can **lose TPS even when bytes win**. Reconstruction state: rice k, rms scale, group scales. |
| **kernel_implications** | New binary GEMV + sparse FMA. Do not use random LUT. Residual topology (scattered vs group-local) is the kernel question; rice is sequential — may need group-local bitmap for SIMD. |
| **applicability** | MoE **expert gate/up** first (Q80). Dense MLP same codec. Hybrid O001: **MLP only**; protect Mamba-2 conv/SSM (recurrent). Do not put this on DeltaNet/attn. |
| **confidence** | **INFERRED** for “beats 2-bit complete” (MEASURED cosine screen). **HYPOTHESIS** for Doctor and for act-select reducing frac. **REFUTED** as a 0.3-base path without a different payload. |
| **transfer** | Encoder transfers (rice_q1). Bar 0.8604 is Q80-specific. O005 q2 failed with localization **protect gate/up** — consistent. O003/O006 language-MoE likely; vision tower **UNKNOWN**. |
| **prior art** | SpQR (sparse high-prec outliers); SqueezeLLM (sensitive non-uniform); PB-LLM (partial binarization); BiLLM residual-approx 1-bit PTQ. Hawking rice residual **stricter accounting** than SpQR’s fp16 outliers. |
| **doctor_risk** | 0.45 HYPOTHESIS (1-bit body). Fast-Doctor 12-item is not a seal. |
| **family_addition** | STRUCTURAL. Extends `base_plus_correction` but **selection=act-hess**, **codec=rice_q1**, **not** q2-affine. Runner grammar today cannot express rice; closest `q1-g128-experts+c0.02` is a **lie** (affine q1 ≠ binary_g). Needs spec token `binary-g128+riceq1`. |
| **info_gain / cost** | 8 / 3 (reuses Q80 xcache recipe). |
| **NEXT_BOTTLENECK** | Native binary+sparse kernel; Doctor not cosine. |

---

### BC-ORGAN-HETERO — per-organ base class + matching repair (gate binary / up rice / down act-LR)

One-line: do **not** put the same residual on every organ; Q80 already screened the split.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **gate:** binary_g 1.127, no residual (MEASURED cosine 0.859–0.893). **up:** binary + rice_q1 @ ~2% = 1.292 (0.864–0.865). **down:** `hgravs01_r160_b3` 1.27 (0.886–0.898) — **not** sparse residual (binary+resid_1.5% only passed at 1.85 legacy). **Mix** 1.230 expert MEASURED. **Always-on 8-bit** → complete **1.431**. Scales/factors of hgravs **in** 1.27. Router/attn/embed/head **8-bit** counted. |
| **stored_bpw** | Q80 **1.431** @ n=8 MEASURED screen. O005 analog if same mix: **0.94985×1.23 + 0.401 = 1.57** DERIVED. Sub-1: **no** at 8-bit always-on. |
| **active_bpw** | O005 @ n=8: **0.540×1.23 + 3.677 = 4.34** DERIVED. NX win is **gather** of this packed expert, not a different expert_bpw. |
| **expected_reachable_bpw** | Expert **1.15–1.35** if rice frac on up drops and/or down rank drops. Complete O005 n=8 **1.49–1.68**. Push to sub-1 only by **swapping the bases** (BC-ACTPQ) not by shrinking residual to 0. |
| **beats naked 2?** | **Yes** vs 2.25 expert / vs uniform q2 that **fails cosine** on up/down. Mixed 1.43 < 2.25. |
| **quality_risk** | **Generation untested.** Cosine bar ≠ Doctor. Dominant failure organ historically **gate** (F1 qwen3-235b) — here gate is the **best** binary organ; **up** is the limit. SwiGLU couples gate×up — independent organ screens can **lie** (need joint SwiGLU cosine; Q80 down scored on **post-SwiGLU**, good). |
| **cheapest_falsifier** | Pack **one layer** mixed (binary gate / rice-up / hgravs-down) vs uniform_b2 vs uniform_b3 on **real x**, then **fast-Doctor** that layer-ablated. Kill if mixed Doctor ≤ uniform_b2 at lower complete_bpw **or** SwiGLU-joint cosine fails while per-matrix cosine passes. |
| **execution_path** | Three kernels: binary GEMV, binary+sparse, low-rank fused (hgravs already has factor matvec in Hawking: `gk_matvec_hgravs` / `q80_hgravs01_factor_matvec` — **serial, slower than geo_tpr64**). Native mixed pack: **false**. FLOPs: down as UV is **r·(in+out)** vs **in·out**; r=160 on 2048×512 is a large factor — **UNKNOWN** whether that is cheaper than 3-bit GEMV on M3 (runtime often kernel-bound). |
| **applicability** | MoE experts. Dense: gate/up/down of MLP. Hybrid: MLP only. Shared experts (O003): treat as **always-on down-proj class** (higher prec). |
| **confidence** | **MEASURED** as organ-cosine screen on Q80. **HYPOTHESIS** as Doctor mechanism. Highest groundedness in this lane. |
| **transfer** | R-organ-inversion INFERRED F0/F1, **not** Odyssey-measured. O005 q2 fail localized to gate/up — **compatible**. Confirm per patient before locking mix. |
| **prior art** | AWQ (protect salient channels); AQLM (additive, not organ-split); Hawking organ inversion. |
| **doctor_risk** | 0.35 HYPOTHESIS (same mix as Q80 screen). |
| **family_addition** | STRUCTURAL `mixed-binary-rice-hgravs-experts`. Not expressible as `mixed-q2q3-experts`. |
| **info_gain / cost** | 9 / 4 |
| **NEXT_BOTTLENECK** | Joint SwiGLU + one-layer Doctor; then kernel. |

---

### BC-CHANNEL-PROMOTE — 0.25–1 bit body, whole **channels** (not scattered weights) at 4–8 bit

One-line: SpQR/PB-LLM/AWQ topology but **column/row** granularity so the hi-prec part is a contiguous GEMV, not a scatter.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Base payload** binary 1.0 **or** SignPQ 0.25 (pair with BC-SIGNPQ). **Channel bitmap** 1 bit/channel: for 2048×768, 2048/1.57e6 ≈ **0.0013 bpw** (negligible). **Hi-prec p** fraction at *b* bits: `p·b`. Example p=0.05, b=8 → **+0.40**. **Scales** for both classes. **No** per-outlier index (bitmap is the index). **nominal** 1+0.05×8 or 0.25+0.40. **complete expert** ≈ 0.70 (SignPQ base) to 1.55 (binary base) + scale. |
| **stored_bpw** | Expert **0.7–1.6** HYPOTHESIS. O005 stored n=8: **1.07–1.92** DERIVED. Stored sub-1 **possible** if base is SignPQ/ActPQ **and** p·b ≲ 0.2. |
| **active_bpw** | Same expert_bpw in the packed tensor. Optional: store hi-prec channels as sidecar, gather only for selected experts (see BC-ROUTE-SIDECAR) — then **stored** still counts sidecar, **active** can drop if sidecar not fused. |
| **expected_reachable_bpw** | Expert **0.6–1.4** HYPOTHESIS. Sub-0.2 complete: **no**. Expert sub-0.2: only if p→0, i.e. this collapses to pure base. |
| **beats naked 2?** | On complete: **yes** if p·b + base < 2.25. Quality: **UNKNOWN** — wins iff activation energy is **channel-concentrated**. If energy is scattered, this is a **worse** topology than rice-scatter at same complete_bpw. |
| **quality_risk** | **up_proj / gate** if salient weights are not whole channels. O005 round8 of **all** experts was Doctor-silent/slightly better (11/12) — not a sensitivity map at 1-bit. Limit organ: **UNKNOWN** until per-channel act-RMS × ‖Wcol‖ ranked vs Doctor. |
| **cheapest_falsifier** | Matched complete_bpw: **channel** hi-prec vs **scattered** rice vs **row** hi-prec, same p·b budget, **real x**, one up_proj. Kill **channel** if held-out act-MSE loses to scattered. Do not increase p first. |
| **execution_path** | Native-credible: two GEMVs (binary body + q4/q8 selected cols) fused, or one kernel with col-dependent dequant. **Better** coalescing than scatter residual. Cost/token: <REDACTED> Must not expand full W. |
| **applicability** | All classes. Best on wide `in` (down_proj 768→2048 / 512→2048) if few cols dominate. Hybrid: not on SSM `A,D` (tiny, protect). |
| **confidence** | HYPOTHESIS. Topology is the experiment; literature (AWQ/SpQR) says **salience is heavy-tailed** but **not necessarily whole channels**. |
| **transfer** | Topology test is cheap per patient; do not transfer p. |
| **prior art** | AWQ (channel scales, not hi-prec keep); SpQR (unstructured outliers); SqueezeLLM; PB-LLM. |
| **doctor_risk** | 0.40 |
| **family_addition** | STRUCTURAL `binary+chp{p}-b8`. Grammar: `+correction` is **not** channel topology — needs `+chan0.05`. |
| **info_gain / cost** | 7 / 3 |
| **NEXT_BOTTLENECK** | Whether salience is channel-aligned on **real** x. |

---

### BC-SIGNPQ-025 — Hamming / sign-block PQ as **0.25-BPW base**, rice residual on leftover

One-line: 8-bit index per 32-D sign block = 0.25 payload; this is how a base actually reaches ~0.3 complete.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Payload** 8/d bpw (d=32 → **0.250**; d=40 → **0.200**; d=48 → **0.167**). **Codebook:** if **binary** codebook, 256×32 bits = **1 KB**/CB. Shared per (layer, organ) across 128 experts: 48×3×1 KB = 144 KB / 28.99e9 W = **0.00004 bpw**. Per-expert CB: ×128 → **0.005**. If **fp16** CB: 256×32×2 = 16 KB; shared **0.0006 bpw**. **Row scale** fp16: 16/in → 16/2048=**0.008** or 16/768=**0.021**. **Residual** rice_q1 0.5–2%: **+0.04–0.20** HYPOTHESIS (not measured on a 0.25-base). **Align/header** ~0.01 UNKNOWN. **nominal** 0.25. **complete expert** ≈ 0.25+0.01+0.00–0.005+0.04–0.20 = **0.30–0.47** without residual, **0.34–0.67** with. |
| **stored_bpw** | Expert **0.30–0.70** HYPOTHESIS. O005 stored n=8: **0.69–1.07** DERIVED → **stored sub-1 is the first honest candidate**. Sub-0.2 complete: **no** (always-on 0.40). |
| **active_bpw** | O005 n=8: **3.84–4.05**. n=4: **2.00–2.22**. n=1: **0.62–0.84** → **active sub-1 only if always-on also ~1-bit**. |
| **expected_reachable_bpw** | Expert **0.30–0.70**. Complete stored O005 n=8 **0.69–1.07**. Expert sub-0.2: d=48 **without** residual **if** quality holds — **same rate as GLM 0.167**, different source (signs not values). |
| **beats naked 2?** | Complete **yes** (0.3–0.7 vs 2.25). Quality vs 2-bit: **UNKNOWN**, high risk. Binary already 0.828 on up — 4× coarser signs will be worse without residual. Residual **must** work or this is GLM-5.2-shaped collapse. |
| **quality_risk** | **up_proj** first, then **gate** (SwiGLU). k-means on **signs** may be a bad alphabet (only 2^d patterns, codebook size 256 covers tiny fraction of 2^32). Limit: codebook coverage. **Do not** entropy-code indices (NS-entropy-coded-pq-indices). |
| **cheapest_falsifier** | One up_proj, **real x**. Fit sign-PQ d=32 k=256 **activation-weighted**. If held-out act-cosine < binary_g (0.828) **minus 0.05**, residual cannot cheaply save it → **kill base**. Then add rice at matched complete_bpw vs uniform_b2. Kill if still < bar **and** fast-Doctor organ fails. |
| **execution_path** | Index GEMV: gather 256-entry **1 KB** table (register-resident — **not** the Type-1 large LUT). Or unpack 32 signs from codeword + scale + dot. **Do not** materialize W. Cost/token UNKNOWN. If residual fused, same scatter as BC-RICE-ACT. |
| **applicability** | Expert + dense MLP. Not embed/lm_head (GLM R0). Not SSM state. Layer-0 may want a different d (NS-layer-zero-is-a-different-source). |
| **confidence** | HYPOTHESIS. Rate arithmetic is DERIVED; quality UNKNOWN. NS-raw-weight-pq-vq **does not auto-kill** (alphabet + act-aware) but is a **warning**. |
| **transfer** | d,k not portable. Shared-CB-across-experts is **not** expert-template tying (local d-blocks). Measure codebook collapse (single-codeword share); NS-row-norm-stratification: 94% figure was a **wrong geometry**. |
| **prior art** | Product quantization; QuIP# E8 (implicit, better native story — BC-LATTICE); AQLM additive (second codebook = this residual, but LUT-heavy); BitNet signs. |
| **doctor_risk** | 0.70 |
| **family_addition** | STRUCTURAL `signpq-d32k256+rice`. Non-native. |
| **info_gain / cost** | 9 / 4 |
| **reopen_if / kill-if** | Kill if act-cosine of base-only < 0.70 on up **or** residual needs >1% fp16 to recover (then complete → 2-bit land). |
| **NEXT_BOTTLENECK** | Base quality at 0.25 **before** spending residual budget. |

---

### BC-ACTPQ-GLM — activation-aware PQ @ ~0.17 BPW as T0 executable + sparse **output** residual

One-line: the **only Hawking sub-1/5 precedent** (0.755 cos @ **0.167 BPW on REAL activations**) as the base; residual repairs the remaining error **in y-space**.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Payload** 8/d: d=48 → **0.1667**; d=32 → 0.25; d=64 → 0.125. **Codebook** fp16 256×48×2 = **24.6 KB**. Shared (layer,organ) across experts: 48×3×24.6 KB = 3.54 MB / 28.99e9 = **0.00098 bpw**. Per-expert: **0.125 bpw** — still OK, but **do not** go per-expert-per-row. **Indices** raw 8-bit (entropy coding ~0, NS). **Scale** optional; k-means is conditional mean — **do not** add post-hoc scalar (NS-posthoc-scalar-gain). **Correction:** sparse ΔW **or** low-rank Δy=U(Vx) (BC-OUT-LR). **nominal** 0.167. **complete expert** 0.17–0.45 with small corr. |
| **stored_bpw** | Expert **0.17–0.45** HYPOTHESIS. O005 stored n=8: **0.56–0.83** DERIVED → **stored sub-1 yes**. Stored sub-0.2: **no** at n≥4. Expert-body sub-0.2: **yes candidate** (this **is** the 0.167 point). |
| **active_bpw** | O005 n=8: **3.77–3.92**. Active sub-1: **no** at 8/4-bit always-on. |
| **expected_reachable_bpw** | Expert **0.17–0.50**. Complete stored O005 n=8 **0.56–0.88**. **Sub-0.2 complete only if always-on ≤ 2-bit** (floor 0.10 + 0.16 = 0.26 still **> 0.2** at d=48; need always-on 1-bit **and** e≤0.16 → complete ≈ 0.20). Treat sub-0.2 **complete** as **not the claim**; sub-0.2 **expert body** is the claim. |
| **beats naked 2?** | Complete **yes**. Quality: GLM 0.755 cos @ 0.167 is **one organ, not Doctor**. Residual must close 0.755→Doctor. Naked 2-bit cosine on Q80 up was 0.801 — 0.755 is **worse than 2-bit cosine**, so **base-only loses quality**; **base+corr must beat 2-bit quality at <2.25 complete**. That is the whole experiment. |
| **quality_risk** | **Proxy trap** if anyone uses gaussian x. **Gate** historically dominant_failure_organ. Uniform PQ across organs **killed**. GLM-5.2 0.75 BPW expert path **failed** — different method, same “sub-bit MoE” slogan; **do not cite as support**. Embeddings: do not apply. |
| **cheapest_falsifier** | Replicate GLM protocol on **O005 one up_proj, real routed x, held-out tokens**. Report complete_bpw **including codebook**. If act-cosine ≲ 0.70, **kill** as a 0.17-base (retreat d). If ≳ 0.75, add residual at **+0.10 complete** and re-score; kill if still < uniform_b2 cosine **and** < binary+rice 1.29. Then one-layer fast-Doctor. |
| **execution_path** | PQ index GEMV with **24 KB** CB (L1, not Type-1 large LUT). Fused dequant-dot. **Not** expand. AQLM-style **multi** codebook (2nd CB = residual) is the literature form — on Apple, **second small CB** maybe OK; **do not** scale to AQLM’s large tables. Cost/token UNKNOWN. |
| **applicability** | MoE experts, dense MLP. Layer-0 separate d. Vision (O003/O006) **UNKNOWN**. O001 MLP only. |
| **confidence** | HYPOTHESIS with **MEASURED precedent** (GLM 0.167/0.755 real x). Highest **sub-0.2 expert-body** leverage in this lane. |
| **transfer** | The **protocol** (act-aware PQ, real x) transfers. The **0.755** number does **not**. Qwen3.8 proxy collapse is the transfer **warning**. |
| **prior art** | PQ; AQLM; GPTVQ; GLM Hawking 0.167. Hypernetworks: **out of lane** (generator bytes). |
| **doctor_risk** | 0.75 |
| **family_addition** | STRUCTURAL `actpq-d48k256+resid`. Non-native. |
| **info_gain / cost** | 10 / 5 (needs real x capture). |
| **NEXT_BOTTLENECK** | Held-out act-cosine at 0.167 on **current patients**; then Doctor. |

---

### BC-OUT-LR — extreme base GEMV + **output-space** rank-r correction `Δy = U(Vx)`

One-line: repair **y**, not scattered **W**; uses the LIVE Q80 down_proj fact; avoids the dead weight-SVD residual lever.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Base** binary 1.13 or SignPQ/ActPQ 0.17–0.30. **U** [out×r], **V** [r×in] at q bits. O005 gate 768×2048, r=16, q=8: 16×(768+2048)×1 B = 45.1 KB vs 1.57e6 W → **+0.229 bpw**. r=8 → **+0.115**; r=32 → **+0.459**. **No** residual index. **Scales** for U,V. **nominal** base + 0. **complete** base + r(in+out)q / (in·out). |
| **stored_bpw** | Expert **0.40–1.40** depending on base+r. With ActPQ 0.20 + r=16 q8: **~0.43**. O005 stored n=8: **~0.81** DERIVED (sub-1). With binary 1.13+0.23: **1.36** (not sub-1). |
| **active_bpw** | Extra **tiny GEMV** always on the active path (r small). Bytes = packed base + U,V of selected experts. |
| **expected_reachable_bpw** | Expert **0.4–1.4**. Down_proj may want **higher r** (Q80 whole-codec r=160 at 1.27 — that was **the codec**, not a residual). As **residual**, r should be **≪** 160 or it becomes the body. Target r=8–32. |
| **beats naked 2?** | Complete yes at r≤32 + sub-1 base. Quality: **UNKNOWN** except down_proj act-LR as **full** codec passed cosine. As residual on a **bad** base, UV can only correct the **r-dimensional** part of the error. If error is full-rank, **kill**. |
| **quality_risk** | **Full-rank residual** (dead_levers SVD energy low — that kill is **exactly** this failure mode in weight space). Prefer **activation-weighted** U,V (like hgravs). Limit: **up_proj** if SwiGLU error is not low-rank. Do not ASVD raw W. |
| **cheapest_falsifier** | On one down_proj and one up_proj: binary_g vs binary_g+UV r∈{8,16,32} at counted complete_bpw vs hgravs-as-body vs uniform_b2. Kill UV-residual if act-cosine gain per extra 0.1 bpw < rice-scatter (BC-RICE-ACT) on the **same** organ. |
| **execution_path** | Native-credible: existing hgravs **factor matvec** plus binary GEMV. Two kernels, or fused. FLOPs: r·(in+out) extra — small vs in·out. Kernel-bound decode: extra dispatch **hurts** unless fused into one CB (DSV4F 5.11 graph collapse is the pattern). Cost/token UNKNOWN. |
| **applicability** | **down_proj first** (Q80 LIVE). Gate/up second. Dense MLP. Hybrid MLP. Not attn QKV until measured (rank of those residuals UNKNOWN). |
| **confidence** | INFERRED on down (LIVE hgravs). HYPOTHESIS as residual-on-binary. Weight-SVD residual remains **dead**. |
| **transfer** | Rank not portable. Organ split: down more likely than gate. |
| **prior art** | LoRA-as-quant-residual; ZeroQuant-V2; hgravs; **not** ASVD (killed). AQLM additive is codebook-LR hybrid. |
| **doctor_risk** | 0.40 (down) / 0.60 (up/gate) |
| **family_addition** | STRUCTURAL `binary+uv-r16-q8`. Not `+correction` (that is sparse W). |
| **info_gain / cost** | 8 / 3 |
| **NEXT_BOTTLENECK** | Residual rank of **up** error on real x. |

---

### BC-ROUTE-SIDECAR — 0.3-bit **fused** base; correction as a **separate** stream, gathered only when selected / marked

One-line: MoE 8/128 already skips unselected **bases**; to make correction **not** inflate active bytes it must **not** live inside the packed expert.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Stored** = base_all_experts + sidecar_corr (all or marked) + bitmap of marked experts (128 bits/layer, negligible) + rice/UV bytes. **Must count sidecar.** **Active DRAM** = 8 selected **bases** + sidecar only for those of the 8 that are marked / predicate-true. If sidecar is **fused into** the expert tensor, active expert_bpw = stored expert_bpw and this proposal **collapses** to BC-RICE-ACT. |
| **stored_bpw** | Expert **0.3 + corr_frac·corr_bpw**. If all experts have 0.4-bpw sidecar: stored **0.70**. O005 stored n=8: **1.07** DERIVED (sub-1 if always-on 8-bit… 0.95×0.70+0.40=**1.06**). |
| **active_bpw** | If fused: same as stored expert. If sidecar + 8/128 gather of sidecar: active expert = **0.3 + (8/128)·0.4 = 0.325** only if you still **load** unselected bases… **No:** gather already loads only 8 experts’ (base+fused-corr). **Fused corr does not change 8/128 math.** The **only** active win of a sidecar is **skipping corr on easy tokens/organs** (hit_rate < 1) or **not storing corr on marked-empty experts**. O005 0 cold + single-expert Doctor-silent → marking experts **likely all-or-nothing**. Real lever: **token-predicate** skip of sidecar (hit_rate). Active expert_bpw = 0.3 + hit_rate·0.4. |
| **expected_reachable_bpw** | Stored expert **0.5–0.9**. Active expert **0.32–0.70** if hit_rate 0.05–1. O005 **model** active still floored by always-on. This is an **expert-traffic** win, not model-active-sub-1. |
| **beats naked 2?** | Stored complete yes. Active vs fused-2-bit: 8/128 of 2.25 = same selection factor; **sidecar skip** is the extra. If hit_rate=1, **no extra vs fused**. |
| **quality_risk** | Predictor miss → quality of **base-only** on hard tokens (the ones you need). Limit: **gate/up** on high-entropy router tokens. Branch divergence in kernel. |
| **cheapest_falsifier** | (1) Per-expert act-cosine of base: if std low, **expert-marking dead**. (2) Token-level ‖Δy‖ of base vs corr: if mass not concentrated on a small token subset, **hit_rate cannot be low**. Kill if 90% of residual energy needs >50% tokens. Use ≥1000 tokens (NS-calibration-88). |
| **execution_path** | Two-stream: always decode base; conditionally decode sidecar. Extra pass = TPS risk (kernel-bound). Native: **no**. Predicate must be **cheap** (already-computed router entropy / ‖x‖). Reconstruction: sidecar residency + hit counter (account as repr state if mandatory). |
| **applicability** | **MoE only** (O003/O005/O006). Dense/hybrid: token-predicate still valid but no 8/128 gather. Shared experts (O003): sidecar **always** on (hit_rate=1) — no skip. |
| **confidence** | HYPOTHESIS. Expert-marking **likely dead** on O005 (uniform + single-expert Δ=0). Token-predicate **untested**. |
| **transfer** | Do not transfer hit_rate. Uniform-routing patients share the “no cold” constraint. |
| **prior art** | Matryoshka / progressive (T0 always, T1 on demand) — **boundary with matryoshka lane**; here the novel bit is **sidecar vs fused** accounting. Mixture-of-quantizers. |
| **doctor_risk** | 0.55 |
| **conventionality** | **ACTIVE_NX** (skip) + STRUCTURAL (sidecar). |
| **family_addition** | `actpq+sidecar-rice` ; runner `+correction` **fuses** — grammar needs `+sidecar`. |
| **info_gain / cost** | 6 / 5 (needs route+token traces) |
| **NEXT_BOTTLENECK** | Residual-energy concentration over tokens. |

---

### BC-LATTICE-LF — lookup-free lattice / incoherence base (QuIP# E8 / QTIP-class) + Hessian residual

One-line: **implicit codebook (0 table bytes)** so 0.3–1.0 bit is honest; Apple-GPU trellis decode is Type-1 dead — only admit a **lane-independent** layout.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Payload** R bits/dim (target 0.25–1.0). **Codebook 0** (E8 / trellis is math). **Hadamard / incoherence** is compute, not bytes (unless stored randomized signs: 1 bit/W — **don’t**; use implicit Hadamard). **Scale** per group/row. **Residual** rice or UV, counted. **Trellis state** if QTIP: if it must be stored per token, **charge it** as repr-attributable state; if only kernel regs, 0. **nominal** R. **complete** R + scale + residual (+state if any). |
| **stored_bpw** | Expert **0.4–1.2** HYPOTHESIS at R=0.25–1.0 + scale 0.01–0.13 + residual 0.05–0.2. Sub-1 stored O005 n=8 possible if R+corr ≲ 0.63. |
| **active_bpw** | Fused in expert pack. Model active still always-on-floored. |
| **expected_reachable_bpw** | Expert **0.5–1.2**. Literature QuIP# **2-bit** is the **naked-2 quality champion** — going **below** 2 bits is the question; 2-bit lattice **is** the baseline to beat, not the proposal. This proposal is **R<2 + residual**. |
| **beats naked 2?** | Complete yes if R+corr < 2.0 (and QuIP# 2-bit already has ~0 codebook so complete ≈ 2 + scale). **Must beat QuIP#-2 / QTIP-2 quality at lower complete**, not just beat `uniform_b2`. That is a **higher bar**. |
| **quality_risk** | QTIP 3-bit quality **leaning Type-2 NO-GO** in-house. 0.3-bit lattice is **very** few points per 8-D (2.4 bits/8-D ≈ 5 levels) — residual does the work. Limit: **incoherence + coarse lattice** may still be gaussian-like mid/late layers (NS-post-hoc-coding: mid/late nearly gaussian, 0.059 decades from Shannon). Layer-0 named exception. |
| **cheapest_falsifier** | One organ, real x: E8/R=1.0 vs binary_g vs uniform_b2 at counted complete (include scales). Kill lattice if act-cosine ≤ binary_g at **higher or equal** complete. Do **not** run Metal trellis TPS (already Type-1 dead). Quality-only first. |
| **execution_path** | Native **only** if decode is **lane-independent bit-arithmetic**, not serial trellis. Current QTIP Metal: **NO-GO Type-1**. QuIP# E8 decode is LUT-free **in principle**; Apple kernel **does not exist** here. Expand-to-fp16 = fake. Cost/token UNKNOWN. Hadamard: extra O(n log n) per vector — **charge FLOPs**. |
| **applicability** | All weight organs except tiny (router 0.03 GB O005 — protect, don’t lattice). Layer-0 maybe **more** non-gaussian → lattice **more** justified there. |
| **confidence** | HYPOTHESIS. Best **complete-accounting** story (0 codebook). Worst **Apple execution** story among this set. |
| **transfer** | Incoherence is architecture-agnostic. Quality of R<2 is not. |
| **prior art** | **QuIP#** (E8, 2-bit); **QTIP** (trellis, implicit); **AQLM** (additive, LUT — avoid large); **BitNet** (not lattice). |
| **doctor_risk** | 0.60 (R=1) / 0.80 (R=0.3) |
| **family_addition** | STRUCTURAL `e8-R025+rice`. Non-native. **Do not** add QTIP-Metal as NX. |
| **info_gain / cost** | 7 / 6 (new kernel if execution) / 3 (quality-only oracle) |
| **NEXT_BOTTLENECK** | Lane-independent layout; else **quality-oracle only**, not an NX. |

---

### BC-BILLM-DUAL — salient/non-salient **dual binarization** (BiLLM/DB-LLM/OneBit class) as ~1.1–1.4 complete, not 0.3

One-line: PTQ 1-bit **quality** path; residual is a **second binary** of the salient residual, not a 16-bit scatter. **Does not reach 0.3 base.** Include so the engine does not confuse it with SignPQ.

| Field | Content |
|---|---|
| **complete_byte_accounting** | **Body** 1 bit/W + group scale (1.13). **Salient mask** (bitmap or N:M). **Second binary** on salient residual: +1 bit × salient_frac + its scale. If salient=10%: **+0.10 + 0.01**. **complete expert ~1.25–1.40**. Compare to rice_q1 2% **1.29** — similar budget, **different** residual alphabet (binary vs rice-q1). |
| **stored_bpw** | Expert **1.20–1.45**. O005 stored n=8 **1.54–1.78**. Not sub-1 at 8-bit always-on. |
| **active_bpw** | Fused. O005 n=8 ~**4.3–4.5**. |
| **expected_reachable_bpw** | Expert **1.2–1.5**. Sub-0.2: **no**. Beats naked 2 **on complete** (yes vs 2.25). |
| **quality_risk** | Literature 1-bit PTQ often needs **QAT** (BitNet) to hold capability. PTQ BiLLM is the cheap test. Limit: **gate** and **generation** (GLM-5.2 1-bit-class failed). |
| **cheapest_falsifier** | BiLLM-style dual-binary vs rice_q1 @ **matched complete_bpw** on one up_proj real x. Kill BiLLM if cosine ≤ rice_q1 (then rice already dominates this budget). |
| **execution_path** | Two binary GEMVs or fused. More native-friendly than rice scatter. Kernel: 2× binary. Cost UNKNOWN. |
| **applicability** | Experts + dense MLP. Not embed. Hybrid MLP. |
| **confidence** | HYPOTHESIS. Likely **dominated by BC-RICE-ACT** at same complete — still worth the **one-organ** test because execution is cleaner. |
| **transfer** | Weak. |
| **prior art** | **BiLLM**, **PB-LLM**, **OneBit**, **DB-LLM**, **BitNet b1.58** (QAT, 1.58 complete not 0.3). |
| **doctor_risk** | 0.55 |
| **info_gain / cost** | 5 / 2 |
| **NEXT_BOTTLENECK** | One-organ bake-off vs rice_q1. |

---

## 5. Harvest order (info/cost, this lane only)

| Rank | ID | Why first | Sub-1 stored O005 n=8 | Sub-0.2 expert body | Beats naked 2 complete | Native |
|---|---|---|---|---|---|---|
| 1 | **BC-ORGAN-HETERO** | Already screened on Q80; stops uniform residual | no (~1.57) | no | **yes MEASURED screen** | no kernel |
| 2 | **BC-RICE-ACT** | Encoding was the Q80 fake-win; act-select is the cheap next bit | no | no | **yes MEASURED** vs uniform_b2 | no |
| 3 | **BC-OUT-LR** | down_proj LIVE hgravs; don’t scatter-residual down | maybe w/ ActPQ base | no | yes if r small | hgravs kernel exists (slow path) |
| 4 | **BC-CHANNEL-PROMOTE** | Cheapest topology discriminator | maybe | no | if p small | better than scatter |
| 5 | **BC-SIGNPQ-025** | First **0.3 complete base** that is honest | **yes candidate** | d=48 maybe | complete yes; quality UNKNOWN | small CB maybe OK |
| 6 | **BC-ACTPQ-GLM** | Only sub-1/5 precedent; only sub-0.2 **expert** lever in this lane | **yes** | **yes candidate** | complete yes; **must beat 2-bit quality** | small CB |
| 7 | **BC-BILLM-DUAL** | Cheap bake-off vs rice at ~1.3 | no | no | yes complete | 2× binary |
| 8 | **BC-ROUTE-SIDECAR** | After a working corr; tests skip | maybe | no | yes stored | two-stream TPS risk |
| 9 | **BC-LATTICE-LF** | Best byte story; worst Apple kernel | maybe | hard | must beat QuIP#2 not just b2 | Type-1 trellis dead |

Do **not** emit `q2-g32-experts+correction` as a frontier proposal — already in `candidate_families.base_plus_correction`. That is affine-q2 + 2–5% fused corr, **not** a 0.3 base, and its default residual topology is unspecified (likely the **legacy 48 b/outlier** failure mode).

---

## 6. Explicit UNKNOWN (do not fill)

- Any **Doctor** score (fast 12-item or full) for any of the above on O001/O003/O004/O005/O006.
- µs/token, FLOPs/token, DRAM/token for binary / rice / PQ / E8 kernels on M3.
- Whether act-select reduces residual **frac** below 2% at fixed cosine.
- Whether salience is **channel-aligned**.
- Sign-PQ or ActPQ cosine on **current** patients (GLM 0.755 is not O005).
- Residual rank of up/gate error.
- Token-wise residual-energy concentration (hit_rate).
- O003/O006 vision-tower sensitivity.
- O001 SSM quant floor.
- Whether a **k=256** gather is Apple-GPU-OK (large LUT is dead; 1–25 KB is untested).
- Generation quality of Q80 mixed 1.43 (explicitly untested).

---

## 7. One-sentence lane verdict

A **0.3-BPW complete base** is only honest via **multi-weight codes** (SignPQ / ActPQ / lattice), not via ternary/binary; a **small rice/UV/channel repair** already **beats naked 2-bit on complete_bpw and on Q80 up_proj cosine** when the residual is **not** u32+fp16; **stored sub-1 on O005** is reachable if expert ≲ 0.63 with 8-bit always-on; **stored sub-0.2 complete is arithmetically forbidden** at ≥4-bit always-on; **active sub-1 is an attn/embed/head problem** (46% of O005 active mass). Doctor, not cosine, is the remaining gate.

---

LANE: base+correction  
PROPOSALS: 9 (BC-RICE-ACT, BC-ORGAN-HETERO, BC-CHANNEL-PROMOTE, BC-SIGNPQ-025, BC-ACTPQ-GLM, BC-OUT-LR, BC-ROUTE-SIDECAR, BC-LATTICE-LF, BC-BILLM-DUAL)  
SCHEMA: mechanism · complete_byte_accounting · stored_bpw · active_bpw · expected_reachable_bpw · quality_risk · cheapest_falsifier · execution_path · applicability · confidence · transfer  
FILES: none (no write tool in this lane; packet is this report)  
GUARDRAILS: complete accounting honored; activation-aware only; Doctor authority; no expand-to-dense fake; stored≠active; UNKNOWN marked.
