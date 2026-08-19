I'll read the campaign packet first and follow its required-output schema for the active-sub1-moe lane.I'll pull the Hawking prior art, patients, and MoE accounting so the proposals stay grounded in this campaign.Hawking's active_bpw math and the patient route maps will decide whether sub-1 is an expert-codec problem or an always-on problem.Checking a few remaining numbers (q2 failure, GLM 0.167 geometry, O003 shared-expert bytes) so the ledgers stay honest.# ACTIVE SUB-1 MoE — novelty lane

**Claim first.** Gather cannot manufacture Hawking `active_bpw < 1` by itself. On the current patients, always-on organs already blow the budget at q4. Sub-1 *Hawking-active* is a joint always-on + expert problem. Sub-1 *expert-path* (selected-expert complete BPW) is a different, easier win. Sub-0.2 Hawking-active is not an MoE trick.

Selection `topk/N` is an **execution** lever (bytes moved, residency). It is **not** a compression stat. Never write `stored_bpw × 0.0625`. Policy `no_fake_active_density`; bible §29 stored≠active.

---

## Accounting law (MEASURED identities)

Hawking receipts:

| axis | formula | O005 q3-g32-experts |
|---|---|---|
| `stored_bpw` | `8 × stored_bytes / N_total` | **4.0253** MEASURED |
| `active_bpw` | `8 × touched_bytes / N_active` | **4.2305** MEASURED |
| `touched_bytes` | `B_ao·N_ao + B_e·(k/N)·N_e + B_shared·N_shared` | **1773.1 MB/tok** |

Gather scales the **expert term of bytes/token**. It does **not** change `B_e` (complete bits on a selected expert). Uniform expert quant ⇒ selected-expert BPW = stored-expert BPW.

**O005 identity (census + gravity, DERIVED):**

```
active_bpw = 0.4596·B_ao + 0.5404·B_e
```

`B_*` = complete bits (payload+scales+biases+tables+offsets+correction+meta+align+container+reconstruction state).

| B_ao complete | B_e needed for Hawking-active < 1 |
|---|---|
| 4.50 (q3-mix always-on, MEASURED) | **impossible** (floor 2.07 even at B_e=0) |
| 2.00 | B_e < **0.15** |
| 1.50 | B_e < **0.58** |
| 1.00 | B_e < **1.00** |
| 0.50 | B_e < **1.43** |

**Sub-0.2 Hawking-active** (`touched < 83.8 MB/tok` on O005): need B_ao ≲ 0.2 **and** B_e ≲ 0.2. Gather does not help the denominator. That is a whole-active-set GLM-function codec, not MoE-NX.

**Physical bytes/token targets (O005, DERIVED from MEASURED N_active=3.353e9):**

| Hawking-active | touched bytes/tok |
|---|---|
| 4.2305 q3-g32 MEASURED | 1773 MB |
| 1.0 | 419 MB |
| 0.2 | 83.8 MB |
| expert-only @ 1.0 complete | 226.5 MB |
| expert-only @ 0.167 | 37.8 MB |
| always-on @ 4.50 | **867 MB** — already 2.07× the entire 419 MB budget |

---

## Patient ledger

| | O005 Qwen3-30B-A3B | O006 Qwen3-VL-30B-A3B | O003 Kimi-VL-A3B |
|---|---|---|---|
| N_total | 30.532B | 31.071B | 16.408B |
| N_active | 3.353B (11%) | 3.892B (12.5%) | 3.362B (20.5%) |
| experts | 128, top-8, 0 shared | same + vision | 64 top-6 + **2 shared** |
| sel/full | **0.0625** | 0.0625 | **0.09375** |
| expert mass | 95% stored | 93% | 88% |
| active_bpw | 0.460 B_ao + 0.540 B_e | 0.534 B_ao + 0.466 B_e (vision in ao) | **0.599 B_ao + 0.401 B_e** (shared in ao) |
| route | entropy 6.16/7.0, **0 cold**, top16=18%, pop=1.33% | 6.23/7.0, 0 cold, 18%, 1.38% | 5.46/6.0, 0 cold, 31%, 2.29% |
| token overlap | **0.408** (vs indep ~0.0625) | 0.403 | 0.376 |
| cross-layer Jaccard | 0.035 ≈ chance | 0.036 | 0.054 |
| mlx | `gather_qmm` **compute**; **full expert body resident** | same | same (+ shared always) |
| Doctor anchor | q3-g32-experts **10/12** CANDIDATE_PASS | 4bit-mlx specimen | 4bit-mlx **12/12** |
| q2 | contract: fail, localize **gate/up + router** | UNKNOWN (transfer from O005) | UNKNOWN |

Route: **no cold-expert compression** (`R-uniform-routing-no-cold-compress`, NS-calibration-88). Frequency-alloc is N/A. **Next-token stickiness is real** (overlap 0.41); **cross-layer reuse is not** (NS-large-expert-cache premise holds).

---

## Prior art / Hawking science (do not rediscover)

| result | what it is | what it is not |
|---|---|---|
| GLM **0.755 cos @ 0.167 BPW** on **real** activations | organ-level function reconstruction | **not** Doctor-sealed generation |
| GLM-5.2 H0.98 @ ~0.98 BPW | integrity SEALED | capability **REFUSED** (semantic collapse) |
| GLM-5.2 weight-space sub-bit | 0.116–0.157 cos @ 0.75 BPW, **< null 0.898** | Type-1 dead (`dead_levers`) |
| Qwen3.8 gaussian-proxy sub-bit | all collapsed 0.5–0.8 BPW, output-div ~0.69 | proxy trap |
| Q80 mixed complete **1.43** (8-bit non-expert) | organ-cosine **screen** (bar 0.8604): binary gate 1.13, rice-q1 2% up 1.29, hgravs01-r160-b3 down 1.27 | **not packed, not generated, no decode kernel** |
| Q80 pairwise expert cos | ~0.004 / subspace overlap ~0.02 | NS-inter-expert-redundancy holds |
| NS-raw-weight-PQ/VQ @ ~1 bit | A1_1p0 + R2_subhalf collapsed 6/6 on qwen3-235b real forward | kills **that family**, not source-changing methods |
| Metal **learned codebook LUT** | Type-1 dead (random LUT gather punished) | lookup-free (QTIP-class) only — but QTIP trellis decode also Type-1 dead |
| mlx 29.3 TPS ≈ dense | kernel-bound, not BW-bound (A3B_RECON) | fewer bytes ≠ TPS unless decode kernel is cheap |

**Native today:** `moe.metal` grouped GEMM **Q4_K_M fused dequant-in-FMA**; mlx `gather_qmm` 2/3/4/8 affine-grouped. **1-bit, PQ LUT, trellis, correction planes, tiers: non-native.**

---

## M1 — Always-on tax (the hidden blocker)

**one-line:** Hawking-active < 1 is impossible while attn+embed+lm_head+router sit at ~4.5 complete; crush always-on **or** stop claiming Hawking-active sub-1.

**complete_byte_accounting**

| bucket | O005 params | count every byte |
|---|---|---|
| attn | 0.906B | payload+group scales/biases+align |
| embed + lm_head | 0.622B | same; tied=false so **both** |
| router | 12.6M | keep high-prec; 4.5 BPW is 7.1 MB — cheap |
| norms | 0.20M | bf16 (16 BPW) is 0.4 MB — leave |
| selected experts | 1.812B | **separate** (M2–M5) |
| unselected experts | 27.18B | **stored only**; not in touched |
| vision (O006) / shared (O003) | see ledger | always-on if on the token path |

**stored_bpw / active_bpw**

- stored: unchanged by this split (experts still dominate stored).
- Hawking-active: `0.46 B_ao + 0.54 B_e` (O005). This mechanism only moves `B_ao`.

**expected_reachable_bpw** (HYPOTHESIS ranges; Doctor UNKNOWN)

| B_ao (attn q2 / embed q3–4 / router q8) | Hawking-active if B_e=4.0 (current experts) | if B_e=1.23 (Q80 screen) | if B_e=0.17 (GLM recon) |
|---|---|---|---|
| ~2.5 (attn~2, embed~4) | ~3.3 | ~1.8 | ~1.24 |
| ~1.5 | ~2.85 | ~1.35 | **~0.81** |
| ~1.0 | ~2.58 | ~1.12 | **~0.55** |

Hawking-active < 1 on O005 needs **B_ao ≲ 1.5 and B_e ≲ 0.6**, or B_ao ≲ 1.0 and B_e ≲ 1.0. O003 is worse (0.60 of active is always-on).

**quality_risk / limiter:** attn (largest always-on organ) and embed/lm_head (vocab). Router ablation zero-kills Doctor (O005 MEASURED); do not touch router. q2-experts already localized to **gate** — attn-q2 is a **new** Doctor surface.

**cheapest_falsifier:** one-layer attn-only affine-q2 (g32) vs q4 on O005 fast-Doctor; if delta_hits ≤ −2, always-on cannot fund sub-1 without a non-affine attn codec. Cost: 1 organ, 1 specimen.

**execution_path:** existing Q4_K / mlx affine grouped for q2–q4 attn. Native. Decode cost: always-on GEMMs already run; bit-cut helps **iff** attn kernel is BW-bound (UNKNOWN on this box; A3B whole-model was kernel-bound).

**applicability:** all patients (dense too). Necessary for MoE Hawking-active < 1; not MoE-specific.

**confidence:** **HIGH on the algebra (MEASURED census)**; **LOW on Doctor-valid B_ao ≲ 1.5**. Why: identity is arithmetic; attn-q2 quality is unmeasured here.

**transfer:** identity transfers (retune fractions). O003 shared experts are always-on — must be in B_ao. O006 text-only (vision dropped) ≈ O005; multimodal path adds vision to B_ao.

---

## M2 — Gathered activation-function codec (GLM 0.167 path)

**one-line:** Code each expert’s **map x↦xW** on **real routed activations**, store all experts, **gather only top-k codes**; Metal LUT forbidden.

**complete_byte_accounting**

| item | stored | touched/token |
|---|---|---|
| per-expert indices/payload | all 128 | **8** |
| per-expert scales/biases | all 128 | **8** |
| **shared codebook / trellis tables** | once | **fully resident every token** (this is the fake-density trap) |
| correction / residual stream | all or route-conditional | only selected |
| always-on organs | as M1 | as M1 |
| unselected expert payload | yes | **no** |
| reconstruction scratch | — | per-token working set; count if not fused |

Nominal 0.167 with a 3-BPW **resident** codebook = **3-BPW active**, not 0.167. GLM 0.167 codebook geometry: **UNKNOWN** (must re-measure complete_bpw on this parent).

**stored_bpw / active_bpw**

- stored ≈ `0.95·B_e_complete + 0.05·B_ao` (O005). Can be **>1** if codebook+all-experts large.
- expert-path BPW = `B_e_complete` (gather does not change it).
- Hawking-active = `0.46 B_ao + 0.54 B_e_complete`.
- physical expert bytes/tok @ 0.167 = **37.8 MB** (O005, DERIVED). Plus always-on.

**expected_reachable_bpw**

| axis | range | evidence |
|---|---|---|
| expert-function recon cos | 0.17 complete **possible as a screen** | GLM 0.755 cos @ 0.167 REAL x |
| expert-function Doctor | UNKNOWN; GLM-5.2 sealed-sub1 **collapsed** | capability ≠ cosine |
| Hawking-active | **still >1** unless M1 lands | algebra |
| Qwen3.8 proxy 0.5–0.8 | **do not use** | all collapsed |

Honest: **expert-path 0.2–0.8 complete as a reconstruction screen**; **Doctor-valid expert-path <1 = UNKNOWN**; Hawking-active <1 **not this mechanism alone**.

**quality_risk / limiter:** **gate_proj** (NS-uniform-subbit, F1 dominant_failure_organ=gate, O005 q2 localize gate). Down more tolerant (Q80; R-organ-inversion). Layer 0 different source (R-layer0). **Doctor, not cosine.** Gibberish still useful if it names the broken expert/organ.

**cheapest_falsifier:** capture **real** routed-x for **one** mid-layer expert (L10, not L0) on ≥1k **held-out** tokens (not the 88-token kill). Fit activation-PQ/QTIP-style **function** codec at target 0.17–0.5 complete (payload+codebook counted). Score `cos(xŴ, xW)` **and** a 12-item fast-Doctor with only that expert replaced. Kill if (a) codebook residency makes complete_bpw ≥ affine-q2, (b) cos < organ bar (Q80 used 0.86; GLM 0.755 is a **different** bar — do not mix), (c) Doctor delta_hits ≤ −2. **Forbidden:** Gaussian/synthetic x.

**execution_path:** **not** PQ LUT-GEMM (Metal learned-codebook Type-1 dead). **not** QTIP trellis decode (Type-1 dead). Credible native: **lookup-free bit layout + fused dequant-in-FMA** (Q4_K-class), or **factored (xU)V** without materializing W (ASVD/low-rank residual Type-1 dead on dense — reopen only if this expert’s activation-Gram is peaked; Q80 down_proj **did** screen-pass hgravs01-r160). Decode FLOPs/tok: 48·8·3·2048·768·2 = **4.83e9** FMA on O005 expert path **plus** dequant. If dequant ALU > saved BW, TPS regresses (A3B kernel-bound).

**applicability:** MoE experts (O005/O006/O003 routed). Not attn (different source). Shared experts (O003) = always-on dense, not gathered.

**confidence:** **MEDIUM on reconstruction screen transferring**; **LOW on Doctor**; **LOW on native kernel**. Why: one real activation-space win exists; every sealed sub-1 GLM **generation** failed; Metal LUT/trellis dead.

**transfer:** O006 language-MoE = O005 geometry (768×2048, 8/128) — **highest**. O003 1408×2048, 6/64, DeepSeek-V3 router — retune. Q80 512×2048, 10/512 — codec hyperparameters do not copy. Dense: no gather term.

---

## M3 — Q80 mixed, gathered (nearest measured recipe, not sub-1 Hawking-active)

**one-line:** Per-organ expert mix that **screen-passed** Q80 at complete **1.43**: binary gate, binary+rice-q1 2% outliers up, activation-weighted low-rank down; gather top-k; protect router/attn.

**complete_byte_accounting** (Q80 MEASURED screen; O005 numbers DERIVED assuming same complete_bpw — **transfer UNKNOWN**)

| organ | codec | Q80 expert_bpw (complete, claimed) | O005 touched/tok if transfer |
|---|---|---|---|
| gate | `binary_group` | 1.1269 | 85 MB |
| up | binary + rice-q1 RMS sparse residual @2%, 8.24 bits/outlier | 1.2918 | 98 MB |
| down | hgravs01 r160 b3 activation-weighted low-rank | 1.27 | 96 MB |
| mixed expert | — | **1.230** | **278 MB** |
| non-expert @8-bit | — | complete **1.431** (Q80 mass 97% expert) | O005 non-expert is **5%** of stored, **46% of active** — **do not copy the 1.43 identity** |
| rice index stream | indices+values+group occupancy | counted in 1.29 | only **selected** experts’ residual pages if packed for gather |
| r160 factors | U,V + scales | counted in 1.27 | fused `(xU)V` or **fake** (materialize W) |
| container/align | — | counted | — |

Q80 identity `0.970·B_e + 0.030·B_non` **does not apply** to O005 (mass fractions differ). O005 complete stored ≈ `0.950·B_e + 0.050·B_ao`.

**stored_bpw / active_bpw** (O005, DERIVED **if** B_e=1.23 and B_ao=4.50)

- stored ≈ 0.95·1.23 + 0.05·4.50 = **1.39**
- Hawking-active = 0.46·4.50 + 0.54·1.23 = **2.73**
- expert-path BPW = **1.23**
- physical touched ≈ 867 + 278 = **1145 MB/tok** (still 2.7 Hawking-active)

With M1 B_ao=1.5: Hawking-active ≈ **1.35**. Still not sub-1. This is the **1.3–2.7 corridor**, not the sub-1 win.

**expected_reachable_bpw:** expert-path **1.15–1.40** if Q80 screen transfers and packing holds; Hawking-active **1.3–2.8** depending on M1. Sub-1 Hawking-active: **no**. Sub-0.2: **no**.

**quality_risk / limiter:** up_proj (binary alone failed Q80 bar; residual required). down only measured post-SwiGLU (Q80). **claim_boundary:** artifact_packed=false, generation=false, decode_kernel=false. Organ cosine ≠ Doctor.

**cheapest_falsifier:** O005 L10, one expert, **real routed x**, run the three Q80 codecs at published complete_bpw (count rice/r160 bytes). Kill if any organ cos < 0.86 **or** packed complete_bpw > 1.5 expert. Then one-layer Doctor. Do **not** skip packing — Q80 explicitly refused the ≤1.5 claim until packed.

**execution_path:** binary: popcount/BitNet-style **or** dequant-to-±scale (latter is a fake if you expand to f16 body). rice residual: **scatter-add of 2% of rows** on selected experts only — decode cost = sparse gather, not dense. down r160: **fused two GEMMs** `(x @ U) @ V`, never materialize 2048×768. Native kernel **does not exist** (Q80 note). mlx affine cannot express this.

**applicability:** MoE SwiGLU experts. O005/O006 first (same 768). O003 1408 retune rank. Hybrid/dense MLP: same mix without gather.

**confidence:** **MEDIUM on screen transfer to O005/O006**; **LOW on packed+Doctor**; **LOW on kernel**. Why: Q80 cleared a cosine bar on a **different** parent; negatives say weight-space sharing is dead but this mix is **per-organ**, not cross-expert.

**transfer:** O006 yes (same expert tensors). O003 maybe. Qwen3.8 dense low-rank is a **different kill** (NS-global-dense-lowrank) — does not auto-kill expert down_proj low-rank.

---

## M4 — Route-conditional correction (T0 gathered + T1 only on selected)

**one-line:** Store a 1-bit/binary T0 for **all** experts plus a sparse residual T1; **execute T1 only for top-k** (and optionally only when router confidence is low). Stored may exceed 1; active pays T0+T1 on 8/128.

**complete_byte_accounting**

| item | stored | touched/token |
|---|---|---|
| T0 signs/scales (binary/ternary) | all experts | top-k |
| T1 residual indices+values (rice/CSR/bitmap — Q80 compared these) | all **or** paged by expert | **top-k pages only** if packed per-expert |
| T1 occupancy metadata | per expert | top-k |
| router (q8) | 25 MB bf16 / ~7 MB q4 | always |
| always-on | M1 | M1 |
| alignment + container | yes | headers of selected pages |
| reconstruction: `y = x·T0 + x·T1_sparse` | — | fused; **do not** write a dense f16 expert |

If T1 is one global blob, gather cannot skip unselected residuals → **fake active**. Must be **per-expert pages**.

Q80 rice vs uint32 vs bitmap: rice won packing; Lloyd-PQ **index** entropy-coding is dead (NS-entropy-coded-pq, 0–0.7%). Rice on **sparse residual positions** is a different source (not Lloyd-optimal PQ indices) — **alive**.

**stored_bpw / active_bpw** (sketch; numbers HYPOTHESIS)

Let T0=1.13 complete (Q80 binary_g), T1=0.16 (Q80 residual budget at 8-bit non-expert was 0.174 — **Q80-specific**).

- stored expert ≈ 1.13+0.16 = 1.29 (if T1 stored for all)
- expert-path active ≈ 1.13+0.16 = 1.29 (selected still decode T1)
- **active savings vs stored appear only if T1 is skipped** (confidence gate, M5) **or** if we compare to a world that moved unselected T1 (residency), not vs Hawking-active identity

Hawking-active still `0.46 B_ao + 0.54 (B_T0+B_T1_selected)`.

**expected_reachable_bpw:** expert-path **1.1–1.5** with T1 always-on-selected; **0.8–1.2** if T1 hit-rate ~30% (UNKNOWN). Hawking-active <1 only with M1. Sub-0.2: T0 alone would need to be the GLM-function codec (M2), not affine-binary.

**quality_risk / limiter:** T0-only on **gate** (binary gate passed Q80; binary up **failed**). Uniform T0 across organs = NS-uniform-subbit. Correction topology matters more than budget (`candidate_families.base_plus_correction`). Post-hoc **scalar gain** on a conditional-mean quantizer is algebraically dead (NS-posthoc-scalar-gain) — T1 must be a **residual of a non-MMSE-orthogonal** base, or an explicit sparse additive (Q80 reconstruction is that).

**cheapest_falsifier:** T0=binary on **up_proj only**, one layer, real x, vs T0+2% rice T1 at **matched complete_bpw**. If T1 does not restore cos across the Q80 bar, topology is wrong — **do not raise budget**. Then Doctor with T1 applied only to selected experts (should match T1-everywhere if packing is correct; mismatch = gather bug).

**execution_path:** T0 fused sign-GEMM; T1 CSR scatter on selected expert pages. mlx: **non-native**. Hawking: no residual kernel today. Decode extra: ~2% of K-dim indexed adds per selected expert. Cheap vs dense q3 **if** the sparse kernel is not launch-bound (UNKNOWN; host dispatch Type-1 dead at 0.5% wall historically — keep T1 inside the expert TG).

**applicability:** MoE. Route-conditional is the MoE-unique part (dense would always pay T1).

**confidence:** **MEDIUM**. Why: Q80 residual is the only packed-budget-aware mix that cleared a bar; O005 q2 fail wants gate protection not global q3. T1-hit-rate unmeasured.

**transfer:** O005↔O006 strong. O003 shared experts should **not** use T0-only (always-on, quality-critical). Cross-expert shared T1 dictionary: likely dead (pairwise cos ~0).

---

## M5 — Confidence-gated Matryoshka gather

**one-line:** Nested T0⊂T1⊂T2; per-token, router margin / entropy picks the tier; report **E[bytes/tok] + p99**, not a single BPW.

**complete_byte_accounting**

| tier | stored | decoded/token |
|---|---|---|
| T0 executable (binary/1.58/function-PQ) | all experts | **always** (selected) |
| T1 correction | all experts, per-expert pages | if `margin(top1,top8) < τ` or layer-entropy high |
| T2 hi-prec channels / exact outliers | small | rare |
| tier id | 2 bits × k × layers | tiny (48×8×2 = 768 bits/tok) |
| always-on | M1 | M1 |

Must count **all stored tiers** in `stored_bpw`. Active = T0 + 1{hit}·T1 + … on selected experts only. Fake if T2 is dense-expanded.

**stored_bpw / active_bpw**

- stored ≥ T0+T1+T2 (will exceed 1 if T2 is fat).
- Hawking-active is a **random variable**. Quote `E[active_bpw]`, `p99`, hit-rate.
- If T1 never hits, T1 bytes are pure stored waste (`matryoshka_tiers` falsifier already in `candidate_families`).

**expected_reachable_bpw:** E[expert-path] **0.7–1.3** if T0 is ~1.1 and T1 hit-rate 20–40% at +0.2 complete. Hawking-active <1 still needs M1. **UNKNOWN** until τ is fit on held-out routes (≥1k tokens, disjoint; NS-calibration-88).

**quality_risk / limiter:** Goodhart on τ (over-skip T1 → gate collapse). Layer 0 may need T1 always (`R-layer0`). Doctor must use **held-out** prompts, not the τ-calibration set.

**cheapest_falsifier:** decode T0-only vs T0+T1 on fast-Doctor (already in family). Then: compute router margin histogram on 1k tokens; if T1-needed tokens (Doctor-fail under T0) are **not** concentrated in low-margin mass, the gate is uncorrelated → **kill M5**, keep M4.

**execution_path:** same as M4 plus a 1-bit-per-expert skip. Prefetch T1 pages for low-margin tokens. Kernel: one launch, predicated T1; not a second pass (dispatch overhead dead).

**applicability:** MoE with a softmax/sigmoid router that exposes a margin. O005 `norm_topk_prob=true` (renorm after top-8) — margin is on **pre-renorm** logits (use those). O003 DeepSeek noaux_tc: different score scale; τ does not transfer.

**confidence:** **LOW–MEDIUM**. Why: family exists; correlation(margin, T0-error) is UNKNOWN and is the whole lever.

**transfer:** O006 yes. O003 retune. Dense: no router margin (could use attn entropy — out of this lane).

---

## M6 — Packed selected-expert working set + next-token prefetch

**one-line:** True NX gather: **stage only top-k quantized tiles** (mlx currently keeps the full body resident); prefetch from the **0.41 adjacent overlap**, not a 64 GB cross-layer cache.

**complete_byte_accounting**

This mechanism does **not** change stored complete_bpw. It changes **DRAM/UMA movement and residency**.

| stream | bytes/tok (O005, at codec B_e) |
|---|---|
| selected expert payload+scales | `N_sel · B_e / 8` = 226.5 MB × B_e |
| unselected | 0 moved if packed; **mlx today moves conceptually 0 extra for compute but RAM holds all 14.5 GB q3 experts** |
| prefetch of persist set | ~0.41 × selected (MEASURED overlap) — extra **if miss-speculated**; expected waste ~0.59 × extra-fetch if naive “prefetch previous 8” |
| always-on | M1 |
| working-set cap | **not 64 GB** (NS-large-expert-cache: 0 evictions, 0 cross-layer reuse) |

Cross-layer cooccurrence **0.065 ≈ chance**. Do not cache experts across layers of the same token.

**stored_bpw / active_bpw**

- stored: whatever codec (q3=4.03 MEASURED).
- Hawking-active: **unchanged** by residency (same weights touched).
- **Physical movement** can drop from “full expert file in RAM” to selected tiles. That is the NX win mlx does not take (`O005_NX_gather`: `full_expert_body_resident: true`).
- Policy objective for MoE is **active-bytes/token**, not stored-density. This is the **execution** half of sub-1.

At q3 B_e=4.0: selected expert **906 MB/tok** moved if packed. At B_e=1.23: **278 MB**. At B_e=0.167: **38 MB**. Always-on extra.

**expected_reachable_bpw:** not a BPW cut. Expected **moved** expert bytes: same as selected-term above. Working-set RAM: `O(k · bytes_per_expert_layer × n_layers)` if streamed layer-by-layer, or `O(k · n_layers · …)` if all layers hot. Layer-streamed resident experts @ q3: 8·3·2048·768·4/8 = **18.9 MB/layer** — tiny. The 14.5 GB is the **unselected** inventory. **This is the largest honest NX gap on O005.**

**quality_risk / limiter:** none representational (same bits). Risk is **correctness of gather indices** and prefetch miss-speculation (wasted BW, not quality). Kernel-bound decode: **may not move TPS** (`R-sparse-active-expert-gather.reopen_if`; A3B 29.3 TPS ≈ dense). Still a **memory** win on 96 GB.

**cheapest_falsifier:** (1) measure `resident_expert_bytes` vs `k/N · expert_bytes` on mlx `QuantizedSwitchLinear` — already MEASURED: full body resident. (2) prototype packed gather for **one** layer; if wall time ≥ dense `gather_qmm`, kernel-bound kill for **TPS**, not for residency. (3) prefetch previous expert set: hit-rate should ≈ 0.41; if measured ≪ 0.3 on 1k tokens, stickiness was prompt-specific (O005 used 573 tokens — **short**, but far above 88).

**execution_path:** Hawking `moe.metal` already has `moe_grouped_gemm_q4` + `moe_gather_combine` + Phase-2 `moe_block_fused`. **Missing:** pack/stage that does not keep 128 experts in the weight tree. mlx `mx.gather_qmm` indexes, does not drop RAM. Prefetch: issue next-token expert tiles overlapping current attn (UMA; `weight prefetcher` Type-1 dead as WILLNEED −29% — **do not** madvise; overlap GPU copy with compute instead).

**applicability:** all sparse MoE. Required for any of M2–M5 to be a **physical** active win rather than an accounting story. Dense: N/A.

**confidence:** **HIGH that residency is the mlx gap (MEASURED)**; **MEDIUM that prefetch hits ~0.4**; **LOW that TPS moves**. Why: NX receipt is explicit; overlap is MEASURED; kernel-bound caveat is MEASURED on the same family.

**transfer:** O006 same. O003: also stage **2 shared always** + 6 routed; shared cannot be dropped. NS-large-expert-cache does **not** kill this (different premise: sequential-token same-layer vs lockstep cross-layer).

---

## M7 — Activation-aware 1-bit / ternary experts + popcount gather

**one-line:** BitNet/1.58-style **lookup-free** experts, scales from **real x**, gathered; not PTQ-on-W, not LUT.

**complete_byte_accounting**

| item | bits | trap |
|---|---|---|
| signs {−1,0,+1} or {−1,+1} | 1.58 or 1 per weight | pack into uint32; count **padding** |
| per-row or per-group scale | f16: +16/g | g=64 → +0.25 BPW; g=32 → +0.50 |
| optional per-expert absmean | f16 × 3 organs × 128 × 48 | ~37 KB — noise |
| T1 residual | M4 | if 1-bit gate fails |
| codebook | **none** (lookup-free — required on Metal) | |
| container | always | |

Complete T0 ≈ **1.25–1.75** typical (1.58+scales), **not** 1.00. Sub-1 complete needs g→∞ (one scale / row): row-scale binary = 1 + 16/K. O005 K=2048 or 768: **1.008–1.021** payload+row-scale, plus container.

NS-ternary-factorization: ternary **lost to VQ at matched rate** on gpt-oss. Reopen only if this parent’s W is ternary-friendly **on a real forward**. NS-raw-weight-PQ is a different family; 1-bit **PTQ on raw W** is still likely dead. Path: **activation-aware scales** (AWQ/GPTQ-class) or QAT/distill (source-changing, not blocked).

**stored_bpw / active_bpw**

- stored expert complete **~1.0–1.6** (honest).
- expert-path same.
- Hawking-active with B_ao=4.5: **2.6–3.1**. With M1 B_ao=1.5 and B_e=1.1: **~1.3**. Sub-1 Hawking-active: **only if** B_ao≲1.0 and B_e≲1.0 (row-scale binary + attn also ~1).

**expected_reachable_bpw:** expert-path **1.0–1.6** Doctor-UNKNOWN; **<1 complete** only with row-scale binary and almost no residual. Sub-0.2: **no** (would require generating bits, M8).

**quality_risk / limiter:** gate (1-bit PTQ). BitNet literature is **trained** 1.58, not PTQ. O005 q2 already failed — 1-bit is strictly harder unless source changes. Embed/lm_head must stay ≥q3–q4 (not this mechanism).

**cheapest_falsifier:** one expert-layer, real x, absmean-row binary vs affine-q2 at **matched complete_bpw**. Kill if binary cos < q2 cos (likely) **and** Doctor on that layer replacement fails. If binary **beats** q2 at matched complete on real x, reopen NS-ternary on this parent.

**execution_path:** **popcount-AND / SIMD sign-FMA**, not dequant-to-f16 (that expands to a dense body = fake). Hawking has **no** BitNet kernel. Metal: viable (lookup-free, unlike PQ). Decode FLOPs similar; ALU cheaper than q3 dequant **if** popcount throughput ≥ FMA (UNKNOWN on M3 Ultra). Gather: same packed working set (M6).

**applicability:** MoE experts; also dense FFN. Worst on gate. Do not put on router.

**confidence:** **LOW** for PTQ-1-bit Doctor; **MEDIUM** as a **native-kernel** direction (lookup-free). Why: BitNet is real prior art; Hawking negatives are hostile to sub-bit PTQ; Metal LUT is dead so this is the remaining 1-bit **execution** story.

**transfer:** architecture-agnostic codec; quality will not transfer without per-parent real-x scales.

---

## M8 — Per-selected-expert generated weights (procedural / tiny hypernet)

**one-line:** Deepest stored sub-0.2 lever: **do not store W**; store seed + generator; **fused** `y = G(seed_e, x)` for top-k only. Account generator bytes **and** reconstruction FLOPs.

**complete_byte_accounting**

| item | stored | touched/token | fake-win mode |
|---|---|---|---|
| per-expert seed | 128 × 48 × d_seed × 2 | k × 48 × … | — |
| shared generator θ | `|\theta|` once | **fully resident** | a 3-BPW generator is a 3-BPW model |
| optional token-cond embedding | small | yes | — |
| reconstructed W scratch | 0 if fused | **N_sel × 2 bytes if materialized** — **counts as active** and kills the win |
| always-on | M1 | M1 | — |

Example scale (HYPOTHESIS, not a prediction): d_seed=256 f16, 128×48 seeds = 3.1 MB; generator 50M f16 = 100 MB. Stored expert-equivalent BPW = `8·(3.1e6+1e8)/28.99e9 ≈ 0.028` **plus always-on**. That is the sub-0.2 **stored-expert** mirage. Active: if you **write W then GEMM**, touched includes 226 MB × 16/B_e_materialized — worse than q4. Only fused implicit matvec counts.

**stored_bpw / active_bpw**

- stored can be **≪ 0.2** on the expert mass (honest only if `|\theta|+seeds+ao` counted).
- Hawking-active = always-on + generator + **any materialized W**. Floor ≈ `0.46 B_ao` even at zero expert storage.
- With B_ao=4.5, Hawking-active floor **2.07**. Sub-0.2 Hawking-active still **requires M1 at 0.2**.
- Decode cost: generator FLOPs **per selected expert per layer**. If G is a 2-layer MLP on each row: can exceed original GEMM by 10–100×. **Must quote FLOPs/tok.**

**expected_reachable_bpw:** stored-expert **0.02–0.2** possible as arithmetic; **Doctor-valid: UNKNOWN, likely fail** (no Hawking success; hypernetworks as full-W replacements are research, not a measured Gravity win). Hawking-active <1 only with M1. This is the **only** credible path to **stored** expert sub-0.2 without a 0.167 function-codec actually packing.

**quality_risk / limiter:** the generator (single organ: all experts share θ → correlated failure). Layer 0. Gate. Doctor will fail first as **language collapse**, not MSE. Localize: which layer’s G.

**cheapest_falsifier:** **one** expert, **one** layer, fit G to predict **rows of W** or **y=xW** on real x. Kill if (a) `|\theta|` for one expert already ≥ storing that expert at q3, (b) fused-FLOPs > 10× GEMM, (c) cos(y,ŷ) < 0.86. Do this **CPU/numpy**, no kernel. If one-expert fails, the family is dead on this parent.

**execution_path:** native only if fused. Materialize-W-then-Q4 = **fake**. No Hawking kernel. Not mlx. Prior art: Ha et al. hypernetworks; LoRA-as-generator; “structured pruning + reconstruct” — NS-expert-merging **kills reconstructing omitted experts from survivors** (rel err ~1). This M8 is **per-expert seed**, not merge.

**applicability:** MoE (k seeds per token, not 128). Worst-case FLOPs scale with k, not N — that’s the MoE gift.

**confidence:** **LOW** (speculative). Why: accounting allows stored sub-0.2; every quality and FLOPs check is unchecked; cheapest falsifier is cheap so **run it**, don’t believe it.

**transfer:** if dead on O005 L10 gate, dead on O006. O003 not worth it until O005 dies or lives.

---

## Scoreboard (what is actually reachable **now**)

| win | now? | mechanism | blocker |
|---|---|---|---|
| Hawking-active < 1 on O005 | **not with current always-on** | M1 **and** (M2 or M3/M7) | attn/embed @ 4.5 complete = 2.07 floor |
| Hawking-active < 0.2 | **no** | whole active set @ ~0.2 | not an MoE-gather win |
| expert-path complete < 1 | **unknown Doctor**; reconstruction precedent 0.167 | M2 | Metal LUT/trellis dead; GLM generation collapsed |
| expert-path ~1.2 complete screen | **nearest recipe** | M3 | not packed; not Doctor; not kernel |
| stored body < 1 | only if experts <1 (95% of mass) | M2/M7/M8 | same |
| physical moved bytes ≪ stored | **yes, unimplemented** | M6 | mlx full-resident; may not move TPS |
| popularity / cold-expert sub-1 | **dead** | — | 0 cold, entropy 6.16/7 |
| shared-template / expert-delta | **dead** unless cos≥0.10 | — | Q80 cos~0; NS-cross-expert |
| proxy-activation sub-bit | **dead** | — | Qwen3.8 collapse |

O003 is **harder** for Hawking-active (shared+vision in B_ao, 0.60 weight). O006 text-only ≈ O005; multimodal path worse.

---

## Ranked cheapest experiments (Doctor is authority)

1. **M1 falsifier** — attn q2 vs q4, one organ, fast-Doctor. Kills Hawking-active<1 without a new attn codec.
2. **M6 measurement** — packed one-layer gather vs mlx resident set. Kills “gather is already done”.
3. **M3 packing on O005 L10 one expert, real x** — three Q80 codecs, complete bytes counted, then Doctor. Kills transfer of the only screen-pass mix.
4. **M2 one-expert activation codec @ 0.17–0.5, real x, codebook residency in complete_bpw**. Kills GLM-0.167 transfer. **No Gaussian x.**
5. **M4 T0 vs T0+T1 matched complete_bpw on up_proj**. Kills residual topology.
6. **M5 margin vs T0-error correlation** on 1k disjoint tokens. Kills confidence gating.
7. **M7 row-scale binary vs q2 matched complete, real x**. Reopen/kill ternary/1-bit PTQ on this parent.
8. **M8 one-expert hypernet size vs q3** (numpy). Kills procedural sub-0.2.

Do not run a 48-layer pack until (3) or (4) survives. Do not quote TPS except in a clean box (physics programmatic). A gibberish artifact that **localizes gate vs down vs router vs layer-0** is a win for the compiler.

---

## Guardrail checklist

| guardrail | how honored |
|---|---|
| complete accounting | every row names payload+scales+codebook+residual+meta+align+container; codebook **residency** called out |
| no fake density | sel/full never used as stored_bpw; Hawking-active identity written |
| activation-aware only | all quality paths require **real routed x**; proxy trap cited |
| Doctor authority | cosine = screen; sealed GLM sub-1 collapsed; localize organ |
| native execution | LUT/trellis Type-1 dead; fused dequant / popcount / (xU)V / packed gather only; materialize-W = fake |
| stored vs active | split in every M; M6 is active-movement without stored change |
| UNKNOWN | codebook geometry of GLM 0.167; O003/O006 q2; T1 hit-rate; attn BW-vs-kernel; BitNet popcount throughput; M8 FLOPs |

**Evidence class of this packet:** HYPOTHESIS on mechanisms; MEASURED on ledgers/routes/q3 anchor/NX residency; DERIVED on identities. Not a Hawking NX win.
