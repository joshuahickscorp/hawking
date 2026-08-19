I'll read the Sub-1 frontier packet first, then produce several shared-basis proposals in its required schema.The packet is loaded. Next I’ll pull Hawking’s existing patients, expert-structure findings, and prior shared-basis work so the proposals stay grounded.There’s already a Qwen80 cross-expert negative receipt. I’ll pull that plus Odyssey prior science and the campaign packet so the proposals don’t rediscover a dead lever.I’ll read the Odyssey patient packets, novelty contracts, and negative-science files next so the proposals cite measured structure rather than architecture folklore.The negatives kill weight-space tying. I’ll check activation-space structure, GLM’s 0.167 BPW result, and whether any shared-codebook path is still live.I have the mass fractions, negatives, and live codecs. Next I’ll write the shared-basis packet with honest accounting and several mechanisms that don’t assume experts are redundant.Writing the harvestable packet with derived bounds, killed-lever exclusions, and eight mechanisms that don’t assume expert redundancy.Lane premise that survives the kills: experts are **not** redundant in weight space. Shared structure, if it exists, is in **(1) the shared activation they all see**, **(2) table/metadata amortization**, **(3) architecture-native shared experts**, not in “experts look alike.”

---

## Accounting law (DERIVED — do not violate)

**O005** (MEASURED census): `f_expert=0.9495`, `f_non=0.0505`, `E=128`, `top-8`, `0` shared, `0` cold, route entropy `6.16/7.00`.  
`complete_stored = 0.9495·e + 0.0505·n`.  
Active receipt convention: `active_bpw = 0.540·e + 0.460·n` (expert is only 54% of **touched** weights).  
`8/128=0.0625` is a **selection** lever, not a compression stat (policy `selected_full_is_selection_not_cost`).

| Target | Expert-only (`n=8`) | `n=4` | `n=2` |
|---|---|---|---|
| stored complete **&lt;1** | `e≤0.628` | `e≤0.841` | `e≤0.947` |
| stored complete **&lt;0.2** | **IMPOSSIBLE** (nonexpert alone `0.404`) | **IMPOSSIBLE** (`0.202`) | `e≤0.104` |
| `active_bpw&lt;1` (vs active params) | IMPOSSIBLE (`0.46·8=3.68`) | IMPOSSIBLE (`1.84`) | `e≤0.148` |

**Q80** (`f_expert=0.9703`): same wall — nonexpert at 8-bit is already `0.238` complete. **Sub-0.2 stored is not an expert-structure problem.**  
Current anchors: O005 `q3-g32-experts` **4.025 / 4.231** stored/active; Q80 mixed screen **1.23 expert / 1.43 complete** @ 8-bit nonexpert (not packed, not Doctor).

**g32 scale+bias identity (O005):** independent f16 scale+bias on experts = **`1.0·f_expert = 0.9495` complete BPW of metadata alone.** Sub-1 stored is impossible while that bin stays per-expert. Sharing/crushing scales is mandatory, not optional.

Geometry-optimal shared factors (O005, all three SwiGLU organs): share the **wide** side (gate/up input `V[r,2048]`; down output `U[2048,r]`). Skinny maps are `[768,r]` × `128` experts.  
`nominal_bits = b` on skinny.  
`expert_payload_bpw ≈ b·r/2048` + shared amortized (`~0.014` at `r=256`, 8-bit shared).  
Per-row f16 scale on skinny ≈ **0.012** expert BPW. Alignment ≈ `18k` tensors × 256 B ≈ **4.7 MB**.

| `r` | `b` skinny | expert payload (DERIVED) | +scales/shared/slop (INFERRED) | whole-model stored `n=8` / `n=4` / `n=2` |
|---|---|---|---|---|
| 256 | 2 | 0.250 | **~0.29** | 0.68 / 0.48 / 0.38 |
| 128 | 2 | 0.125 | **~0.16** | 0.55 / 0.35 / 0.25 |
| 64 | 2 | 0.0625 | **~0.09** | 0.49 / 0.29 / **0.19** |
| 64 | 1 | 0.031 | **~0.06** | 0.46 / 0.26 / **0.16** |

Those last two rows are the only **honest sub-0.2 stored** path in this lane, and only if nonexpert ≤2-bit. Organ quality at those `r` is **UNKNOWN**.

---

## Dead on this premise — do not re-propose

| Kill | Premise | Still live |
|---|---|---|
| **NS-inter-expert-redundancy** | mean pairwise `W` cosine `~1e-4` | reopen iff cosine `≥0.10` |
| **QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE** L10 `n=96` | pairwise cosine `0.004` / `-6e-5`; **subspace_overlap_top32 `0.025/0.020`**; `k90=84/96` | weight-space shared rank-32 **dead on Q80**; clustering needs `K≈E` |
| **NS-cross-expert-and-cross-layer-tying** | template energy `0.2513` vs orthogonal null `0.2500`; same-index layer tying = control @ `1e-7` | reopen iff row-norm off-diag cosine `≥0.10` |
| **NS-expert-merging** | best survivor rel-err `~0.89–1.0` | — |
| **dead_levers: cross-layer weight delta** | cosine `≈0` | trained-for only |
| **NS-raw-weight-pq-vq-at-one-bit** | `1.0075` and `0.493` BPW collapsed 6/6 real forward | only if the **source changes** (activation-projected / Hessian / QAT) |
| **NS-kronecker** | Van Loan flat at depth | **layer-0 named exception** |
| **NS-entropy-coded-pq-indices** | Lloyd indices `0–0.7%` gain | non-Lloyd / biased indices |
| **dead_levers: learned codebook LUT** | Apple GPU gather punished | lookup-free / tiny register CB / GEMM factors |
| **dead_levers: low-rank residual / ASVD** | SVD energy low | **hgravs01 activation-weighted is LIVE** on Q80 `down_proj` (`0.886–0.898` cos @ **1.27 BPW**, `r160 b3`) |
| **GLM-5.2 sub-bit expert path** | `0.116–0.157` cos @ `0.75` BPW vs null `0.898` | student dense map `0.0104` BPW (**not weight-space**) |
| **GLM-5.2 H0.98 / R0** | `~0.88–0.98` complete **SEALED + SEMANTIC_COLLAPSE** | cosine ≠ Doctor |
| **NS-uniform-subbit** | organs do not fail together | organ-conditioned (gate first) |
| **0 cold** O005/O003/O006 | no popularity skew | cannot drop experts |

**Qwen3.8 gaussian-proxy sub-bit is an artifact.** GLM **`0.755 cos @ 0.167 BPW` on REAL activations** is the only sub-1/5 **screen** precedent — not a Doctor pass.

---

## SB1 — `ACT-SHARED-V`  
**one-line:** Per layer, one activation-PCA basis on real `X`; per-expert skinny maps on that basis. Experts may be orthogonal as `W` and still share **input covariance**.

| field | content |
|---|---|
| **complete_byte_accounting** | **payload** skinny `U_e`/`V_e`: `b·r/2048` BPW of expert mass. **tables** shared wide factors `V_gate/up[r,2048]`, `U_down[2048,r]` @ `b_v` (8-bit default): `~0.014·(r/256)·(b_v/8)`. **scales** per-row f16 on skinny `~0.012`. **biases** `0` if scale-only. **offsets/correction** `0` in the base variant. **tier/router-meta** `r`, `b`, organ tags, per-expert nothing (no family-id). **alignment** `~5 MB`. **container** Gravity header/index **UNKNOWN** (budget `≤0.01`). **recon state** = packed skinny + shared factors; **not** densified `W`. **nominal_bits=`b`**; **complete = payload+tables+scales+align+container**. |
| **stored_bpw** | expert complete **~0.09–0.29** at `r∈{64,128,256}`, `b=2` (table above). Whole-model **~0.49–0.68** if `n=8`; **~0.29–0.48** if `n=4`. |
| **active_bpw** | vs active params: **`0.54·e + 0.46·n`** → **~1.9 at `e=0.16,n=4`** (NOT sub-1). vs body: touched skinny `8/128` + one `Vᵀx`/layer. O005 bytes/token INFERRED **~0.8 GB** at `r=128,b=2,n=4` vs q3 **1.77 GB**. |
| **expected_reachable_bpw** | **stored sub-1: YES (DERIVED)** if skinny holds. **organ sub-0.2: MAYBE** at `r≤128,b≤2` (quality UNKNOWN). **stored sub-0.2: NO** unless nonexpert ≤2-bit. **active_bpw&lt;1: NO** unless attn also ~2-bit. |
| **quality_risk** | **gate/up** break first (NS-uniform-subbit; O005 q2 localization; F1 `dominant_failure_organ=gate`). Down is the **tolerant** organ (Q80 hgravs01 already `r160`). If `XXᵀ` energy at `r` is low, all experts lose the same complement — systematic, not per-expert. Embed/lm_head/router **out of scope** (keep ≥4/8-bit). |
| **cheapest_falsifier** | One layer, one organ, **real** `X` (≥256 tokens, existing Q80 xcache or O005 hook). SVD of `X`, energy `r/2048`. Reconstruct `WX ≈ U(VX)` **without** coding `U`. **Kill if** output cosine `&lt;0.86` (Q80 bar) at `r=256`, or `&lt;0.75` (GLM 0.167 analog) at `r=64`. If uncoded PCA already dies, quantizing `U` cannot save it. **Do not use Gaussian `X`.** |
| **execution_path** | Native GEMM, **lookup-free**. Per layer: `z=Vx` once (serves gate+up), then gather **8** skinny maps. Down: 8 skinny then shared `U`. FLOPs gate: `r·2048 + 8·768·r` vs `8·768·2048` → **`r/2048` of the second GEMM + one shared first GEMM** (`r=128` ≈ **8× fewer MAC** on gate). **Forbidden:** materialize dense `W_e=U_eV` then `gather_qmm`. Kernel: fused `pca_gather_qmm` (new); mlx `gather_qmm` is the wrong object. Decode cost/token = <REDACTED> skinny GEMV per selected expert per organ + 1 shared GEMV/layer. |
| **applicability** | **MoE expert organs** (O005/O006 8/128; Q80 10/512 even more amortization; O003 6/64). Dense: degenerates to per-layer ASVD — **Type-1 dead** as residual codec; do not run as “shared-basis.” Hybrid: attn/SSM not this object. |
| **confidence** | **HYPOTHESIS**, highest prior in-lane: same `X` is a fact of the residual stream; does **not** need `W` cosine (so NS-inter-expert does not apply). Quality at sub-0.2 `r` **UNKNOWN**. |
| **transfer** | O005→O006 **likely** (same topology, uniform, 0 shared). Q80: 512-way amortization **better**, but L10 weight subspace overlap says **do not** share a rank-32 **weight** basis. GLM/O003: still valid, stack with SB4. Not two-Qwen universal until O003. |

**Prior art:** GPTQ/AWQ (activation-aware PTQ); ASVD / SVD-LLM / SliceGPT (activation PCA — residual form killed here, **stack** form untested); VeRA / LoRA-FA (shared projection, per-adapter skinny); hgravs01 (LIVE, per-expert).

---

## SB2 — `META-SHARED-SCALES`  
**one-line:** Share group/row scales across experts (and optionally layers) so metadata stops eating `~1` complete BPW. Directions stay independent.

| field | content |
|---|---|
| **complete_byte_accounting** | Attacks **scales/biases**, not payload. Today g32 f16 scale+bias = **0.95 complete BPW** on O005. **tables** = 1 scale template / `(layer, organ)` or a K-family scale codebook. Template bytes: `48×3×(768·2048/32)×4 ≈ 28 MB` → **0.007 BPW**. **payload** unchanged (q2/q3/binary). **offsets** none. **meta** = `family_id` 4–7 bits/expert if K-family (`48×128×1 B ≈ 6 KB`). **correction** optional per-expert scale residual (count it). **alignment/container** as today. |
| **stored_bpw** | If payload stays q3-g32 (`~3`) + shared scales (`0.007`) vs current q3-g32 complete **4.025**: **~3.1** if the extra `~0.95` was scale+bias (INFERRED from group geometry; MEASURE the live mlx scale bin before claiming the delta). Combined with 2-bit payload: complete expert **~2.01** (payload) **+0.02** meta → whole-model **`0.95·2.03+0.05·8≈2.33`** still **&gt;1**. **This mechanism does not reach sub-1 alone.** It is the **gate** that lets SB1/SB6/SB8 reach sub-1 honestly. |
| **active_bpw** | Same payload density; slightly less scale traffic. Not an active-sub-1 lever. |
| **expected_reachable_bpw** | **−0.2 to −0.95 complete** vs independent g32 scales (range = how much of the 0.95 bin is really live in the q3 artifact — **UNKNOWN until bin-split**). Floor: shared template + per-expert 8-bit scale residual. |
| **quality_risk** | Scale mismatch on **gate** rows with huge dynamic range. NS-row-norm-stratification: the “94% one codeword” premise **failed** (single-codeword share `0.027`); row norms still span widely on F1. Sharing **across experts** of the **same row index** is a different question. Down likely fine. |
| **cheapest_falsifier** | Dump per-expert group-absmax on **1 layer, gate**. Cosine of scale-vectors across 128 experts; rel-err of layer-mean template vs per-expert. **Kill if** median rel-err of `W / scale_template` vs `W / scale_e` on real `WX` cosine drop `≥0.02`. No training. |
| **execution_path** | Native: current affine grouped kernel with a **broadcast scale** (one load / organ / layer). Zero extra FLOPs. Better than LUT. mlx `quant_predicate` already special-cases router; extend to shared scale buffer. |
| **applicability** | All grouped codecs, MoE **and** dense. Biggest on MoE (`E` multiplier). |
| **confidence** | **DERIVED** that the bin is huge; **HYPOTHESIS** that templates transfer across orthogonal experts (magnitudes can agree when directions do not). |
| **transfer** | Universal metadata lever. Measure per parent (F1 row-norm span may not match O005). |

**Prior art:** AWQ per-channel scales; SqueezeLLM sensitive-value; GGUF qK scale layouts; candidate_families `metadata_codec=shared` (already in grammar, **not native**).

---

## SB3 — `HGRAVS-SHARED-TALL`  
**one-line:** Take LIVE `hgravs01` (activation-weighted low-rank) and share the **expensive** factor across experts.

| field | content |
|---|---|
| **complete_byte_accounting** | Q80 `down` unshared `r160 b3`: `3·160·(2048+512)/(2048·512)=1.172` + scales → measured **1.27**. Sharing skinny `V` on down: **~0.94** (weak). Sharing **tall** `U[2048,r]` on down: **`3·160/2048=0.234`** + amortized U `~0.001`. Gate/up: share **wide input** (SB1). **payload** 3-bit factors; **scales** per-factor; **tables** shared factor; **correction** none or 0.25–1% rice (Q80 residual ledger: rice+q1 outliers **counted**; 2% already over-budget vs 1.30 allowance). |
| **stored_bpw** | Down organ **~0.25–0.40** complete if tall-share holds (DERIVED geometry + LIVE unshared 1.27). Mixed with binary_g gate (Q80 **1.13** @ `0.86–0.89` cos) is the current 1.23 mix — this aims to **replace** the 1.27 down and **cut** gate below 1.13 via SB1. |
| **active_bpw** | Same as SB1 with `b=3,r=160`: not active-sub-1 at 4-bit attn. |
| **expected_reachable_bpw** | Beat Q80 expert **1.23** toward **0.4–0.8** mixed complete-expert if tall-share cosine holds. Sub-0.2: only with `b≤2,r≤128` (leaves hgravs’ 3-bit comfort zone — **UNKNOWN**). |
| **quality_risk** | Sharing down’s **output** basis: experts may write **orthogonal residual directions** (weight overlap 0.025 analog). If so, tall-share dies and you only get skinny-share (~1.0, not a win). Gate at `r160` unshared was **not** the Q80 winner (binary_g was); forcing low-rank on gate may lose to binary+rice. |
| **cheapest_falsifier** | Q80 xcache, 4 `(layer,expert)` pairs already in `QWEN80_DOWN_PROJ_FRONTIER_SWEEP`. Fit one `U` on concatenated activation-weighted downs of 8 random experts; score held-out experts’ `WX` cosine vs per-expert `hgravs01_r160_b3`. **Kill if** shared-`U` cosine `&lt;0.86` or does not beat unshared at **matched complete_bpw**. |
| **execution_path** | Two GEMMs, LUT-free, **already the hgravs01 story**. Sharing `U` = one `U` buffer / layer / organ, gather skinny. Do not unpack to dense. Kernel-bound: `r=160` vs `in=512` is already a compute cut on down. |
| **applicability** | First **Q80 down_proj** (only organ with a LIVE low-rank screen). Then O005/O006 down. Gate/up only after SB1 uncoded PCA passes. |
| **confidence** | **INFERRED** from LIVE unshared hgravs01 + geometry. Tall-share quality **HYPOTHESIS**. |
| **transfer** | Down-tolerant (organ inversion) may transfer; tall-share of output basis may **not** (Q80 overlap negative is on gate/up, down overlap **UNKNOWN**). |

**Prior art:** hgravs01; LoRA/LoftQ/LQ-LoRA; QuIP# LDLQ low-rank; ASVD (killed as residual, not as stacked-expert factor share).

---

## SB4 — `SHARED-EXPERT-DELTA`  
**one-line:** On archs that **already** have a shared expert, code `W_e = a_e W_shared + Δ_e` with `Δ` low-bit / low-rank / sparse. Not “experts copy each other.”

| field | content |
|---|---|
| **complete_byte_accounting** | **tables** = shared expert body (already stored; **do not double-count**). **payload** = `{a_e}` (f16 scalar or 768-d row scale) + `Δ_e` codec. If `Δ` is 1-bit+per-row scale: `1.0+0.012`. If `Δ` is SB1 skinny on residual: `b·r/2048`. **offsets** if sparse `Δ` (rice/CSR; Q80 rice residual **8.24 bits/outlier** — 2% outliers already **+~0.96 BPW**, easy to blow the budget). **correction** = `Δ`. **meta** `a_e`. |
| **stored_bpw** | **UNKNOWN** until `‖Δ‖/‖W_e‖` is measured. If relative residual `≥0.9`, this is just 1-bit `W_e` + wasted shared ref → **no win**. If residual `0.3` and 2-bit `Δ`, expert complete **~0.6** plus shared (already in `n` or expert mass). O003: shared is **2.7%** of params (MEASURED); GLM-4.5-Air shared ≈ one extra expert (`1/129` of routed+shared). |
| **active_bpw** | Shared expert is **always-on** (GLM `y += shared(x)`; O003 2 shared). Active includes `100%` of shared + `top-k` of `Δ`. Honest active = shared bytes + selected `Δ` bytes. |
| **expected_reachable_bpw** | **Wide.** Win iff residual energy is small. Speculative organ **0.3–1.0**. Sub-0.2 only if `Δ` itself takes SB1/SB8. |
| **quality_risk** | Routed experts may be **complements** of the shared expert, not copies — residual **larger** than `W_e`. Gate `Δ` most fragile. Layer-0 dense on GLM is a **different source** (NS-layer-zero / NS-kronecker exception) — do not fold into this codec. |
| **cheapest_falsifier** | One MoE layer. For each organ, `a_e=⟨W_e,W_sh⟩/‖W_sh‖²`, residual energy. **Kill if** median `‖Δ‖/‖W_e‖ ≥ 0.8` on real-`X` output cosine (proj onto shared fails `&lt;0.70` cos). Activation-aware: score `ΔX` not `Δ`. |
| **execution_path** | Native: existing shared-expert GEMM **plus** gathered `Δ` GEMM. `y = a_e·y_shared + Δ_e x` with `y_shared` already computed. Extra cost = selected-`Δ` only. No densify. O005/O006: **N/A** (`shared_experts=0`). |
| **applicability** | **O003** (2 shared, 64 routed, top-6), **GLM-4.5-Air / glm4_moe** (1 shared, 128 routed, top-8, layer0 dense). Not O005/O006. |
| **confidence** | **HYPOTHESIS**. Architecturally motivated; weight-space negatives do **not** speak to `W_e` vs `W_shared` (never measured on Odyssey). |
| **transfer** | Only shared-expert MoEs. Do not transfer a Qwen-no-shared fail/pass. |

**Prior art:** DeepSeek-V2/V3 shared-expert design; GLM4-MoE; residual adapters; LoRA around a frozen base.

---

## SB5 — `FUNC-FAMILY` (router-row + `WX` clustering)  
**one-line:** Cluster experts by **function on real `X`** (and by router-row cosine), not by `vec(W)`. Families share a prototype; members store a delta. **Do not assume** a family exists.

| field | content |
|---|---|
| **complete_byte_accounting** | **meta** `family_id` `⌈log2 K⌉` bits/expert/layer (`K=8 → 3 bits`; `48·128·3/8 = 2.3 KB`). **tables** `K` prototypes. If prototypes are fp16 full `W`, `K/E` of bf16 body: `K=8` → **0.95 complete BPW already** — **fake win**. Prototypes **must** be SB1 bases or 4-bit. **payload** per-member `Δ` at `b_Δ`. **correction** = `Δ`. Count prototype+`Δ`+id+scales. |
| **stored_bpw** | Only a win vs SB1 if `K` bases beat 1 basis enough to **drop `r` or `b_Δ`**. Otherwise strictly worse (K copies of `V`). |
| **active_bpw** | Gather 8 members; at most 8 family bases (usually fewer). |
| **expected_reachable_bpw** | **Likely zero.** Q80 `k90=84/96` in **weight** space. This proposal exists to **measure the functional analog**, not because families are expected. |
| **quality_risk** | False families → merged experts (NS-expert-merging died at rel-err `~1`). Prototype collapse = capability collapse localized to that family. |
| **cheapest_falsifier** | **(a)** `128×128` cosine of **router rows** per layer (O005 `gate.weight[128,2048]`, seconds). **(b)** pairwise cosine of `W_e X` on captured `X` (counterfactual apply-all, not only routed). **Kill if** both mean off-diag `&lt;0.10` (same threshold as NS-inter-expert `reopen_if`). If (a) clusters and (b) does not, families are a **routing** story, not a weight story — stop. |
| **execution_path** | Same as SB1 with `K` bases. Native iff prototypes are GEMM factors, not LUT. |
| **applicability** | MoE with `E≥32`. First O005 L0 + L24 (layer-0 may differ). |
| **confidence** | **HYPOTHESIS, prior to kill.** Run falsifier before any pack. |
| **transfer** | If O005 dies, try O003 (shared experts may induce true families). Do not retry on O006 after O005 kill (same family). |

**Prior art:** MC-MoE / HC-SMoE / MergeMoE / D2-MoE / NAEE (merge/cluster) — **conflict with NS-expert-merging**; only reopen with the functional cosine evidence. Product-quantization **codebooks** are SB6, not families.

---

## SB6 — `AQLM-SHARED-CB` (Hessian, **tiny**, LUT-free)  
**one-line:** One additive codebook per `(organ, layer)` — or one global — fit on **activation-projected** subvectors. Sharing is a **byte identity**, not a similarity claim.

| field | content |
|---|---|
| **complete_byte_accounting** | **The packet trap:** `0.2-bit` indices + fat tables = table BPW. Per-expert AQLM `M=4,K=4096,d=16` fp16: `4·4096·16·2=512 KB` × `128×3×48 ≈ 9.4 GB` = **2.46 BPW tables alone** → **worthless**. Shared 1 CB/organ/layer: `48×3×512 KB ≈ 74 MB` = **0.019 BPW**. Shared **global** 1 CB: `0.5 MB`. **payload** indices `(n/d)·log2(K)` bits. `d=8,K=16` → `0.5` BPW indices; `d=16,K=16` → `0.25`; `d=32,K=16` → `0.125`. **scales** per-block. **NS-entropy-coded-pq-indices** forbids Lloyd-index entropy as a win. **Must change source:** code `U_e` from SB1 or Hessian-weighted `W`, not raw `W` (NS-raw-PQ **dead** at 1.0 and 0.49). |
| **stored_bpw** | Honest organ **0.15–0.40** if CB shared + `d≥16,K≤16` + SB1 projection. Raw-weight 1-bit PQ: **dead**. |
| **active_bpw** | Index gather of 8 experts. If `K≤16,d≤8`, CB is **register-resident FMA** (not the Apple LUT death). Larger K re-opens **dead_levers: learned codebook**. |
| **expected_reachable_bpw** | Sub-1 organ **yes** as table-amortization. Sub-0.2 organ **only** on projected `U_e` with `log2(K)/d ≤ 0.2` (`K=16,d=80` or `K=4,d=20`) — quality **UNKNOWN**, likely gate-fail. |
| **quality_risk** | Raw PQ/VQ @ ~1 bit **collapsed** (NS-raw-weight-pq-vq). Qwen3.8 **proxy** PQ “wins” collapsed (output-div `~0.69`). Gate first. GLM-5.2 expert PQ `0.12` cos. |
| **cheapest_falsifier** | Fit **one shared** `d=8,K=16` CB on SB1 `U` of 16 experts, **real `X` Hessian diag** (GPTQ-style, 256 tokens). Encode held-out 16 experts. **Kill if** `WX` cosine `&lt;0.75` at complete (indices+CB+scales) `≥0.5` BPW, or if a **per-expert** CB of same complete_bpw wins by `≥0.03` cos (then sharing hurts). |
| **execution_path** | **LUT-free only:** additive binary/ternary codebook (`C ∈ {−1,0,+1}^{d×K}`) + scales → FMA. Or `K≤16` broadcast. **Not** QTIP Metal trellis (Type-1 proxy dead). **Not** large fp16 LUT gather. If the kernel unpacks to dense `W` before GEMM, **fake**. |
| **applicability** | MoE expert stack (amortization ∝ `E`). Q80 `E=512` is the table-win jackpot. Dense: sharing across **layers** only (NS-cross-layer-tying killed **weight** tying, not CB tying). |
| **confidence** | **HYPOTHESIS.** Table math is **DERIVED**; quality blocked unless source is projected/Hessian. |
| **transfer** | Table identity transfers. Fit quality does **not** (need per-parent real `X`). |

**Prior art:** AQLM (Egiazarian 2024); PQ (Jégou 2011); Residual PQ; QuIP# / QTIP (lookup-free target); GPTQ Hessian; **do not cite** GPTQ-on-raw-1bit as live here.

---

## SB7 — `QUIP-HADAMARD + 1-bit skinny`  
**one-line:** Implicit Hadamard (0 bytes) on the shared activation basis, then lookup-free 1-bit/ternary on skinny maps. Rotation is the shared object.

| field | content |
|---|---|
| **complete_byte_accounting** | **tables** Hadamard **0 bytes** (implicit). Optional learned rotation `R[2048,2048]` is **16 MB/layer fp16** = `48×16 MB=768 MB` = **0.20 complete BPW** — **do not store a dense `R`**. Use Walsh–Hadamard / QuaRot / SpinQuant **implicit** or a **low-rank** `R` counted as tables. **payload** 1-bit skinny `r/2048`. **scales** per-row `~0.012`. **NS-ternary-factorization** killed ternary **factors vs VQ**, not BitNet-style ternary **weights**. |
| **stored_bpw** | With SB1 `r=128`, 1-bit: expert **~0.09**. Whole-model `n=4` **~0.29**; `n=2` **~0.19**. |
| **active_bpw** | 1-bit gather of 8 skinny + Hadamard on `x` (`O(d log d)`). vs active params still attn-dominated. |
| **expected_reachable_bpw** | Organ **0.06–0.15**. Sub-0.2 stored whole-model only with 2-bit nonexpert. Quality may match 2-bit unrotated (QuIP# claim) — **UNKNOWN on these patients**. |
| **quality_risk** | 1-bit PTQ on **raw** `W` is in the NS-raw-PQ blast radius. Here bits sit on **already-projected** skinny maps. Gate. BitNet is **trained** ternary; we are PTQ. |
| **cheapest_falsifier** | After SB1 uncoded PCA passes: 1-bit RTN vs 1-bit after Hadamard on `U`, same complete_bpw, real `WX`. **Kill if** incoherence gain `&lt;0.01` cos (rotation is then a no-op). |
| **execution_path** | Native: `H x` fused (FWHT), then 1-bit GEMV (BitNet-style, **scale + popcount/xor**, no LUT). Apple GPU **likes** this more than codebook gather. Do not run QTIP trellis. |
| **applicability** | Any linear; highest leverage on SB1 skinny (smaller 1-bit object). |
| **confidence** | **HYPOTHESIS** (QuIP#/QuaRot external; not measured on Odyssey MoE). Cheap. |
| **transfer** | Rotation quality often transfers; still re-score gate vs down. |

**Prior art:** QuIP / QuIP#; QuaRot; SpinQuant; BitNet b1.58; OneBit / BiLLM / PB-LLM (1-bit PTQ — treat as **risk**, not proof).

---

## SB8 — `STACK-SUB02` (the aggressive composition)  
**one-line:** SB1 `r∈{64,128}` + SB2 shared scales + SB7 Hadamard + **organ-conditioned residual on gate/up only** + leave router/attn/embed protected. This is the only in-lane path that can print **organ ≤0.2 complete** without lying.

| field | content |
|---|---|
| **complete_byte_accounting** | Sum every bin. Worked example O005, `r=64`, `b=1` skinny, 8-bit shared `V`/`U`, per-row f16 scale, **1% rice+q1 residual on gate+up only**, down naked 1-bit: **payload** `1·64/2048=0.031`. **tables** `~0.004`. **scales** `0.012`. **correction** Q80 identity: 1% outliers × **~8.24 bits** / weight-equivalent ≈ **`0.01×8.24=0.082` on gate+up mass** = `2/3` of experts → **+0.055** complete-expert. **offsets** rice stream (included in 8.24). **meta** `r,b`, residual bitmap/rice header. **alignment** `~5 MB`. **container** `≤0.01`. **recon** packed 1-bit skinny + shared factors + sparse repair **on selected channels**, never dense `W`. **nominal_bits=1**; **complete_expert ≈ 0.03+0.004+0.012+0.055+0.01 ≈ 0.11**. Whole-model: `0.95·0.11 + 0.05·n = 0.10+0.05n` → **`n=2` gives `0.20`**. |
| **stored_bpw** | Organ **~0.11** (INFERRED composition). Whole-model sub-0.2 **only** with `n≤2`. With `n=8`: **~0.50** (sub-1, not sub-0.2). |
| **active_bpw** | vs active params: `0.54·0.11+0.46·n`. `n=4` → **1.90**; `n=2` → **0.98**. **Active sub-1 and stored sub-0.2 are the same nonexpert problem.** Touched expert bytes: `8/128` of 1-bit skinny + residual hits + 1 `Vx`/layer. |
| **expected_reachable_bpw** | **Organ 0.08–0.20** (range = residual rate 0–2% + `r=64–128`). **Whole-model 0.20–0.55**. GLM screen **0.167 @ 0.755 cos** is the analog — **expect screen-pass / Doctor-fail** at the bottom of the range. |
| **quality_risk** | **Gate residual under-budget** (Q80 binary_g **clears** bar; binary+2% **over-budget**; gate needs more repair than 1%). **Down 1-bit** may be OK (tolerant) or not (Q80 binary_g down **failed** bar `0.81–0.83`; needed hgravs01). So down in the stack should stay **SB3 `r≥128,b=2`**, which **raises** down organ to `~0.13` and mixed expert to **~0.14–0.18**. Doctor: GLM H0.98 **collapsed at ~0.98 complete**. A 0.11 organ cosine of 0.75 is **not** a capability claim. Localize: report which of `{gate,up,down,layer-band,family}` broke. |
| **cheapest_falsifier** | Do **not** pack the stack first. Sequence: (1) SB1 uncoded `r=64` cosine; (2) 1-bit skinny; (3) add 1% gate residual; (4) **one** fast-Doctor **only if** cosine ≥0.75 on all three organs. Kill at the first step that fails. Use **real `X`**. |
| **execution_path** | Fused: `FWHT(x) → Vx → 1-bit gather skinny → optional residual scatter-add on gate`. Residual is **sparse channel repair**, not densify. Kernel cost: extra CSR/rice decode **per selected expert** — can dominate if residual is scattered (Q80: topology of correction is the falsifier, not bit width). If rice decode &gt; GEMM, switch to **block** residual or drop it (count the kernel, not just bytes). |
| **applicability** | O005/O006/Q80 expert body. GLM/O003: add SB4 as the `Δ` source. Nonexpert 2-bit is a **different lane** (attn/embed sensitivity; GLM sub-bit embed **catastrophic**). |
| **confidence** | **SPECULATIVE** as a Doctor-valid sub-0.2 body. **DERIVED** as the unique in-lane complete-accounting path to organ 0.2. |
| **transfer** | Stack transfers as a **recipe**; every `r`/`residual%` retuned. O003 first if SB4 residual is small (free delta budget). |

**Prior art:** GLM `0.755@0.167` real-act screen; Q80 binary + rice residual; AQLM+LoRC; BitNet + sparse repair; Matryoshka T0=skinny 1-bit, T1=gate residual (tiers family).

---

## SB9 — `CROSS-LAYER-V` (not weight tying)  
**one-line:** Tie the **activation PCA basis** across a block of layers (residual stream moves slowly). **Not** `W^{ℓ}_e ≈ W^{ℓ+1}_e` (killed).

| field | content |
|---|---|
| **complete_byte_accounting** | One `V` per block of `L` layers instead of per layer: tables `/L`. At `L=4`, SB1 tables `0.014→0.0035`. Skinny maps **untied** (NS-cross-layer-tying). |
| **stored_bpw** | Incremental **−0.01** expert. Not a sub-1 driver; a table-trim on SB1. |
| **active_bpw** | `V` can stay resident across the block (cache). Tiny. |
| **expected_reachable_bpw** | **~0.01** complete. Only do it if SB1 tables matter (`b_v=16` or huge `r`). |
| **quality_risk** | Layer-0 is a **different source** — never tie L0 into L1–L4. Mid/late more plausible. |
| **cheapest_falsifier** | Principal-angle of `V_ℓ` vs `V_{ℓ+1}` on real `X`. **Kill if** mean overlap `&lt;0.5` at `r=128`. |
| **execution_path** | Same kernel as SB1; `V` buffer reused. |
| **applicability** | Depth ≥16 MoE. Skip L0. |
| **confidence** | **HYPOTHESIS**, small stakes. |
| **transfer** | Residual-stream smoothness may be universal; still exclude L0. |

**Prior art:** NS-cross-layer-tying (killed **weights**); QuaRot single rotation for whole model; SpinQuant per-layer vs fused.

---

## Native / stored vs active (judge the executable object)

| Mechanism | Executable object | Expand-to-dense? | Apple-GPU |
|---|---|---|---|
| SB1/3/8/9 | packed skinny + shared factor GEMM | **forbidden** | GEMM — live |
| SB2 | broadcast scale | no | live |
| SB4 | shared GEMM + gathered `Δ` | no | live |
| SB5 | K bases + Δ | no, if bases are factors | live |
| SB6 `K≤16` binary CB | FMA codebook | no | live **only** tiny/LUT-free |
| SB6 large LUT / QTIP trellis | — | — | **dead** |
| SB7 | FWHT + 1-bit GEMV | no | live |

Runtime is often **kernel-bound**. SB1 at `r=128` cuts **MACs** ~8× on gate, so it can win even when bandwidth is not the limiter. A 0.2-bit payload that **unpacks to fp16 `W`** is a **fake** sub-0.2 (count unpack DRAM + FLOPs).

---

## Attack order (cheapest information first)

1. **SB5 falsifier (a)(b)** on O005 L10 analog — minutes. Expect **KILL**. Records that functional families are absent.  
2. **SB2 scale-template cosine** — minutes. Either harvest `~0.5–1.0` complete BPW or kill sharing of scales.  
3. **SB1 uncoded PCA** `r={256,128,64,32}` on real `X`, gate **and** down — the lane’s go/no-go.  
4. If (3) lives at `r=256`: **SB3 tall-share** on Q80 down (xcache exists).  
5. **SB4 residual energy** on O003/GLM only.  
6. Bit-code skinny (SB7/SB6 tiny) only after uncoded PCA holds.  
7. **SB8** residual on gate, then **Doctor**. Cosine is a screen. GLM `0.755@0.167` is the screen analog; GLM `~0.9` complete **collapsed** is the Doctor analog.  
8. Nonexpert 2-bit is **out of this lane**; without it, **stored sub-0.2 and active_bpw sub-1 will not happen**. Say so rather than hide it in expert BPW.

**Doctor authority:** fast 12-item battery is a screen, not a seal. On fail, name `{organ, layer-band, expert-set, channel}` — a gibberish 0.2 artifact that **localizes gate-L0** is still a win.

---

## Transfer (cohort)

| Patient | SB1 | SB2 | SB3 | SB4 | SB5 | SB6 | SB7 | SB8 |
|---|---|---|---|---|---|---|---|---|
| **O005** 8/128, 0 shared, uniform | primary | primary | after SB1 | N/A | falsify | after SB1 | after SB1 | primary stack |
| **O006** sibling + vision | same language-MoE | same | same | N/A | skip if O005 killed | same | same | same; vision **out** |
| **O003** 6/64, **2 shared** | yes | yes | yes | **primary** | retry even if O005 dies | yes | yes | SB4+SB8 |
| **Q80** 10/512, shared, ΔNet | max amortization | max scale bin | **LIVE down** | maybe | weight-space already **no** | max table win | yes | yes; protect ΔNet/attn |
| **GLM-4.5-Air** 8/128+1 shared, L0 dense | yes | yes | unknown | **primary** | maybe | care (GLM-5.2 PQ dead) | yes | L0 **separate codec** |
| dense/hybrid | no (ASVD-dead) | yes | no | no | no | layer-CB only | maybe | no |

Universal: **SB2** (metadata), **activation-PCA as the shared object** (not expert templates).  
Not universal: families, shared-expert deltas, Q80 rank-32 weight subspace.

---

**Lane verdict:** Weight-space sharing is **dead** on measured MoEs. The credible sub-1 mechanism is **shared activation basis + per-expert skinny maps + shared scales**, executed as GEMM, counted complete. That can put **expert organs** in `~0.09–0.29` complete BPW (DERIVED geometry; quality UNKNOWN below `r=256`). **Whole-model sub-0.2 is blocked by the 3–5% nonexpert** unless that lane also goes ≤2-bit. Treat GLM `0.167` as a **real-activation screen precedent**, not a capability precedent.
