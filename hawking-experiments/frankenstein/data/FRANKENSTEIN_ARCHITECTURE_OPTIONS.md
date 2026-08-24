# Frankenstein Architecture Options

Research pass on whether a more structural ("Frankenstein-ier") technique beats the currently-running
small reversible residual bridge approach, given our sealed constraints. Consult profile, read-only,
2023-2026 literature + local sealed contracts.

## SUMMARY

**Plain answer:** under the sealed constraints (hidden-size mismatch 6144/7168→4096, frozen DeepSeek
256-expert top-6 router, training-free/small-fit budget, reversible additive adapters), **there is no
well-precedented "Frankenstein-ier" technique that clearly beats small always-on residual bridges at
fixed transplant points.**

Weight merging, passthrough frankenmerges, sparse upcycling, BTX, and native guest-expert slots are
either **shape-illegal** or **budget/router-illegal**. The one real upgrade in the same family is a
**conditional multi-adapter hub** (a tiny router outside the native MoE) — and only worth building
**after** V0's fixed bridges validate.

Sealed locals checked (read-only): `DSV4F_TRANSPLANT_POINTS.json` (`direct_weight_transplant: false`),
`DSV4F_LATENT_BRIDGE_CONTRACT.json`, `KIMI_STRATEGIC_BRIDGE_CONTRACT.json`, `FRANKENSTEIN_FUSION_OPERATION.json`
(already forbids weight average / direct transplant / dual-donor residency).

---

## TECHNIQUES SURVEYED (with citations)

### 1. Model merging / frankenmerging (SLERP, TIES, DARE, passthrough)

| Method | Idea | Key cites |
|--------|------|-----------|
| SLERP / soups | Interpolate / average same-shape weights | mergekit; NVIDIA merge intro 2024 |
| Task arithmetic | τ = θ_ft − θ_pre | Ilharco et al. 2023 |
| TIES | Trim, elect signs, merge | Yadav et al. 2023 |
| DARE | Drop-and-rescale deltas | Yu et al. 2023/24 |
| Passthrough | Concatenate layers → exotic depth | mlabonne HF blog 2024; Goliath-style frankenmerges |
| SOLAR DUS | Same-arch depth splice + CPT | Kim et al. arXiv:2312.15166 |

**Cross-architecture?** **No for weight ops.** Same hidden size, heads, layer layout required. Surveys
(e.g. arXiv:2603.09938) treat exact architectural correspondence as an assumption; cross-arch is an open
problem. HF merge FAQ: identical architecture required. Passthrough still needs compatible layer tensors
— you cannot splice a 6144-d GLM block into a 4096-d residual.

**Cost:** cheap if shapes match; **zero utility** here without rewriting the whole donor into DSV4F
geometry (no longer "merging"). **Risk:** interference, dead averages when models diverged. **Scale:**
7B-70B same-family dense; not foreign MoE graft.

**Verdict: RULED OUT** — confirms sealed `stream_portion_and_average_weights: IMPOSSIBLE`.

---

### 2. Sparse upcycling

**What:** Komatsuzaki et al., arXiv:2212.05055 (ICLR 2023) — copy **your own** dense MLP into E identical
experts, random-init router, continue train. Gains at roughly **+10-60%** of original dense pretrain
budget. Drop-Upcycling (2025) improves diversity when naive clones stay uniform.

**Foreign experts?** **Not in the recipe.** Experts start as clones of the **same** model's FFN at the
**same** hidden size. Projecting GLM experts (H=6144) into DSV4F slots (H=4096, intermediate 2048) is a
different, unpublished research program + router rebalance + substantial fine-tuning.

**Cost:** real continued pretrain, not closed-form. **Risk:** quality dip after surgery, uniform routing,
load imbalance. **Scale:** T5 XL / ViT-L / 7B-class — not proven as foreign graft into a 284B/13B MoE
under a small-fit budget.

**Verdict: NOT APPLICABLE** for GLM/Kimi-sourced experts.

---

### 3. BTX / BTM

**What:** Sukhbaatar et al., arXiv:2403.07816 — branch seed → domain train → pack FFNs as MoE experts,
average attention, MoE-finetune routing (domain experts tens-hundreds of B tokens; mix stage ~80B on
Llama-2 7B).

**Requires same architecture as seed.** Cannot import GLM/Kimi weight tensors. Router fine-tuning
conflicts with the frozen-router / promotion-gate constraint.

**Verdict: RULED OUT** as a donor-graft path.

---

### 4. Routed "guest expert" grafting (native MoE expansion)

**The idea we were probing:** a projected donor module as expert 256...256+K; teach DeepSeek's router to
pick it conditionally, instead of an always-on residual add.

**Precedent gap:** no clean 2023-2026 recipe for *extending a trained large MoE router with
foreign-dimension experts*. Related work:
- Router logit deltas / test-time rerouting (arXiv:2510.14853) — tweak **existing** logits, don't add experts.
- PEFT LoRA-MoE / LoRAMoE (Dou et al. ACL 2024, arXiv:2312.09979), MoLoRA, AdapterFusion — a **separate**
  small router over adapters on a **frozen** backbone; deliberately avoid native router surgery.
- MoE collapse literature — hot/cold experts, dead experts, aux-loss fragility.

**What it would actually take:** H→guest→H projection; expand `W_router` 256→256+K; capacity/top-k
accounting; a load-balance loss; freeze or heavily regularize the original columns; and it would still
face a promotion-gate reject for changing routing behavior.

**Verdict: HIGH RISK / DEFER.** The safe analogue is an **external adapter MoE**, not native slot expansion.

---

### 5. Depth / width extension (LLaMA Pro, SOLAR)

| Method | Mechanism | Cost class |
|--------|-----------|------------|
| LLaMA Pro (Wu et al. arXiv:2401.02415) | Interleave new blocks, zero-init identity, freeze base, train new blocks on domain corpus | ~80B tokens; ~2830 H800 GPU-h (their 7B→8.3B) |
| SOLAR DUS (Kim et al. arXiv:2312.15166) | Duplicate/splice same-arch layers, continued pretrain | CPT-scale |

Inserting "GLM capability blocks" still requires rewriting into **4096-d DSV4F interfaces** — a large
residual module with worse reversibility and latency than a sealed adapter archive.

**Verdict: POOR FIT** for the training-free/small-fit + additive/reversible program law.

---

### 6. Adapter / LoRA hubs & mixture-of-adapters (closest match)

| Technique | Mechanism | Cites |
|-----------|-----------|-------|
| Residual adapters | Always-on bottleneck residual | Houlsby 2019 |
| LoRA | Low-rank ΔW | Hu 2021 |
| AdapterFusion | Freeze task adapters; learn fusion over outputs | Pfeiffer 2020 |
| AdaMix | Mixture of adaptation modules | Wang 2022 |
| MoLoRA / LoRAMoE | Token router over LoRA experts; knowledge-preserving allocation | Zadouri 2023; Dou 2024 |
| MoA / MoE-LoRA family | Heterogeneous/soft/sparse PEFT MoE | arXiv:2506.05928 |

**Current plan** = single always-on residual `y = x + A(x)` at named sites, fit closed-form / small
supervised loss to projected donor activations — squarely the PEFT residual-adapter family, already
sealed in the fusion op.

**Stronger same-family variant:** a soft/top-1 gate over `{A≈0, A_glm, A_kimi, ...}` with a **tiny
external router** (not the 256-expert native router). Addresses always-on interference and GLM+Kimi
co-location without touching native MoE routing.

**Verdict: BEST POST-V0 UPGRADE** — not a replacement for validating V0.

---

### 7. Cross-arch fusion without weight merge (FuseLLM)

Wan et al. arXiv:2401.10491: fuse **distributions** of heterogeneous LLMs into a target (~1.8B tokens,
~33h on 8×A100 for 7B). Cross-arch works because fusion happens on logits, not weights. HeteroFusion
(2026) strengthens topology-aware transfer under mismatch.

**Relation to us:** our sealed path is the **latent-local cousin** — project activations, fit reversible
residuals, stream one donor at a time. Supports "projections + small train beat weight merge under
mismatch" as a general principle; does **not** justify full-body continued-pretrain of a 13B-active MoE
as the default here.

---

## FEASIBILITY AGAINST OUR CONSTRAINTS

| Technique | Dim mismatch | Frozen native router | Small-fit budget | Reversible / additive | Proven at ~284B/13B MoE scale? | Verdict |
|-----------|--------------|----------------------|-------------------|-------------------------|--------------------------------|---------|
| Always-on residual bridge (current) | Handles it (H→4096) | Yes | Yes | Yes | Pattern proven; donor graft is novel but sound | **PRIMARY** |
| Multi-adapter hub (external router) | Handles it | Yes | Mostly | Yes | PEFT scale | **BEST V1+** |
| FuseLLM-style body distill | Handles it | Only if router stays frozen | Marginal-No | Partial | 7B class | Optional later |
| Native guest experts | Only with projection | **No** | No | Hard | No | Defer |
| Sparse upcycling (own weights) | Wrong problem | No | No | N/A | Not a foreign graft | Out |
| BTX/BTM foreign donor | No | No | No | Partial | 7B same-arch only | Out |
| mergekit / TIES / DARE / SLERP | **No** | N/A | Useless here | N/A | Same-shape only | Out |
| Passthrough frankenmerge | **No** | N/A | Useless here | Poor | Same-shape only | Out |
| LLaMA Pro / SOLAR depth extension | Only if rewritten to 4096-d | If MoE untouched | No | Weaker | Dense 7-11B only | Poor |

---

## RECOMMENDATION (ranked risk/reward)

| Rank | Action | Risk / reward |
|------|--------|----------------|
| **0** | **Validate the current V0 always-on reversible residual bridge** (already sealed as the fusion op) | Lowest structural risk; reward is bounded but gateable and stackable (GLM then Kimi, disjoint block_ids) |
| **1** | After V0: conditional multi-bridge hub (mixture-of-adapters) at the same sites | Medium engineering, PEFT-scale data; fixes always-on interference / GLM×Kimi collision — **only worth building if** V0 shows secondary regressions or a real bridge-site collision |
| **2** | Optional small FuseLLM-flavored / teacher-forced curriculum **on top of** bridges | Higher data cost than closed-form; still no router surgery |
| **3 — park** | mergekit, foreign sparse upcycling, BTX-from-donors, native guest experts, depth-grafted donor blocks | Shape-, budget-, or router-illegal; higher risk for uncertain (or structurally impossible) gain |

**Explicit answer to "can we do more Frankenstein-y stuff":** Not under these constraints, no. The
current approach is the right conservative best-practice given the real dimension mismatch and the
frozen-router requirement. Expanding native MoE expert count with projected GLM matrices is **not** a
free win — it would reintroduce load-balance collapse risk while still needing the exact same projection
math the residual bridge already does. The one legitimate upgrade path (a small external router choosing
between bridges) is worth revisiting only after V0's fixed-bridge result is in hand.

---

## CONFIDENCE

| Claim | Confidence |
|-------|------------|
| Weight merge/passthrough cannot cross the hidden-size mismatch | High |
| Sparse upcycling = same-model growth + continued training, not foreign grafting | High |
| Native guest-expert router expansion is under-precedented / high risk at our scale | High-Medium |
| Current residual bridge is the correct primary path | High |
| Multi-adapter hub is the best post-V0 upgrade | Medium-High |
| No hidden 2023-2026 technique dominates under our training-free/small-fit budget | High |
