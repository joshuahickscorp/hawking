# Ascension Gravity Research Registry

**Programme:** Hawking Ascension (Apple-first Gravity co-design; dual-Qwen self-optimization)  
**Contract:** research + registry only — no live Qwen download, streaming, or Gravity packing  
**Template:** same research-registry pattern as `FRANKENSTEIN_ARCHITECTURE_OPTIONS.md`  
  (real citations, honest feasibility, no invented numbers, explicit ADMIT/DEFER/REJECT)  
**Authority bible:** `HAWKING_ASCENSION_BIBLE.md` §3 (physical model), §5.1–5.12 (portfolio), §6 (Gravity gate)  
**Local evidence base:** tonight’s / this campaign’s sealed DSV4F Metal receipts (M3 Ultra)  
**Status:** RESEARCH_REGISTRY — decisions for future Qwen Gravity packing, gated on Proto-Frankenstein offload  
**Date:** 2026-08-06

---

## SUMMARY

This pass applies the Frankenstein architecture-options research pattern to the twelve Ascension mechanisms in bible §5.1–5.12. The goal is **not** implementation against Qwen weights (none are local yet) but a **decision registry**: what must change Gravity packing, what belongs in runtime/kernel only, and what is deferred or rejected under the governing B/F/U/R/D/S/K/C model.

**Plain answer for Gravity packing of Qwen3-Coder (30B-A3B, then Next-80B):**

| Must be decided **before** final Gravity packing | Can land later in runtime/kernel without re-packing | Do **not** bake into first Qwen Gravity |
|---|---|---|
| 5.1 data-local packed layout (scale/codebook/tile/alignment) | 5.2 expert/projection wave scheduling | 5.5 delta projection (drift risk, unproven) |
| 5.7 FLOP-substitution-friendly bit layout (native unpack, not full decode) | 5.3 cross-session weight amortization (HCLI scheduler) | 5.8 low-rank/basis as default body format |
| 5.12 primary kernel grammar (custom SIMDgroup path first) | 5.4 cross-token invariant reuse (pipelines, graphs, RoPE tables) | 5.9 active-expert reduction as packing default |
| hot/cold expert **grouping slots** (layout only; policy later) | 5.6 multi-token/block execution (accounting + draft/verify) | 5.10 conditional depth / functional replacement |
| | **5.11 command-graph collapse (already measured)** | |

**Tonight’s real DSV4F evidence that already answers mechanisms (not hypothetical):**

| Mechanism | Measured fact (sealed) | Receipt |
|---|---|---|
| **5.11 Command-graph collapse** | Attention complete block: **21 CB / 21 waits → 1 CB / 1 wait**, still **21 dispatches**; host-wall **p50 62 579 → 54 271 µs = 13.28% win**; p99 also improved; **promoted** under “p50 better and p99 not worse” | `DSV4F_P4A_LAYER0_ATTENTION_TOPOLOGY_SWEEP-v1.json` |
| **5.1 / 5.7 / 5.12 FP4 matvec** | Source-native FP4 E2M1FN×2 + E8M0 fused matvec, **1 CB / 1 dispatch**, **max abs error 0.0** vs CPU oracle, GPU **1 773 µs** (2048×4096 expert gate linear) | `fp4-metal-component-probe-receipt.json` |
| **5.1 / 5.7 / 5.12 FP8 matvec** | Source-native FP8 E4M3FN + E8M0 fused matvec, **1 CB / 1 dispatch**, **max abs error 0.0**, GPU **830 µs** (1024×4096 control linear) | `fp8-metal-component-probe-receipt.json` |
| **5.1 / 5.12 SIMDgroup split-K** | Raw-weight SIMDgroup split-K candidates: FP4 component p50 speedup **24.3×** vs serial authority; FP8 **30.0×** — **NOT_PROMOTED** (component-only; not source-forward / not runtime) | `DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP-CANONICAL-v1.json` |
| **5.1 / 5.7 act quant** | SIMDgroup block candidate GPU p50 **95 µs** vs fixed authority **5 967 µs** (~**62.8×** component) with **byte-exact** act+scale vs CPU oracle — component QAT candidate only | `DSV4F_ACT_QUANT_SIMDGROUP_SWEEP-v1.json` |
| **5.2 / 5.11 layer topology tax** | L0–L1 full Metal forward: **169 dispatches / 20 CB / 20 waits**, wall **≈17.98 s**; L0–L2: **276 / 26 / 26**, **≈22.9 s**; full BOS L0–L42 greedy: **3 654 dispatches / 265 CB** | multi-layer receipts + `KERNEL_BROKERS_TUNING_PLAN.md` |
| **5.2 MoE wave inventory** | Per full MoE layer P6+P7: **63 dispatches / 4 CB** (18 FP4 expert + 3 FP8 shared + 8 act quant + glue); expert W1/W3 and W2 already concurrent-group eligible | P6 topology + brokers plan |
| **mHC precision (DSV4-only)** | mHC attn pre dominates attention micro-profile (**76.61 ms GPU / 2 disp** of 101.55 ms total at pos-1); strict Darwin-DD control domain required — **not a Qwen organ** | P4B complete attention receipt |
| **Prior GLM collapse (cross-family)** | Functional MoE path: **76 → 1 CB/token**, **6.3 → 7.1 TPS (1.12×)** measured wall | `HAWKING_ASCENSION_CLOSED.json` |

**Honesty boundary:** Qwen weights are **not** present. Every “measured result” below is either (a) DSV4F/Metal evidence that **transfers as a mechanism class**, or (b) literature precedent, or (c) explicitly `PROTOTYPE_PENDING` / `NOT_MEASURED_ON_QWEN`. No fabricated Qwen TPS, BPW, or capability scores.

---

## PHYSICAL MODEL FRAMEWORK (§3)

For every model and every candidate, Hawking measures:

| Symbol | Meaning | Roof it feeds |
|---|---|---|
| **B** | physical bytes / token | memory roof = bandwidth / B |
| **F** | executed operations / token | compute roof = achieved ops/s / F |
| **U** | achieved hardware utilization | effective ops/s and effective BW |
| **R** | reuse across experts, tokens, sessions | multiplies useful work per load |
| **D** | sequential dependency depth | critical-path roof with S, K |
| **S** | synchronization / submission cost | critical-path roof |
| **K** | KV / recurrent / state traffic | critical-path + memory |
| **C** | capability and correctness | hard gate on every promotion |

```text
ceiling = min(memory roof, compute roof, critical-path roof)
capability: every promoted path must preserve C
```

**Improvement axes (bible):** use more of the chip · move fewer bytes · execute fewer ops · reuse each load · shorten the serial graph · reduce sync · compress state · preserve capability.

**Implication for this registry:**

- A mechanism that only shrinks file size but **raises active B** or **destroys C** is REJECT.
- A mechanism that only reduces F while leaving B and S dominant on decode is DEFER for first Gravity if it forces a packing change.
- Mechanisms that change **layout, scale locality, tile geometry, expert grouping, basis format, KV format, or kernel grammar** are Gravity-gate material (ADMIT_TO_GRAVITY).
- Mechanisms that only change **scheduler, command topology, session multiplexing, draft/verify** are ADMIT_TO_RUNTIME unless they force resident tensor layout.
- Kernel microarchitecture (SIMDgroup, fused decode, split-K) is ADMIT_TO_KERNEL; it still informs Gravity when it requires a specific packed layout.

**Qwen geometry assumed for future packing (public HF/config, not local weights):**

| Model | Role | Public shape (literature / HF card) |
|---|---|---|
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | executor | ~30B MoE, **128 experts / top-8 active**, SwiGLU experts, GQA, long ctx (family 32K–256K class) |
| `Qwen/Qwen3-Coder-Next` | reviewer | ~80B-class hybrid; exact local config **not sealed in this worktree** — treat as “larger MoE / denser control” until config hash is admitted |

DSV4F (Terra) differs (256 experts / top-6 + shared, FP4 experts / FP8 control, mHC). Shared mechanism classes transfer; **mHC-specific rows do not apply to Qwen**.

---

## THE 12-MECHANISM REGISTRY (§5.1–5.12)

Record schema (bible §4):

```text
mechanism · hypothesis · expected B/F/U/R/D/S/K effect · source geometry
prototype status · measured result · capability risk · Gravity implication
runtime implication · reopen condition · verdict
```

Verdicts used: `ADMIT_TO_GRAVITY` · `ADMIT_TO_RUNTIME` · `ADMIT_TO_KERNEL` · `DEFER` · `REJECT`  
(A mechanism may carry **multiple** admits when effects span layers.)

---

### 5.1 Data-local packed execution

| Field | Content |
|---|---|
| **Mechanism** | Fused quant decode + scale + dot; no decoded-weight materialization; register-resident accumulation; threadgroup/codebook caching; packed vector loads; multi-row SIMDgroup reductions |
| **Hypothesis** | Decode bandwidth is dominated by materializing dequantized weights. Fusing unpack into the matvec inner loop cuts B roughly by the compression ratio and raises U on memory-bound decode |
| **Expected B/F/U/R/D/S/K** | **B↓** (no full FP16/FP32 weight buffer) · **F** similar or slightly ↑ (decode ops) but useful progress/s ↑ · **U↑** on BW-bound GEMV · **R↑** if scales/codebooks cached in TG · **D/S** neutral if still one dispatch · **K** neutral |
| **Source geometry** | Qwen MoE experts + attention projections at native or PTQ container layout; DSV4F precedent: E2M1×2 + E8M0 / E4M3 + E8M0 blocks |
| **Prototype status** | **MEASURED on DSV4F components** (not Qwen). Authority fused matvec kernels live in `matmul.metal` |
| **Measured result (real)** | FP4 expert gate linear: **max abs error 0.0**, 1 dispatch, GPU **1 773 µs**. FP8 control linear: **max abs error 0.0**, GPU **830 µs**. SIMDgroup split-K candidates **24–30×** component p50 vs serial authority (**not promoted**). Act-quant SIMDgroup candidate **~62.8×** vs 5 967 µs fixed authority with **byte-exact** act+scale |
| **Capability risk** | **Medium if association order changes**; **Low** if bit-exact unpack + sealed scale placement vs source/CPU oracle. Wrong scale locality silently kills C |
| **Gravity implication** | **ADMIT_TO_GRAVITY:** tensor ordering, scale/codebook **co-located with tiles**, alignment, tile boundaries, expert grouping so kernels never repack or fully decode at token time |
| **Runtime implication** | Authority path stays fused; candidate SIMDgroup geometries A/B under parity before serve flip |
| **Reopen condition** | New quant grammar (e.g. Q4_K vs native FP4/FP8) or measured promote of split-K that requires different tile pitch |
| **Literature (2023–2026)** | Marlin FP16×INT4 fused kernel (Frantar et al., arXiv:2408.11743; near-ideal ~4× up to batch 16–32); GPTQ (Frantar 2022); AWQ; llama.cpp / BaseRT-class fused dequant-in-inner-loop on Apple Silicon (BaseRT arXiv:2607.00501: “dequantisation directly into the inner loop rather than materialising dequantised weights”) |
| **Verdict** | **ADMIT_TO_GRAVITY** + **ADMIT_TO_KERNEL** |

---

### 5.2 Projection and expert waves

| Field | Content |
|---|---|
| **Mechanism** | QKV waves; top-k expert gate/up waves; device activation; down waves; shared-expert concurrency; route-weighted combine |
| **Hypothesis** | Serial projection/expert graphs leave U low. Wave-concurrent independent matvecs raise occupancy and cut wall without changing math if combine order is preserved |
| **Expected B/F/U/R/D/S/K** | **B** same · **F** same · **U↑** · **R↑** (activation reused across concurrent experts) · **D↓** within layer · **S↓** if waves share CBs · **K** neutral |
| **Source geometry** | Qwen3-Coder-30B: **128 experts, top-8**; DSV4F: 256 / top-6 + 1 shared |
| **Prototype status** | **PARTIALLY MEASURED on DSV4F** (P6 concurrent groups exist; multi-layer still high CB tax). Qwen wave geometry **PROTOTYPE_PENDING** |
| **Measured result (real)** | DSV4F full MoE layer inventory: **63 disp / 4 CB** (18 FP4 expert matvec = 6×(W1+W3+W2)); brokers plan documents W1/W3 and W2 **begin_concurrent_group**. L0–L1 still **20 CB / 169 disp** end-to-end — wave packing incomplete at whole-token graph |
| **Capability risk** | **Medium:** reduction/combine association; exact top-k IDs must stay hard-gated. Wrong concurrent write hazards destroy C |
| **Gravity implication** | **ADMIT_TO_GRAVITY (layout only):** pack expert W1/W3/W2 tiles for wave-friendly strides; optional shared-expert co-location. Do **not** freeze wave width into artifact metadata beyond geometry |
| **Runtime implication** | **ADMIT_TO_RUNTIME + ADMIT_TO_KERNEL:** scheduler emits expert waves; Metal concurrent groups; measure occupancy / achieved FLOPS / BW / wall / p95/p99 per bible |
| **Reopen condition** | Qwen top-8 gather width or learned route compose changes wave inventory |
| **Literature** | MegaBlocks / grouped GEMM MoE; vLLM/SGLang expert parallelism; DeepSeek-V3/V4 open inference stacks with concurrent expert execution |
| **Verdict** | **ADMIT_TO_RUNTIME** + **ADMIT_TO_KERNEL**; **partial ADMIT_TO_GRAVITY** (expert tile grouping / strides only) |

---

### 5.3 Cross-session weight amortization

| Field | Content |
|---|---|
| **Mechanism** | Apply each weight tile to 1/2/4/6/8 independent HCLI sessions; measure aggregate accepted TPS, per-session p99, weight bytes/session/token, FLOPS, fairness, starvation |
| **Hypothesis** | On unified memory Apple Silicon, resident weights can be reused across sessions, amortizing B_effective = B_weights / N_sessions while preserving per-session KV isolation |
| **Expected B/F/U/R/D/S/K** | **B_eff↓** with N · **F** same per session · **U↑** if microbatching fills GPU · **R↑↑** (primary lever) · **D** may ↑ slightly (batch deps) · **S** may ↑ (multi-seq encode) · **K↑** with concurrent sessions (KV) |
| **Source geometry** | HCLI multi-agent / multi-session; Qwen long-ctx makes K the limiter at high N |
| **Prototype status** | **PROTOTYPE_PENDING on Qwen.** Bible marks this a **primary HCLI mechanism**. Continuous-batching literature is mature; local multi-session Gravity measure does not exist for Qwen |
| **Measured result** | **NOT_MEASURED_ON_QWEN.** Related local: GLM expert-wave / residency work (collapse wins elsewhere); DSV4F static expert residency is **metadata-only**, `physical_active_bytes_per_token: NOT_MEASURED_NO_NATIVE_RUNTIME` |
| **Capability risk** | **Low for weight sharing** (read-only weights); **High for KV/session isolation bugs** and fairness starvation under RED/CRITICAL pressure governor |
| **Gravity implication** | **No re-quant required.** Optional: multi-seq-friendly KV page layout (see 5.6/K). Weights stay single resident copy |
| **Runtime implication** | **ADMIT_TO_RUNTIME:** session multiplex scheduler; measure ladder N=1..8; separate fairness receipts |
| **Reopen condition** | First sealed dual-Qwen or multi-session HCLI coexist_bench on Apple Silicon |
| **Literature** | Orca continuous / iteration-level batching (OSDI 2022); vLLM PagedAttention (SOSP 2023) + continuous batching; FlashDecoding (Stanford CRFM 2023) for long-ctx decode batches |
| **Verdict** | **ADMIT_TO_RUNTIME**; **DEFER** multi-session Gravity KV page format until first N≥2 measure |

---

### 5.4 Cross-token invariant reuse

| Field | Content |
|---|---|
| **Mechanism** | Reuse only proven invariants: pipelines, argument tables, tensor descriptors, quant tables, scale tables, hot experts, static RoPE data, command graphs, KV allocations, route-transition statistics. **Do not** reuse prompt-dependent activations without proof |
| **Hypothesis** | Host encode / pipeline lookup / graph build dominate S on short kernels; reusing PSO, arg buffers, and static tables cuts S and host wall without touching C |
| **Expected B/F/U/R/D/S/K** | **B/F** neutral · **U↑** slightly (less host stall) · **R↑** (pipelines, tables) · **D** neutral · **S↓↓** · **K** neutral if KV arena reused correctly |
| **Source geometry** | All Metal decode graphs |
| **Prototype status** | **PARTIALLY MEASURED.** Multiple DSV4F receipts set `pipelines_precompiled_before_warmup: true`; FP4/FP8 probes show pipeline_lookup_us **25–33 ms** on cold path vs encode **~250 µs** — cold lookup tax is real |
| **Measured result (real)** | P4A 1-CB candidate encode p50 **45 µs** with precompiled pipelines; component probes host_wall ≫ GPU when pipeline cold. Multi-layer still pays high wait count (S) — invariant reuse incomplete at full graph |
| **Capability risk** | **Low** for PSO/tables; **Critical** if prompt-dependent activations or route IDs are wrongly reused (silent wrong experts) |
| **Gravity implication** | Static RoPE, quant tables, scale tables stored as **first-class immutable organs** (no token-time metadata parse) |
| **Runtime implication** | **ADMIT_TO_RUNTIME + ADMIT_TO_KERNEL:** persistent pipelines, argument tables, optional replayable graphs (ties to 5.11) |
| **Reopen condition** | Any candidate that wants activation reuse across tokens without a proof → automatic REJECT until formal argument |
| **Literature** | Metal PSO / argument buffer best practices; CUDA graph / cupy graph capture analogues; vLLM persistent engine workers |
| **Verdict** | **ADMIT_TO_RUNTIME** + **ADMIT_TO_KERNEL**; **ADMIT_TO_GRAVITY** for static tables / RoPE / quant metadata organs |

---

### 5.5 Delta projection execution

| Field | Content |
|---|---|
| **Mechanism** | \(h_{t+1}=h_t+\Delta h\), \(W h_{t+1}=W h_t + W\Delta h\); exploit sparsity/low-rank in \(\Delta h\) |
| **Hypothesis** | Consecutive hidden states are similar; projecting only \(\Delta h\) reduces F and possibly B if \(\Delta h\) is sparse |
| **Expected B/F/U/R/D/S/K** | **F↓** if sparse · **B↓** if sparse touch · **U** uncertain · **R** on cached \(Wh_t\) · **D** may ↑ (dependency on previous full state) · **S** may ↑ (reset/checkpoint) · **K** may ↑ (must store \(Wh_t\)) |
| **Source geometry** | Any linear projection; most attractive on huge vocab lm_head or dense layers |
| **Prototype status** | **NO local prototype.** Literature for true delta-W·h reuse in production decode is thin vs speculative decoding / low-rank adapters |
| **Measured result** | **NOT_MEASURED.** Do not invent sparsity rates |
| **Capability risk** | **High:** drift accumulation, adversarial continuations, reset policy failures — bible explicitly requires drift / reset / adversarial / full-token capability tests |
| **Gravity implication** | **Do not** invent a delta-native weight layout for first Qwen artifact |
| **Runtime implication** | Research harness only after BASE_TRUE_TPS path exists |
| **Reopen condition** | Sealed prototype shows ≥X% F reduction **and** full-token C parity with bounded drift + explicit reset |
| **Literature** | Residual streaming / incremental computation ideas; LoRA is ΔW not Δh; speculative methods are preferred production alternative (see 5.6). No 2023–2026 mainstream decode stack ships general Δh projection as default |
| **Verdict** | **DEFER** (research-only); **REJECT** for first Qwen Gravity packing |

---

### 5.6 Multi-token and block execution

| Field | Content |
|---|---|
| **Mechanism** | Prefill blocks; independent-session microbatches; MTP / draft proposals; verified token blocks; branch evaluation; blockwise state update. Keep **BASE_TRUE_TPS**, **BLOCK_EXECUTED_TPS**, **ACCELERATED_ACCEPTED_TPS** separate. No future-token leakage |
| **Hypothesis** | Prefill and speculative verify convert serial decode into wider GEMM / parallel verify, raising U and accepted tokens/s without changing the base model distribution when verification is exact |
| **Expected B/F/U/R/D/S/K** | Prefill: **U↑**, **B** amortized · Spec/MTP: **F↑** on draft+verify but **accepted tokens/s↑** · **R↑** on shared weights · **D** changes shape (tree/block) · **S** may ↑ · **K↑** for multi-token KV writes |
| **Source geometry** | Qwen prefill; optional MTP head if present on a future model; draft model or self-spec head |
| **Prototype status** | **PROTOTYPE_PENDING for Qwen.** DSV4F runtime spine notes MTP auxiliary **excluded** from base topology |
| **Measured result** | **NOT_MEASURED_ON_QWEN.** Do not convert any DSV4F component µs into accelerated TPS |
| **Capability risk** | **High if verification skipped**; **Low** for rejection-sampled speculative decoding (exact distribution). Prefill chunking risk is position/RoPE bookkeeping |
| **Gravity implication** | **DEFER packing changes** for MTP-specific organs until a model admits an MTP head. Prefill-friendly **KV page / block layout** may be admitted with multi-session (5.3) |
| **Runtime implication** | **ADMIT_TO_RUNTIME** with mandatory triple TPS accounting; never report accelerated as BASE_TRUE |
| **Reopen condition** | Sealed Qwen prefill block bench + optional draft/verify with acceptance stats |
| **Literature** | Speculative decoding (Leviathan et al. 2023; Chen et al. 2023); Medusa (Cai et al. 2024); EAGLE (Li et al. 2024); Multi-Token Prediction (Gloeckle et al. ICML 2024); FastMTP (2025); SGLang MTP support (2025); DeepSeek-V3 cascaded MTP |
| **Verdict** | **ADMIT_TO_RUNTIME** (accounting + prefill/spec paths); **DEFER** Gravity MTP organs; **REJECT** any metric conflation of accelerated vs base |

---

### 5.7 FLOP substitution

| Field | Content |
|---|---|
| **Mechanism** | Replace general FP work with bit extraction, integer unpack, table lookup, precombined scales, permutation, structured accumulation, small codebook selection. Target **useful model progress/s**, not max FLOP count |
| **Hypothesis** | On decode GEMV, many “FLOPs” are really unpack + scale. Doing them as integer/table ops can raise effective progress/s even when raw FLOP counters fall |
| **Expected B/F/U/R/D/S/K** | **B↓** with narrower loads · **F** redefined (fewer FP mul/add, more bitwise) · **U↑** on BW path · **R↑** with TG codebooks · **D/S/K** neutral |
| **Source geometry** | Quant matvecs (INT4/FP4/FP8), act quant, rope tables |
| **Prototype status** | **MEASURED on DSV4F fused unpack matvecs and act quant** |
| **Measured result (real)** | Fused FP4/FP8 matvec authority (table/bit decode in-kernel) parity-clean. Act quant is **glue that currently costs too much FP/TG** at authority TG=32 (P4B: **17.46 ms GPU / 5 disp** for tiny traffic); SIMDgroup candidate **95 µs** component — shows substitution/geometry matters more than “more FLOPs” |
| **Capability risk** | **Medium:** table contents and scale combine order must match source codec exactly |
| **Gravity implication** | **ADMIT_TO_GRAVITY:** store codebooks / scale grids for **direct kernel consumption**; forbid token-time scale reconstruction |
| **Runtime implication** | **ADMIT_TO_KERNEL:** prefer fused unpack; promote candidates only with byte-exact or V2.1 gates |
| **Reopen condition** | New codec (e.g. Q4_K, AQLM codebook) with different lookup geometry |
| **Literature** | Marlin; BitNet-style bit-linear lines; QuIP# lattice codebooks (Tseng et al. ICML 2024); AQLM additive quantization (Egiazarian et al. ICML 2024); Apple quant-aware Metal kernel guidance (WWDC custom ML kernels) |
| **Verdict** | **ADMIT_TO_GRAVITY** + **ADMIT_TO_KERNEL** |

---

### 5.8 Low-rank and basis execution

| Field | Content |
|---|---|
| **Mechanism** | Native low-rank factors, basis coefficients, product-quantized basis, compact residual, protected outlier channels. **Reject any path that reconstructs the full dense matrix before compute** |
| **Hypothesis** | Extreme compression via basis/PQ can cut B below 4 bpw while preserving C if residual/outliers protected |
| **Expected B/F/U/R/D/S/K** | **B↓↓** · **F** depends (two small GEMVs vs one) · **U** kernel-dependent · **R** on basis tables · **D** may ↑ (multi-stage) · **S** may ↑ · **K** neutral |
| **Source geometry** | Optional second-stage compression of Qwen weights; not required if native/PTQ 4-bit already meets residency |
| **Prototype status** | **NO Qwen prototype.** Local DSV4F path uses **source-native FP4/FP8**, not AQLM/QuIP# rewrite |
| **Measured result** | **NOT_MEASURED for basis-exec on Qwen/DSV4F serve path.** Do not cite component FP4 as “low-rank” |
| **Capability risk** | **High** at ≤2–3 bpw without careful calibration; outlier channel damage is a known failure mode |
| **Gravity implication** | **DEFER** as default body. If later admitted: Gravity must store **factors + residual + outliers** for direct exec (**no dense reconstruct**) — bible hard rule |
| **Runtime implication** | Only after residency ceiling fails under native quant |
| **Reopen condition** | Resident memory gate fails for Qwen-80B-class on target Mac **and** basis kernel beats fused quant matvec at iso-C |
| **Literature** | QuIP / QuIP#; AQLM; LoRA/adapters as ΔW (training-time, not full-body basis exec); CALDERA-style low-rank compression (NeurIPS 2024 line) |
| **Verdict** | **DEFER**; **REJECT** any “decode to dense then GEMV” design |

---

### 5.9 Active-expert reduction

| Field | Content |
|---|---|
| **Mechanism** | Margin-conditioned top-k reduction; weak-expert surrogate; early route pruning; shared correction expert; expert-group functional replacement |
| **Hypothesis** | Some tokens have a dominant expert margin; running fewer than top-8 (Qwen) or top-6 (DSV4) experts can cut B and F with small C loss if gated |
| **Expected B/F/U/R/D/S/K** | **B↓** · **F↓** · **U** mixed · **R** on hot experts · **D/S** slightly ↓ · **K** neutral |
| **Source geometry** | Qwen top-8/128; DSV4 top-6/256 |
| **Prototype status** | DSV4F **static frequency ranking only** (`static-expert-residency-receipt-v2.json`): hot banks **CANDIDATE_ONLY**, **no cache/prefetch measurement**, **no capability gate** |
| **Measured result** | Layer-4 diagnostic frequencies exist (e.g. experts 205, 168, 53, …) — **not** a reduction correctness proof. **NOT_MEASURED** dynamic margin skip |
| **Capability risk** | **Very high** without whole-model capability + route-stability gates (bible §5.9 explicit) |
| **Gravity implication** | **ADMIT_TO_GRAVITY only for hot/cold tier slots / grouping** so residency policy can pin hot experts — **not** for permanently dropping experts from the artifact |
| **Runtime implication** | **DEFER** dynamic top-k reduction until capability suite exists; hot-expert cache policy is runtime |
| **Reopen condition** | Held-out capability suite shows non-inferiority at reduced k with stable routes across seeds/prompts |
| **Literature** | MoE pruning / expert dropping; router margin confidence; draft-and-verify over experts is uncommon vs token speculation; dead-expert literature warns load imbalance |
| **Verdict** | **DEFER** dynamic reduction; **partial ADMIT_TO_GRAVITY** for hot/cold **layout tiers only** |

---

### 5.10 Conditional depth and functional replacement

| Field | Content |
|---|---|
| **Mechanism** | Verified layer skipping; early exit + correction; shared/recurrent blocks; conditional attention; functional expert/attention replacement |
| **Hypothesis** | Tokens do not need full depth; verified skip or early exit reduces D and F |
| **Expected B/F/U/R/D/S/K** | **F↓** · **B↓** (skipped layer weights) · **D↓** · **S↓** · **U** mixed · **K** careful with KV for skipped layers · **C** at risk |
| **Source geometry** | Full Qwen stack depth; DSV4F 43 layers |
| **Prototype status** | **NO production path in Hawking Ascension bootstrap.** Functional transfer / Frankenstein is a **separate** programme (out of scope; do not touch live frankenstein evidence) |
| **Measured result** | **NOT_MEASURED** for Qwen layer skip. Do not reuse Frankenstein residual-bridge numbers as Ascension skip proof |
| **Capability risk** | **Very high** without verification (LayerSkip-style self-spec or explicit verify). Training-free skip on stock Qwen is unsafe as default |
| **Gravity implication** | **REJECT** for first Qwen Gravity (would require alternate graphs / early-exit heads not in stock weights) |
| **Runtime implication** | **DEFER** to post-TG research with verified skip only |
| **Reopen condition** | Model ships native early-exit / MoD checkpoints **or** sealed verify-correct scheme matches base distribution |
| **Literature** | Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258); LayerSkip (Elhoushi et al. 2024, arXiv:2404.16710; Meta); early-exit surveys |
| **Verdict** | **DEFER** (verified research only); **REJECT** for first Qwen Gravity packing |

---

### 5.11 Command-graph collapse

| Field | Content |
|---|---|
| **Mechanism** | Many submissions/layer → few/layer → ≤8/token → ≤3/token → replayable token graph → persistent causal loop. Measure **wall and p99**, not command count alone |
| **Hypothesis** | CPU-visible waits and CB submit tax dominate wall when GPU intervals are short; collapsing CBs reduces S and host wall even when GPU sum is flat |
| **Expected B/F/U/R/D/S/K** | **B/F** neutral · **U↑** (less host stall) · **R** on persistent graphs · **D** same math depth · **S↓↓** · **K** neutral |
| **Source geometry** | All Metal token graphs (DSV4F, Qwen, GLM) |
| **Prototype status** | **MEASURED and promoted (topology) on DSV4F P4A attention block**; prior GLM token path measured; full multi-layer DSV4F still far from ≤3 CB/token |
| **Measured result (real)** | **P4A Layer-0 complete attention topology sweep (M3 Ultra, 7 clean trials):** baseline **21 CB / 21 encoders / 21 waits / 21 dispatches**, host-wall p50 **62 579 µs**, p99 **63 527 µs**; candidate **1 CB / 21 encoders / 1 wait / 21 dispatches**, host-wall p50 **54 271 µs**, p99 **54 510 µs** → **p50 win 13.28%**, **p99 win 14.19%**; GPU interval sum almost unchanged (~54.0 → ~53.9 ms) — win is **S**, not F. Status: `PASS_REAL_METAL_P4A_ONE_CB_COMPLETE_PARITY_TOPOLOGY_WIN_NOT_RUNTIME`, **promoted: true**. **Still not** a full runtime/TPS claim. **Residual tax:** L0–L42 greedy forward still **265 CB / 3 654 dispatches**. **Cross-family:** GLM functional MoE **76 → 1 CB/token**, **1.12× TPS** |
| **Capability risk** | **Low** if encoder order preserves dependencies (P4A kept ordered encoders). **High** if same-encoder concurrent hazards are assumed without proof |
| **Gravity implication** | Soft: prefer **kernel grammar / stage boundaries** that allow multi-encoder single-CB encode; recovery points declared so partial graphs can drain. No weight repack |
| **Runtime implication** | **ADMIT_TO_RUNTIME + ADMIT_TO_KERNEL (P0).** Ladder: many → few → ≤8 → ≤3 → replayable → persistent causal loop. Always report wall+p99 |
| **Reopen condition** | Full-token graph still >8 CB after wave fusion; or p99 regresses under collapse |
| **Literature** | Metal multi-encoder command buffers; CUDA graphs; BaseRT fusion of operator sequences; Hawking GLM wait-collapse lessons (`HAWKING_ASCENSION_CLOSED.json`) |
| **Verdict** | **ADMIT_TO_RUNTIME** + **ADMIT_TO_KERNEL** (evidence-backed); **ADMIT_TO_GRAVITY** only for declaring stage/recovery grammar, not tensor bytes |

---

### 5.12 Metal implementation portfolio

| Field | Content |
|---|---|
| **Mechanism** | Compare: custom SIMDgroup Metal · Metal tensor / cooperative-tensor ops · MPS · MPSGraph · generated exact-geometry shaders. Exact model geometry + one protected authority. **No path wins by assumption** |
| **Hypothesis** | Exact MoE quant geometries (FP4/FP8 blocks, top-k gather) are poorly served by generic MPS GEMM; custom SIMDgroup wins decode GEMV; MPSGraph may win some fused dense prefill ops |
| **Expected B/F/U/R/D/S/K** | Portfolio chooses max **U** at iso-**C**; B fixed by layout; S depends on graph API |
| **Source geometry** | Qwen exact shapes once admitted; DSV4F already has authority vs candidate portfolio |
| **Prototype status** | **Custom SIMDgroup + authority serial measured on DSV4F.** MPS/MPSGraph **head-to-head on same Qwen tensors: NOT_MEASURED** |
| **Measured result (real)** | Custom authority fused matvecs parity-clean. SIMDgroup split-K and act-quant candidates show large **component** speedups (**24–30×**, **~63×**) but sealed as **NOT_PROMOTED** / component-only. Brokers plan: authority is correctness-first, not speed-tuned. mHC requires **custom** Darwin-DD control path (Qwen N/A) |
| **Capability risk** | **Medium** when switching libraries (reduction order, NaN handling, FP mode). Protected authority + V2.1 / byte-exact gates mandatory |
| **Gravity implication** | **ADMIT_TO_GRAVITY:** pack for the **primary** grammar (custom fused quant SIMDgroup/authority family). Do not pack for MPS-only layouts that force token-time conversion |
| **Runtime implication** | **ADMIT_TO_KERNEL** portfolio A/B harness (already scaffolded in `broker_kernel_ab`); MPS/MPSGraph as **baseline arms**, not assumed winners |
| **Reopen condition** | Sealed head-to-head on Qwen geometry where MPSGraph beats custom at iso-C on a hot organ |
| **Literature** | Apple MPS / MPSGraph (WWDC transformer acceleration); Metal TensorOps / custom ML kernels (WWDC 2024–2026); MLX; llama.cpp Metal; BaseRT “Metal GPU API without intermediate framework” (arXiv:2607.00501) arguing hand-written fused shaders for quant LLM decode |
| **Verdict** | **ADMIT_TO_KERNEL** (custom primary); **ADMIT_TO_GRAVITY** (layout for custom grammar); **DEFER** declaring MPS/MPSGraph winners until iso-geometry receipts |

---

## Registry scoreboard

| § | Mechanism | Verdict | Gravity-blocking? | Tonight’s real evidence? |
|---|---|---|---|---|
| 5.1 | Data-local packed execution | **ADMIT_TO_GRAVITY + ADMIT_TO_KERNEL** | **Yes** | FP4/FP8 fused matvec; SIMDgroup; act quant |
| 5.2 | Projection / expert waves | **ADMIT_TO_RUNTIME + KERNEL** (+ layout partial) | Partial (tile grouping) | P6 63 disp/4 CB; concurrent groups |
| 5.3 | Cross-session amortization | **ADMIT_TO_RUNTIME** | No (KV pages later) | Not on Qwen |
| 5.4 | Cross-token invariant reuse | **ADMIT_TO_RUNTIME + KERNEL** (+ static tables Gravity) | Partial | Precompiled PSO; cold pipeline tax |
| 5.5 | Delta projection | **DEFER / REJECT first Gravity** | No | None — honest DEFER |
| 5.6 | Multi-token / block | **ADMIT_TO_RUNTIME**; DEFER MTP organs | No for v1 | None on Qwen |
| 5.7 | FLOP substitution | **ADMIT_TO_GRAVITY + KERNEL** | **Yes** | Fused unpack; act-quant cost profile |
| 5.8 | Low-rank / basis | **DEFER**; REJECT dense reconstruct | No for v1 | None as basis-exec |
| 5.9 | Active-expert reduction | **DEFER** dynamic; hot/cold **layout** admit | Partial tiers | Frequency ranks only |
| 5.10 | Conditional depth / replacement | **DEFER / REJECT first Gravity** | No | None (Frankenstein OOS) |
| 5.11 | Command-graph collapse | **ADMIT_TO_RUNTIME + KERNEL** | Grammar only | **21→1 CB, 13.3% p50** |
| 5.12 | Metal portfolio | **ADMIT custom KERNEL + Gravity grammar** | **Yes** (primary grammar) | SIMDgroup vs authority; MPS untested |

**Research-phase completion check (bible §4):** every mechanism that could **materially affect Gravity packing** has a decision: **5.1, 5.2(partial), 5.4(static), 5.7, 5.9(tiers), 5.11(grammar), 5.12** admitted; **5.5, 5.8, 5.10** deferred/rejected for first artifact; **5.3, 5.6** runtime-first.

---

## GRAVITY GATE DESIGN (§6)

### Gate law

Research findings that affect executable representation **must** be incorporated **before** final Qwen packing. A final Gravity artifact **must not** require token-time:

```text
weight repacking · full decode · layout conversion · scale reconstruction
metadata parsing · expert reindexing
```

Every artifact declares: source revision/hashes · stored bytes · complete BPW · resident bytes · active bytes/token · ops/token · sequential depth · synchronization · KV/state · capability · kernel grammar · family identity · exact restore command.

### Per-organ decisions for first Qwen Gravity (30B-A3B executor)

| Organ | Decision | Driven by | Notes |
|---|---|---|---|
| **Representation** | Source-faithful or sealed PTQ container **already fused-kernel-native** | 5.1, 5.7, 5.12 | Prefer one grammar end-to-end; no dual layout |
| **Complete BPW** | Report **complete** and **active** BPW separately | §6, 5.9 | Smaller complete BPW that raises active traffic → REJECT |
| **Protected precision** | Keep router, norms, embeddings, lm_head at protected precision until capability gate | 5.7, 5.9 | Do not force extreme quant on control organs |
| **Layout** | Row/tile order matches **fused matvec authority** (decode in-registers) | 5.1 | No token-time transpose |
| **Tile geometry** | Fixed tile pitch + alignment for SIMDgroup/TG ladders used by primary kernels | 5.1, 5.12 | Candidate geometries may vary TG; storage pitch stable |
| **Scale / codebook locality** | Scales/codebooks **co-located** with weight tiles; immutable tables as organs | 5.1, 5.4, 5.7 | No scale reconstruction |
| **Expert grouping** | Pack for **top-8 wave** gather; optional hot-expert bank grouping by layer | 5.2, 5.9 | Grouping ≠ dropping experts |
| **Hot / cold tier** | Declare tier **slots** in artifact; policy fills at runtime from route stats | 5.3, 5.9 | Frequency ranks are candidates only |
| **Basis / residual format** | **Not used** in v1 body | 5.8 | Reopen only on residency failure |
| **KV / state format** | Prefill-block-friendly pages; multi-seq ready; no quant conversion at token time | 5.3, 5.6, K | Long-ctx (Luna) makes this P0 later |
| **Kernel grammar** | **Custom Metal fused quant SIMDgroup/authority family** primary; stage boundaries allow **≤ few CB/token** encode | 5.11, 5.12 | MPSGraph optional arm, not packing authority |
| **Recovery points** | Per-layer and per-wave drain points; host intermediate handoff **forbidden** on promoted path | 5.11, 5.4 | Receipts already track `host_intermediate_handoff` |

### Explicit non-decisions (must not sneak into v1 packing)

- Delta-h projection buffers as primary path (5.5)  
- Permanent expert deletion / reduced-k without capability suite (5.9)  
- Layer-skip graphs without verify (5.10)  
- AQLM/QuIP# full-body rewrite before native path measured (5.8)  
- Any layout that needs token-time repack to satisfy a kernel  

### Declared ceilings discipline

When Qwen artifacts are later built, each must ship a filled table:

```text
memory roof   = measured_or_rated_bandwidth / B_active
compute roof  = achieved_ops_per_s / F
critical path = 1 / sequential_latency(D, S, K)
ceiling       = min(three roofs)  subject to C preserved
```

Use DSV4F lesson: **GPU µs can be fine while wall is awful** if S is high (P4A: GPU flat, wall −13.3% from CB collapse alone).

---

## FILES

| Path | Role |
|---|---|
| `workspace/docs/plans/ascension/ASCENSION_GRAVITY_RESEARCH_REGISTRY.md` | **This registry** |
| `/Users/scammermike/Downloads/HAWKING_ASCENSION_BIBLE.md` | Governing §3, §5, §6 |
| `workspace/campaign/evidence/models/frankenstein/FRANKENSTEIN_ARCHITECTURE_OPTIONS.md` | Research-registry template |
| `workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md` | Dispatch inventory + broker priorities |
| `workspace/campaign/records/runs/deepseek-v4/DSV4F_P4A_LAYER0_ATTENTION_TOPOLOGY_SWEEP-v1.json` | **21→1 CB, 13.3% p50** |
| `workspace/campaign/records/runs/deepseek-v4/fp4-metal-component-probe-receipt.json` | FP4 fused matvec parity |
| `workspace/campaign/records/runs/deepseek-v4/fp8-metal-component-probe-receipt.json` | FP8 fused matvec parity |
| `workspace/campaign/records/runs/deepseek-v4/DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP-CANONICAL-v1.json` | SIMDgroup split-K component sweep |
| `workspace/campaign/records/runs/deepseek-v4/DSV4F_ACT_QUANT_SIMDGROUP_SWEEP-v1.json` | Act quant SIMDgroup candidate |
| `workspace/campaign/records/runs/deepseek-v4/DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v2.json` | Attention stage profile / mHC cost |
| `workspace/campaign/records/runs/deepseek-v4/static-expert-residency-receipt-v2.json` | Hot/cold candidate ranks only |
| `receipts/dsv4f_multi_layer_gpu_forward_*.json` | Multi-layer CB/dispatch wall tax |
| `workspace/campaign/evidence/systems/hawking/HAWKING_ASCENSION_CLOSED.json` | GLM 76→1 CB measured |

**Not touched (contract):** `lab/operators/frankenstein_*`, live frankenstein evidence, remote/push/PR, Qwen downloads.

---

## CONFIDENCE

| Claim | Confidence |
|---|---|
| P4A 21→1 CB host-wall p50 win is **13.28%** and is real sealed evidence for 5.11 | **High** |
| Fused FP4/FP8 matvec parity (max abs 0) is real and supports 5.1/5.7 | **High** |
| SIMDgroup candidates are real **component** wins but **not** runtime promotions | **High** |
| First Qwen Gravity must decide layout/scale/tile/grammar before packing (5.1/5.7/5.12) | **High** |
| 5.5 delta projection and 5.10 conditional depth are unsafe defaults for stock Qwen | **High** |
| 5.3 cross-session amortization is the right primary HCLI lever once models are resident | **High** (literature + bible; local Qwen measure pending) |
| 5.9 dynamic expert reduction will not pass capability gates without a dedicated suite | **High-Medium** |
| Custom Metal will beat MPS/MPSGraph on exact quant MoE decode GEMV | **Medium** (strong DSV4F custom path; **no** sealed MPS head-to-head yet — portfolio still required) |
| Qwen3-Coder-30B public shape 128 experts / top-8 is correct planning geometry | **High** (HF model card / public writeups); exact Next-80B geometry **Medium** until config hash sealed |
| No mechanism above requires inventing Qwen TPS numbers to decide Gravity packing | **High** |

---

## Recommended next steps (outside this research pass)

1. **After Proto-Frankenstein offload:** admit Qwen3-Coder-30B config + small tensor windows; freeze Gravity organ table from this registry.  
2. **Kernel lane:** promote 5.11 collapse along full token graph (target ≤8 CB/token, then ≤3) with wall+p99 receipts.  
3. **Kernel lane:** act-quant + FP4/FP8 SIMDgroup A/B through `broker_kernel_ab` until parity_pass ∧ speed_improved → CandidateReady (never silent ServePromote).  
4. **Runtime lane:** N=1..8 session amortization harness (5.3) with fairness.  
5. **Do not** open 5.5/5.8/5.10 packing work until residency or capability ceilings force them.

---

*End of registry. Models produce candidates; protected controller + receipts produce verdicts.*
