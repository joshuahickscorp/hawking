> **THROUGHPUT BIBLE — ACTIVE (lean working doc).** Split 2026-05-31 from `throughput_bible_2026_05_30.md`. This is the current-state + live-forward half. Deep strategy reference (full catalog §2, superseded envelope §3, physical floor §7, system-shift moat §8), completed wins, and the kill ledger live in **[bible_archive.md](bible_archive.md)**. Canonical kill registry (every dead lever + Type + evidence) → **[../reports/dead_levers.md](../reports/dead_levers.md)**. Body sections below are verbatim as authored 2026-05-30/31; supersede with a dated successor, do not edit findings in place.

---

# `dismantle` — The Throughput Bible · ACTIVE

*Lean working reference: the thesis (§0), the methodology gate (§1), the current measured state (§3.0 — ~31 tps anchor, decode kernels closed), a status ledger of every lever's fate, the critical path (§4), the honesty caveats (§5–§6), and the breadth axis (§9). Full strategy essays + completed wins + the fence store: [bible_archive.md](bible_archive.md).*

---

## 0. The reconciled foundation (read this first or the rest misleads)

**Decode is ~85% GPU-busy. The kernels are the wall.** The "58–70% inter-dispatch idle" framing from two prior diagnoses was a measurement artifact (first a command-buffer round-trip mismeasurement, then a token-count ÷64-vs-÷32 error caught via `sample_argmax_f32`=32). The decisive physical check: both dismantle and llama.cpp read the same ~1.9 GB of Q4_K_M weights per token, so dismantle's clean ~39 tps means ~70 GB/s wall-averaged ≈ ~47% of the 150 GB/s peak. A 70%-idle GPU would require pushing all 1.9 GB during the busy 30% at ~155% of peak — impossible. The GPU is busy ~85% of the wall, running the Q4_K GEMVs at **~37–47% of peak vs llama's ~60% and MLX's higher still.** The +21% ffn_down→predec win confirms it: removing scale-decode *compute inside a kernel* couldn't help a gap-bound system, so the system is kernel-bound.

**Every speedup is one of three axes. This is exhaustive:**

| axis | mechanism | ceiling on this axis | who it helps |
|---|---|---|---|
| **1 — stream weights faster** | raise % of peak bandwidth (better kernels) | **hard wall ~85% of peak** → ~66 tps at Q4_K_M | the dense per-token cost |
| **2 — stream fewer bytes** | lower bits/weight; quant KV; smaller model | no % ceiling — raises the tps wall proportionally | the dense per-token cost |
| **3 — more tokens per stream** | amortize reads across tokens (speculation) | workload-bound multiplier on top of 1×2 | effective throughput, not per-forward |

The original program chased a phantom (the megakernel, attacking a 12–15% gap whose ceiling is ~+15%, not the +70–120% projected when the 67% gap was believed real). The corrected program is: **axis 1 to llama/MLX efficiency → axis 2 to cut the denominator → axis 3 to multiply on the product's code workload.**

**The hardware truth behind all of it:** the M3 Pro's 150 GB/s is the binding constraint — 25% below M1/M2 Pro, a fraction of M3/M4 Max (400) or Ultra (800). The same Q4_K_M kernels at 64% efficiency would do ~134 tps on an M3 Max. Nothing below changes that; it's the real ceiling the software ceilings sit under.

---

## 1. Methodology — the gate that stops a third misdiagnosis (do first, always on)

You were sent the wrong direction twice by measurement. Make it structurally impossible. Every headline number passes four physical invariants *before* it drives a decision, wired into `analyze_tcb_trace.py` as asserts that refuse to print a violating result:

1. **Busy-time bandwidth ≤ peak (150 GB/s).** This alone kills any "N% idle" claim that implies >100% of peak. The cheapest, highest-value check you have.
2. **Σ per-kernel GPU time ≈ measured GPU-busy** (within ~5%). Catches "other"-bucket and accounting drift.
3. **Token count from `sample_argmax_f32` dispatches**, never `completion_tokens`. (Session A's fix — keep it.)
4. **Bit-identical greedy parity gates every kernel change.** Already in place; never relax it for an exact lever.

Calibrate the homemade analyzer *once* against an Instruments Metal System Trace to anchor it to ground truth, then trust the calibrated tool. Report tps with a fixed thermal protocol (the bench.py contamination you've seen is real — log SoC temp, run a warmup, report median of N clean trials). **Every lever below quotes its win against this gate, not against a contaminated window.**

---

## 3. The stacked tps envelope (honest, with the ifs)

### 3.0 — Status correction (read before trusting the envelope below)

> **Added 2026-05-31. The §3 envelope below is unchanged from its 2026-05-30 authoring and is now PROVISIONAL — two corrections supersede how to read it. The original numbers are left intact for the audit trail; do not consume them without this block.**
>
> **Correction 1 — Stage 2 (MLX-class kernels) is CLOSED for DECODE, short of the ~50 target.** The overnight kernel haul (`plans/overnight_build_queue_2026_05_31.md` CLOSEOUT, commit `3cb5944`) exhausted the decode-kernel micro-optimization track: vectorized `uint4` unpack (A5 — Type-1, loads already simdgroup-coalesced), occupancy / threadgroup tuning (A6 — Type-1, BW-bound and oversubscribed), simdgroup-matrix decode (A7 — Type-1, MMA is a compute lever while decode is BW-bound and M=1 underfills the tile 7/8), and access-order weight layout (A10 — Type-1, built + measured −16.8%, `reports/a10_layout_repack_design.md`) all died. The A4 per-kernel profile (commit `f2a6a4f`) shows the predec GEMVs are 89.4% of decode at ~56% of peak BW; the lone bandwidth win was A6.5 f16-scales (+6–9%, opt-in `DISMANTLE_QWEN_PREDEC_F16SCALES`, commit `0899137`). **Conclusion: the Q4_K predec decode GEMV is at (or at the practical limit of) the Apple-GPU memory-model optimum for batch=1 decode.** Kernel micro-opt did NOT reach the ~50 dense-parity target; clean dec_tps is **~31** (≈**33–34** with the f16s flag = ~31 plus the +6–9% A6.5 delta). The path to >34 dense tps now runs ONLY through **fewer bytes** (Q3_K wiring, QTIP, or the L1.4 data-aware-SVD reframe if it resurrects) or the **spec / stateful axes** — NOT decode kernels. *Whether to keep ~50 as the dense target is an OPEN strategic decision for an attended session; this block records only that the kernel path did not reach it, not that ~50 is unreachable.*
> - **Scope limit (do not over-generalize):** this closure is for **DECODE (batch=1, M=1) only.** It does NOT apply to **PREFILL (M>1)**, where simdgroup-matrix / MMA still helps — batched prefill / TTFT (Stage 5, silicon #8) remains a separate **LIVE** lever (A7's own note: "Prefill MMA is a SEPARATE live lever … not decode"). "Kernels are dead" is false; "the *decode* GEMV micro-opt track is exhausted" is the precise claim.
>
> **Correction 2 — the §3 baseline anchor is RESOLVED to ~31** (was UNRECONCILED ~39 vs ~31). ✅ **2026-05-31 clean-room batch** (`tools/bench/clean_room_batch.sh` §B, Claude quit, greedy temp=0, 256 tok, locked fast-path): **clean dec_tps = 29.12** → −6.1% vs ~31, −25.3% vs ~39. **The ~31 recent anchor is real; the ~39 envelope was optimistic.** Consistent with A1 (paired 30.94→31.55) and A4 (31.0 median). *(Single clean run; the canonical figure is a thermal-protocol median via `clean_bench.sh` ×N — expect ~29–31.)* **Remaining attended task:** the §3 envelope below and every "% of the way to ~50" is drawn off the now-superseded ~39 anchor and must be **re-projected from ~31** — the *anchor* is settled; the *re-projection* is NOT done here. Baselines: `reports/clean_room_baselines_2026_05_31.md`.
> - **Contamination warning (first-class).** A running Claude / agent session inflates dec_tps — prior sessions measured the effect as high as **~4–5×** (the bench-contamination finding), and even the mild case here (the EAGLE paired bench reported **36.9** vs ~31 clean) is the same effect. **Absolute tps from any bench taken during an active session is untrustworthy; only PAIRED RELATIVE deltas are valid** (every overnight-haul win is reported as a paired delta for exactly this reason).

---

*The ~39-anchor envelope tables that followed §3.0 (the "% of the way to ~50" projections) are SUPERSEDED by the corrections above and moved verbatim to [bible_archive.md](bible_archive.md) §3 for the audit trail. Re-projection from the ~31 anchor is an open attended task.*

---

## Lever status ledger (the fence — every lever's fate at a glance)

*One row per lever so the active doc shows what was tried and how it closed without carrying the body. Full evidence is the cited archive section + the canonical registry [../reports/dead_levers.md](../reports/dead_levers.md). Legend: `LIVE` = forward work; `SHIPPED` = landed (see archive Completed-wins ledger); `FUTURE` = oracle-gated, not yet run; `DEAD-T1` = Type-1 (a fact about reality — never re-test); `DEAD-T2-parked` = Type-2, alive only behind a named oracle; `HELD` = built, below ship gate.*

### Live / working set
| lever | status | note | where |
|---|---|---|---|
| QTIP 3-bit lookup-free trellis (byte-cut) | **LIVE** | the sole sub-Q4 byte-cut bet; quality-oracle-gated | archive §2 Axis-2; `plans/qtip_bytecut_design_2026_05_31.md` |
| n-gram / SAM speculation | **LIVE** | τ=1.43 on code, free CPU draft, lossless; the kept spec option | archive §2 Axis-3, §8 L2.3 |
| prefix cache (exact KV reuse) | **SHIPPED · default-on** | the moat leg; ~84% prefill cut, bit-identical | wins ledger; archive §8 L1.2 |
| online draft tuning (L3.1) | **LIVE · GO** | user-ngram warm-start; +148% on repetitive code; gated | archive §8 L3.1 |
| architecture-family breadth (§9.1) | **LIVE · next** | the do-first breadth move after the live work; parameterize the loader | §9.1 (this doc) |
| imatrix mixed-precision | **FUTURE** | the cheap byte cut, +12–20%; code-imatrix KL/PPL oracle | archive §2 Axis-2, §7.3.a |
| prefill-MMA / TTFT (rows>cols) | **LIVE** | simdgroup-MMA is GO for prefill (M>1); dormant in the predec path, needs a predec-MMA twin | dead_levers (Q4_K batched MMA); `plans/p1_prefill_mma_integration_handoff_2026_05_31.md` |
| fused quantized-KV attention | **FUTURE** | neutral at short ctx, big at >16K | archive §2 Axis-2, §7 |
| on-device LoRA personalization (L3.3) | **LIVE · capability bet** | the brand moat; on-device training | archive §8 L3.3 |
| KV working-set *lossless mode* | **LIVE (no-op escape hatch)** | ships regardless; eviction itself is dead (below) | archive §8 L1.1 |
| scheduler / host-loop, energy axis, load path (L4.x) | **LIVE/FUTURE** | recover earned tps; own the joules-per-token brand | archive §8 L4.1–L4.3, §7.5 |
| early-exit / dynamic depth (L2.1) | **FUTURE** | L-conf on a 3B; argmax-agreement oracle | archive §8 L2.1, §7.3.b |
| workload-adaptive quant (L3.2) | **FUTURE** | per-user imatrix; oracle-gated | archive §8 L3.2 |
| co-design / heal the model (L5.1) | **FUTURE · unifier** | trains-in the structure the post-hoc levers couldn't find | archive §8 L5.1 |
| native dismantle format (§9.3) | **DEFERRED capstone** | consolidation + bounded layout, NOT a tps unlock; build last | §9.3 (this doc) |

### Dead / parked (the kill fences — do not re-derive)
| lever | verdict | one-line cause | evidence |
|---|---|---|---|
| decode-kernel micro-opt (A5 uint4 / A6 occupancy / A7 MMA / A10 layout) | **DEAD-T1** | the Q4_K predec decode GEMV is at the Apple-GPU memory-model optimum (M=1, BW-bound); A10 layout −16.8% | §3.0 (this doc); dead_levers; closeout `3cb5944` |
| EAGLE-3 trained draft head | **NO-GO** | τ=0.877 < 2.5 gate; the free n-gram (τ=1.43) beats it; spec net-negative on Qwen-3B+code | dead_levers; archive §2 Axis-3 |
| L1.1 KV working-set eviction (StreamingLLM/H2O/SnapKV) | **DEAD-T1** | attention mass is diffuse — 99% needs 78–92% of positions on Qwen-3B+code | dead_levers; archive §8 L1.1 |
| L1.2 semantic cache (beyond exact prefix) | **DEAD-T2-parked** | +1.48 mean / +0.00 median over the shipped exact cache on the git-history proxy; re-run on real file-interleaved logs | dead_levers; archive §8 L1.2 |
| L1.3 cross-layer weight delta | **DEAD-T1** | layers orthogonal (cos≈0), delta anti-compressible (std-ratio 1.61); the data-aware reframe also died | dead_levers; archive §8 L1.3 |
| L1.4 low-rank + residual codec | **DEAD-T1** | not low-rank (top-64 = 3–9% FFN energy); data-free AND data-aware SVD both died | dead_levers; archive §8 L1.4 |
| L1.5 learned per-model codebook | **DEAD-T1** | raw k-means = random LUT gather; no HW gather on Apple Silicon | dead_levers; archive §8 L1.5 |
| L2.2 contextual FFN sparsity | **DEAD-T1** | block-256 / permute / dynamic all die — 99% energy needs 39–53% of neurons, scattered | dead_levers; archive §8 L2.2 |
| L3.1 vocab norm-bound screen | **DEAD-T1** | 0% certified-fast-path; high-norm head rows are rare tokens (anti-correlation) | dead_levers; archive §8 L3.1 |
| Q3_K sub-Q4 decode byte-cut | **DEAD-T1** | compute-bound (33 GB/s = 22% peak); fewer bytes ≠ speed when not BW-bound | dead_levers |
| W4A8 default decode path | **HELD** | quality-blocked (20% bit-id) + paired 1.115× < 1.20× gate | dead_levers |
| host-side per-dispatch family (ICB / concurrent-encoder / PSO / megakernel) | **DEAD** | CPU encode 0.51% of wall; the gap is real GPU-side | dead_levers |
| MLA Phase-4 simdgroup attention | **DEAD** | −1.7–2.5% regression; attention is only 2.4% of decode | dead_levers |
| older kills (f16 residual, MoE serial/megakernel, Q5_0 shuffle, Q8-KV layer-diff, Eagle5 v1, spec ExactShared, LM-head simdmat, predec-4r) | **DEAD/parked** | each with its own evidence + resurrection check | dead_levers |

---

## 4. The critical path (so "do it all" stays focused)

Dependencies matter: some levers gate others. This is the order that maximizes expected tps per unit risk.

**Stage 0 — Instrument & oracle (days, mostly free, much of it parallel on Colab/local).** Wire the §1 invariant gate + one Instruments calibration. Run the three cheap offline oracles now, before any kernel: (a) **spec acceptance** on real code transcripts (decides n-gram and EAGLE-3 viability — the single most informative number for the whole axis-3 ceiling); (b) **SVD lm_head recall** (decides if screening adds anything beyond lm_head→predec); (c) **mixed-precision KL/PPL** from a code-calibrated imatrix (decides the cheap byte cut). These are NumPy/llama.cpp-tool afternoons and they rank the rest.

**Stage 1 — Bank the cheap, exact wins (~1 session, M3).** predec everywhere + ILP default + LM-head→predec + hoist-constants audit. ~+8–12%, bit-identical, zero risk. → ~43–45 tps.

**Stage 2 — The primary lever: MLX-class Q4_K GEMV (the core of the program, M3).** ⚠ **STATUS 2026-05-31 — CLOSED for DECODE, short of the ~50 target (see §3.0).** The decode-kernel micro-opt track is exhausted: vectorized unpack (A5), occupancy (A6), simdgroup-matrix decode at M=1 (A7), and access-order layout (A10) are all Type-1 dead for batch=1 decode; the lone win was f16-scales (A6.5, +6–9%, opt-in). Clean dec_tps ~31 (≈33–34 with f16s) — the decode GEMV is at the Apple-GPU memory-model optimum. Remaining dense tps comes from fewer bytes (Q3_K / QTIP) or the spec / stateful axes, NOT decode kernels. **This closure is DECODE-ONLY — simdgroup-matrix / MMA stays a LIVE lever for PREFILL (M>1, Stage 5 TTFT).** *(The original Stage-2 plan is retained verbatim below for the record; it is the now-closed decode path.)* Vectorized unpacking → multi-row register blocking → simdgroup-matrix decode → split-K → threadgroup tuning → coalesced repack. Parity-gated, each A/B'd against busy-time bandwidth. This is the work that takes you from ~47% to llama's ~60% (→~50, high confidence) and reaches for MLX's ~70–80% (→~55–64). **This is where the bulk of the time goes and where the dense ceiling is decided.**

**Stage 3 — Cut the denominator (Colab→M3).** Once Stage 2 has the kernels bandwidth-bound: ship **mixed-precision** first (cheap, existing codecs, code-calibrated imatrix) for ~+12–20%; then build the **QTIP 3-bit** lookup-free kernel for ~+30–40% if its quality oracle passed. → ~70–85 dense.

**Stage 4 — Speculation, the multiplier (M3, then Colab).** Ship **n-gram/SAM** first (free draft, lossless, code-ideal) gated on the Stage-0 acceptance oracle. Then, with the kernels bandwidth-bound (idle compute available) and acceptance confirmed on code, **train EAGLE-3 on Colab** (FR-Spec 32K vocab, fusion layers {2,18,33}) and integrate. → effective triple digits on the product's code workload.

**Stage 5 — Product-specific (M3).** The target coding workload does file-context querying → **long context** → build the **fused quantized-KV attention** kernel (real byte cut at >16K, neutral short) and optimize **prefill/TTFT** (large micro-batch + simdgroup-matrix, which shines at M>1 prefill — the easy MMA regime). These don't move the short-prompt benchmark but they move the product's actual latency.

**Deliberately deprioritized (with evidence):** the **megakernel / persistent GPU loop** — attacks the ~12–15% gap (ceiling ~+15%), weeks of coupled hard problems, and the persistent variant is blocked on Metal (no `grid.sync()`; cross-threadgroup = atomic-spin with no forward-progress guarantee). Revisit *only* if Stage 2 plateaus well below 60% and the Instruments trace shows a fixable structural gap. **CPU gap-fill** — near-dead now that the gap is ~12–15%, not 67%.

---

## 5. The Colab / local division of labor

You said Colab+ is available; that unlocks the offline-heavy levers that were previously gated:

- **Colab (offline, GPU-heavy, one-time):** EAGLE-3 / MTP draft training (~hours); QTIP quantization + incoherence fine-tuning; code-calibrated imatrix computation; pruning + distillation healing; the SVD/entropy oracles (faster than local).
- **M3 (the engine itself):** all Metal kernel work (axis 1); mixed-precision precision-assignment + bench; the n-gram/SAM runtime and the fused quantized-KV kernel; integrating any Colab-trained draft into the Rust/Metal runtime; *all* benchmarking under the §1 gate (only the M3 numbers are real).

The boundary is clean: **Colab produces artifacts (a trained draft, a quantized weight file, an imatrix); the M3 produces the kernels and the verdicts.** Check whether SpecForge/SpecBundle already ships a Qwen2.5-3B EAGLE-3 draft before training your own — it may save the Colab run entirely.

---

## 6. Caveats (the bible stays honest)

- **The physics is fixed.** ~66 tps is the Q4_K_M wall on 150 GB/s; ~99 is the 3-bit wall. Triple-digit *dense* tps is not available on this chip at usable quality. Triple-digit *effective* tps requires speculation on a favorable (code) workload. An M3/M4 Max would ~2.6× every dense number — the largest single lever is hardware, not software.
- **The stretch above 60% is real but not free.** MLX proves >60% on small models, but batch=1 decode is the worst regime for the matrix units (M=1 underfills the tile); the ~70–80% stretch is genuine kernel engineering with medium confidence, not a switch.
- **Speculation is workload-conditional and, for EAGLE-3, regime-gated.** The mlx-lm caveat is explicit: small fast models on Apple Silicon may not amortize draft overhead. n-gram/SAM de-risks this (free draft); EAGLE-3 is gated on post-Stage-2 idle compute + a code-acceptance oracle. Quote effective tps with the workload named.
- **Mixed-precision and QTIP are quality trades, oracle-gated.** Auto bit-allocation beats naive only ~70% of the time; 3B quality degrades hard below ~3 bits. KL/PPL on a real code corpus is the gate, and it's run on Colab before any kernel.
- **QTIP competes with kernel efficiency.** Its trellis decode adds compute; at high % of peak that can pull you back toward compute-bound, so the axis-1 × axis-2 stack is sub-multiplicative. Mixed-precision (affine codecs) stacks more cleanly.
- **Only M3 numbers under the §1 gate count.** Colab-side perplexity and acceptance are necessary oracles, not throughput claims. The bench.py series should report dense tps under the thermal protocol; effective (spec-on) tps is reported separately with the workload stated.
- **The diagnosis itself is the most load-bearing claim, and it's solid:** the busy-time-bandwidth invariant rules out a large idle gap independent of any tool, and the +21% predec win corroborates kernel-boundedness directly. If a future trace ever shows busy-time BW implying a large idle fraction, that violates §1.1 and means the measurement is wrong, not the physics.

---

### The one-paragraph version
Decode is kernel-bound, not gap-bound. Spend the program on three axes in order: make the Q4_K GEMV kernels MLX-class (→~50 high-confidence, ~55–64 stretch); cut the denominator with code-calibrated mixed-precision then 3-bit QTIP (→~70–85 dense, with a ~99 wall); and multiply with speculation on the product's code workload — n-gram/SAM first (free, lossless), EAGLE-3 once the kernels are bandwidth-bound (→ effective triple digits on copy-heavy code). Do the offline training/quantization on Colab, the kernels and the verdicts on the M3, and gate everything behind the four physical invariants so the measurement never lies to us a third time. The deterministic ceiling on this chip is ~99 tps; the hardware is the real wall above that.

---

## §9 — The Breadth Axis (model coverage, layout, and the deferred native format)

*Companion section to the Throughput Bible. The bible's other axes ask "how fast / how alive can this engine be." This axis asks "how does dismantle avoid being stuck on one model — and is there a native file format worth building?" It exists because single-model specialization is fragile (the model obsolesces; users want choice), but unbounded breadth turns dismantle into a worse general engine. Same discipline as the rest of the bible: every lever carries mechanism, benefit, exact-vs-quality-trade, effort tier, confidence, sequencing gate, and an honest throughput verdict. The load-bearing correction is stated once up front and enforced throughout: **a file format does not fix tps; only fewer bytes/token or cheaper decode does. The format's real throughput contribution is the bounded "last-mile" layout win (§7), not a new tier of speed.***

---

### 9.0 — The framing (read before trusting any "format fixes tps" intuition)

Throughput at batch=1 decode is set by exactly two things: **how many bytes cross the memory bus per token**, and **how cheaply the kernel decodes them.** Storage layout that never changes either of those is free to redesign and will not move tps. So the test for *any* breadth/format idea is one question: **does this change the bytes that flow, or the decode cost of those bytes?**

- If **yes** → it is a real throughput lever (and it belongs in the byte-cut axis, or the bounded layout lever below).
- If **no** → it is ergonomics, consolidation, or coverage. Valuable, but it will not add tps by existing.

By that test:
- **A new container holding the same quantized weights → zero tps change** (same bytes flow). This is the trap: do not build a format expecting speed; the speed was never in the container.
- **A weights-on-disk layout that matches the kernel's exact access order → real but bounded tps** (raises *achieved* bandwidth toward the memory-controller ceiling; this is the §7 "last mile," single-digit-to-low-double-digit %, exact, no quality cost). And crucially this is a **repack pass**, available *without* inventing a format.
- **A better quantization or trained-in structure → real tps** — but that is QTIP / co-design (the byte-cut and L5.1 axes), and it can ride in *any* container. The format does not create it.

The measured ground truth this rests on: decode is at the Apple-GPU memory-model optimum (~31 tps, kernels closed), the ceiling is the M3's 150 GB/s and the read-once nature of batch=1 decode, and the prior layout attempt (A10) measured **−16.8%** because a layout helps *only* when it matches the kernel exactly. So: **the missing throughput is not trapped in GGUF.** It is the bus and the workload. The format makes you use the bus you have slightly better and holds your specialization elegantly; it does not hand you a bus you do not have.

**Therefore this axis is, in priority order:** (1) architecture-family breadth — the high-value, do-first expansion that ends single-model fragility *without* a new format; (2) the access-order layout repack — a real, bounded, exact tps lever that lives in §7 and needs no format; (3) the native format — a deferred **capstone** that consolidates the engine's specialization (codebook, LoRA deltas, calibration, cache state) and bakes in the layout win, explicitly **not** a tps unlock, built last once its payload exists.

---

### 9.1 — Architecture-family generalization (the breadth move — do this first, after the live bible work)

- **Mechanism:** generalize dismantle from "Qwen2.5-3B specifically" to "any dense Llama-family transformer." Qwen2.5, Llama, Mistral, Gemma, Phi and most of what matters are the same architectural shape (dense decoder, RMSNorm, RoPE, SwiGLU/GeGLU FFN, GQA/MHA attention). The work is **parameterizing what currently may be hardcoded** — hidden dim, layer count, head counts (Q/KV for GQA), vocab size, intermediate dim, RoPE base/scaling, norm variant (RMSNorm vs LayerNorm), activation (SiLU vs GELU), tie-vs-untied embeddings — and reading them from the model's existing GGUF metadata. The kernels already perform the operations; this exposes the dimensions instead of baking them in.
- **Benefit:** ends single-model fragility and delivers **breadth-of-models for free within the family** — a user can run Llama-8B or Mistral-7B instead of Qwen-3B on the *same* kernels, *same* stateful machinery (prefix cache, draft tuning), *same* (eventual) format. Critically, **the whole moat applies to every model in the family equally** — personalization on a *menu* of models beats personalization on one. This is "widen the specialization to the architecture, stay specialized in the philosophy."
- **Exact?** E (no numerical change to any single model's output; this is parameterization, not approximation). Each newly-supported model still passes the bit-identity / parity gate against a reference.
- **Effort tier:** **MODERATE, mostly DETERMINISTIC** (not research-gated like the kernels were). Weeks, not months. The risks are integration breadth (each architectural knob is a code path to test), not unknown physics. Bigger models (7–8B) also pull more bytes/token, so re-confirm the bus math per model — a 7B at Q4 is ~2× the bytes of the 3B, so its dense tps will be proportionally lower; that is expected, not a regression.
- **Confidence:** **H** that it works (the operations exist; this is plumbing). M on how many architectural variants are clean vs. fiddly (GQA group counts, RoPE scaling schemes, and tied-embedding handling are the usual sharp edges).
- **Sequencing gate:** AFTER the live bible work (batched-verify spec unblock, QTIP quality oracle, anchor/consistency pass). It does not block on them, but it should not pre-empt them — they are the throughput-defining moves; this is the coverage move.
- **Throughput verdict:** NEUTRAL per-model (it does not speed up any single model; it adds *which* models you can run). The stateful/personalization wins then apply across all of them.
- **Validation oracle (cheap, before broad rollout):** pick ONE second model in the family (e.g. Llama-3.x-8B or Mistral-7B GGUF), wire the parameterized loader, and confirm (a) it loads from GGUF metadata with no hardcoded-dim failures, (b) it passes greedy parity against a reference (llama.cpp output on the same prompt/seed), (c) the existing prefix-cache and draft-tuning machinery attach unchanged. If those three hold on the second model, the family generalization is sound; expand the supported list incrementally, parity-gating each.
- **Where:** M3 (engine — loader + kernel parameterization + per-model parity).

#### 9.1.a — Scope discipline (what breadth is NOT)
Breadth is **architecture-family**, not "run everything." The moment dismantle tries to support every model and format llama.cpp does, it becomes a worse llama.cpp — generality is *their* moat, not yours. Explicitly out of scope unless a later strategic decision reopens them: non-dense architectures (MoE routing, Mamba/SSM, encoder-decoder), exotic quant formats outside what the byte-cut axis builds, and non-Apple hardware. The rule: **widen to the architecture you are already good at; do not widen to a worse version of the general engine.**

---

### 9.2 — Access-order layout repack (a real, bounded tps lever — lives in §7, needs no format)

- **Mechanism:** lay the weights on disk (or in a one-time repacked artifact) in the **exact order the decode kernels stream them**, with alignment and grouping tuned to the Apple GPU's coalescing and the M3's DRAM row-buffer behavior, so the memory controller is not fighting the access pattern. Raises *achieved* bandwidth toward the ~85–90% practical ceiling.
- **Benefit:** a genuine throughput win at the layout level — exactly the "last mile of the bus" from §7. Closes part of the gap between current achieved bandwidth (~56% of peak per A4) and the practical ceiling.
- **Exact?** E (pure data movement; bit-identical).
- **Effort tier:** SMALL-to-MODERATE. The prior attempt (A10) is **not dead — it measured −16.8% because the layout did not match the kernel exactly.** A layout helps only when it is co-designed with the precise kernel access pattern; the lever is "get the match right," not "try layout." So the work is: profile the exact streaming order of the current predec GEMVs, repack to match, re-measure.
- **Confidence:** M (the mechanism is real and documented in §7; the magnitude on Apple's specific controller is "measure, don't quote" — could be a few %, could be ~10%, and A10 shows it can go *negative* if mismatched).
- **Sequencing gate:** independent — can be tried any time as a repack pass. **Does NOT require the native format** (it is a repack of the weights you already load). It will later be *baked into* the format (§9.3), but it is available first as a standalone repack.
- **Throughput verdict:** **GENUINE but BOUNDED** — single-digit-to-low-double-digit %, exact, no quality cost. This is the *entire* legitimate "format fixes tps" claim, and it is the last 10%, not the missing 50%.
- **Validation oracle:** the §1 busy-time-bandwidth meter with/without the reordered layout, parity-gated. Positive Δ achieved-GB/s ⇒ keep; A10's −16.8% is the cautionary baseline for a mismatched layout.
- **Where:** M3 (offline repack + bench). Cross-reference: this is the §7.1.a lever (layout = access order); recorded here because it is the *only* part of "a native format" that touches throughput.

---

### 9.3 — The native dismantle format (DEFERRED CAPSTONE — consolidation + bounded layout, NOT a tps unlock)

- **What it actually is:** a versioned, dismantle-native artifact that holds, as **one living per-user file**, what GGUF structurally cannot: the quantized weights **+** a learned per-model codebook (if QTIP/codebook lands) **+** a low-rank/residual split (if co-design produces one) **+** the user's personalization LoRA deltas **+** the user-calibrated importance/quant map **+** prefix-cache / KV state metadata — with the access-order layout (§9.2) baked in.
- **Benefit (stated honestly, by category):**
  - **Consolidation / ergonomics (the real reason to build it):** GGUF is a *general, static* container — it has nowhere to put dismantle's specialization, so today that specialization is bolted on via sidecars and runtime state. A native format makes the engine's "alive" artifacts (codebook, LoRA, calibration, cache) **first-class and versioned together**, instead of duct-taped. This is the philosophy-aligned win: the format treats the model as a living, personalized, structured artifact because that is what dismantle treats it as.
  - **Bounded throughput (via §9.2 only):** baking the access-order layout in gives the §7 last-mile win as a property of the file. This is the *single-digit-ish* layout gain, nothing more.
  - **What it does NOT do:** it does not make Q4_K weights smaller (only a better *quant* does — QTIP, which rides in any container), it does not break past the memory-controller ceiling, and it does not change that batch=1 decode reads each weight once. **The format is layout + container, not new physics.**
- **The honest proof that it is not a tps unlock:** the planned **GGUF→dismantle converter** (below) is *lossless*. If conversion is lossless, the dismantle file contains no *information* GGUF lacks — it is the same weights in a better arrangement plus room for your extras. A lossless converter being *possible* is the proof that the format is re-layout + payload-attachment, not magic decompression. Throughput from a container that holds the same information is a category error.
- **GGUF→dismantle converter (the adoption bridge — build WITH the format):** a one-command `dismantle convert model.gguf → model.dismantle` that re-lays-out the weights (the §9.2 win) and attaches/initializes the specialization payload. **Rationale (correct, per the founder's instinct):** users with large GGUF libraries are wary of change and will not re-download; a frictionless converter lets them opt in, keep their models, and adopt incrementally. It is also a de-risking demo of the whole format (round-trip GGUF→dismantle→parity proves the format holds the weights faithfully).
- **Exact?** E for the weight/layout content (lossless round-trip, parity-gated); the personalization payload is Q (by design — it is the adaptation).
- **Effort tier:** **LARGE** — a format is a spec + writer + reader + converter + versioning + migration tooling, and **formats are forever** (you maintain it and its migrations indefinitely). But it is *engineering, not research* — estimable, not physics-uncertain. Plan a multi-week-to-month v1.
- **Confidence:** H that it is buildable; the open question is *worth-it timing*, not feasibility.
- **Sequencing gate (critical — do NOT build early):** **build it LAST, once its payload exists.** Today, half the intended contents are unproven — QTIP's codebook is pending its quality oracle, low-rank died on the stock model (alive only via co-design L5.1), personalization is the untested bet. **A format is the crystallization of decisions not yet made.** Building it now means designing a container for contents that may not exist; building it after QTIP / co-design / personalization resolve makes it the natural, consolidating home for confirmed wins. **Gate: native format begins only after (a) the byte-cut path is settled (QTIP GO/NO-GO), (b) the personalization oracle has reported, and (c) co-design (L5.1) is at least scoped.** Until then, sidecars + the standalone §9.2 repack carry the load.
- **Throughput verdict:** NEUTRAL-plus-bounded — the only tps it adds is the §9.2 layout win it bakes in; everything else it does is consolidation and ergonomics. **Do not attribute future tps hopes to the format; attribute them to QTIP and co-design, which the format merely stores.**
- **Where:** spec + tooling (M3 for the engine-side reader/writer; converter is CPU).

---

### 9.4 — Sequencing within the Breadth axis (relative to the rest of the bible)

1. **Finish the live bible work first** (batched-verify spec unblock, QTIP quality oracle, anchor reconciliation + the §6/roadmap consistency pass + ~50-goalpost decision). The Breadth axis is the post-reevaluation expansion, not a detour from the throughput-defining moves.
2. **Then §9.1 — architecture-family generalization.** The high-value breadth move: ends single-model fragility, multiplies the moat across a menu of models, moderate/deterministic effort, needs no format. Validate on one second model, then expand parity-gated.
3. **§9.2 — access-order layout repack** can be slotted opportunistically any time (it is a §7 lever); it is the only throughput-bearing piece of the format story and it stands alone as a repack.
4. **§9.3 — native format LAST**, gated on QTIP + personalization + co-design resolving, built as the consolidating capstone (with the GGUF converter as the adoption bridge), explicitly framed as layout + container, not a tps unlock.

---

### 9.5 — The one correction to carry forward

**Better resource use at the format level is real, but it is the last 10%, not the missing 50%.** The missing throughput is not trapped in GGUF — it is the M3's 150 GB/s bus and the read-once nature of batch=1 decode. The only levers that genuinely move tps are **fewer bytes** (QTIP, co-design) and **more tokens per byte** (speculation). A native format (a) widens nothing by itself — breadth comes from §9.1 architecture generalization, and (b) speeds up only by the bounded §9.2 layout amount it bakes in. Build the format for what it actually does — **consolidation, ergonomics, a frictionless GGUF on-ramp, and the last-mile layout win** — and it is genuinely worth building, *later*. Build it expecting it to fix tps, and it will disappoint you for a reason that was never its fault.

*Companion docs: the Throughput Bible §0–§8 (physics, kernel ceiling, stateful shift), §7 (physical floor incl. the layout lever this section cross-references), and the roadmap. This §9 is the breadth + coverage axis, queued behind the live throughput work.*

---

## Where the rest lives

The deep strategy reference and the fence store are in **[bible_archive.md](bible_archive.md)** (verbatim, long by design):

- **Completed-wins ledger** — every shipped throughput win with its commit hash.
- **§2 — the full lever catalog** (Axis 1/2/3 tables, with each lever's mechanism / exact / ceiling / confidence / oracle).
- **§3 — the stacked tps envelope** (the ~39-anchor projections, **SUPERSEDED** by §3.0 above; kept for the audit trail).
- **§7 — Approaching the Physical Floor** (7.0–7.9: the bus last-mile, heterogeneous spatial speculation, sub-3-bit, the power/thermal envelope, the model-level lever, the verified-facts ledger).
- **§8 — The System-Level Shift** (8.0–8.4: the stateful/adaptive/specialized moat, the fifteen levers L1.1–L5.1, the tiering, and **§8.3.1 the Kill Protocol** + its worked-example kill ledger).

Canonical kill registry (the authoritative fence, updated when any lever dies): **[../reports/dead_levers.md](../reports/dead_levers.md)**.

*Note: `CLAUDE.md` cites the Kill Protocol as "§8.3.1 of `throughput_bible_2026_05_30.md`" — that path is now a redirect stub; the protocol is verbatim in [bible_archive.md](bible_archive.md) §8.3.1.*
