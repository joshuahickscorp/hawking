> **STATUS: CANONICAL — operative throughput strategy as of 2026-05-30.**
> Supersedes the gap/megakernel framing in [silicon_architecture_audit_2026_05_29.md](silicon_architecture_audit_2026_05_29.md) (Topics 2 & 7) and the contaminated 26.6-tps anchor.
> Reconciliation crosswalk (this doc ↔ audit plan ↔ silicon-builds): [throughput_bible_reconciliation_2026_05_30.md](throughput_bible_reconciliation_2026_05_30.md).
> Empirical backing for axes 1–2: [../silicon-builds/SUMMARY.md](../silicon-builds/SUMMARY.md) (#8 simdgroup-MMA LIVE, #13 zero-copy LIVE, #16/#17 byte-cut prize).
> Body below is verbatim as authored 2026-05-30. Do not edit in place — supersede with a dated successor.

---

# `dismantle` — The Throughput Bible

*The single consolidated reference for pushing single-stream Qwen2.5-3B decode on M3 Pro to its limit. Everything viable is in here, tiered by axis, with a mechanism, a realistic ceiling, a confidence, a run-before-you-build oracle, a dependency, and whether it runs on Colab (offline) or locally on the M3. This is a map of the whole territory — not a claim that we do it all at once. The critical path at the end says what to do in what order. Maximal scope; the physics ceilings still bind, and we exploit them rather than wish them away.*

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

## 2. The full lever catalog

Notation: **Ceiling** = realistic standalone delta (deltas don't simply add — see §3). **Exact?** = bit-identical greedy (E) or quality trade (Q). **Where** = Colab (offline/GPU-heavy/one-time) or M3 (engine/kernel/bench).

### Axis 1 — Stream the weights faster (raise % of peak; wall ~85% → ~66 tps at Q4_K_M)

| lever | mechanism | acts on | exact | ceiling | conf | oracle (before building) | where |
|---|---|---|---|---|---|---|---|
| **predec everywhere + ILP default** (Phase 1) | finish hoisting Q4_K scale-decode to load; make 2-row ILP default on q/o/ffn_down; LM-head→predec | the 82% predec-kernel time | E | **+8–12%** | **H** | already proven (ffn_down +21%, 2r +4%); just generalize | M3 |
| **hoist-constants audit** | the predec lesson generalized: stop recomputing any per-dispatch constant (RoPE tables, norm consts, setup) | per-token fixed compute | E | +1–4% | M | grep the per-token path for inline constants | M3 |
| **vectorized nibble unpacking** | load weights as `uint4`/packed, unpack 8 nibbles + convert via SIMD instead of scalar | predec-kernel compute | E | part of +30–50% | **H** | micro-bench one kernel scalar vs vectorized | M3 |
| **multi-row register blocking (4–8 rows/simdgroup)** | more independent accumulator chains hide memory latency; M3 Dynamic Caching gives register-heavy kernels better occupancy than M1/M2 | latency hiding | E | part of +30–50% | **H** | sweep rows-per-simdgroup vs register pressure | M3 |
| **simdgroup-matrix decode** | use `simdgroup_float8x8` tiles for the dequant-accumulate; fill the tile by processing multiple output rows (M=1 underutilizes — the hard part) | the core matmul | E | the MLX-class step | **M** | prototype on one GEMV; measure busy-time BW | M3 |
| **split-K reduction for big FFN GEMVs** | split the reduction dim across threadgroups to saturate all ~18 cores (mlx-qsdpa does this for attention) | core occupancy on gate/up/down | E | a few % | M | A/B split factor on ffn_down | M3 |
| **per-quant-type threadgroup tuning** | size threads-per-threadgroup as a multiple of `threadExecutionWidth`, tuned per kernel (llama's documented edge) | occupancy | E | a few % | **H** | sweep; it's a constant search | M3 |
| **coalesced predec layout repack** | offline-repack weights+scales for fully-coalesced vectorized loads (use `q4k_fast`) | effective BW | E | a few % | M | inspect access pattern; A/B repack | M3 |
| **stacked-QKV single GEMM** | concat [Wq;Wk;Wv] into one matrix/dispatch (distinct from the +1.68% concurrent-encoder attempt) | core saturation + 1 dispatch | E | +1–3% | M | one-off: build stacked matrix, bench | M3 |
| **GPU-side sampling** | keep argmax/sampling on GPU to drop a per-token CPU round-trip (a dispatch + sync) | the ~12–15% gap | E | +1–2% | L | count the per-token sync; the gap is small | M3 |

**Axis-1 honest ceiling:** matching llama (~60–64%) → **~50 tps, high confidence (existence proof).** Reaching MLX-class (~70–80%) → **~55–64 tps, medium confidence** — and the evidence says this headroom is *most* available precisely for small models like a 3B (MLX's lead over llama widens at small sizes, collapses at 27B+ where both saturate bandwidth). The hard wall at Q4_K_M is **~66 tps**; you cannot stream these bytes faster on 150 GB/s. The winning Metal idiom across MLX/llama is consistent: **fuse + inline-dequant + single-dispatch + simdgroup ops.**

### Axis 2 — Stream fewer bytes (lower the denominator; no % ceiling)

| lever | mechanism | acts on | exact | ceiling | conf | oracle | where |
|---|---|---|---|---|---|---|---|
| **imatrix mixed-precision (existing codecs)** | code-calibrated importance matrix → per-tensor bit allocation (sensitive→Q5/Q6, robust→Q3) using the Q3/Q4/Q5/Q6 codecs you already have; **no new kernel** | bytes/token | Q (tiny) | ~3.8–4.0 eff bits → **+12–20%** | **M-H** | compute imatrix on code corpus; quantize; KL/PPL vs Q4_K_M | Colab→M3 |
| **QTIP 3.0–3.25 lookup-free trellis** | near-Gaussian-optimal sub-4-bit; lookup-free contiguous decode (no gather — Apple-friendly); spends surplus ALU for fewer bytes | bytes/token | Q | ~3.0 bits → **+30–40%** (net, minus decode compute) | M | quantize on Colab; KL/PPL; **must be bandwidth-bound first** | Colab→M3 |
| **fused quantized-KV attention** (mlx-qsdpa pattern) | read 4/8-bit KV inline in one dispatch, no FP16 buffer; **only matters at long context** (TailorAI file context) | KV bandwidth + memory at long ctx | Q (KV) | ~0% short ctx; **big at >16K** (mlx-qsdpa 1.7× @128K) | M | measure KV's byte share at the product's real context lengths | M3 |
| **structured pruning + distillation** | prune layers/dims or distill to a faster student; heal on Colab — the most direct "fewer bytes" (smaller model) | total bytes/token | Q (real) | model-dependent, potentially **1.3–2×** | L | prune candidate, eval downstream task vs full | Colab→M3 |
| **QTIP 2-bit** | push to ~2 bits | bytes/token | Q (heavy) | larger, but 3B quality floor | **L** | KL/PPL — likely fails the gate on a 3B | Colab |

**Axis-2 note:** mixed-precision is the *cheap* byte cut (no new kernel, just precision assignment + your existing codecs, calibrated on code) and stacks cleanly with axis 1 because affine dequant adds little compute. QTIP is the *deep* byte cut (new kernel, lower bits, better quality-per-bit) but its trellis decode competes with axis-1 efficiency, so it pays only once the kernels are bandwidth-bound. 3-bit raises the tps wall from ~66 to **~99 (practical ~85%)**, which is what makes triple digits physically reachable. imatrix below Q5 reduces perplexity 10–30% vs naive — but per-tensor auto-allocation beats naive only ~70% of the time (20% it loses), so it is **data-dependent and oracle-gated, not automatic.**

### Axis 3 — More tokens per weight-stream (amortize via speculation; workload-bound multiplier)

| lever | mechanism | exact | ceiling (code workload) | conf | oracle | where |
|---|---|---|---|---|---|---|
| **n-gram / suffix-automaton (PLD/REST/SAM-Decoding)** | match suffix against an index of prompt+generation; **~zero GPU draft cost** (CPU automaton); verify batch reuses weights once across K | E | **1.5–2.5×** on copy-heavy code; ~1× (free) on prose | **M-H** | replay real code transcripts; mean accepted length ≥~2.5 ⇒ win | M3 |
| **EAGLE-3 / EAGLE-3.1 draft head** | single-layer draft with tri-layer feature fusion (layers {2,18,33} for 36L); FR-Spec vocab truncation to ~32K for the 152K head; lossless tree-verify | E | **3–6×** in lit (6.47× HumanEval); see caveat | **M** | train on Colab (~1.5h H100-class); bench acceptance on M3 | **Colab→M3** |
| **batched-verify as compute-amortizer** | the verify GEMM decodes each weight once, reused across K positions — amortizes the proven scale-decode *compute*, not just bandwidth | E | multiplier on accepted runs | M | already characterized (your batched-vs-serial verify) | M3 |
| **the 4.1× spec work in flight** | continue; orthogonal to 1–2 | E | stacks | — | your existing measurement | M3 |
| **MTP heads (DeepSeek-style)** | native multi-token-prediction draft, no separate post-train | E | alt to EAGLE-3 | L | compare to EAGLE-3 in the same harness | Colab |

**The Apple-Silicon spec caveat (foregrounded honestly).** Speculation is the highest-ceiling axis, but it's the most regime- and workload-dependent here. The mlx-lm EAGLE-3 analysis found that for *small, fast, quantized* models on Apple Silicon, per-token time is already low enough that draft overhead struggles to pay off. This couples to our corrected diagnosis: while the kernels are compute-bound (~85% busy), there is **little idle GPU for an EAGLE-3 draft to run in** — the draft competes for the same scarce compute. Two consequences:
- **n-gram/SAM speculation is the safe first spec lever** — its draft is a CPU automaton with ~zero GPU cost, so it works regardless of GPU compute headroom, it's lossless, and code's high copy-rate is the ideal case.
- **EAGLE-3 pays off best *after* Phase 2** makes the kernels bandwidth-bound (then there's idle compute for the draft), and **on code specifically** (where acceptance is highest). With FR-Spec vocab truncation to handle the 152K head. It is lossless and the ceiling is large, but on a 3B the realized gain will be below the H100 headlines — **gate it on a real-transcript acceptance oracle before training.**

---

## 3. The stacked tps envelope (honest, with the ifs)

Anchored at ~39 tps clean (Q4_K_M, ~50% of peak, ~1.9 GB/token, theoretical-100% ≈ 78 tps). Deltas interact; this is reasoned composition, not multiplication.

**Dense decode — what a deterministic benchmark (your bench.py) measures:**

| stage | dense tps | ~% of peak | confidence |
|---|---|---|---|
| now (clean) | ~39 | ~50% | measured |
| + Phase 1 (predec everywhere, ILP, hoist) | ~43–45 | ~55% | **H** |
| + Phase 2 core (vectorize, multi-row, coalesce → llama) | **~48–52** | ~60–64% | **H — llama proves it** |
| + Phase 2 stretch (simdgroup-matrix, split-K → MLX-class) | ~55–64 | ~70–80% | M (decode is the hard MMA regime) |
| **wall at Q4_K_M** | **~66** | ~85% | physical |
| + mixed-precision (imatrix, ~3.9 bits, code-calibrated) | ~60–72 | — | M (data-dependent) |
| + QTIP 3-bit (once bandwidth-bound) | **~70–85** | — | M (quality + compute-gated) |
| **wall at 3-bit** | **~99** | ~85% | physical |

So the **committed, high-confidence dense target is ~50 tps** (≈2× the program's start, matching the best comparable engine). The **strong dense target is ~70–85 tps** (MLX-class kernels + sub-4-bit), past the Q4_K_M wall via fewer bits. The dense ceiling on this chip even at 3-bit is ~99 tps.

**Effective decode — with speculation on the product's code workload (NOT a deterministic number):**
- n-gram/SAM on copy-heavy code: ~1.5–2.5× on accepted spans → effective **~100–180 tps** stacked on a ~70–85 dense base, on favorable spans, degrading to the dense rate on novel prose.
- EAGLE-3 on code, post-bandwidth-bound: potentially higher, with the small-model caveat shaving the H100 headlines.

**I will not collapse this into one headline number.** Each multiplier is real but carries an "if" (kernel stretch needs simdgroup-matrix mastery; QTIP needs to be bandwidth-bound + pass quality; spec needs the code workload and, for EAGLE-3, idle compute). Stated as an aspiration with its three ifs, the maximal stack — MLX-class kernels × 3-bit × code-spec — plausibly reaches **~100–180 effective tps on copy-heavy code**, with ~50 dense as the floor you can quote unconditionally and ~70–85 dense as the strong realistic target. The honest one-liner: **the program's deterministic ceiling on this chip is ~99 tps (3-bit, max kernel efficiency); everything above that is speculation's workload-conditional multiplier.**

---

## 4. The critical path (so "do it all" stays focused)

Dependencies matter: some levers gate others. This is the order that maximizes expected tps per unit risk.

**Stage 0 — Instrument & oracle (days, mostly free, much of it parallel on Colab/local).** Wire the §1 invariant gate + one Instruments calibration. Run the three cheap offline oracles now, before any kernel: (a) **spec acceptance** on real code transcripts (decides n-gram and EAGLE-3 viability — the single most informative number for the whole axis-3 ceiling); (b) **SVD lm_head recall** (decides if screening adds anything beyond lm_head→predec); (c) **mixed-precision KL/PPL** from a code-calibrated imatrix (decides the cheap byte cut). These are NumPy/llama.cpp-tool afternoons and they rank the rest.

**Stage 1 — Bank the cheap, exact wins (~1 session, M3).** predec everywhere + ILP default + LM-head→predec + hoist-constants audit. ~+8–12%, bit-identical, zero risk. → ~43–45 tps.

**Stage 2 — The primary lever: MLX-class Q4_K GEMV (the core of the program, M3).** Vectorized unpacking → multi-row register blocking → simdgroup-matrix decode → split-K → threadgroup tuning → coalesced repack. Parity-gated, each A/B'd against busy-time bandwidth. This is the work that takes you from ~47% to llama's ~60% (→~50, high confidence) and reaches for MLX's ~70–80% (→~55–64). **This is where the bulk of the time goes and where the dense ceiling is decided.**

**Stage 3 — Cut the denominator (Colab→M3).** Once Stage 2 has the kernels bandwidth-bound: ship **mixed-precision** first (cheap, existing codecs, code-calibrated imatrix) for ~+12–20%; then build the **QTIP 3-bit** lookup-free kernel for ~+30–40% if its quality oracle passed. → ~70–85 dense.

**Stage 4 — Speculation, the multiplier (M3, then Colab).** Ship **n-gram/SAM** first (free draft, lossless, code-ideal) gated on the Stage-0 acceptance oracle. Then, with the kernels bandwidth-bound (idle compute available) and acceptance confirmed on code, **train EAGLE-3 on Colab** (FR-Spec 32K vocab, fusion layers {2,18,33}) and integrate. → effective triple digits on the product's code workload.

**Stage 5 — Product-specific (M3).** TailorAI does file-context querying → **long context** → build the **fused quantized-KV attention** kernel (real byte cut at >16K, neutral short) and optimize **prefill/TTFT** (large micro-batch + simdgroup-matrix, which shines at M>1 prefill — the easy MMA regime). These don't move the short-prompt benchmark but they move the product's actual latency.

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

## §7 — Approaching the Physical Floor
*Appendix to the Throughput Bible. Scope: once everything in §1–§6 has landed (MLX-class kernels, sub-4-bit quant, speculation), what remains to push single-stream Qwen2.5-3B decode to the actual physical limit of the M3 Pro. Every Apple-Silicon claim in this section was verified against current sources; where a fact would have been an assumption, it is flagged as such with the verification. Confidence tags and oracles are mandatory per lever, same as the bible. This is the layer **under** kernel efficiency — the bus itself, the other silicon, the power envelope, and the model — and it is where the last 10–40% to the wall lives. Read §7.0 first; it is the completeness argument that tells you when you are done.*
---
### 7.0 — The completeness theorem (so you can prove you've reached the limit)
For single-stream (batch=1) decode of a fixed model on fixed silicon, throughput is governed by **one resource: the memory bus.** Every compute unit on the SoC — GPU, CPU (P/E cores), AMX, Neural Engine — shares the same unified-memory bus, verified: Apple Silicon places "the CPU, GPU, Neural Engine, and media blocks on a single system on a chip that shares high-bandwidth unified memory," and an HPC measurement study found "both CPU and GPU can reach comparable memory bandwidth to the unified memory." There is no private fast path for any unit.
Because decode at batch=1 reads each weight **exactly once per token** (arithmetic intensity ≈ 1 FLOP/byte), the time floor per token is:
```
t_token  ≥  (bytes that must cross the bus) / (achievable sustained bus bandwidth)
```
and there are **exactly four** ways to reduce it. This list is exhaustive — there is no fifth mechanism:
1. **Raise the denominator's speed** — get achievable sustained bandwidth closer to the memory-controller's true ceiling (kernel + memory-layout engineering). §7.1.
2. **Shrink the numerator's element size** — fewer bits per weight (quant). Bible §2; extended in §7.3.
3. **Shrink the numerator's element count** — read fewer weights per token (skip layers/experts, smaller model). §7.3, §7.6.
4. **Reuse each crossing for more tokens** — amortize a single weight-read across K tokens (speculation; the only batch=1-legal reuse). Bible §3; extended in §7.2.
**Corollary (why "use all the silicon" is mostly a trap):** adding compute units (ANE, AMX, more CPU cores) cannot raise the ceiling of mechanism 1, because they contend for the *same* bus that is already the bottleneck. They help **only** by (a) doing non-bus-bound work in time-overlap with the GPU (§7.2), (b) reducing what crosses the bus (mechanisms 2–3), or (c) not stealing the GPU's power budget (§7.4). Any plan that "parallelizes the weight reads across GPU+ANE" is physically void.
**Corollary (compute is not the wall, with margin):** the GPU's raw arithmetic throughput vastly exceeds what bandwidth demands. Qwen2.5-3B decode is ≈ 2 × 3.1e9 = ~6.2 GFLOP/token. The M3 Pro GPU (14 or 18 cores depending on bin) delivers on the order of ~5–9 FP32 TFLOPS / ~10–18 FP16 TFLOPS (an HPC study measured ~2.9 FP32 TFLOPS on the smaller M4 GPU as calibration; the M3 Pro's larger core count puts it higher). Even at the low end that is a compute ceiling of **>1,500 tps**, vs a bandwidth ceiling of ~78 tps at Q4_K_M — a >20× margin. *Consequence:* spending compute to save bandwidth is almost always net-positive, even an expensive codec. The only way compute becomes the wall is an inefficient kernel (e.g. the pre-predec scalar scale-decode that the +21% win fixed) or an absurdly expensive per-byte codec — not the silicon running out of FLOPs.
**The physical floor, stated numerically.** With achievable bandwidth at ~88% of 150 GB/s (≈132 GB/s — see §7.1 for why ~85–90% is the realistic DRAM ceiling, not 100%):
| effective bits/weight | bytes/token (approx) | dense tps at ~88% bus | regime |
|---|---|---|---|
| 4.5 (Q4_K_M today) | ~1.9 GB | ~69 | the current quant's wall |
| 3.0 (QTIP/mixed) | ~1.27 GB | ~104 | sub-4-bit wall |
| 2.5 (aggressive mixed) | ~1.06 GB | ~125 | quality-floor edge for a 3B |
| 2.0 (likely breaks a 3B) | ~0.85 GB | ~155 | below usable quality |
**So the deterministic dense ceiling on this chip-and-model at usable quality is ~100–125 tps** (3.0–2.5 effective bits, ~88% bus). Effective throughput beyond that is *only* mechanism 4 (speculation) on a favorable workload — plausibly ~200–250 effective tps on TailorAI's copy-heavy code spans, decaying to the dense rate on novel prose. There is no path to higher dense throughput on this hardware without a smaller/sparser model (§7.6) or different silicon. **An M3/M4 Max (400 GB/s) would ~2.6× every number in the table** — the single largest lever in existence for this workload, and the honest top-line truth.
---
### 7.1 — Mechanism 1: the last mile of the bus (exact, no quality cost — do these first)
The ~85%-of-peak figure quoted as the kernel wall in the bible is "typical good kernels," not the memory controller's true sustained ceiling, which is closer to ~88–90% for a pure streaming read. Closing that gap is **bit-identical-greedy** engineering with zero quality trade — the literal definition of approaching the physical limit. These act on the *achievable bandwidth* term, on top of the §6 kernel work.
#### 7.1.a — Physical memory layout = kernel access order (the highest-value last-mile lever)
DRAM (LPDDR5 here) delivers far higher bandwidth on accesses that hit an already-open row buffer than on accesses that force a row activation. Standard DRAM behavior; the **magnitude on Apple's specific memory controller is not publicly documented**, so this is a measure-it lever, not a quoted-percent lever. The action: lay the weight tensors out in physical memory in *exactly* the order the decode kernels stream them, so consecutive reads stay within open rows and the prefetcher sees a clean sequential pattern. You already have the offline-repack tool (`q4k_fast`); extend it to a whole-model decode-order layout pass.
- **Exact?** E (pure data movement; bit-identical).
- **Confidence:** M (mechanism certain; Apple-specific magnitude unverified — could be 1% or could be 5%+).
- **Oracle:** on a capture, instrument achieved GB/s during the busy window with and without the reordered layout (the §1 busy-time-bandwidth meter). No model change, so parity is trivially preserved.
- **Where:** offline repack (Colab or M3) + M3 bench.
#### 7.1.b — Wired (resident, non-pageable) weights via `mlock`, NOT huge pages
**Verified correction to avoid an error:** macOS on Apple Silicon does **not** support superpages / huge pages — only 16 KB pages. The `VM_FLAGS_SUPERPAGE_SIZE_2MB` / `_ANY` flags exist in headers but **fail on M1/M2/M3** (superpages are an x86_64-only macOS feature). Do **not** prescribe huge pages; it will not compile-to-effect on this hardware. Two true facts replace it:
1. Apple Silicon's native **16 KB** page is already 4× the Linux-default 4 KB, so each TLB entry covers 4× the span — baseline TLB reach is better than a naive Linux port would assume. There is no further page-size lever to pull.
2. The real, available lever is **wiring**: `mlock()` (or `madvise(MADV_WILLNEED)` for warmup) to keep the 1.9 GB weight map resident so it is never reclaimed or re-faulted mid-stream. This helps **first-token / after-idle latency**, not steady-state throughput (once resident in 18 GB, the weights stay resident between sub-50 ms tokens regardless). The bible already noted this is steady-state-neutral; it belongs here only as TTFT hygiene.
- **Exact?** E.
- **Confidence:** H that huge pages are unavailable (verified); H that `mlock` helps only TTFT, not steady-state tps.
- **Oracle:** page-fault counter on token 2+ (should already be ~0). If it's not, wiring helps; if it is, this is TTFT-only.
- **Where:** M3 (runtime).
#### 7.1.c — Minimize read→write turnaround and intermediate activation traffic
Every switch between reading weights and writing activations costs DRAM read/write-turnaround cycles, and intermediate activations that spill to DRAM add traffic. Keep activations on-chip (registers / threadgroup memory) across fused ops; batch the few unavoidable writes.
- **Honest magnitude bound:** the weights utterly dominate the byte budget. Per-layer intermediate-activation traffic is ~KB-scale (hidden=2048 → 4 KB f16; intermediate=11008 → ~22 KB f16; a handful of round-trips per layer ≈ ~100–200 KB/layer × 36 ≈ ~4–7 MB/token) against ~1.9 GB of weights — i.e. **~0.2–0.4% of bytes/token.** So activation-fusion's *bandwidth* benefit is near-zero. This is a **second, independent reason the megakernel is not a bandwidth play** (the first being that the inter-dispatch gap is only ~12–15%). The turnaround-cycle effect is real but also small at this weight:activation ratio.
- **Exact?** E. **Confidence:** H that the ceiling here is <1%. **Oracle:** count activation bytes written to DRAM/token; confirm it's MB not GB. **Where:** M3.
#### 7.1.d — Fewer, larger contiguous reads (amortize per-tensor pipeline fill/drain)
With ~200 weight tensors streamed per token, each tensor's read has a pipeline fill before it saturates and a drain at the end. Coalescing logically-separate weights that are always read together into one contiguous matrix (the stacked-QKV idea, generalized: also gate+up, which you already fuse) reduces the count of fill/drain boundaries.
- **Exact?** E. **Confidence:** M (real; magnitude depends on how many boundaries remain after §6 fusion). **Oracle:** count distinct weight-tensor reads/token before vs after; bench busy-time BW. **Where:** offline layout + M3.
**§7.1 honest envelope:** collectively, the last-mile bus levers plausibly move achieved bandwidth from ~85% toward ~88–90% of peak — **on the order of +3–6% dense**, exact, no quality cost. Small in percent, but at the physical limit this *is* the remaining budget on mechanism 1, and it is free of quality risk. Layout-as-access-order (7.1.a) is the one to try first.
---
### 7.2 — Mechanism 4 extended: heterogeneous **spatial** speculation (use the idle silicon correctly)
The one genuinely additive role for the other compute units: run the **speculation draft on a different unit, time-overlapped with the GPU's verify**, so the draft's cost hides behind GPU work instead of competing for GPU compute serially. This directly addresses the Apple-Silicon spec caveat (with the kernels at ~85% GPU-busy there is little idle GPU for an EAGLE-3 draft to live in) by moving the draft off the GPU.
#### 7.2.a — Draft on CPU (NEON / AMX), verify on GPU — the practical version
The n-gram / suffix-automaton draft (bible §3) is already a CPU automaton with ~zero GPU cost; this is the safe default and needs nothing new. For a *learned* draft (EAGLE-3), running its small forward on the CPU P-cores (NEON, or the AMX matrix coprocessor which exists on M-series per the HPC review) overlapped with the GPU verify is the tractable heterogeneous path.
- **Critical tension (verified):** the M3 Pro's **6 P-cores form one cluster that runs at a single frequency, and macOS lowers that frequency as more threads load the cluster** — and the whole SoC shares a power/thermal budget, so heavy CPU work during decode **can pull the GPU's clock down**. This means a CPU-side draft is not free: it can slow the GPU verify it was meant to overlap. The win is real only if the draft is light enough that the net (faster effective tokens) beats the cost (lower GPU clock + bus contention from the draft's own weight reads).
- **Exact?** E (lossless spec verify). **Confidence:** M (mechanism sound; the power-budget interaction makes it measure-or-die). **Oracle:** with `powermetrics` / `macmon` logging GPU frequency, run the draft on CPU during a GPU verify and confirm GPU clock holds AND end-to-end accepted-tokens/sec rises. **Where:** M3.
#### 7.2.b — Draft on the ANE — high ceiling, high difficulty, do not assume it's easy
**Verified, to prevent a serious error:** the Neural Engine is **not Metal-programmable** ("Metal cannot be used to program the ANE… there is currently no public framework for directly programming the ANE"). The **only public path is Core ML**, which is a black-box scheduler — you cannot force ANE placement, inspect its programs, or run arbitrary ops; models often must be redesigned to fit, with op/shape constraints (e.g., a documented Stateful-API failure when a state-tensor width isn't 32-aligned) and inter-engine context-transfer overhead. The headline "38 TOPS" is INT8 that the ANE **dequantizes to fp16 internally** (~19 TFLOPS fp16 effective). Private-framework routes that bypass Core ML exist and are real but heroic (the Orion project drives the ANE via `_ANEClient`/`_ANECompiler`/MIL IR for training; Draw Things shipped ANE-as-an-accelerator-inside-a-custom-runtime for 8-bit models, ~1.8× on M4) — they are months of reverse-engineering, not a switch.
- **Why it could still matter:** if (and only if) you can express the EAGLE-3 draft as a Core ML model (or via a private path), it runs on silicon that is otherwise idle during decode and that has its **own** execution while the GPU verifies — the cleanest possible spatial overlap, and Apple explicitly cites "opening up the CPU and the GPU to execute non-ML workloads while ANE is executing" as a benefit. FR-Spec vocab truncation (→ ~32K) helps fit the 152K-head constraint.
- **Exact?** E (lossless verify). **Confidence:** L (the integration risk dominates, not the math). **Oracle:** before any of this, (1) convert the draft to Core ML, confirm it actually places on ANE (trace for `ANE`/`H11ANE` symbols, not `MPS`/`MTL`), measure its standalone latency + the round-trip cost back to the Rust/Metal runtime per speculation step; (2) only proceed if (draft-on-ANE latency + transfer) < the GPU idle it overlaps. **Where:** Colab to train, M3 (Core ML) to integrate.
**§7.2 framing:** n-gram/SAM on CPU is the safe, free spatial-spec win (do it). EAGLE-3 on CPU is the medium path, gated on the GPU-clock measurement. EAGLE-3 on ANE is the high-ceiling/high-risk path, gated on Core ML actually placing it on the ANE and the transfer overhead being smaller than the overlap. The multiplier from all of these is on **effective** tps and is **workload-conditional** (high on code, ~1 on novel prose) — never quote it without naming the workload.
---
### 7.3 — Mechanisms 2 & 3 extended: below 3-bit, and fewer layers (quality trades, oracle-gated)
These bend the numerator further than the bible's 3-bit QTIP, but every one is a quality trade gated on KL/perplexity against a **code-representative** corpus, not bit-identical.
#### 7.3.a — Per-tensor-type aggressive mixed precision toward ~2.5 effective bits
Push the FFN bulk to 2-bit where the importance matrix says it survives, while keeping attention, the (tied) embedding/lm_head, and outlier-sensitive tensors higher. This is the bible's imatrix lever taken to its quality limit, and it's what moves the wall from ~104 (3.0-bit) toward ~125 tps (2.5-bit). Uses your existing Q2_K/Q3_K/Q4_K/Q5_K/Q6_K codecs — **no new kernel**, just a precision-assignment search.
- **Verified caveat:** llama.cpp's own auto bit-allocation beats naive only ~70% of the time (it *loses* ~20% of the time, sometimes badly), so per-tensor allocation is **data-dependent and must be oracle-selected, not assumed.** And a 3B degrades hard below ~3 bits on average — 2-bit is viable only on a *subset* of robust tensors, never globally.
- **Exact?** Q (small if well-allocated). **Confidence:** M. **Oracle:** compute a code-corpus imatrix on Colab; build candidate mixed-precision assignments; KL/PPL each vs Q4_K_M; keep only those inside the quality gate. **Where:** Colab (imatrix + quantize) → M3 (bench).
#### 7.3.b — Dynamic depth / confidence-gated early exit (fewer layers for easy tokens)
Mechanism 3 in its purest form: run only the first E of 36 layers for tokens the model is already confident about, full depth otherwise — reducing **average** bytes/token independent of bit-width. Training-light (CALM/LayerSkip-style early-exit calibration) and stackable with quant.
- **Exact?** Q (changes outputs unless paired with a verify; with a full-model verify it can be made lossless, at verify cost). **Confidence:** L on a 3B (small models have less per-token "easy/hard" headroom than the 7B+ models where this was shown). **Oracle:** on a code capture, measure argmax-agreement between E-layer and full-model logits across E; compute the average-depth reduction at your tolerated agreement. If easy tokens don't cluster, it dies (same discipline that killed block-256 sparsity). **Where:** Colab (calibrate) → M3.
#### 7.3.c — W4A8 unblocked via Hadamard/rotation (small, activation-side)
Your W4A8 path is held (quality-blocked at 20% bit-identical on outlier channels). RHT/QuaRot rotation is the standard fix — it removes the outlier channels that break int8 activation quantization, and the rotation can be absorbed offline into adjacent linears (computational-invariance trick) for ~zero decode cost. Payoff is **activation-side bytes**, which are small at batch=1, so this is a minor throughput lever — but it may *unblock* a capability you've already built, and the rotation front-end is the same one QTIP wants.
- **Exact?** Q (W4A8 is a quality trade; rotation is exact). **Confidence:** M that rotation unblocks W4A8; L that the throughput gain is large (activations are a small byte share). **Oracle:** bake RHT, re-measure W4A8 bit-identical rate and KL. **Where:** Colab (rotation) → M3.
---
### 7.4 — The power/thermal envelope (literally "the processing power available")
The user's phrase deserves a literal answer: the *available* processing power is itself dynamic, and you can lose or gain ~clock by how you use the SoC.
#### 7.4.a — Protect the GPU's clock (the heterogeneous-execution backstop)
**Verified:** macOS — not a simple thermal governor — controls P-core and GPU frequency, and "in many circumstances P cores and GPUs aren't run at maximum frequency even when there's no immediate thermal reason." Cluster frequency drops as thread count rises, and the SoC shares a power budget. **Action:** during steady-state decode, keep CPU work minimal and on the right QoS so the GPU is granted maximum clock and the full power budget. This is the direct counterweight to §7.2 — heterogeneous CPU work must *net* win against the GPU-clock it costs.
- **Exact?** E (scheduling, not math). **Confidence:** M (the effect is real and measurable; the magnitude is workload- and macOS-version-specific). **Oracle:** `powermetrics`/`macmon` GPU-frequency log under (i) GPU-only decode vs (ii) decode + your CPU host work; if (ii) shows lower GPU MHz, trim or re-QoS the host work. **Where:** M3.
#### 7.4.b — Sustained vs burst, and the benchmark honesty rule
M-series chips throttle under sustained load (well-documented for the Pro parts under heavy GPU). A 64-token bench in a cool window over-reports vs sustained generation. **Action (extends §1's protocol):** report both a burst number (short, cool) and a sustained number (long run to thermal steady-state), with `powermetrics` temp/clock logged, so the tps you quote is the tps the product delivers. Plugged-in vs battery and Low-Power-Mode also move the ceiling — fix them in the protocol.
- **Exact?** E. **Confidence:** H that sustained < burst; magnitude is chassis/ambient-specific. **Where:** M3.
---
### 7.5 — Make wall-clock equal GPU-busy (close the residual ~12–15%)
Once §6 has the kernels at the bus limit, the last gap between *measured* tps and the *kernel* limit is host overhead — the ~12–15% that is now NOT inter-dispatch GPU ramp (that was the artifact) but genuine host glue: sampling, tokenizer/detokenizer, the n-gram automaton, KV bookkeeping, command-buffer submission, CPU↔GPU sync. At 100+ tps, a 1 ms/token host cost is >10% of the budget.
- **Levers (all E):** a persistent, zero-allocation host decode loop (reuse the per-decode arenas you have); **GPU-side sampling/argmax** to remove a per-token CPU round-trip + sync; tightest-possible command-buffer reuse so the GPU never waits on host encode (the encode itself is already cheap — this is about *latency on the critical path*, not encode cost); pin the host loop to a P-core and keep everything else off it (ties to §7.4.a).
- **Confidence:** M-H (this is well-trodden host-loop engineering with a clear ceiling = the host fraction). **Oracle:** the §1 cross-check `host_wall − sum_gpu_us` per token *is* this budget; drive it toward zero and watch tps approach the busy-time-bandwidth-implied ceiling. **Where:** M3.
---
### 7.6 — The model-level lever (the only thing that multiplies dense throughput beyond quant)
Stated plainly because the question is "the physical limit," and at the limit the **parameter count is the limit.** Inference tricks cannot read fewer than (model size in bits) per token at batch=1 except by skipping or batching; the remaining degree of freedom is to make the model itself smaller or genuinely sparse.
- **Distillation to a smaller student.** A well-distilled 1.5B at 3-bit is ~0.6 GB → roughly **double** the 3B's dense token rate on the same bus, because it halves the numerator at the source. This is not an inference optimization; it's a different model, and it lives entirely on whether TailorAI's quality bar tolerates it. For a *product*, "is a faster, slightly-weaker model good enough for the common case?" is a legitimate and possibly decisive engineering question.
- **Structured pruning + heal.** Remove whole layers/heads/dims (not contextual single neurons — that died on scattering) and heal on Colab. Reduces total bytes/token at some quality cost.
- **A natively low-bit / sparse model** (BitNet-style ternary, or a ReLU-fied FFN that *is* sparse so mechanism-3 skipping finally works) — these need (re)training, so they're a Colab-scale model project, not a runtime change, but they are the only way to break the ~2-bit PTQ quality floor that caps §7.3.a.
- **Exact?** Q (real quality trade). **Confidence:** M that a smaller model doubles speed (arithmetic); L on whether quality suffices (product-specific). **Oracle:** distill/prune a candidate on Colab; eval on TailorAI's real downstream tasks (not perplexity) vs the 3B. **Where:** Colab (train) → M3 (serve).
---
### 7.7 — Consolidated lever table (§7 additions only; bible §2–§3 levers not repeated)
| # | lever | mechanism (of the 4) | exact? | acts on | realistic ceiling | conf | oracle (before building) | where |
|---|---|---|---|---|---|---|---|---|
| 7.1.a | layout = kernel access order | 1 (bus speed) | E | achieved BW (row-buffer locality) | +1–5% (measure) | M | busy-time GB/s with/without reorder | repack + M3 |
| 7.1.b | `mlock` wiring (NOT huge pages) | 1 | E | TTFT/after-idle, not steady tps | ~0% steady; TTFT only | H | page-fault count token 2+ | M3 |
| 7.1.c | read/write turnaround + activation on-chip | 1 | E | turnaround cycles | <1% (weights dominate) | H | activation bytes/token (MB not GB) | M3 |
| 7.1.d | fewer/larger contiguous reads | 1 | E | per-tensor fill/drain | a few % | M | tensor-read count/token | repack + M3 |
| 7.2.a | draft on CPU, verify on GPU | 4 (reuse) | E | effective tps (code) | workload mult; gated by GPU clock | M | GPU MHz holds + accepted-tok/s up | M3 |
| 7.2.b | draft on ANE, verify on GPU | 4 | E | effective tps (code) | high ceiling, high risk | L | Core ML places on ANE + transfer < overlap | Colab+M3 |
| 7.3.a | per-tensor 2-bit-where-robust mixed precision | 2 (fewer bits) | Q | bytes/token → ~2.5 bit | wall ~104→~125 tps | M | code-imatrix KL/PPL per assignment | Colab→M3 |
| 7.3.b | dynamic depth / early exit | 3 (fewer layers) | Q (or E w/ verify) | avg bytes/token | model-dependent | L | E-vs-full argmax agreement on code | Colab→M3 |
| 7.3.c | RHT to unblock W4A8 | 2 (activation bytes) | Q | activation bytes (small) | minor tps; unblocks W4A8 | M | W4A8 bit-id rate + KL after RHT | Colab→M3 |
| 7.4.a | protect GPU clock (trim host CPU) | enabler for 1 | E | available GPU MHz | recovers throttle loss | M | GPU MHz: GPU-only vs +host | M3 |
| 7.4.b | sustained-vs-burst bench protocol | measurement | E | honesty of quoted tps | — | H | long-run tps + temp/clock log | M3 |
| 7.5 | wall-clock = GPU-busy (host loop, GPU sampling) | 1 (realize the ceiling) | E | residual ~12–15% host gap | up to ~10%+ | M-H | `host_wall − Σgpu_us` → 0 | M3 |
| 7.6 | smaller/distilled/sparse model | 2+3 (smaller numerator) | Q | total bytes/token | up to ~2× dense | M arith / L quality | downstream-task eval vs 3B on Colab | Colab→M3 |
---
### 7.8 — Sequencing the floor-approach (after the bible's §1–§5 land)
1. **Exact, no-quality-cost, do first:** §7.5 (wall-clock = GPU-busy) and §7.1.a (layout = access order). These approach the physical bus limit with zero quality risk and need no training. §7.4.a/b (protect GPU clock + honest sustained bench) run alongside as standing discipline.
2. **Effective-tps multiplier, gated:** §7.2.a (CPU-draft spatial spec) — free with the n-gram automaton you'll already have; gate the learned-draft version on the GPU-clock measurement.
3. **Push the numerator, quality-gated:** §7.3.a (2-bit-where-robust mixed precision) once §6 has the kernels bandwidth-bound; then §7.3.b and §7.3.c behind their oracles.
4. **High-ceiling research, only if justified:** §7.2.b (draft on ANE) — pursue only if the Core ML placement + transfer oracle clears; it's months, not days.
5. **Product decision, not an optimization:** §7.6 (smaller model) — evaluate in parallel as a strategic question; it's the only lever that doubles dense throughput, and it trades quality the product may or may not afford.
**The honest bottom line for §7:** the exact levers (§7.1, §7.4, §7.5) get you from "MLX-class kernels" to the true bus ceiling — a further single-digit-to-low-double-digit percent with no quality cost. The quality-gated numerator levers (§7.3) move the dense wall from ~104 to ~125 tps. Speculation (§7.2) is the only thing that puts **effective** throughput into the ~200–250 range, and only on TailorAI's code workload. The **deterministic dense ceiling on this M3 Pro at usable quality is ~100–125 tps**; everything above is either speculation's workload multiplier or a different model (§7.6) — and the largest lever of all remains the one piece of hardware we can't change here: a 400 GB/s Max-class chip would ~2.6× the whole table.
---
### 7.9 — Verified-facts ledger (so Claude Code ingests no assumptions)
Every Apple-Silicon claim this section relies on, with its status:
- **Unified bus shared by all units; CPU and GPU reach comparable UMA bandwidth** — verified (SoC architecture sources; HPC measurement study). Basis for §7.0.
- **M3 Pro = 150 GB/s** — verified earlier in the project (Apple spec; 25% below M1/M2 Pro). Basis for all floor numbers.
- **Compute ≫ bandwidth headroom (>20×)** — derived from ~6.2 GFLOP/token and measured M-series GPU TFLOPS (HPC study calibration); robust to GPU-core-bin uncertainty. Basis for "expensive codecs are net-positive."
- **macOS Apple Silicon has NO huge/superpages; 16 KB pages only; `VM_FLAGS_SUPERPAGE` fails on M1/M2/M3** — verified (multiple developer-forum + gist confirmations). Prevents the huge-page error in §7.1.b.
- **ANE is not Metal-programmable; Core ML is the only public path, a black-box scheduler with op/shape constraints and transfer overhead; "38 TOPS" INT8 dequantizes to ~19 TFLOPS fp16; private-framework bypass exists but is heroic** — verified (hollance/neural-engine; Apple ML research; Orion; Draw Things engineering; arXiv ANE paper). Basis for §7.2.b's L confidence.
- **M3 Pro = 6 P-cores in one cluster at a single macOS-set frequency that drops as thread count rises; SoC shares a power budget; macOS (not a simple thermal governor) controls GPU/CPU clocks and often runs below max** — verified (Eclectic Light Company analyses). Basis for the §7.2/§7.4 power-budget tension.
- **`powermetrics`/`macmon` expose GPU frequency/power/temp without kexts** — verified. The measurement instrument for §7.4.
- **llama.cpp auto mixed-precision beats naive ~70% of the time (loses ~20%)** — verified (llama.cpp discussion #18531). Basis for §7.3.a being oracle-gated.
- **Real-world Apple-Silicon decode hits 60–80% of theoretical bandwidth** — verified (optimization guide). Basis for the ~85–90% achievable-bus ceiling.
- **Marked "measure, not quoted":** DRAM row-buffer-locality gain on Apple's specific controller (§7.1.a); exact percent of each last-mile lever; dynamic-depth headroom on a 3B (§7.3.b) — all flagged as oracle-first, not assumed.
