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
