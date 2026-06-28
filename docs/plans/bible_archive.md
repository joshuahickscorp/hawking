> **ARCHIVE / REFERENCE — the long companion to [bible_active.md](bible_active.md)** (split 2026-05-31 from this file's predecessor `throughput_bible_2026_05_30.md`). This doc holds the deep strategy reference + completed wins + the full kill ledger; it is allowed to be long. Current state, the lever status ledger, the critical path, and the breadth axis (§9) live in the **[active doc](bible_active.md)**. Canonical kill registry: **[../reports/dead_levers.md](../reports/dead_levers.md)**. The original 2026-05-30 header + mission statement follow verbatim.

---

> **STATUS: CANONICAL — operative throughput strategy as of 2026-05-30.**
> Supersedes the gap/megakernel framing in [silicon_architecture_audit_2026_05_29.md](silicon_architecture_audit_2026_05_29.md) (Topics 2 & 7) and the contaminated 26.6-tps anchor.
> Reconciliation crosswalk (this doc ↔ audit plan ↔ silicon-builds): [throughput_bible_reconciliation_2026_05_30.md](throughput_bible_reconciliation_2026_05_30.md).
> Empirical backing for axes 1–2: [../silicon-builds/SUMMARY.md](../silicon-builds/SUMMARY.md) (#8 simdgroup-MMA LIVE, #13 zero-copy LIVE, #16/#17 byte-cut prize).
> Body below is verbatim as authored 2026-05-30. Do not edit in place — supersede with a dated successor.

---

# `hawking` — The Throughput Bible

*The single consolidated reference for pushing single-stream Qwen2.5-3B decode on M3 Pro to its limit. Everything viable is in here, tiered by axis, with a mechanism, a realistic ceiling, a confidence, a run-before-you-build oracle, a dependency, and whether it runs on Colab (offline) or locally on the M3. This is a map of the whole territory — not a claim that we do it all at once. The critical path at the end says what to do in what order. Maximal scope; the physics ceilings still bind, and we exploit them rather than wish them away.*

---

---

## Completed-wins ledger (shipped — with commit hashes)

*The throughput wins that landed, in rough chronological order. Hashes verified against `git log` on 2026-05-31. Deltas are paired (contamination-robust) unless noted. This is the "what worked" half of the fence; the dead half is the kill ledger (§8.3.1) + [../reports/dead_levers.md](../reports/dead_levers.md).*

| win | effect | commit(s) | note |
|---|---|---|---|
| full Metal pipeline (TCB) | 1.32 → ~18 dec_tps | `b11dc94` | the baseline engine on Qwen-3B-Q4_K_M |
| opt-in Q4_K LM head | — | `81509c9` | LM-head on the Q4_K GEMV path |
| vocab-prune LM head (32K) + compound w/ Q4_K + corpus-derived | → ~22.4 dec_tps | `9a54f27`, `4e33ca2`, `0c3b736` | prune the 152K head to the in-use vocabulary |
| Q4_K v4_predec (pre-decoded sub-block scale table) | +39.8% paired, bit-identical | `8a67a63` (infra), `c325068` (wire), `6f0209e` (default-on) | the single biggest decode lever; kills ~332K decode-ops/token |
| path-to-50 predec: 2r default-on + fused gate+up + ffn_down→predec | +32.5% decode, bit-identical | `d2343cc` | 2-row ILP default |
| Q4K_FAST sidecar (160B sub-block-contiguous layout) | +28.1% paired | `4fc7e59` (infra), `2ff8198` (wire) | mutually exclusive with predec (predec wins) |
| f16-scales predec (FFN gate+up pair) | +6–9% decode, opt-in | `7d614f8`, `0899137` | A6.5 — the lone bandwidth win of the overnight kernel track; `HAWKING_QWEN_PREDEC_F16SCALES` |
| pruned-Q4_K LM head through predec GEMV | +2.0%, bit-identical | `0e6eb14` | route the pruned head through predec |
| in-RAM prefix cache (exact KV reuse) | ~84% prefill cut; default-on | `ebfc57a` (opt-in), `9e03270` / `fc93ea0` (default-on + LRU + disk-tier) | the moat leg (§8 L1.2) |
| batched Q4_K predec GEMM (spec verify) | verify-perf | `6be1057`, `e25c033` | amortizes scale-decode compute across K positions |
| GPU pruned-Q4K LM head in batched verify | bit-identical | `010827b` | engages with VOCAB_PRUNE + Q4K_LMHEAD |
| L3.1 draft-tuning (user-ngram warm-start) | GO; +148% on repetitive code; gated | `a0f7a6e`, `680cb35`, `354d718` | the propose-first user-draft loop; default-off body |
| A4 per-kernel decode profile | predec GEMVs 89% of decode @ ~56% peak | `f2a6a4f` | the profile that closed the decode-kernel track |

*The decode-kernel micro-opt track that did NOT pan out (A5/A6/A7/A10, overnight closeout `3cb5944`) is recorded as Type-1 kills in [../reports/dead_levers.md](../reports/dead_levers.md) and summarized in [bible_active.md](bible_active.md) §3.0.*

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
| **fused quantized-KV attention** (mlx-qsdpa pattern) | read 4/8-bit KV inline in one dispatch, no FP16 buffer; **only matters at long context** (whole-file / codebase context) | KV bandwidth + memory at long ctx | Q (KV) | ~0% short ctx; **big at >16K** (mlx-qsdpa 1.7× @128K) | M | measure KV's byte share at the product's real context lengths | M3 |
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


---

## 3. The stacked tps envelope — SUPERSEDED ~39-anchor projections

*Moved here verbatim from the original §3. These projections are drawn off the now-superseded ~39 tps anchor; the live current-state corrections (anchor → ~31, decode kernels closed) are in [bible_active.md](bible_active.md) §3.0. Kept for the audit trail; re-projection from ~31 is an open attended task.*

Anchored at ~39 tps clean (Q4_K_M, ~50% of peak, ~1.9 GB/token, theoretical-100% ≈ 78 tps). ⚠ *SUPERSEDED — the anchor is now RESOLVED to ~31 (clean-room 2026-05-31, §3.0 Correction 2); this ~39 envelope and its %s need re-projection from ~31.* Deltas interact; this is reasoned composition, not multiplication.

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
**So the deterministic dense ceiling on this chip-and-model at usable quality is ~100–125 tps** (3.0–2.5 effective bits, ~88% bus). Effective throughput beyond that is *only* mechanism 4 (speculation) on a favorable workload — plausibly ~200–250 effective tps on the target workload's copy-heavy code spans, decaying to the dense rate on novel prose. There is no path to higher dense throughput on this hardware without a smaller/sparser model (§7.6) or different silicon. **An M3/M4 Max (400 GB/s) would ~2.6× every number in the table** — the single largest lever in existence for this workload, and the honest top-line truth.
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
- **Distillation to a smaller student.** A well-distilled 1.5B at 3-bit is ~0.6 GB → roughly **double** the 3B's dense token rate on the same bus, because it halves the numerator at the source. This is not an inference optimization; it's a different model, and it lives entirely on whether the target workload's quality bar tolerates it. For a *product*, "is a faster, slightly-weaker model good enough for the common case?" is a legitimate and possibly decisive engineering question.
- **Structured pruning + heal.** Remove whole layers/heads/dims (not contextual single neurons — that died on scattering) and heal on Colab. Reduces total bytes/token at some quality cost.
- **A natively low-bit / sparse model** (BitNet-style ternary, or a ReLU-fied FFN that *is* sparse so mechanism-3 skipping finally works) — these need (re)training, so they're a Colab-scale model project, not a runtime change, but they are the only way to break the ~2-bit PTQ quality floor that caps §7.3.a.
- **Exact?** Q (real quality trade). **Confidence:** M that a smaller model doubles speed (arithmetic); L on whether quality suffices (product-specific). **Oracle:** distill/prune a candidate on Colab; eval on the target workload's real downstream tasks (not perplexity) vs the 3B. **Where:** Colab (train) → M3 (serve).
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
**The honest bottom line for §7:** the exact levers (§7.1, §7.4, §7.5) get you from "MLX-class kernels" to the true bus ceiling — a further single-digit-to-low-double-digit percent with no quality cost. The quality-gated numerator levers (§7.3) move the dense wall from ~104 to ~125 tps. Speculation (§7.2) is the only thing that puts **effective** throughput into the ~200–250 range, and only on the code workload. The **deterministic dense ceiling on this M3 Pro at usable quality is ~100–125 tps**; everything above is either speculation's workload multiplier or a different model (§7.6) — and the largest lever of all remains the one piece of hardware we can't change here: a 400 GB/s Max-class chip would ~2.6× the whole table.
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

---

## §8 — The System-Level Shift

*A standalone strategic + engineering plan, companion to the Throughput Bible (`plans/throughput_bible_2026_05_30.md`) and the roadmap. The bible answers "how fast can the kernels go on this hardware" (physics ceiling: ~50 dense parity, ~100–125 with everything stacked). This plan answers a different question: "what is hawking, as a system, that the incumbents structurally cannot be." Same discipline as the bible — every lever has a mechanism, an offline oracle to run before any kernel, an Apple-GPU/M3 feasibility verdict, an exact-vs-quality-trade label, a confidence tag, and an honest energy verdict. The dreaming stays defensible.*

---

### 8.0 — The thesis (the actual shift)

Every incumbent inference engine — llama.cpp, MLX, vLLM, Ollama — is built to be **general, stateless, and static**:
- **General:** it must serve any model, on any hardware, for any user.
- **Stateless:** it recomputes from scratch each request and throws the history away.
- **Static:** it treats the weight file as fixed, immutable input — a blob to stream.

These three properties are not laziness; they are *requirements* of being a general-purpose engine. And every one of them leaves performance on the table.

**hawking's shift is to refuse all three.** It runs on **one machine, for one user, on an observable workload, with persistent state, and a model that molds to that user.** Every lever in this plan is downstream of that refusal. This is the system-level move; the individual techniques are how it cashes out. The incumbents cannot copy the core of it, because copying it would mean abandoning the generality that is their entire value proposition.

**Why this is the right bet (the honest version).** The bible already proved you cannot win on raw tps — MLX is Apple's own and beats you, llama.cpp is free and years-tuned, and the physical ceiling is parity, not dominance. "Faster" is a treadmill. But the *adaptive/stateful/specialized* dimension is empty: no shipping local engine treats the model as alive. You reach "all" not by winning two races you can't win (kernels vs MLX, coverage vs llama.cpp), but by being **fast enough** (parity, which the tps lane delivers) and then **owning the entire dimension everyone else abandoned by design.** To a user, an engine that runs near-MLX speed, handles enormous context, reuses repeated computation, sips power, and quietly molds to their patterns privately — that reads as "all," and it is a *defended position*, not a treadmill.

**The brain frame, kept honest.** The reason a brain runs on ~20W and a GPU doesn't is that the brain has no bus: memory and compute are co-located, so it never pays the von Neumann tax that *is* hawking's measured bottleneck (decode is bandwidth-bound = paying the bus tax the brain avoids). That is a beautiful diagnosis. But the brain's *solution* — co-located analog compute, spiking sparsity, no clock — is **hardware**, owned by the neuromorphic/in-memory-compute industry, not buildable in software on a fixed M3. So this plan does **not** claim a brain-grade hardware paradigm shift. It claims the two things the brain frame legitimately motivates and that you *can* build: exploit the brain's *software-transferable* principles (sparsity, predict-only-the-surprise, statefulness), and own **energy-per-token** as a measured, branded axis nobody else competes on.

**The energy caveat, stated once and applied throughout.** On a fixed M3, decode is bandwidth-bound, and the GPU draws roughly similar power whether it is computing or stalling on memory. So "use less energy → fit more in the same envelope" only holds where you are **compute-throttled or thermally-throttled**, not across the board. Energy efficiency is a real, underexploited dimension to *own and market*, but the "more fits in the same power" lever is narrower than it feels. Each lever below carries an explicit **Energy verdict**: GENUINE (draws less power / does less work), NEUTRAL (helps speed, not power), or THERMAL (helps only by reducing sustained-throttle).

---

### 8.1 — The fifteen levers (grouped by system layer)

Notation per lever: **Mechanism** · **Benefit to the project** · **Exact?** (E = bit-identical greedy, Q = quality trade) · **Oracle** (run on a capture before any kernel) · **Apple-GPU/M3 feasibility** · **Energy verdict** · **Confidence** (H/M/L) · **Where** (Colab offline / M3 engine).

---

#### Layer 1 — The data that crosses the bus (attack the von Neumann tax)

##### L1.1 — KV cache as a living working set
- **Mechanism:** attention is sparse — most cached tokens contribute almost nothing to the current token. Rank cached tokens by importance (attention-sink positions à la StreamingLLM; cumulative-attention "heavy hitters" à la H2O; recent-window + pooled-importance à la SnapKV) and **evict or compress** the low-value ones, keeping the KV at a bounded working-set size instead of a linearly-growing blob.
- **Benefit:** the single biggest capability unlock for a coding tool. Turns whole-file / whole-codebase context from choking at ~32K into running at 200K+ within the same 18 GB, and cuts KV-read bandwidth at long context. Directly serves the file-context use case (whole-file / codebase coding).
- **Exact?** Q (approximate attention; bounded quality loss, tunable by working-set size). A "lossless mode" (keep all, no eviction) stays available for correctness-critical runs.
- **Oracle:** on a real long-context coding capture, replay attention and measure, per layer, what fraction of cached tokens receive ≥99% of cumulative attention mass. If a small bounded set dominates (expected — this is the documented finding), eviction is safe; quantify the context-length-vs-quality curve before any kernel. *This is the StreamingLLM/H2O finding re-measured on your model and workload.*
- **Apple-GPU/M3 feasibility:** favorable. Eviction is bookkeeping (drop/relocate KV blocks); compression reuses your existing quant codecs on KV tensors. The fused quantized-KV-attention pattern (read 4/8-bit KV inline, no FP16 buffer — cf. mlx-qsdpa) is the kernel to mirror. No random gather of weights.
- **Energy verdict:** GENUINE at long context (fewer KV bytes moved = less bus energy and less work); NEUTRAL at short context.
- **Confidence:** H (documented wins; clean oracle). **Where:** M3 (runtime + kernel), oracle is offline/either.

##### L1.2 — Cross-prompt computation reuse (prefix + semantic caching)
- **Mechanism:** coding workloads re-send the same files, imports, and scaffolding constantly. Cache the KV/activations for shared **prefixes** so an unchanged prefix is never recomputed (vLLM's prefix caching; a single-user local engine is the *ideal* case). Then extend past exact-match into **semantic caching**: embed recent contexts and recognize "I have computed something near-identical before → reuse," with an exact-match verification step before trusting a near-hit.
- **Benefit:** eliminates whole forward passes on the redundant 70–90% of a coding session. A stateless engine discards this every request; statefulness is the entire advantage.
- **Exact?** E for prefix caching (a matched prefix is bit-identical). Q for semantic-cache *acceptance* unless gated by a verify (then E). Default to verified reuse for greedy-exactness.
- **Oracle:** on real session transcripts, measure (a) average shared-prefix length across consecutive requests (prefix-cache hit rate) and (b) near-duplicate rate of contexts under an embedding-similarity threshold (semantic-cache potential). High prefix overlap on code is expected; quantify before building. No kernel needed to measure.
- **Apple-GPU/M3 feasibility:** favorable. Prefix caching is KV-block retention keyed by token hash — pure bookkeeping over your existing per-decode arenas. Semantic caching adds a small local embedding index (fits in unified memory).
- **Energy verdict:** GENUINE (a skipped forward pass is power *not* drawn — among the strongest true energy wins here, because it removes work entirely, not just moves it faster).
- **Confidence:** H for prefix caching, M for semantic caching. **Where:** M3 (runtime).

##### L1.3 — Cross-layer weight delta-encoding
- **Mechanism:** adjacent transformer layers' weight matrices are often correlated. Store layer L+1 as layer L plus a **delta** that is more compressible (lower entropy → fewer bits, or low-rank) than the full matrix.
- **Benefit:** fewer *unique* bytes/token across the bus — a direct hit on the measured bandwidth bottleneck, stacking with quant.
- **Exact?** Q (lossy if the delta is quantized; can be made near-exact at higher delta precision).
- **Oracle (decisive, run first):** compute pairwise inter-layer weight similarity on Qwen2.5-3B (cosine / low-rank-residual energy between consecutive layers' projections). If deltas are substantially more compressible than the originals, proceed; **if Qwen's layers are too independent, this lever dies cheaply** with zero kernel written. Same discipline that killed block-256 sparsity.
- **Apple-GPU/M3 feasibility:** decode reads base + delta contiguously (no gather) if laid out together. Adds decode compute (cheap — you have surplus ALU per the bible's >20× compute headroom).
- **Energy verdict:** GENUINE if it lands (fewer bytes moved); NEUTRAL otherwise.
- **Confidence:** L (genuinely under-explored; magnitude unknown on this model until the oracle runs). **Where:** Colab (analysis + repack) → M3.

##### L1.4 — Low-rank + compressible residual, jointly with quant
- **Mechanism:** decompose each weight as W ≈ UV (tiny low-rank part capturing the smooth/dominant structure) + a residual. Because UV removed the structure, the residual quantizes to **fewer bits** at equal quality. Read the rank-r part + the low-bit residual.
- **Benefit:** lower effective bits/weight than quant alone. The under-exploited part is the *joint* design — let UV and the quantizer each handle what they are good at, rather than quantizing the raw matrix.
- **Exact?** Q (lossy; tunable by rank and residual bits).
- **Oracle:** for rank r ∈ {16, 32, 64}, SVD each weight, quantize the residual at 2–3 bits, measure KL/perplexity vs Q4_K_M on a code corpus. If (r-part bytes + residual bytes) < Q4_K_M bytes at equal quality, proceed.
- **Apple-GPU/M3 feasibility:** the UV GEMV is contiguous/coalesced; the residual uses existing codecs. Competes with QTIP for the same byte budget — **build at most one of {L1.4, QTIP}**, chosen by oracle.
- **Energy verdict:** GENUINE if fewer bytes; NEUTRAL otherwise.
- **Confidence:** M. **Where:** Colab → M3.

##### L1.5 — Learned per-model codebook
- **Mechanism:** llama.cpp uses a universal quant grid for every model on earth. hawking fits a codebook (k-means / lattice) to *this specific model's* weight distribution, offline, once.
- **Benefit:** more quality per bit than a one-size-fits-all grid — a specialization general engines cannot afford per-model.
- **Exact?** Q (lossy; better quality-per-bit than fixed-grid at the same bits).
- **Oracle:** fit the codebook on Qwen's weights; measure KL/perplexity vs Q4_K_M at matched bits. Proceed only if the per-model grid beats the universal grid by a margin worth a custom decode path.
- **Apple-GPU/M3 feasibility:** **this is the danger lever** — a learned codebook implies codebook *lookups*, which is exactly what makes IQ-quants slow on Apple (random gather, no hardware gather instruction). Only viable if the codebook is tiny enough to live in threadgroup memory AND the decode stays mostly contiguous (lookup-free codes preferred, à la QTIP's bitshift trellis). If it forces per-element random LUTs, it loses to contiguous Q4_K — **kill it at the feasibility gate, not after building.**
- **Energy verdict:** GENUINE if fewer bytes and the decode is cheap; risk of NEUTRAL-or-worse if lookups dominate.
- **Confidence:** L (the Apple-GPU decode feasibility is the binding constraint, not the quality). **Where:** Colab (fit) → M3 (only if feasible).

---

#### Layer 2 — The computation itself (do less, like the brain)

##### L2.1 — Per-token compute budgeting (early exit / mixture-of-depths)
- **Mechanism:** easy tokens are resolved by the first E of 36 layers; hard tokens use the full stack. Confidence-gated dynamic depth (CALM/LayerSkip-style).
- **Benefit:** cuts *average* bytes/token independent of bit-width — and it is one of the **few genuine energy wins**, because skipped layers are compute and bus traffic genuinely not performed.
- **Exact?** Q (changes outputs unless paired with a full-model verify; with verify it can be made lossless at verify cost).
- **Oracle:** on a code capture, measure argmax-agreement between E-layer and full-model logits across E; compute the average-depth reduction at your tolerated agreement. If easy tokens cluster, it works; **if they don't (small-model headroom is limited — this is the L-confidence risk), it dies on the oracle.**
- **Apple-GPU/M3 feasibility:** favorable (skipping layers is fewer contiguous weight reads; no new access pattern). The early-exit head is small.
- **Energy verdict:** GENUINE (less compute drawn AND fewer bytes moved).
- **Confidence:** L on a 3B (the depth headroom may be thin). **Where:** Colab (calibrate exit) → M3.

##### L2.2 — Contextual sparsity at the right granularity (the resurrected lever)
- **Mechanism:** most FFN neurons do not fire for most tokens (the brain's event-driven sparsity). Predict the active set per token and read only those weights. The block-256 version **died** (active neurons scattered, 0.2% @ 99% recall). The resurrection is **offline co-activation permutation** — reorder neurons so co-firing ones land in contiguous, GPU-friendly blocks — at a finer granularity (32/64) than block-256.
- **Benefit:** fewer weight reads/token; sparsity in the form the GPU tolerates.
- **Exact?** Q (predicted sparsity; recall-gated).
- **Oracle (decisive):** on the existing FFN capture (`_capture/q3b_ffn.bin`), build the co-activation matrix, permute, and measure best-case byte-cut at 99% recall at block sizes {32, 64, 128}. Must clear ~30%+ at an aligned block to justify a kernel. **Prior says it likely re-dies on Qwen's SwiGLU (not ReLU → not natively sparse); run it to confirm-or-kill cheaply.**
- **Apple-GPU/M3 feasibility:** only favorable if permutation yields large contiguous droppable runs; otherwise it degenerates into the gather problem the Apple GPU punishes. Predictor overhead adds dispatches (bad — but Layer-4 scheduling helps).
- **Energy verdict:** GENUINE if it lands (fewer reads + less compute).
- **Confidence:** L (fights SwiGLU's lack of true sparsity). **Where:** Colab (permute) → M3.

##### L2.3 — Predict-only-the-surprise (speculation as the core loop)
- **Mechanism:** the brain mostly predicts and only spends energy on prediction *error*. Speculative decoding is exactly that: draft cheaply, verify in one batched pass, pay full cost only where the draft was wrong. n-gram/suffix-automaton draft (free, CPU, lossless, ideal for code's copy-rate) as the safe default; EAGLE-3 (3.27–6.47× in the literature, top end on code) once the kernels are bandwidth-bound and there is idle compute for the draft — gated on a real-transcript acceptance oracle.
- **Benefit:** amortizes the dominant per-token cost across accepted tokens; the multiplier *past* dense parity. The batched verify reuses each weight across K positions, amortizing scale-decode compute (the proven bottleneck) across the time dimension.
- **Exact?** E (lossless with exact-match verify).
- **Oracle:** replay real coding transcripts; measure mean accepted length for n-gram/SAM (≥~2.5 ⇒ strong win, ~1 on prose ⇒ free); for EAGLE-3, measure early-draft acceptance on code before committing the Colab training run.
- **Apple-GPU/M3 feasibility:** n-gram draft is a CPU automaton (~zero GPU cost). EAGLE-3 has the small-fast-model caveat (little idle GPU while compute-bound) — hence the Layer-3/Layer-4 dependency. Verify is a K-wide GEMM your kernels already shape.
- **Energy verdict:** GENUINE on accepted-heavy spans (fewer total forward passes per accepted token); the n-gram draft adds trivial CPU energy.
- **Confidence:** M-H on code (workload-conditional). **Where:** M3 (n-gram now) + Colab (EAGLE-3 training).

---

#### Layer 3 — The model molds to its owner (the moat the incumbents cannot enter)

##### L3.1 — Online vocabulary + draft specialization
- **Mechanism:** the engine watches the user's live token distribution and adapts: prune the output head to the vocabulary actually in use; tune the speculative draft on the user's accept/reject history.
- **Benefit:** compounding speedups **specific to this user** and impossible to copy — a general engine cannot specialize to one user's distribution.
- **Exact?** Vocabulary pruning is E if paired with a certifiable screen (cf. the bible's low-rank lm_head screening — exact greedy when the true argmax is in the screened set). Draft tuning is E (lossless verify).
- **Oracle:** measure the user's effective vocabulary coverage over a session (what fraction of 152K is ever the argmax) and the SVD-screen recall for exact greedy. Both are NumPy-afternoon measurements.
- **Apple-GPU/M3 feasibility:** favorable; vocab screen is contiguous, draft tuning is offline-ish (periodic, on accumulated history).
- **Energy verdict:** GENUINE for vocab pruning (the lm_head is ~10% of bytes/token; reading less of it is less bus energy); NEUTRAL-to-GENUINE for draft tuning (better acceptance = fewer forwards).
- **Confidence:** M-H. **Where:** M3 (runtime, with periodic light offline updates).

##### L3.2 — Workload-adaptive quantization
- **Mechanism:** recalibrate the quantization importance matrix (and mixed-precision assignment) on the **user's actual data**, not a generic calibration set. The bits go where *this user's* workload needs precision.
- **Benefit:** better quality-per-bit on the workload that matters; a privacy-preserving personalization a cloud engine cannot do per-user.
- **Exact?** Q (quality trade; tuned to be better for this user than generic).
- **Oracle:** compute a code-corpus imatrix vs a generic imatrix; KL/perplexity on held-out user-style data. Proceed if the user-calibrated mix beats generic by a margin worth the re-quant cost.
- **Apple-GPU/M3 feasibility:** uses existing codecs; the re-quant runs offline/periodically (Colab or background on M3).
- **Energy verdict:** NEUTRAL (same bytes, better-placed; not a power lever).
- **Confidence:** M (auto bit-allocation beats naive only ~70% of the time — oracle-gated). **Where:** Colab/background → M3.

##### L3.3 — On-device continual personalization (LoRA on usage)
- **Mechanism:** periodically fine-tune small LoRA adapters on the user's accumulated usage (their codebase, their style), entirely on-device. The model learns the user, privately.
- **Benefit:** **the moat.** "It gets better the more you use it" — the one line no cloud or wrapper can say, because they cannot train on the user's private data on the user's machine. This is the brand, the differentiator, and the reason the whole stateful thesis pays off commercially.
- **Exact?** Q (the model's behavior changes by design — this is the point, not a regression).
- **Oracle:** this is a *capability* bet, not a perf lever — validate on the target workload's real downstream tasks (does a LoRA'd-on-the-user's-repo model measurably help on that repo's tasks?) vs the base model. Quality eval, not perplexity.
- **Apple-GPU/M3 feasibility:** LoRA fine-tuning on Apple Silicon is feasible (mlx-lm supports it; on-device training is the path). Adapter hot-swap at inference is cheap (small matrices). Training runs in the background / overnight; Colab is the heavy-lift fallback for bigger adapters, but the *private* version must be on-device.
- **Energy verdict:** training draws energy (a cost, not a saving) — but it is amortized and optional/scheduled (overnight, plugged-in). Inference with a LoRA is NEUTRAL.
- **Confidence:** M (mechanism proven; the product value depends on whether per-user adaptation measurably helps — likely yes for a focused coding workload). **Where:** M3 (on-device, the privacy point) + Colab (fallback for larger adapters).

---

#### Layer 4 — The system around the model (the parts nobody treats as the engine)

##### L4.1 — The scheduler as a first-class citizen
- **Mechanism:** treat dispatch order, CPU/GPU/ANE placement, and the host loop as a *tunable system*, not glue. Includes: closing the wall-clock-vs-GPU-busy residual (the ~12–15% host gap — zero-alloc persistent loop, GPU-side sampling, tightest command-buffer reuse), spatial speculation (n-gram draft on CPU overlapped with GPU verify), and **protecting the GPU clock** by keeping host work light.
- **Benefit:** converts the engine from "fast kernels with overhead around them" into a tight system; recovers throughput the kernels already earned but the host wastes.
- **Exact?** E (scheduling).
- **Oracle:** the bible's `host_wall − Σgpu_us` per token *is* this budget; drive it to zero. For placement, `powermetrics`/`macmon` GPU-frequency under GPU-only vs +host work — confirm the GPU clock holds (the **verified power-budget tension**: the M3 Pro's 6 P-cores share one macOS-set frequency that drops as thread count rises, and the SoC shares a power budget, so heavy host work can pull the GPU clock down).
- **Apple-GPU/M3 feasibility:** all software; the ANE path is gated (ANE is Core ML–only, not Metal-programmable — high-ceiling, high-effort, separate gate).
- **Energy verdict:** THERMAL/GENUINE (a tighter loop finishes sooner and idles, drawing less total energy per token; protecting the GPU clock avoids throttle-waste).
- **Confidence:** M-H. **Where:** M3.

##### L4.2 — Energy as a measured, branded axis (joules-per-token)
- **Mechanism:** instrument and optimize **quality-per-watt / joules-per-token**, and publish it the way you publish tps. Most engines optimize speed; almost none lead on energy. For a local laptop tool, "runs cool, won't torch your battery, sips power" is real and on-brand with privacy/local-first.
- **Benefit:** a competitive flag **nobody is flying**, dead-on-brand, and a content/marketing axis alongside the bench.py series. Distribution and differentiation, not just engineering.
- **Exact?** N/A (a measurement + positioning discipline).
- **Oracle:** instrument joules-per-token on the M3 (`powermetrics`/`macmon` expose GPU/CPU/package power, frequency, temp — no kexts). Establish the baseline today; report it alongside tps in the bench series.
- **Apple-GPU/M3 feasibility:** measurable today.
- **Energy verdict:** this *is* the energy axis. **Honest scope (per §0's caveat):** the levers that genuinely move joules-per-token are the ones that **do less work** — L1.2 (skipped forwards), L2.1 (skipped layers), L2.2 (skipped neurons), L2.3 (fewer forwards via spec), and tighter scheduling (L4.1). Levers that only move bytes *faster* (kernel efficiency) are largely **NEUTRAL** on energy because a bandwidth-bound GPU draws similar power busy or stalled. So "fit more in the same energy envelope" is true specifically for the do-less levers, not for the speed levers — **market the do-less wins as the energy story; do not claim kernel speedups as energy savings.**
- **Confidence:** H (it is a measurement + positioning move). **Where:** M3.

##### L4.3 — The storage / load path
- **Mechanism:** model load time, cold-start, memory residency (`mlock` wiring — **not** huge pages, which Apple Silicon does not support), and access-order weight layout. The felt-performance layer.
- **Benefit:** "instant" feel and sustained-clock behavior — users notice load time and responsiveness more than they notice 39-vs-50 tps. Felt performance is perceived quality.
- **Exact?** E.
- **Oracle:** measure cold-start and first-token latency; page-fault count on token 2+ (should be ~0 once resident). Access-order layout validated by busy-time bandwidth (cf. bible §7.1).
- **Apple-GPU/M3 feasibility:** all available; **verified constraint** — macOS Apple Silicon is 16 KB pages only, `VM_FLAGS_SUPERPAGE` fails, so the lever is wiring + layout, not page-size.
- **Energy verdict:** NEUTRAL (latency/felt-performance, not power).
- **Confidence:** M-H. **Where:** M3.

---

#### Layer 5 — The boundary dissolves (co-design)

##### L5.1 — Heal the model into the engine
- **Mechanism:** stop treating the model as fixed input. Fine-tune on Colab into a shape that is simultaneously **accurate and ideal for hawking's kernels**: GPU-mappable structured sparsity (so L2.2 finally works), fusion-friendly layer structure, a residual stream fine-tuned to tolerate low precision (the f16-residual-error finding is the signal that there is headroom here), and a layer structure amenable to cross-layer delta (L1.3) or low-rank (L1.4).
- **Benefit:** the most literal "moldable foundation" — model and engine stop being separate artifacts. The wins from L1.3–L1.5 and L2.1–L2.2 get **baked in** (trained-for) rather than bolted-on (extracted post-hoc), which is the difference between a lever that "could come back small on Qwen" and one you *engineered* to be large.
- **Exact?** Q (a different, healed model — validated to match or beat the base on real tasks).
- **Oracle:** this is the unifier, so its oracle is downstream — once L1.3/L1.4/L2.2 oracles report what structure *would* help, a small healing fine-tune tests whether the model can be pushed into that structure without quality loss (eval on real tasks, not perplexity).
- **Apple-GPU/M3 feasibility:** healing runs on Colab; the *result* is a model file hawking serves. No new runtime constraint.
- **Energy verdict:** GENUINE downstream (it makes the do-less levers land), via the levers it unlocks.
- **Confidence:** M (the mechanism is sound; it is the long-horizon unifier, highest-effort).
- **Where:** Colab (heal) → M3 (serve).

---

### 8.2 — Tiering — what this means for the roadmap

| tier | levers | what it is | why now |
|---|---|---|---|
| **The stateful core (the moat)** | L1.1, L1.2, L3.1, L3.2, L3.3 | adaptive/stateful/specialized — the half incumbents structurally cannot build | changes *what the engine is*; defensible; mostly E or capability-bets; high-leverage and unspent |
| **The bus levers** | L1.3, L1.4, L1.5 | shrink unique bytes/token, offline, oracle-gated | stack on quant; each cheap to kill before a kernel |
| **The compute levers** | L2.1, L2.2, L2.3 | do less work per token (the brain principle) | the genuine energy wins live here; L2.3 (spec) is the multiplier past parity |
| **The system layers** | L4.1, L4.2, L4.3 | scheduler, energy axis, load path — felt performance + brand | recover earned throughput; own the energy flag nobody flies |
| **The unifier** | L5.1 | co-design model↔engine | long-horizon; makes the bus/compute levers land by design |

**The honest grading, held throughout:** the stateful core (L1.1, L1.2, L3.1, L3.3) is the strong, defensible bet — I'd stake the thesis on it. The bus and compute levers are real but each is **oracle-gated and several are L-confidence** (L1.3 cross-layer delta, L1.5 learned codebook, L2.1 early-exit, L2.2 contextual sparsity) — they may come back small or dead on Qwen, and the discipline is to *measure before building*, exactly as block-256 taught. L4.2 (energy) is a positioning win with a narrow true-savings scope. Not every item is "groundbreaking"; the strong ones stand on merit and the speculative ones are flagged as the experiments they are. That honesty is what makes the plan actionable instead of inspirational.

---

### 8.3 — Sequencing (the build order)

**Phase A — Oracles first (days, mostly offline, run in parallel; no kernels).**
The whole point of the discipline. Run, on real captures/transcripts:
1. **L1.2 prefix/semantic-cache hit-rate** on coding transcripts — likely the highest-ROI, lowest-risk win; measure first.
2. **L1.1 KV working-set** attention-mass concentration on long-context captures.
3. **L2.3 spec acceptance** (n-gram/SAM mean accepted length on code) — sets the entire multiplier ceiling.
4. **L3.1 vocab coverage + SVD-screen recall** (exact-greedy).
5. The **kill-or-keep** offline oracles: L1.3 inter-layer similarity, L2.2 co-activation permutation byte-cut, L1.4 low-rank+residual KL, L1.5 per-model codebook quality+feasibility, L2.1 early-exit agreement. Each is a NumPy afternoon; each either greenlights a kernel or dies cheaply.

**Phase B — Ship the stateful core that cleared its oracle (the moat, the differentiated work).**
Prefix caching (L1.2) and KV working-set (L1.1) first — biggest capability + energy wins, mostly E. Then online vocab/draft specialization (L3.1). Then begin the on-device LoRA loop (L3.3) — the brand-defining capability — as a parallel track (training is offline/overnight).

**Phase C — The do-less compute levers (the energy story).**
n-gram/SAM speculation (L2.3) now (free, lossless); EAGLE-3 once kernels are bandwidth-bound (Phase D of the bible) and its acceptance oracle clears. Early-exit (L2.1) and contextual sparsity (L2.2) only if their oracles cleared — these are the genuine joules-per-token wins to market via L4.2.

**Phase D — The system layers (recover earned throughput + own the brand axis).**
Scheduler tightening (L4.1) to close the host gap; energy instrumentation + publishing (L4.2) alongside the bench series; load-path (L4.3) for felt performance.

**Phase E — The bus levers that cleared their oracle, then the unifier.**
Whichever of L1.3/L1.4/L1.5 cleared (build at most one of the byte-cut codecs); then L5.1 co-design to bake the structural wins in by training-for-them.

**The first move:** Phase A, item 1 — the prefix/semantic-cache hit-rate oracle on real coding transcripts. It is the cheapest, highest-confidence, most-differentiated win (statefulness the incumbents can't match), it is a genuine energy win (skipped forwards), and it needs no kernel to validate. If the hit rate is high — expected for code — it jumps the queue ahead of everything.

#### 8.3.1 — Kill Protocol (mandatory before any NO-GO)

Block-256 sparsity, then four more levers (L1.3/L1.4/L1.5/L2.2), died on offline oracles. The oracles were rigorous; the risk they do *not* by themselves guard against is **killing the idea when only one *form* of it was tested.** So, as a standing rule for this project:

**Before any lever is marked NO-GO (here or in `reports/dead_levers.md`), the kill MUST record three things:**

1. **Type-1 vs Type-2.** *Type-1* = died on a measured property of reality that no implementation cleverness changes (a delta with higher variance than the original is not compressible by *any* method; an FFN active set that is ~half the neurons is not "sparse"; Apple Silicon has no hardware gather). *Type-2* = died in the *form* tested, but a different formulation attacks the same goal (a different basis, **data-aware vs data-free**, **gather-free vs gather**, **trained-for vs extracted-post-hoc**).
2. **The reframe considered.** State the specific alternative formulation explicitly — even if only to reject it. "We tested form X; the obvious reframe is Y."
3. **Why the reframe also dies, OR a pointer to the reframe's oracle.** A Type-2 reframe earns "alive" **only** with a named, cheap (offline / CPU NumPy-scale) oracle that could kill it. No nameable oracle → the lever stays dead; **never resurrect on vibes.** And **never re-test a recorded Type-1** — its death is a fact about reality, not about our effort.

The expected shape on a well-trained *stock* model: most post-hoc structural levers are **Type-1** (the structure was trained *out*, or never existed), and the honest survivors are the **gather-free codec (QTIP)** and the **trained-for co-design path (L5.1)**. That is not a failure — the protocol exists to catch the *rare* Type-2 (here, activation-aware low-rank) before it is buried with the Type-1s, and to stop the same Type-1 being re-proposed a third time.

**Kill ledger — the four Phase-A kills, retro-filled:**

| Lever | Type | Form tested that died | Reframe considered | Reframe's fate (oracle pointer or why-dead) |
|---|---|---|---|---|
| **L1.3** cross-layer delta | **Type-2 (narrow)** | **Data-free, weight-space** delta `W[L+1]−W[L]`; cosine ≈ 0 at distance 1 *and* 35, std-ratio 1.61 (anti-compressible), affine `α*≈0`. Weight-space is Type-1 dead at any distance. | **Data-aware (activation-weighted) cross-layer reference**: is `W[L+1]`'s action on its *real* inputs partly captured by the already-resident `W[L]`? Frobenius cosine ≠ data-norm cosine. | **Oracle written, not yet run:** `tools/bench/oracle_dataaware_lowrank.py` (L1.3 section). Prior is *against* it (weight orthogonality + full-ish activation rank); and a GO collapses into L1.4/L5.1, not a standalone lever. Confirm-or-kill cheaply, then stop. |
| **L1.4** low-rank+residual | **Type-2 (strong)** | **Data-free SVD** (Frobenius energy): top-64 captures 3–9% (FFN) / ~26% (attn); residual std ~0.95 → `U,V` are dead overhead. | **Activation-aware SVD** (ASVD / SVD-LLM): SVD on `W·C^{1/2}`, `C=E[xx^T]`. Frobenius is the *wrong* objective for an inference codec — a weight can be full-rank in Frobenius yet low-rank in the data norm. **The original oracle ran the naive form the literature explicitly improves on.** | **Oracle written, not yet run:** `oracle_dataaware_lowrank.py` (L1.4 section) — measures data-weighted energy@r and the functional-error-at-matched-bytes vs Q4_K. The one genuinely-too-early kill; decisive offline. (Target = Q4_K-recon, a lower bound; a marginal GO defers to the f16/AWQ lane.) |
| **L1.5** learned codebook | **Type-1** (gather form) | **Raw k-means** index→value LUT = per-element **random gather**; no HW gather on Apple Silicon → killed at the feasibility gate before quality. Hardware wall, basis-independent. | **Gather-free learned code** (QTIP bitshift-trellis / lattice): codes *computed* arithmetically on contiguous bits, no LUT read. | **Survives the contiguity gate by construction — but it is QTIP**, already the documented survivor and routed to the GPU/quality lane (build at most one byte-cut codec). No *new* resurrection of L1.5. Cheapest next offline check = trellis reconstruction-MSE vs Q4_K at matched bits, left for attended QTIP work (a wrong trellis sim is worse than none). |
| **L2.2** contextual FFN sparsity | **Type-1** | **block-256** (0.2% @99%); **PowerInfer static hot/cold** (0% permanently cold); **static co-activation permutation** {32,64,128} (1.5/0.8/0.3% vs 30% bar). Plus **dynamic per-token gather** also dies: 99% energy needs **39–53% of neurons** (active set is *not* small), top-200 hot neurons reshuffle **22–25% Jaccard/token** (no static layout packs them), and scattered per-token selection on Q4_K 256-superblocks *is* the Apple-hostile gather. The load-bearing "few neurons" premise is measured **false**. | **Trained-for sparsity** (L5.1): finetune toward ReLU / structured block sparsity so the exploitable structure *exists*. | **Real but expensive co-design**, already the documented L5.1 path — multi-day finetune + quality risk, and it has **no cheap oracle** (its test *is* the training run + real-task eval). It converts a dead *extraction* lever into a speculative *training* bet. **Do not re-propose post-hoc extraction.** |

---

### 8.4 — The honest bottom line

The System-Level Shift is real, and it is the right bet — not because it is a paradigm shift for inference engines writ large (the brain-grade version of that is hardware you don't build), but because the **stateful/adaptive/specialized dimension is empty**, and owning it is a defended product position that the general engines gave up by design. hawking reaches "all" by being **fast enough** (the tps lane → parity) and then being the **only local engine that treats the model as alive** — managing context as a working set, reusing computation across a session, molding to its user privately, doing less work like a brain, and sipping power as a measured, branded virtue. Keep the engine; reframe what it is *for*. Every lever above earns its place by an oracle, not by enthusiasm — and the strong ones (caching, KV working-set, on-device personalization, speculation) are strong enough that they don't need the hype.

*Companion docs: `plans/throughput_bible_2026_05_30.md` (the physics + kernel ceiling, incl. §7 the physical floor), `plans/roadmap_2026_05_30.md` (in-flight). This plan is the systems + product axis orthogonal to raw tps.*

---


---

## The canonical kill registry

Every dead lever — with its Type-1 / Type-2 classification, killing evidence, killing memory, and a resurrection check — is maintained in **[../reports/dead_levers.md](../reports/dead_levers.md)**. The §8.3.1 worked-example ledger above is the four Phase-A structural kills (L1.3 / L1.4 / L1.5 / L2.2) retro-filled under the Kill Protocol; the registry is the full, authoritative list and is the doc to update when a new lever dies. Before re-spawning any lever, read its resurrection check there. **Never resurrect a Type-1 kill** (its death is a fact about reality); **never resurrect a Type-2 on vibes** — only behind its named, cheap oracle.
