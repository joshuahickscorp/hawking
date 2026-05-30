# Throughput Bible — reconciliation crosswalk

**Date:** 2026-05-30
**Reconciles three docs:**
- `plans/throughput_bible_2026_05_30.md` — **CANONICAL** (corrected diagnosis + 3-axis frame + critical path)
- `plans/silicon_architecture_audit_2026_05_29.md` — audit plan, 8 topics (framing superseded-in-part; topic catalog still useful)
- `silicon-builds/SUMMARY.md` — empirical build/bench of 16 solutions (2026-05-29 evening)

This doc exists so nothing is orphaned: every silicon solution (#1–#17) and every audit topic (1–8) is mapped to a Bible axis or marked dead, and every place the Bible *overrides* the audit plan is stated explicitly.

---

## 1. Chronology — why three docs, which one wins

1. **Audit plan (2026-05-29, AM).** Enumerated 8 silicon-level hypotheses. Framed on two premises now known false: *(a)* "the 36 ms/token gap between Σgpu_us and decode wall is REAL and GPU-side" → megakernel (Topic 2) is THE lever; *(b)* cross-engine concurrency (Topic 5) is "the highest-ceiling unaudited lever." Anchored at **26.6 dec_tps**, ceiling **63**.
2. **Silicon-builds (2026-05-29, PM).** Turned 15 candidates into real code and benched them. Empirically **killed** the cross-engine front (AMX/ANE/CPU) and the entire memory-placement front (heap/super-page/mlock/prefetch/multi-queue), and **confirmed** two live levers (#8 simdgroup-MMA, #13 zero-copy) + the byte-cut prize (#16/#17). This is the audit plan's Phase A/B executed for most topics.
3. **Bible (2026-05-30).** Corrects the *core diagnosis*: the gap was a measurement artifact (token-count ÷64 vs argmax-proven ÷32, plus an earlier command-buffer round-trip mismeasurement). **Decode is ~85% GPU-busy → kernel-bound.** Reorganizes the survivors into 3 axes with a dependency-ordered critical path, and formally deprioritizes the megakernel.

**Verdict:** the Bible is canonical. The audit plan's **topic catalog and per-topic dead-front conclusions remain valid reference**; its **framing** (the gap is the enemy, megakernel is the lever, cross-engine is top-ceiling, 26.6 anchor, 63 ceiling) is **superseded**.

---

## 2. Where the Bible overrides the audit plan (explicit supersessions)

| audit-plan claim (location) | Bible correction (§) | status |
|---|---|---|
| "The 36 ms/token gap is REAL and GPU-side; code-only investigation exhausted" (§0, Topic 2) | gap was ÷64-vs-÷32 token-count artifact; busy-time-BW invariant rules out a large idle fraction; decode ~85% busy (§0, §1.1) | **SUPERSEDED** |
| Topic 2: megakernel collapses the gap → "100 dec_tps capped at 63 = 2.4× e2e," "lean harder into megakernel" | megakernel attacks the ~12–15% gap; ceiling ~+15%, not +70%; deprioritized (§0, §4) | **SUPERSEDED** — also empirically: 8-layer fused = 4.4× slower (commits `e03ce26`/`dc7fdf2`/`a9c6280`) |
| Topic 7: speculative cross-layer co-issue fills idle gap (GO if "gap-was-idle") | GPU is busy ~85% → ~no idle to fill | **DEAD** per Bible |
| Topic 1: hard ceiling = **63 dec_tps** (1.93 GB / 120 GB/s) | ~66 tps at Q4_K_M (~85% of 150 peak); and the denominator is not fixed — **~99 tps wall at 3-bit** (§0, §2 axis-2, §3) | **REFRAMED** — ceiling is per-bitwidth, not a single number |
| Anchor = **26.6 dec_tps** (predec default-on, but contaminated/closed window) | Bible anchors **~39 tps clean** under the §1 thermal protocol | **OPEN** — must unify before any go/no-go (see §5) |
| Topic 5: cross-engine is "the single biggest lever in this plan" | single-engine; no second engine helps the bus-bound, dependency-chained decode (§0 hardware truth) | **SUPERSEDED** — silicon-builds killed every arm empirically |

---

## 3. Full lever crosswalk (Bible ↔ silicon #N ↔ audit Topic)

### Axis 1 — stream weights faster

| Bible axis-1 lever | silicon-builds | audit topic | state / note |
|---|---|---|---|
| predec everywhere + ILP default + LM-head→predec | (shipped: predec default-on `6f0209e`; `_2r` ILP opt-in) | — | **Stage 1** — generalize what's proven |
| hoist-constants audit | — | — | **Stage 1**, new |
| vectorized nibble unpacking | part of **#8** | Topic 8 (Q4_K layout) | **Stage 2** |
| multi-row register blocking (4–8 rows) | part of **#8** | **Topic 6** (M3 Dynamic Caching) | **Stage 2** — Topic 6 is folded in here |
| simdgroup-matrix decode | **#8 simdgroup-MMA — LIVE** (+10–20% batched/prefill, bit-identical) | Topic 8 | **Stage 2 primary** — prototype exists in `silicon-builds/dismantle-q4k-mma/` |
| split-K reduction (big FFN GEMVs) | — | Topic 6 | **Stage 2** |
| per-quant-type threadgroup tuning | — | Topic 6 | **Stage 2** — constant search |
| coalesced predec layout repack | q4k_fast (shipped) / #8 repack | Topic 8 | **Stage 2** |
| stacked-QKV single GEMM | rel. #14 multi-queue (DEAD) & concurrent-QKV (+1.68%, held) | — | **Stage 2**, low conf |
| GPU-side sampling | **#7 GPU top-K sampler — REDUNDANT** (greedy argmax already on-GPU) | Topic 4 | only temp>0 gap remains, unbenched |

### Axis 2 — stream fewer bytes

| Bible axis-2 lever | silicon-builds | audit topic | state / note |
|---|---|---|---|
| imatrix mixed-precision (existing codecs) | **#16 mixed-prec — PRIZE** (naive RTN dead; quantifies +31% Q3 / +71% Q2) + **#17 AWQ-lite** (importance-aware reaches it) | — | **Stage 3** — AWQ in flight; needs model-level PPL gate |
| QTIP 3.0–3.25 lookup-free trellis | NEW | — | **Stage 3** — Colab; only pays once bandwidth-bound |
| fused quantized-KV attention | **#15 int4 KV — CONDITIONAL** (cosine 0.998; 1.2 GB→340 MB @32K) + Q8-KV (landed) | Topic 4 (SLC/KV) | **Stage 5** — quality-gated, long-context only |
| structured pruning + distillation | NEW | — | low conf, Colab |
| QTIP 2-bit | NEW | — | likely fails 3B quality floor |

### Axis 3 — more tokens per stream (speculation)

| Bible axis-3 lever | silicon-builds | prior work in repo | state / note |
|---|---|---|---|
| n-gram / suffix-automaton (PLD/REST/SAM) | — | track C n-gram was BLOCKED (parity bug on repetitive prompt) | **Stage 4 first** — free CPU draft, lossless |
| EAGLE-3 / 3.1 draft head | — | eagle5 port on this branch (`2f037a2`, `e25c033`, `6be1057`); **head NOT trained** (empty checkpoints) | **Stage 4** — Colab train, gate on acceptance oracle |
| batched-verify compute-amortizer | — | batched predec GEMM (`6be1057`, `e25c033`) | in flight |
| MTP heads (DeepSeek-style) | — | — | alt to EAGLE-3, low conf |

### Orthogonal (not a decode-tps axis, but LIVE)

| item | silicon-builds | note |
|---|---|---|
| zero-copy mmap MTLBuffer loader | **#13 — LIVE** (1673× faster bind, −1.9 GB RSS, bit-identical) | load-path / TTFT / RSS; adopt regardless of axis work; helps slm coexistence |

---

## 4. Dead-front crosswalk — Bible "deprioritized/dead" ↔ silicon DEAD ↔ audit topic

Every dead silicon solution and every dead audit topic, accounted for. The Bible and the silicon-builds session agree on all of these.

| front | Bible position | silicon-builds | audit topic | cause of death |
|---|---|---|---|---|
| megakernel / persistent GPU loop | deprioritized (ceiling ~+15%; Metal has no `grid.sync()`) | — | **Topic 2** (was "Strong GO") | gap is ~12–15% not 67%; 8-layer fused 4.4× slower |
| CPU gap-fill | near-dead | **#5 CPU offload** | — | non-GEMM = 3.2% of wall, on serial chain |
| ICB single-pass forward | host overhead off crit path | **#4 ICB** | Topic 2 | host encode = 0.27% of per-dispatch budget |
| argbuf pre-compiler | same | **#9** | Topic 2 | host encode negligible (= #4) |
| AMX GEMV | single-engine | **#1 hybrid AMX, #11 AMX-fused** | Topic 5 | AMX 1.1–2.5× slower than GPU; can't ingest Q4_K |
| ANE / CoreML FFN | single-engine | **#6 ANE (measured)** | Topic 5 | bandwidth-bound at 56 GB/s, 4–7× slower than GPU FFN |
| multi-command-queue | single-engine | **#14** | Topic 5/7 | GPU saturated per decode kernel (32–1376 TGs ≫ 16-TG crossover) |
| super-page mempool | (weights already placed) | **#2** | Topic 3 | arm64 refuses 2 MB pages (kr=4); streaming already TLB-amortized |
| MTLHeap residency v2 | (weights already placed) | **#3 heap-v2** | Topic 3/4 | heap 8.5% slower than separate shared buffers even batched |
| weight prefetcher | (weights warm in RAM) | **#10** | — | WILLNEED −29% once warm; weights always warm in 18 GB |
| mlock allocator | inverts coexistence | **#12** | — | pins 10% RAM from slm |
| speculative cross-layer co-issue | no idle to fill | — | **Topic 7** | GPU busy ~85%; depends on Topic 2 "gap-is-idle" which is false |

---

## 5. Open questions to settle in Stage 0 (before any code)

1. **Anchor reconciliation.** Audit plan = 26.6 dec_tps; Bible = ~39 clean. These differ by both bench contamination (Claude-open inflation per `bench_contamination`) and clean-window protocol. Lock one canonical anchor under the §1 thermal protocol (warmup + median-of-N + SoC temp log) so every downstream go/no-go uses the same baseline.
2. **The three offline oracles** (Bible §4 Stage 0): spec-acceptance on real code transcripts (ranks the entire axis-3 ceiling), SVD lm_head recall, mixed-precision KL/PPL on a code imatrix. NumPy/llama.cpp-tool afternoons; they rank the rest of the program.
3. **§1 invariant gate.** Wire the four physical invariants into `analyze_tcb_trace.py` as hard asserts (busy-BW ≤ 150 GB/s; Σkernel ≈ busy ±5%; argmax token count; bit-identical parity), and calibrate once against an Instruments Metal System Trace. Note: `tools/bench/analyze_tcb_trace.py` is already modified in the working tree — check what's there before adding.

---

## 6. Accounting check (nothing orphaned)

- **Silicon solutions #1–#17:** all 16 placed (#1 dead, #2 dead, #3 dead, #4 dead, #5 dead, #6 dead, #7 redundant, **#8 live→axis1**, #9 dead, #10 dead, #11 dead, #12 dead, **#13 live→orthogonal**, #14 dead, **#15 conditional→axis2**, **#16+#17 prize→axis2**).
- **Audit topics 1–8:** all placed (1 reframed-ceiling, 2 superseded/dead, 3 dead, 4→axis2-KV + GPU-sampler-redundant, 5 dead, **6→axis1 multi-row/tuning**, 7 dead, **8→axis1 layout/#8**).
- **Bible levers with no prior prototype** (flagged NEW, Colab-side): QTIP 3-bit & 2-bit, structured pruning/distillation, hoist-constants audit, n-gram/SAM runtime, EAGLE-3 trained head.
