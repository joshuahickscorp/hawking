# Staging-session worklist — build the architecture now, derisk every later lever

**Purpose.** A parallel "staging" session that builds the scaffolding + enabling
infrastructure for the whole remaining Throughput-Bible program, so later work is
"fill in the kernel body + bench," not "design from scratch." Paste this whole file
as the opening prompt for a fresh chat.

**Parallelization model (3 lanes that don't block each other):**
- **CPU-scaffold lane (do most of this NOW):** kernel stubs + signatures, dispatch
  wiring behind env gates, parity-test skeletons, bench harnesses, docs, housekeeping.
  None of this needs the GPU — many items run in parallel.
- **GPU-bench lane (serialized):** each lever's bit-identical/atol parity + paired
  bench. One at a time, whenever the M3 GPU is free (it's currently tied up by the
  EAGLE capture; benches wait for that). Paired deltas are contamination-robust, so
  a quick A/B can run even under light contention.
- **Cloud lane (separate machine):** AWQ/QTIP quantization, EAGLE training.

**Ground rules (carried from the main program):**
- §1 methodology gate on every number (`tools/bench/analyze_tcb_trace.py` — exits 2
  on a physics violation). Decode is **kernel-bound, ~85% busy**; the gap to llama
  is *bandwidth efficiency* (~41% of peak vs ~60%), not parallelism.
- Exact kernels gate on **bit-identical greedy** (`tools/bench/path_to_50_verify.sh`
  pattern); quality-trade kernels gate on **atol 1e-3 fp16**.
- Commits: inline `git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com'`,
  **no Claude/Generated-by attribution**, never force-push.
- Don't touch files another session owns mid-flight (coordinate on `colab/02_*` and
  the Q4_K kernel files).

---

## TIER 0 — Enabling infrastructure (build FIRST; unblocks all of Tier 1–3)

**0.1 xctrace profiling + export pipeline.** *The Stage-2 unblock.* The homemade
TCB trace is split-CB-distorted (the §1 gate rejects it), so kernel levers are
guesses (4r was a dead guess). `tools/bench/mst_capture.sh` already wraps
`xcrun xctrace record --template "Metal System Trace"` (xctrace 16.0 is CLI). Build
the **export + analysis** half: `xctrace export` → a per-kernel GPU-occupancy /
stall / achieved-bandwidth breakdown for the decode pass, so the next lever targets
the real bottleneck. *Independent, CPU+one-GPU-capture. Effort: M. Gate: matches
`analyze_tcb_trace.py` busy-fraction within 5% (calibrates the homemade tool).*

**0.2 §1-gated clean paired-bench harness.** Generalize `path_to_50_verify.sh` into
one command that takes {env-A, env-B}, runs interleaved paired trials, prints median
+ delta + the §1 gate verdict, writes JSON. Every Tier-1 lever reuses it. *Indep.
CPU. Effort: S.*

**0.3 Local torch venv (python3.12).** `python3.12 -m venv /opt/torchenv && pip
install torch safetensors numpy`. Unblocks `tools/eagle5_forward_dump.py` + the
`eagle5_forward_parity` test **locally** — so the EAGLE forward-parity diagnostic
(3.2) needs no Colab. *Indep. CPU+network. Effort: S.*

**0.4 Map the "other" kernel bucket.** The §1 gate flagged ~5.6% unmapped (fails
INV2). Expand `static_kernel_name` (`crates/dismantle-core/src/metal/mod.rs`) so
every decode dispatch is named and traces pass the gate. *Indep. CPU. Effort: S.*

**0.5 Long-context test harness.** No long-context timing exists; build a harness
that decodes/prefills at 4K/16K/32K context and reports per-token + KV byte-share.
**Gates** Tier-2 fused-KV (proves where KV bandwidth actually bites). *Indep. CPU +
GPU-bench. Effort: M.*

---

## TIER 1 — Kernel levers (Stage 2/3): scaffold now, bench when GPU free

For each: build the **kernel stub + dispatch behind an env gate + a parity-test
skeleton + a bench entry**, so only the kernel body + the bench remain.

**1.1 Prefill MMA port (recommended first).** Port the LIVE `#8` simdgroup-MMA
prototype (`silicon-builds/dismantle-q4k-mma`: `gemm_q4k_mma`, `gemm_q4k_mma_nwide`)
into the production batched/prefill path (`gemm_q4_k_m_batched_v3w`, dispatched
`kernels/mod.rs:595+`, used at `qwen_dense.rs:1063` / `batch_prefill` :1328).
Proven **+10–20% bit-identical** on M>1 (the *easy* MMA regime). Speeds **TTFT** —
directly relevant to the product's file-context workload. *Indep of decode/EAGLE/
Colab. Effort: M-L. Gate: bit-identical prefill + paired prefill bench.*

**1.2 f16 predec scales.** predec expands 16 B packed scales → 64 B f32/block
(+33% bytes; 192 vs 144). Store predec scales as **f16** (32 B → 160 B/block, −17%)
to cut the dominant-GEMV bandwidth. Touch `ensure_q4k_predec_cache`
(`qwen_dense.rs:2407`) + a `half`-scales kernel variant + atol-1e-3 parity.
*Indep. Effort: M. Gate: atol 1e-3 + token-identical greedy + paired bench.*

**1.3 Vectorized nibble unpack.** Load Q4_K weights as `uint`/`uint4` and unpack 8
nibbles with vector ops instead of per-`uchar`, for better coalescing toward peak
BW. Variant of the predec kernels in `shaders/quant.metal`. *Indep. Effort: M. Gate:
bit-identical + paired bench.*

**1.4 Stacked-QKV single GEMM.** Concat `[Wq;Wk;Wv]` into one matrix/dispatch
(load-time repack + one GEMM) — fewer dispatches + better core saturation. Distinct
from the prior +1.68% concurrent-encoder attempt. *Indep. Effort: M. Gate:
bit-identical + paired bench. Expected +1–3%.*

**1.5 LM-head → predec.** Route the Q4_K LM head (currently `gemm_q4_k_m_v3_8r`,
~4–5% of decode) through the predec cache. *Indep. Effort: S-M. Gate: bit-identical
+ bench. Expected +1–2%.*

**1.6 Q3_K fast-GEMV kernel.** *Unlocks the byte-cut.* Oracle proved Q3_K quality is
viable (+4.7% PPL, −11% bytes from a clean source) but dismantle has **no fast Q3_K
kernel** — a Q3_K model runs the generic path (~19 tps, slower than Q4). Build a
Q3_K predec/GEMV (Q3_K block layout differs from Q4_K's 144 B). Then a Q3_K model is
faster, realizing the byte-cut. *Indep. Effort: L. Gate: bit-identical vs CPU Q3_K
deq + paired bench vs Q4_K_M.*

**1.7 simdgroup-matrix decode (the MLX-class headline; HARD, multi-session).** The
real path from ~41% → ~60–80% of peak on the *decode* GEMV (M=1 underfills the 8×8
tile — fill it by processing multiple output rows). This is the big Stage-2 lever
and the one that needs 0.1's profiling to target. *Indep. Effort: XL. Gate:
bit-identical + busy-time-BW under §1.*

---

## TIER 2 — Long-context / product (Stage 5)

**2.1 Fused quantized-KV attention.** From `silicon-builds/dismantle-int4kv` (#15,
per-channel int4, cosine 0.998). Read 4/8-bit KV inline in one attention dispatch
(no f16 KV buffer). Neutral at short ctx, real byte-cut + memory win at >16K — the
product's file-context regime. *Gated by 0.5. Effort: L. Gate: real-model PPL at
long ctx + paired bench at 16K/32K.*

**2.2 Prefill/TTFT micro-batch.** Large prefill micro-batch + the 1.1 MMA (M>1
shines). Reduces time-to-first-token on long prompts. *Overlaps 1.1. Effort: M.*

---

## TIER 3 — Spec-decode (Stage 4) — independent of the current EAGLE retrain

**3.1 n-gram / SAM runtime + real-transcript oracle.** Oracle A gave τ=1.43 (NO-GO)
on a *repo-source* corpus, but a real code-completion workload (high copy-rate) may
clear ~2.5. Build the lossless prompt-lookup/suffix-automaton speculator (CPU
automaton, ~zero GPU draft cost) as a runtime option, AND re-run
`tools/bench/oracle_spec_accept.py` on **real product transcripts**
(`llama-tokenize -f transcripts | oracle_spec_accept.py`) to decide GO/NO-GO before
building. *Indep of EAGLE. Effort: M (runtime) / S (oracle). Gate: τ≥2.5 on real
transcripts, then lossless bit-identical + paired bench.*

**3.2 EAGLE forward-parity fix.** Per `plans/eagle_forward_parity_handoff.md`: once
0.3 (local torch) is up, run `eagle5_forward_dump` + `eagle5_forward_parity` on the
trained head to confirm runtime==PyTorch, then align `speculate/eagle5_forward.rs`
or the accept loop. (The `num_blocks=2` retrain may already fix it — verify first.)
*Needs 0.3. Effort: M-L.*

---

## TIER 4 — Land + housekeeping (low-risk, reduces future friction)

**4.1 Land the verified wins.** The path-to-50 **+32.5%** (24→32 dec_tps,
bit-identical) + the 2r-default-flip + the §1 gate tooling are **uncommitted** in
the working tree — secure them in clean commits. *Indep. CPU. Effort: S.*

**4.2 MEMORY.md consolidation.** Over the 24.4 KB load limit (entries truncate
silently). Run `/consolidate-memory` — merge dupes, prune the index. *Indep. Effort: S.*

**4.3 ROADMAP + dead-lever registry.** Update `reports/dead_levers.md` + a roadmap
with current state: shipped (predec +32.5%), dead (4r, naive-mixed-prec, host-
dispatch family), in-flight (EAGLE num_blocks=2, byte-cut-pending-Q3-kernel),
queued (this list). Stops future sessions re-treading. *Indep. Effort: S.*

**4.4 Repo hygiene.** Decide policy on: the ~190 MB committed EAGLE corpus (LFS? keep?),
the 629 MB untracked `silicon-builds/`, the `*.parquet` force-adds. *Indep. Effort: S.*

---

## Suggested staging order (max parallelism)
1. **Immediately, in parallel (CPU):** 0.2, 0.3, 0.4, 4.1, 4.2, 4.3 — small, independent, unblock the rest.
2. **Then scaffold (CPU):** 1.1, 1.2, 1.5, 1.6, 2.1, 3.1 — stubs + gates + parity skeletons (no bench yet).
3. **Build 0.1 (profiling)** → use it to rank which Tier-1 body to write first (don't guess like 4r).
4. **GPU-bench lane (when free):** parity + paired bench each scaffolded lever, one at a time, profiling-ranked.
5. **0.5 + 2.1** for the long-context/product front; **3.2** once 0.3 is up.
6. **1.7 (simdgroup-matrix decode)** is the XL headline — schedule it as its own multi-session push after 0.1 shows the exact stall.

**Highest-confidence first wins:** 1.1 (proven +10–20% prefill), 4.1 (secure +32.5%),
1.6 (unlocks the proven byte-cut). **Biggest ceiling:** 1.7 (MLX-class decode) — but
gate it on 0.1's profiling so it's targeted, not guessed.
