# path-to-50 gap diagnosis — 2026-05-29

**Task:** execute `HANDOFF_consolidated_path_to_50.md` REVISED FIRST MOVE —
locate the real decode gap to llama.cpp's 50.4 tps before any kernel grind.
**Box:** M3 Pro 18 GB, Qwen-2.5-3B-Instruct-Q4_K_M, locked env
(`TCB + VOCAB_PRUNE=32000 + Q4K_LMHEAD + FFN_DOWN_Q4K + Q4K_PREDEC`).
**Contamination:** Chrome + Claude desktop open; all absolute tps figures
deflated ~6%. Decisions use paired ratios (contamination cancels).

## Bottom line

1. **The 2× gap is REAL** (paired ratio, not contamination): dismantle
   **25.08** dec_tps vs llama.cpp **47.35** tg64, same box, back-to-back =
   **0.53×**. llama.cpp clean is 50.4; here 47.35 (the ~6% haircut both pay).
   The handoff's hopeful "clean dismantle is closer to 50" is refuted.

2. **The handoff's GB/s gate that "refuted the kernel gap" is itself an
   artifact.** It read `bench-kernel` *min* latencies (e.g. ffn 11008×2048 =
   122 µs → "104 GB/s = 69% of peak") and concluded the GEMV is near-peak.
   But `bench-kernel`'s `dispatch_threads` (`metal/mod.rs:612-626`) does
   `new_command_buffer → commit → wait_until_completed` **per dispatch**.
   That ~130 µs floor is the **command-buffer round-trip**, not the kernel.
   Proof: the min latency is **flat at ~110-170 µs across a 44× byte range**
   (256×2048 = 0.29 MB → 11008×2048 = 12.7 MB) — the *smallest* matrix is the
   *slowest*. A bandwidth-bound kernel would scale with bytes. So we do NOT
   actually know the Q4_K kernel's true DRAM efficiency from `bench-kernel`,
   and "the GEMV is already 69-84% of peak, don't grind it" is unsupported.

3. **Decode is one TokenCommandBuffer, one `commit_and_wait` per token**
   (`qwen_dense.rs:2992` create, `:3760` commit; the per-layer loop dispatches
   into that single TCB). So the CB round-trip is paid **once** per token
   (~150 µs = negligible). This re-confirms every prior host-side kill — ICB,
   encode-fusion, commit stalls, PSO, concurrent-encoder are all moot for the
   gap. The gap is not on the host.

4. **In-decode `gpu_prod` trace at the current locked env:** GPU-busy =
   **17.62 ms/token**, wall = **41.8 ms/token** → only **42% of decode wall is
   inside measured kernel GPU-time**; ~58% is not. (May-24 trace was 43%; the
   predec default-on flip and FFN_DOWN_Q4K did not change the shape of this.)
   `gemm_q4_k_m_v3_8r` + the unmapped predec kernel (`gemv_q4_k_v4_predec_
   pinned_tcb`, lands in "other") dominate GPU-busy; norms/rope/attn/silu are
   <8% combined.

## What the gap is (and is not)

| candidate | status | evidence |
|---|---|---|
| FFN GEMV memory bandwidth | **NOT the gap** (and not measurable via bench-kernel) | flat min vs 44× bytes; bench-kernel floor is CB round-trip |
| Command-buffer round-trips / commits | **NOT the gap** | 1 commit/token; round-trip ~150 µs/token |
| CPU encode / ICB / PSO / host | **NOT the gap** | prior measured 0.2-1.5 ms; reconfirmed (1 TCB) |
| **Inter-dispatch GPU idle at M=1** | **the live suspect** | 17.6 ms GPU-busy vs 40 ms wall, within ONE command buffer, host already excluded |

The ~22 ms/token unaccounted for sits **inside a single command buffer**,
after CPU encode, between ~320 serial dependent dispatches. At M=1 (single
token) each GEMV fills a fraction of the GPU and cannot overlap its neighbor
(residual-stream dependency), so the GPU stalls in launch/drain between
dispatches. This is the classic decode latency-bound regime.

### The one thing this session could NOT settle

Whether that ~22 ms is (a) genuine GPU inter-dispatch idle, (b) a `gpu_prod`
counter-sample-barrier artifact (`withBarrier:true` per dispatch), or (c)
under-measured kernel time. A synthetic TCB microbench (`tests/tcb_dispatch_
cost.rs`, added this session) shows in-TCB marginal per-dispatch cost of
~7 µs (q/o) / ~35 µs (ffn) and is **flat whether 1 or 32 distinct weight
buffers are cycled** — because freshly-allocated buffers are zero pages and
Apple Silicon memory compression serves all-zero reads without touching DRAM.
So the microbench measures dispatch launch + compute floor, not real DRAM
traffic, and cannot close the kernel-vs-gap fork. **This is the same wall the
May-24 sessions hit; it needs a Metal System Trace (Instruments).**

## Recommended lever

All interpretations point to the **same lever family: fewer, fatter serial
dispatches at M=1** — i.e. kernel fusion / megakernel.

- **The megakernel is NOT dead as a concept.** The 8-layer STOP
  (`megakernel_revival_nlayer_bench_2026_05_29`) failed because it used **f16
  weights (4× the DRAM bytes) + a single threadgroup (1 of ~18 cores)** — two
  penalties unrelated to the fusion idea. A megakernel that keeps weights
  **Q4_K-inline** and uses **multiple threadgroups** directly attacks the
  inter-dispatch idle. The N-layer scaffold (`qwen3b_megakernel_nlayer`) is
  committed and parity-green; it is the starting point.
- **Ceiling:** if the ~22 ms inter-dispatch idle were fully removed, decode →
  ~18 ms/token = **~56 tps**, past llama.cpp's 50.

## The cheap gate before the multi-week grind

llama.cpp hits 50 tps with the **same per-op dispatch model** (it is not a
megakernel). That means its per-dispatch GPU overhead is ~half ours — the
question is *why*. The decisive, cheap-relative-to-a-megakernel step is the
**attended Metal System Trace** the May-24 session already flagged as held:
capture one `kernel_mul_mv_q4_K_f32` (llama.cpp) and one dismantle Q4_K
dispatch at 11008×2048 from a clean window, diff occupancy / threadgroup
sizing / inter-dispatch gap. That settles whether the win is "launch our
existing kernels back-to-back with no drain" (fusion) or "our kernel is
genuinely under-occupied at M=1" (kernel rewrite) — before weeks of work.

## Cheap fusion VALIDATED the lever (2026-05-30)

Before any megakernel grind, a single cheap fusion was shipped to test the
hypothesis that **dispatch-count reduction at M=1 is the lever**: a new
`gemm_q4_k_v4_predec_pair` kernel computes FFN **gate + up** in ONE dispatch
(they share the post-norm activation), behind `DISMANTLE_QWEN_FFN_GATEUP_FUSE=1`.
This removes 36 dispatches/token (1/layer).

- **Parity: bit-identical** 32-tok greedy off vs on (same arithmetic per row).
- **Paired bench (6 interleaved trials, locked env, 64 tok):**
  OFF median **29.21** vs ON median **31.55** dec_tps = **+8.0%**, zero trial
  overlap (every ON > every OFF). All 86 workspace lib tests pass.

**This confirms the diagnosis:** fusing just ONE dispatch pair per layer buys
+8%. The gap really is per-dispatch / inter-dispatch overhead at M=1, and the
fix is fewer, fatter dispatches. Gate+up fusion is now **DEFAULT-ON** (opt out
`DISMANTLE_QWEN_FFN_GATEUP_FUSE=0`), like the predec default-on flip.

### k+v fusion benched FLAT — the win scales with dispatch SIZE

The same pair kernel was wired for **k+v** (k_proj/v_proj are also Q4_K, same
256×2048 shape, same input), behind `DISMANTLE_QWEN_KV_FUSE` (kept **DEFAULT-OFF**).
3-way interleaved bench: NONE 28.91 → gate+up 31.51 (+9.0%) → +k+v 31.46 (**flat**).
Bit-identical throughout. **k+v fusion adds nothing** — k/v are tiny (256 rows,
~tens of µs), so removing their dispatch saves negligible inter-dispatch drain,
whereas the big gate+up dispatches (11008 rows, ~300 µs) carry a large drain.

**Refined lever:** fuse the BIG dispatches (FFN), not the small attention
projections. The megakernel should prioritize the FFN block. This also nuances
the "uniform per-dispatch overhead" picture — the recoverable overhead scales
with dispatch duration, so a megakernel's payoff comes mostly from chaining the
big GEMVs without drain, not from cutting dispatch count per se.

## ⚠️ MAJOR CORRECTION (2026-05-30) — the gap is ~15%, NOT 58%; kernels ARE the wall

The "58% inter-dispatch gap" above is **WRONG** — a token-count artifact. The
`gpu_prod` trace captures only **32 tokens** (proven: `sample_argmax_f32` fires
1/token and appears exactly 32 times) but the analyzer divided total GPU time by
`completion_tokens=64`, halving GPU-busy. Corrected:

| config | GPU-busy/token | wall/token | gap |
|---|---|---|---|
| no fusion | **35.25 ms** | 41.8 ms | ~16% |
| gate+up fused | **29.03 ms** | 33.2 ms | ~12% |

**Decode is ~85% GPU-busy — the kernels are the wall.** Effective bandwidth ≈
**55 GiB/s = 37% of peak**; llama.cpp runs ~60% of peak. So our Q4_K GEMVs run at
~half llama's efficiency — the lever IS kernel efficiency (this vindicates the
handoff keystone's *direction*; only its bench-kernel GB/s *measurement* was a
CB-round-trip artifact). The gate+up +8% was a kernel-efficiency win (GPU-busy
35.3→29.0 via ILP), which is why tiny k+v was flat. (`analyze_tcb_trace.py` now
auto-derives traced-token count from the argmax/sample count.)

## SECOND WIN: ffn_down → predec = +21%

ffn_down was the #1 GPU consumer (in `v3_8r` at 46%) but, unlike the projections,
was **not on the predec kernel** — it re-decoded Q4_K sub-block scales inline
every dispatch (43 blocks/row × 2048 rows). Its predec scale table already
existed at load (`ffn_down_q4k_predec` for requant'd layers; `predec_cache` for
native-Q4_K layers) but was unused. Wired both paths to `gemv_q4_k_v4_predec`
(`DISMANTLE_QWEN_FFN_DOWN_PREDEC`, **default-on**):

- **Bit-identical** 32- and 64-tok greedy (different prompts).
- **+21–22%** decode tps (paired, ffn_down was scale-decode-compute-bound, not
  BW-bound — which is why 37%-of-peak looked low).

## THIRD WIN: predec 2-row ILP = +4%

predec is now 82% of GPU time, so the pair kernel's ILP trick was generalized to
single GEMVs: `gemm_q4_k_v4_predec_2r` computes **2 output rows of the same matrix
per simdgroup** with 2 independent accumulator chains, sharing the single x load —
hiding DRAM latency for q/o/ffn_down. Behind `DISMANTLE_QWEN_PREDEC_2R` (kept
**opt-in/default-off** — newest, highest register pressure, smallest win; user
flips after review). Bit-identical; **+4.1%** paired.

## Cumulative this session

| stage | flag | delta | bit-identical |
|---|---|---|---|
| baseline | — | — | — |
| + gate+up fusion | `FFN_GATEUP_FUSE` (default-on) | +8% | ✓ |
| + ffn_down predec | `FFN_DOWN_PREDEC` (default-on) | +21% | ✓ |
| + predec 2-row ILP | `PREDEC_2R` (opt-in) | +4% | ✓ |

**Total ≈ +34%** (none → all-on). Paired ratio to llama.cpp **0.50 → 0.68**
(heavy-contamination window; cleaner windows ~0.77, ~39 vs 50.4 tps). The gap is
~⅓ closed. Remaining: LM head (v3_8r, last big GEMV not on predec, ~4%), then
deeper kernel work (simdgroup-MMA-style Q4_K, or the FFN megakernel) for the rest.

## Artifacts produced

- `reports/traces/qwen3b_decode_gpu_prod_2026_05_29.json` — current-env in-decode trace.
- `crates/dismantle-core/tests/tcb_dispatch_cost.rs` — TCB batched-dispatch microbench (reusable).
- `shaders/quant.metal` `gemm_q4_k_v4_predec_pair` + `kernels/mod.rs`
  `gemv_q4_k_v4_predec_pair_pinned_tcb` + qwen_dense wiring behind
  `DISMANTLE_QWEN_FFN_GATEUP_FUSE=1` → **+8.0%**, bit-identical.
- Paired bench: dismantle 25.08 / llama 47.35 = 0.53 (pre-fusion baseline).
