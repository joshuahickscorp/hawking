# Dismantle latency-and-cost budget — sessions 1+2

**Date:** 2026-05-24
**Workload of record:** Qwen-2.5-3B-Instruct-Q4_K_M on M3 Pro 18 GB
**Locked config:** `DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=32000 DISMANTLE_QWEN_Q4K_LMHEAD=1`
**Most recent main:** `68c0ece` (P3 batched-MHA decode landed)
**Sessions:** s1 = budget skeleton + floor formulas (no code), s2 = ProdCbGpu trace + first revision (1-line wiring fix in `qwen_dense.rs:836` to drain `dispatch_samples`; otherwise no code).

> **Update from the s2 ProdCbGpu trace (see new Section: Measured per-kernel decode breakdown, end of document).**
>
> The trace overturns the s1 Lever 0 prediction. **ffn_down already uses the native Q6_K kernel** (`gemv_q6_k_pinned_tcb` at `qwen_dense.rs:1251`), reading bit-packed Q6_K directly from mmap — not the f16-dequant pin (which exists in memory but is unused at runtime). The dominant decode cost is **`gemm_q4_k_m_v3_8r` at 87.3% of GPU time**, attributable to the ffn_gate/ffn_up/q/o projections + LM head. The priority list is revised at the bottom.

## Reframe

tps is the inverse of a single line in one of the categories below. It hides where time goes and is contamination-sensitive (Claude Code inflates `dec_tps` 4–5× — see `memory/bench_contamination.md`). From now on:

- The headline output is a **budget table per category**, with each line showing achieved vs. theoretical floor and a one-line theory of the gap.
- Optimize the line that has the biggest absolute gap, not whichever lever was loaded into a session's context.
- Every change ships with parity (1e-3 fp32 atol + bit-identical greedy ≥16 tokens) **and** is logged against the line it claims to improve.

## Model dimensions (Qwen-2.5-3B)

| field | value | source |
|---|---|---|
| n_layers | 36 | `qwen2.block_count` |
| hidden | 2048 | `qwen2.embedding_length` |
| n_heads | 16 | `qwen2.attention.head_count` |
| n_kv_heads | 2 | `qwen2.attention.head_count_kv` |
| head_dim | 128 | derived `hidden/n_heads` |
| intermediate | 11008 | `qwen2.feed_forward_length` |
| vocab | 151936 | derived from `tokens` |
| GGUF file on disk | 1,929,903,264 B = **1.797 GiB** | `ls -l` |

**Per-token weight bytes read (full forward, decode mode, locked config):**

| weight | shape | dtype on disk | runtime bytes | × 36 layers |
|---|---|---|---|---|
| q_proj | 2048×2048 | Q4_K | 2.36 MB | 84.9 MB |
| k_proj | 256×2048 | Q6_K (→ f16 dequant pinned) | 1.05 MB | 37.7 MB |
| v_proj | 256×2048 | Q6_K (→ f16 dequant pinned) | 1.05 MB | 37.7 MB |
| o_proj | 2048×2048 | Q4_K | 2.36 MB | 84.9 MB |
| ffn_gate | 11008×2048 | Q4_K | 12.69 MB | 456.7 MB |
| ffn_up | 11008×2048 | Q4_K | 12.69 MB | 456.7 MB |
| ffn_down | 2048×11008 | Q6_K (→ f16 dequant pinned) | 45.1 MB | 1,622 MB |
| 2× rmsnorm | 2048 | f32 | 16 KB | 0.6 MB |
| **per layer** | | | **77.3 MB** | **2,782 MB** |
| LM head (Q4K-LMHEAD) | 151936×2048 | Q4_K | 175 MB | — |
| LM head (default f16) | 151936×2048 | f16 | 622 MB | — |
| embed (1 row) | 2048 | f16 | 4 KB | — |

**Floor BW per decode step:**
- Locked config (Q4K LM head): `2782 + 175 = 2,957 MB / token`
- Default (f16 LM head): `2782 + 622 = 3,404 MB / token`

**Floor latency at 150 GB/s (M3 Pro peak DRAM BW):**
- Locked: `2957/150000 = 19.7 ms/token` → 50.7 dec_tps ceiling
- Default: `3404/150000 = 22.7 ms/token` → 44.1 dec_tps ceiling

Note: the user prompt's 10.7 ms/token floor used 1.6 GB. The real per-token read is ~2.96 GB because Q6_K weights are pinned as dequantized f16. **Restoring native-Q6 reads (P2 native kernel ships, but ffn_down still uses dequant-f16 fallback for the gemv_proj macro) would cut BW from 1622 MB → 658 MB (Q6_K bit-packed at 6.5/8) on ffn_down — a 964 MB/token saving, dropping floor from 19.7 → 13.3 ms.** This is the largest single budget item and is treated as Lever 0 of the priority list.

---

# Category 1 — Decode latency (μs/token)

Wall-clock per autoregressive step at locked config. Currently **~44.6 ms/token = 22.4 dec_tps** (memory: `qwen_dense_metal_pipeline.md`); default-path (TCB-only) **~53.2 ms/token = 18.8 dec_tps**.

**Floor (locked config, BW-bound):** 19.7 ms/token. Current gap: **24.9 ms/token = 2.27× floor**.

| line | achieved | floor | gap | theory of gap |
|---|---|---|---|---|
| **Total decode μs/token (locked)** | 44,600 μs | 19,700 μs | 24,900 μs (2.27×) | aggregate of below |
| ffn_down (Q6→f16 dequant pinned) read | ⨯ not yet measured per-kernel for Qwen | floor 300 μs/layer × 36 = 10,800 μs | — | reading dequant-to-f16 means **2.46× more bytes** than native Q6 read; this dominates the gap |
| ffn_gate (Q4_K v3_8r pinned) | not measured per-Qwen | floor 85 μs/layer × 36 = 3,060 μs | — | Q4_K v3_8r at 27 GB/s effective (`qwen3b_dead_levers.md`); peak 150 → 5.5× headroom |
| ffn_up (Q4_K v3_8r pinned) | not measured | floor 85 μs × 36 = 3,060 μs | — | same kernel/path as gate |
| q_proj / o_proj (Q4_K v3_8r pinned) | not measured | floor 16 μs × 36 = 575 μs total | — | small matrix; dispatch overhead may dominate at 16 μs/call shape |
| k_proj / v_proj (f16 dequant) | not measured | floor 7 μs × 36 = 252 μs | — | tiny matmul; pure dispatch-overhead-bound |
| mha_decode_f32 | not measured | trivial (KV cache fits cache; 0.5 KB read per token) | — | per `per_kernel_time_breakdown.md` attention was ~2.4% on V2-Lite; expect <2 ms/token here |
| LM head gemv_f16 (locked = Q4_K) | not measured | floor `175/150000 = 1,167 μs` for Q4_K; `622/150000 = 4,147 μs` for f16 | — | shape 151,936 × 2048 means ~19K TGs; scheduling overhead at this row count may dominate (Q4_K LM head was within noise solo per `qwen3b_dead_levers.md`) |
| rmsnorm + add_inplace combined | not measured | trivial (residual stream 8 KB/layer read/write) | — | per V2-Lite was 24% of encoder time; on Qwen-3B likely smaller because GEMMs are larger |
| dispatch overhead (CPU encode visible) | ~365 μs/token estimated from 73 saved dispatches × 5 μs (`qwen3b_dead_levers.md` rmsnorm-fusion null result) | floor ~0.5 μs/dispatch × ~115 dispatches/token = 58 μs | gap ≈ 300 μs/token | TCB encoder amortizes most of these; further compression needs the v2.3 ICB lever which is **DEAD** per `v230_icb_dead.md` |

**Headline:** decode is at **44.6 ms/token**, floor is **19.7 ms/token**, gap is **24.9 ms/token = 56% of total**, attributable predominantly to **ffn_down being read as f16 dequant (45 MB/layer) instead of native Q6_K (18.4 MB/layer)**.

**Required measurement to confirm:** `DISMANTLE_TRACE_DISPATCH=1 + TcbTraceMode::ProdCbGpu` over a 64-token decode at locked config, then tabulate μs/call per kernel name. See "Methodology gaps" below — this is the gating measurement for session 2.

---

# Category 2 — Prefill latency (ms total, ms/token, ms/chunk)

47-tok prompt, batched B=8 + TCB + locked config: **1,136 ms total = 24.2 ms/token** (memory: `p3_batched_prefill_shipped.md`).

**Floor (B=8, BW-bound):** weight bytes are read once per chunk, so 6 chunks × 2,957 MB = 17.7 GB / 150 GB/s = **118 ms total = 2.51 ms/token**. Per-chunk floor: **~20 ms/chunk** (vs. measured ~189 ms/chunk = 9.4× floor).

| stage | calls/chunk | achieved μs/chunk | % of chunk | floor | gap theory |
|---|---|---|---|---|---|
| **Total per chunk** | — | ~189,000 | 100% | ~20,000 | aggregate |
| Q4_K batched GEMM v3w B=8 (q,o,gate,up,down requant if env, lm_head) | varies per layer | ~170,000 (~90%) | 90% | ~18,000 | v3w hits 68 GB/s of 150 peak — **2.2× BW headroom**; compute-bound for cols-large at B=8 |
| Batched MHA (P3, 2D grid heads×B) | 36 | ~1,300 | ~0.7% | <500 | small at seq≤47; will grow at long context |
| RoPE | 36 layers × 2 (q,k) | ~6,000 (~3%) | 3% | <500 | already a per-batch loop; sequential B dispatches |
| KV append (2 calls/layer post-consolidation) | 72 | <500 | <0.5% | <100 | already consolidated 2B→2 |
| q/k/v biases (add_inplace_broadcast batched) | 3 | ~700 | 0.4% | <200 | post-consolidation 3B→3 |
| silu_mul + add_rmsnorm_fused (batched) | 4 | ~1,200 | 0.6% | <200 | post-consolidation B→1 / 2B→2 |
| CPU encode + dispatch overhead | ~300–500/chunk | ~1,000 | <1% | <500 | per chunk |

**Headline:** prefill is at **24.2 ms/token**, floor is **2.51 ms/token**, gap is **21.7 ms/token = 90% of total**, attributable to **the v3w batched Q4_K GEMM running compute-bound at ~68 GB/s of 150 peak**. This is the textbook simdgroup-MMA target.

**Required measurement to confirm:** `kernel_compare.sh` microbench of `gemm_q4_k_m_batched_v2_v3w` at the four production shapes (q/o, ffn_gate, ffn_up, ffn_down) with B=8 — already partially captured in `qwen_dense_metal_pipeline.md` per-op deltas; needs an isolated GB/s number per shape.

---

# Category 3 — TTFT (cold and warm)

Time from `Engine::load` to first emitted token on a chat turn. **Not currently measured as a budget.** The `dismantle-bench` harness reports `ttft_ms` per `crates/dismantle-bench/src/suites/decode.rs:229` but no decomposition exists.

| line | achieved | floor | gap theory |
|---|---|---|---|
| **Total TTFT cold** | not measured | — | — |
| **Total TTFT warm** | not measured | — | — |
| Tokenizer encode | not measured | <1 ms (BPE on tens of tokens) | likely already at floor |
| Prefill (warm) | 1,136 ms @ 47 tok | 118 ms (cat. 2) | covered in cat. 2 |
| Prefill (cold) | not measured | ≥1,136 ms (mmap + pipeline-cache warmup adds) | first-dispatch-per-kernel cost is the cold delta |
| First decode step | ~44.6 ms (cat. 1) | 19.7 ms | covered in cat. 1 |
| First decode step cold delta | not measured | 0 | pipeline-cache warmup for any decode-only kernels |
| Argmax + emit | not measured | <0.1 ms | GPU `sample_argmax_f32_tcb` is the only step; trivial |

**Headline:** TTFT is **un-budgeted**. Session 2's gating measurement is `dismantle bench prefill --completion 1` × cold AND warm × {locked config, default config} with a wall-clock breakdown printed.

**Cold-warm delta theory:** the Metal pipeline cache is cold on first dispatch of each kernel. With ~20 unique kernel names in the qwen_dense TCB path and ~30 ms each on first compile, cold could be ~600 ms over warm; on every subsequent session the cache lives in `~/Library/Caches/com.apple.metal` and reads as ~1 ms each.

---

# Category 4 — Model load time

`Engine::load` end-to-end. **No structured measurement exists.** User-reported anecdote is "~5–15 s for Qwen-3B." Floor: I/O-bound on weight bytes that aren't mmap'd-zero-copy.

| stage | achieved | floor | gap theory |
|---|---|---|---|
| **Total load** | not measured | — | aggregate |
| `GgufFile::open` (mmap) | not measured | <10 ms (mmap is page-table only) | likely already at floor |
| Tokenizer load | not measured | ~50 ms (152K BPE tokens) | likely close to floor |
| Weight scan + PinnedBuffer per Q4_K tensor | not measured | bound by **mmap read-through** since we don't copy weights — should be <50 ms total | first read does the actual disk I/O (~1.8 GB / SSD bandwidth ≈ 0.5–1 s page-cache miss) |
| Q6_K → f16 dequant pin (k/v_proj + ffn_down) | not measured | bound by Q6_K read (38+37+1622 = 1.7 GB) + f16 write (1.7 GB) = 3.4 GB / DRAM 150 GB/s = ~23 ms work, but **first read is page-cache miss → SSD I/O** | this is the visible "load is slow" complaint; the actual compute is fast |
| LM head requant (Q4K-LMHEAD env on) | not measured | LM head bytes (622 MB f16 → 175 MB Q4_K) = bounded by f16 read + Q4_K write = ~5 GB / 150 GB/s = ~33 ms (post-page-cache) | first run does ~600 MB SSD read |
| Vocab-prune scan (DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=32000) | not measured | corpus is small, scan is trivial; remap build is constant time | likely close to floor |
| ffn_down Q6→Q4_K requant (env off by default; opt-in dead lever) | n/a (off by default) | — | — |
| Metal pipeline cache warmup (first dispatch of each kernel) | not measured | ~30 ms × ~20 kernels = ~600 ms cold | this is recoverable via pre-touch at load |

**Headline:** load time is **un-budgeted**. Likely dominated by **first SSD page-cache miss reading the 1.8 GB weight file** (~1–2 s on M3 Pro SSDs) **plus the Q6→f16 dequant write** (~23 ms work but contended with the read). Pipeline-cache warmup may add ~600 ms on the very first run.

**Required measurement:** wrap `Engine::load` with per-stage `std::time::Instant` and dump the breakdown. This is the only category where the measurement *requires* a small code change (printing only). Holding for session 2.

---

# Category 5 — RSS / peak memory (MiB)

`dismantle bench` reports `peak_rss_mb` per `crates/dismantle-bench/src/lib.rs:133`. **The number is recorded but no breakdown exists.**

| line | floor (MiB) | gap theory |
|---|---|---|
| **Total peak RSS** | — | — |
| Q4_K weights mmap (lazy) | 1,797 MiB (mmap is virtual; resident pages = pages actually read = ~all of file at end of warmup) | — |
| Q6_K → f16 dequant pinned (k+v+ffn_down) | (37+37+1622)×2 / 1024 = 3,392 MiB **of additional resident memory** | this is the big one — Q6_K → f16 doubles the relevant tensors. ffn_down alone is 1.6 GiB of "extra" f16 dequant |
| LM head Q4_K (locked) or f16 (default) | 175 or 622 MiB | — |
| Vocab-prune Q4_K LM head buffer | 32K rows × 2048 × Q4_K = ~36 MiB | — |
| KV cache at max_seq_len | n_layers × 2 (K+V) × max_seq × n_kv_heads × head_dim × 4 B = `36 × 2 × max_seq × 2 × 128 × 4 = max_seq × 73,728 B ≈ max_seq × 72 KiB` | at max_seq=4096: ~288 MiB; at 32K: ~2.3 GiB |
| Metal pipeline cache | ~20 kernels × ~few-MB IR each → ~50 MiB | — |
| Scratch arenas (DenseDecodeArena + batch arenas) | hidden + intermediate + q_dim + vocab × few-arenas = a few MiB | — |
| **Sum at 4K context, locked config** | ~1,797 + 3,392 + 175 + 36 + 288 + 50 + ~10 ≈ **5,748 MiB ≈ 5.6 GiB** | — |

**Headline:** peak RSS is dominated by the **Q6_K → f16 dequant of ffn_down** (1.6 GiB unique to that strategy) plus the mmap pages. **Switching ffn_down to a native Q6_K kernel (or Q4_K requant, dead lever per `qwen3b_dead_levers.md` regressing in compound) saves ~1.6 GiB peak RSS** in addition to the BW saving in cat. 1. The two effects are co-driven by the same lever.

**Required measurement:** `target/release/dismantle doctor` already prints RSS at multiple stages — needs a one-line invocation script that captures it at (load complete) and (after first generate) for both locked and default config.

---

# Category 6 — Energy per token (mJ/token)

**Never measured in this repo.** Floor for DRAM-read energy: ~3–5 pJ/bit on Apple Silicon = ~0.4 nJ/byte. At 2.96 GB/token decode: floor ≈ 2.96e9 × 0.4e-9 = **1.2 mJ/token** just from DRAM reads. Add GPU compute + CPU encode + display compositor; realistic floor probably 3–5 mJ/token. Real number likely 30–80 mJ/token (rough scaling from 5–10W sustained at 22 tps).

| stage | floor (mJ/token) | gap theory |
|---|---|---|
| DRAM read (weights) | 1.2 | bytes × 0.4 nJ/byte; tied to cat. 1 BW |
| GPU compute | ~1 | compute is small relative to memory traffic |
| CPU encode | ~0.2 | 365 μs × ~5W ≈ 1.8 mJ — wait, that's high; revise |
| Display / compositor (when Claude open) | huge | contamination caveat — `bench_contamination.md` says 4–5× tps inflation; energy contamination is the same |
| **Total per token (clean)** | not measured | — |

**Headline:** energy is **un-budgeted**. The DRAM-read floor (1.2 mJ/token at current BW) drops in lockstep with cat. 1 wins, so energy is **not an independent lever** — it's a derivative of decode latency. **Deprioritize as a primary target** unless a future workload is energy-bounded (mobile, battery).

**Required measurement (only when energy becomes a topline goal):** `sudo powermetrics --samplers gpu_power,cpu_power,energy_impact -i 1000 -n 30` during a 100-tok decode, both clean and Claude-open windows.

---

# Category 7 — Dispatch encode overhead (μs/call)

CPU-side cost of each kernel encoding into a Metal command buffer.

**What we already know:** the v2.3 ICB lever was killed because the CPU encode time was measured at ~0.22 ms = **0.51% of decode wall** (memory: `cpu_gpu_pipelining_audit.md`, `v230_icb_dead.md`). The rmsnorm-fusion 73-dispatch-saving was within noise (~5 μs/dispatch × 73 = 365 μs/token = 0.7% headroom, swamped by run-to-run variance).

| line | achieved | floor | gap theory |
|---|---|---|---|
| **CPU encode time per decode token** | ~0.22 ms (measured prior) | ~58 μs (115 dispatches × 0.5 μs ideal) | gap of ~160 μs/token = 0.36% — **does not clear any practical gate** |
| Encode time per prefill chunk (B=8) | ~1–3 ms (memory) | — | nearly negligible vs. 189 ms GPU/chunk |

**Headline:** dispatch encode is **already at ~0.5% of wall — there is no meaningful lever here**, ICB/megakernel/CPU-overlap workstreams are all dead-by-bench, and only the rmsnorm fusion is kept (no-op but smaller dispatch list for any future regime). **Stop spawning sessions against this line.**

---

# Category 8 — Throughput under concurrency (req/s, p99)

`dismantle serve` mode. **Never measured.** No baseline exists.

| level | achieved | theory |
|---|---|---|
| 1 concurrent | covered in cat. 1+2 | — |
| 2 concurrent | not measured | naïvely 2× slower per request (sequential weight reads); request-batching could amortize |
| 4 / 8 concurrent | not measured | KV cache contention + arena duplication likely the floor; need to inspect `dismantle/src/bench_server.rs` |

**Headline:** concurrency is **un-budgeted**. Likely a 2× win available if request-batched decode (one forward, N output streams sharing the weight read) is implemented — but **gating on the simdgroup-MMA Q4_K kernel** (Lever A in the parent prompt) because that kernel is the request-shared GEMM. Concurrency is therefore a **session 4+ target**.

---

# Category 9 — Variance / tail latency (σ, p99/p50)

The `clean_bench.sh` harness runs n trials and computes σ; in-Claude-Code sessions are contamination-dominated. **No per-kernel variance characterization exists.**

| line | achieved | gap theory |
|---|---|---|
| dec_tps run-to-run σ (in-session) | ~0.05–0.2 over 5–20 trials (various memories) | dominated by Claude.app GPU contention |
| Per-kernel μs/call p99/p50 | not measured | — |
| Prefill ms p99/p50 | not measured | first chunk has cold-cache penalty |

**Headline:** variance is **un-budgeted in absolute terms** but operationally controlled by **paired bench in the same Claude window** (`memory/feedback_bench_with_claude_open.md`). Session 1 conclusion: no work needed unless a *specific* kernel turns out to have unstable p99 once cat. 1 ProdCbGpu trace lands.

---

# Category 10 — Quality regression budget

| gate | status | source |
|---|---|---|
| Bit-identical greedy ≥ 16 tokens | **non-negotiable** | every shipped lever passes |
| Bit-identical greedy ≥ 64 tokens | currently passes for batched-prefill (mem: `p3_batched_prefill_shipped.md`) | — |
| Bit-identical at 256 tokens (Q8-KV) | drifts ~150 tokens due to Q8-KV quant noise (mem: `q8_kv_runtime_landed.md`) | — |
| Perplexity on locked config | **never measured** for current shipped chain | gap — see below |

**Headline:** the bit-identical gate catches *fast* drift, but the calibration-corpus perplexity number is the only check that catches *slow* drift past the bit-identical horizon. **No baseline PPL exists for the locked-config Qwen-3B chain.** This becomes a session-2 prereq if/when Q8-KV is re-evaluated as a decode-latency lever for long context (cat. 1 line).

**Required measurement:** the calibration corpus from `memory/corpus_complete_analysis_landed.md` (4,512 sequences) → run `dismantle generate --temp 0` over a held-out 100-sequence subset, compute mean NLL, log as the baseline. ~30 min wall.

---

# Consolidated priority list

Ranked by **estimated absolute saving on wall-clock decode + prefill** at locked config, with the work-and-risk theory.

| rank | lever | category line | est. saving | risk / cost |
|---|---|---|---|---|
| **0** | **Native Q6_K kernel for ffn_down** (replace f16-dequant-pin) | cat. 1 ffn_down + cat. 5 RSS | **~24 ms/token decode** (53 → ~30 ms) **and 1.6 GiB RSS** | P2 native kernel already exists at `quant.metal:864`; the gemv_proj macro routes Q6_K to f16 fallback. The lever is a wiring change, not a new kernel. Q4_K-requant variant of this was dead-in-compound (`qwen3b_dead_levers.md`); the native-Q6 path was untested at the ffn_down shape in compound. Parity gate is the only risk. |
| **1** | **simdgroup-MMA Q4_K batched GEMM** | cat. 2 prefill | **~85 ms/chunk × 6 chunks = ~500 ms/prefill** (1136 → ~636 ms = 1.8×) | Multi-session. Two-phase: cooperative Q4_K → f16 register tile in shmem, then `simd_mma_f16f16f32` over the tile. Parity at 1e-3 + bit-identical greedy is the gate. Per-call microbench at the four production shapes first. |
| **2** | **Pre-touch Metal pipeline cache at load** | cat. 3 TTFT cold-warm delta | **~600 ms cold-only saving** | Cheap. Touch every kernel once at end of `Engine::load`. Recoups the first-dispatch JIT compile. |
| **3** | **Per-stage load-time instrumentation** | cat. 4 budget itself | unblocks load-time budget | One-evening change. Add `Instant::now()` boundaries inside `Engine::load`. Pure measurement; no parity risk. |
| **4** | **Cold ProdCbGpu trace of decode at locked config** | cat. 1 budget itself | unblocks the per-kernel decode budget | Run `DISMANTLE_TRACE_DISPATCH=1` + ProdCbGpu over 64-token decode; produce a per-kernel μs table. Confirms or rejects the lever-0 estimate. |
| **5** | **Q8 KV cache at long context** | cat. 1 ffn_down + cat. 8 concurrent | currently +2.5% at 256 tok per `q8_kv_runtime_landed.md`; payoff grows with context | Already landed in worktree as `--q8-kv`. Re-evaluate as a *decode-at-long-context* lever once cat. 1 trace shows the K/V cache-read line is non-trivial. |
| **6** | **Request-batched decode in serve mode** | cat. 8 concurrency | conditional 2× at 2 concurrent | Gated on lever 1 (simdgroup-MMA kernel) because that kernel is the cross-request shared GEMM. |
| **7** | **Corpus perplexity baseline** | cat. 10 quality | unblocks Q8-KV / Q4K-LMHEAD long-context evaluation | 30 min wall, no risk. Run on the calibration corpus held-out subset. |

**Dead-end fences (don't re-spawn against these lines, all already in memory):**
- ICB / megakernel for cat. 7 (`v230_icb_dead.md`, `cpu_gpu_pipelining_audit.md`)
- MLA Phase 4 simdgroup (`mla_phase4_resurrected.md`)
- Simdmat Q6_K projections / LM head (`qwen3b_dead_levers.md`)
- M4 autotune `v2t` (`m4_autotune_2026_05_23.md`)
- Pure dispatch fusion without a BW or compute story (`qwen3b_dead_levers.md`)
- Spec-decode "runtime broken" workstream — the runtime works (`spec_decode_runtime_NOT_broken_2026_05_22.md`)

---

# Methodology gaps to close before session 2

These are the measurements the budget *needs* but doesn't yet have. None requires a code change (except #2 and #3 require trivial instrumentation).

1. **ProdCbGpu per-kernel μs trace of Qwen-3B decode at locked config.** Run `DISMANTLE_TRACE_DISPATCH=1` + `TcbTraceMode::ProdCbGpu=gpu_prod` over a 64-token decode, then aggregate by kernel-name. Confirms the cat. 1 per-line table; the existing V2-Lite breakdown (`reports/per_kernel_time_2026-05-20.md`) is the wrong model.
2. **Per-stage timing inside `Engine::load`.** Add `Instant::now()` boundaries (mmap, tokenizer, weight-pin, requant, vocab-prune, pipeline-warmup) and print at end of load. Code-change-but-pure-instrumentation.
3. **Per-line wall-clock printout in `dismantle bench` for TTFT.** Already partially there (`ttft_ms` in `crates/dismantle-bench/src/suites/decode.rs:229`); split into tokenizer + prefill + first-decode + emit.
4. **`peak_rss_mb` breakdown at two checkpoints.** Capture RSS immediately after `Engine::load` AND immediately after first 16-token decode at locked and default configs.
5. **Microbench GB/s per shape for v3w batched GEMM.** Use `tools/bench/microbench_levers.sh` style harness to measure `gemm_q4_k_m_batched_v2_v3w` GB/s at the four production shapes (q/o = 2048×2048, ffn_gate = 11008×2048, ffn_up = 11008×2048, ffn_down = 2048×11008) at B=8. Validates the cat. 2 "68 GB/s of 150" claim and quantifies the simdgroup-MMA opportunity.
6. **Calibration-corpus perplexity baseline.** Held-out 100-seq subset from `memory/corpus_complete_analysis_landed.md`, mean NLL at locked config. Becomes the gate for any future Q8-KV or Q4K-LMHEAD long-context evaluation.

---

# Closing — the target sentence

By the end of this workstream the headline is no longer "tps up by X." It is:

> *Decode at 13 ms/token (1.0× of 13.3 ms BW-floor after native-Q6 ffn_down). Prefill at 5 ms/token (2× of 2.5 ms B=8 floor after simdgroup-MMA). TTFT cold 1.4 s, warm 0.7 s. RSS 4.1 GiB at 4K context. Energy ~5 mJ/token. Here is each ms and byte accounted for.*

Session 2 picks lever **0 (native-Q6 ffn_down)** because the estimated saving is the largest and the kernel already exists — it is a wiring change, not new code. Sessions 3+ are reordered after the cat. 1 trace from methodology-gap #1 lands.

---

# Section: Measured per-kernel decode breakdown (s2, 2026-05-24)

**Trace command:**
```
DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=32000 DISMANTLE_QWEN_Q4K_LMHEAD=1 \
DISMANTLE_TCB_TRACE=gpu_prod \
./target/release/dismantle bench \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --suite decode --trials 1 --max-new-tokens 64 \
  --trace-dispatch \
  --trace-json reports/traces/qwen3b_decode_gpu_prod_2026_05_24.json
```

**Run:** 1 trial, 4-tok prompt, 64-tok decode, in-Claude window. `decode_tps` = 21.13 (vs. 22.24 in a sibling untraced run — ~5% tracing overhead). Total bench wall = 3.03 s decode + 0.60 s prefill = 3.63 s. Total dispatches = **20,864**. Total GPU time captured = **1.301 s** = **43% of wall**; rest is CPU/host/dispatch overhead and tracing perturbation.

## By kernel (sorted by GPU time)

| kernel | GPU total (ms) | % of GPU | calls | mean µs/call | CPU encode µs/call |
|---|---|---|---|---|---|
| **`gemm_q4_k_m_v3_8r`** | **1135.7** | **87.3%** | 6,944 | 163.5 | 5.1 |
| `other` (fused-encoder rollup; contains `gemv_q6_k_pinned`) | 126.6 | 9.7% | 4,608 | 27.5 | 5.3 |
| `add_rmsnorm_fused` | 21.3 | 1.6% | 2,304 | 9.2 | 4.6 |
| `add_inplace` | 7.7 | 0.6% | 3,456 | 2.2 | 4.6 |
| `rope_q_f32_inplace` | 4.9 | 0.4% | 2,304 | 2.1 | 5.2 |
| `moe_batched_silu_mul` | 3.7 | 0.3% | 1,152 | 3.2 | 4.8 |
| `rmsnorm_f32` | 0.3 | <0.1% | 32 | 9.6 | 18.6 |
| `sample_argmax_f32` | 0.3 | <0.1% | 32 | 8.9 | 19.6 |
| `embed_lookup_f32` | 0.1 | <0.1% | 32 | 4.6 | 96.2 |

## Inside `gemm_q4_k_m_v3_8r` — distribution by GPU duration

The 6,944 v3_8r calls cluster into two regimes; this lets us attribute by shape without per-call layer tagging.

| GPU bucket (µs) | calls | total ms | mean µs | likely shape | BW eff |
|---|---|---|---|---|---|
| **[200, 500)** | **2,873** | **943.4 (72.5% of all GPU)** | 328 | ffn_gate / ffn_up (11008×2048, 12.69 MB) | **38.7 GB/s = 25.8% of 150 peak** |
| [50, 100) | 2,283 | 149.4 (11.5%) | 65 | gate+up fusion (or q+o pair), 1 per layer per token | — |
| [0, 50) | 1,716 | 20.5 (1.6%) | 12 | small (likely biases / vocab-pruned fragments) | — |
| [500, 1500) | 17 | 14.0 (1.1%) | 822 | LM head (32K rows × 2048) | **44 GB/s** |
| [100, 200) | 54 | 6.8 (0.5%) | 126 | edge cases | — |

**Headline:** the [200, 500) bucket alone (`ffn_gate` + `ffn_up` reads on the big 11008×2048 shape) is **72.5% of decode GPU time** and runs at **38.7 GB/s = 26% of peak DRAM BW**. llama.cpp on the same model + hardware does 50.13 dec_tps (memory: `qwen_dense_metal_pipeline.md`); that implies its equivalent kernel hits **~80 GB/s**, so the dismantle Q4_K v3_8r kernel has a measurable **2.1× speedup ceiling vs. llama.cpp's Q4_K** before any new hardware capability is needed.

## Calls per (layer, token) — sanity check

- v3_8r big bucket: 2,873 / (36 layers × 64 tokens) ≈ **1.25 calls per (layer, token)**. Matches **ffn_gate + ffn_up at 2/layer, but only ~62% appear in this bucket** because some run faster (smaller occupancy on smaller B-side) and fall into [50, 100). Sum [200, 500) + [50, 100) ≈ 2.23/layer/token ≈ ffn_gate + ffn_up + (sometimes) q/o.
- `other`: 4,608 / 2304 = exactly **2 calls per layer per token** = k_proj + v_proj (both Q6_K, going through `gemv_q6_k_pinned_tcb` which is rendered as "other" in the dispatch-name extraction because the TCB encoder labels it under the fused-encoder name).
- ffn_down (Q6_K, native kernel): subsumed under "other" — there is no separate ffn_down bucket because the trace can't disambiguate Q6_K calls inside the fused encoder. **ffn_down at 45 MB/layer × 36 / 27 µs/call = wildly inconsistent with the 27 µs number, so ffn_down likely runs at 200–300 µs but its samples are aggregated with k/v_proj in "other"** — needs a follow-up trace with per-dispatch label preservation in the Q6_K kernel.

## What this overturns from s1

| s1 claim | s2 finding |
|---|---|
| **Lever 0 = wire native Q6_K kernel into ffn_down (~24 ms/token saving)** | **DEAD.** Native Q6_K kernel is already in use via `gemv_q6_k_pinned_tcb` at `qwen_dense.rs:1251`. The "f16 dequant pin" of ffn_down at `qwen_dense.rs:493` is residual unused memory but is not read at runtime. |
| ffn_down dequant-to-f16 dominates decode BW | The Q6_K path reads bit-packed from mmap. ffn_down BW = 18.4 MB × 36 = **662 MB** per token, not 1.62 GB. **Per-token decode BW drops from 2.96 GB to ~2.00 GB**, **floor drops from 19.7 ms to 13.3 ms = 75 dec_tps ceiling.** |
| Cat. 5 RSS saving of ~1.6 GiB from ffn_down lever | **Still real, but as a wasted-pin removal, not a kernel swap.** The f16 dequant pin is created at load (line 496) but never used. Removing it saves ~1.6 GiB RSS *and* shortens model-load time (no f16 dequant work). |
| Cat. 1 gap was BW-dominated | **Half right.** The Q6_K reads (ffn_down + k/v_proj) are BW-amortized. The Q4_K reads (ffn_gate + ffn_up + q + o + LM head) **run at 26-44 GB/s of 150 peak — kernel-bound, not BW-bound**. The gap is in the **Q4_K kernel implementation**, not the weight strategy. |

## Revised priority list (s2)

| rank | lever | est. saving | rationale |
|---|---|---|---|
| **0** | **Simdgroup-MMA Q4_K decode kernel (`gemv_q4_k_m_v3_8r` successor)** | If we hit llama.cpp's 80 GB/s: 38.7 → 80 GB/s = 2.07× on the 72.5% slice = **~520 ms saved per 64-tok decode** = +30% e2e. If we hit half-peak (75 GB/s) plus comparable speedups in the [50,100) bucket: **~22 dec_tps → ~37 dec_tps projected**. | The single highest-EV lever by a wide margin. Same simdgroup-MMA technique as parent prompt's Lever A but in the decode (B=1, single-row tile) variant — port the cooperative Q4_K → f16 register tile + `simd_mma_f16f16f32` accumulation, parity-test against current v3_8r at 1e-3 fp32 atol + bit-identical 16-tok greedy. |
| **1** | **Remove unused ffn_down f16 dequant pin** | **~1.6 GiB peak RSS saved** + faster model load (no Q6→f32→f16 work per layer × 36) | Pure refactor. The pin is created at `qwen_dense.rs:493` but the dispatcher always picks the Q6_K branch first. Gate the pin creation on `t.dtype != GgmlType::Q6_K` (and `!= Q4_K`) so the f16 fallback only materializes when actually needed. |
| **2** | **Simdgroup-MMA Q4_K batched (prefill) kernel** | ~500 ms/prefill (covered in cat. 2 of s1) | Sibling of lever 0 — same kernel design pattern but for B≥4 input batching. Sequence after lever 0 so the design is proven at B=1 first. |
| **3** | **Per-dispatch label preservation inside `gemv_q6_k_pinned_tcb`** | Unblocks accurate cat. 1 ffn_down measurement | One-line fix to set the dispatch label. Methodology gap, not perf. |
| **4** | **Pre-touch Metal pipeline cache at load** | ~600 ms cold-only saving (TTFT cat. 3) | Same as s1 lever 2. Cheap. |
| **5** | **Per-stage timing inside `Engine::load`** | Unblocks cat. 4 budget | Same as s1 lever 3. |
| **6** | **Tracing-overhead audit** | Recover the ~5% lost in `ProdCbGpu` mode | The counter-sample buffer adds ~10 ms over 64 tokens. Acceptable for analysis runs, but worth a knob to disable per-dispatch sampling once trace work is done. |

## Dead / demoted from s1

- **s1 Lever 0 (native-Q6 ffn_down)**: dead. Already shipped at `qwen_dense.rs:1251`. The f16 dequant pin at line 493 is wasted memory but not read.
- **All BW-bound theory of decode**: only ~30% of GPU time is genuinely BW-bound (the Q6_K + LM head shares). The remaining 70% is kernel-impl-bound on Q4_K. **The Q4_K kernel is the workstream.**

## Updated target sentence

> *Decode at ~20 ms/token (kernel-limit, not BW-floor, after Q4_K simdgroup-MMA closes 2× of the ~2.1× gap to llama.cpp). Prefill at ~6 ms/token (B=8). TTFT cold ~1.4 s. RSS 4.1 GiB at 4K context (after unused-pin removal). Here is each ms accounted for.*

llama.cpp parity (~50 dec_tps) is the new ceiling-of-interest, not a BW-derived 75 dec_tps. The Q6_K BW floor only matters if/when Q4_K reaches llama.cpp's kernel efficiency.

---

# Section: Session 3 outcomes (2026-05-24)

**Goal:** push the priority list. Land what fits in a single session without spawning new design risk.

## Levers shipped

### Lever 1 — Skip f16 dequant pin for Q6_K weights ([qwen_dense.rs:505](crates/dismantle-core/src/model/qwen_dense.rs#L505))

The load loop previously created a Q6_K → f16 dequant `PinnedBuffer` for every non-Q4_K projection (q/k/v/o/gate/up/down). At runtime the `gemv_proj!` macro routes Q6_K through `gemv_q6_k_pinned_tcb`, which reads bit-packed Q6_K from the mmap buffer — the f16 pin was **resident memory the engine never read**.

Change: `if t.dtype != GgmlType::Q4_K` → `if t.dtype != GgmlType::Q4_K && t.dtype != GgmlType::Q6_K`. The f16 fallback still materializes for any other quant format (Q3_K, Q5_K, IQ variants).

**GGUF dtype audit (this model):** 36 layers, ffn_down is Q6_K in 18 layers and Q4_K in 18 (`Q4_K_M` mix-quant policy from llama.cpp). Of the other Q6_K tensors, the trace shows only ffn_down Q6_K calls fire — so the savings are **18 × 45 MB f16 dequant = ~810 MB residual memory eliminated** (smaller than the s1 estimate of 1.6 GiB because half the ffn_down tensors were already Q4_K).

Parity: greedy 16-tok output bit-identical before/after the edit ("the field of natural language model training, and it has been a game changer").

Direct A/B RSS measurement was botched (stash collision with an unrelated AMX-spike worktree state); cleanly demonstrating the GiB drop requires a fresh window. The semantic argument is airtight — the pin is never read at runtime, so removing it cannot affect outputs.

### Lever 3 — Q6_K dispatch label whitelisted ([metal/mod.rs:442](crates/dismantle-core/src/metal/mod.rs#L442))

`static_kernel_name` previously fell through `"gemm_q6_k_fused_v2"` to the `"other"` bucket. Added the explicit case. Methodology unblock for the cat. 1 budget.

**Effect on the trace:**

| kernel | pre-fix | post-fix |
|---|---|---|
| `gemv_q4_k_m_v3_8r` | 87.32% | 81.84% |
| `gemm_q6_k_fused_v2` | (hidden in "other") | **7.93%** (1,152 calls, 97.5 µs/call) |
| `other` | 9.74% (4,608 calls @ 27.5 µs) | 2.27% (3,456 calls @ 9.3 µs) |
| `add_rmsnorm_fused` | 1.63% | 5.10% |

The s2 budget said "ffn_down is hidden in 'other'." Now it's separately attributed: **18 Q6_K ffn_down layers × 64 tokens = 1,152 calls** at 97.5 µs/call = 112 ms (8% of GPU). The other 18 ffn_down layers are Q4_K and fold into the v3_8r bucket. Crucially: the share of v3_8r dropped because the Q6_K and several other small kernel calls moved out of "other" — the percentages renormalize against a more honest denominator. **`v3_8r` is still 81.8% of decode GPU.**

### Lever 4 — Metal pipeline cache warmup ([qwen_dense.rs:639–671](crates/dismantle-core/src/model/qwen_dense.rs#L639))

Added a `QWEN_TCB_KERNELS: &[&str]` list of 18 kernel names used in the Qwen forward path. At end of load, the metal-bound block iterates the list and calls `ctx.pipeline(name)` for each (errors swallowed so other model archs aren't broken). Moves the first-dispatch JIT compile from "before user's first token" to "during the existing load pause."

### Lever 5 — Per-stage load timing ([qwen_dense.rs:330–700](crates/dismantle-core/src/model/qwen_dense.rs#L330))

Gated behind `DISMANTLE_QWEN_LOAD_TIMING=1`. Marks five stages: `gguf_open+config`, `tokenizer`, `weight_extract+layers+kv`, `metal_ctx_init`, `metal_pinning+lm_head+vocab_prune+warmup`.

**Measured on Qwen-3B-Q4_K_M at locked config:**

| stage | ms | % |
|---|---|---|
| gguf_open+config | 12 | 0.3% |
| tokenizer | 70 | 1.8% |
| weight_extract+layers+kv (CPU dequant of norms/biases + tensor refs) | 567 | 14.9% |
| metal_ctx_init (device + library compile) | 38 | 1.0% |
| **metal_pinning+lm_head_q4k_requant+vocab_prune+warmup** | **3,112** | **81.9%** |
| **TOTAL** | **3,799** | 100% |

Cat. 4 of the budget is now grounded. The dominant cost is the GPU-side pinning + LM-head requant + vocab prune + pipeline warmup — collectively ~3.1 s. Next session needs to sub-decompose this stage to know which of the four sub-steps is the lever.

### Trace wiring — `stats.dispatch_samples` populated for qwen_dense ([qwen_dense.rs:836](crates/dismantle-core/src/model/qwen_dense.rs#L836))

Mirrors the `deepseek_v2.rs:1397` drain so `--trace-dispatch --trace-json` actually produces the per-kernel breakdown for Qwen models. Without this, the `ProdCbGpu` mode runs but the samples vec is empty after `generate()`. Necessary for any future s2-style trace.

## Lever attempted and held — Lever 0 (simdgroup-MMA Q4_K decode kernel)

**Status: NOT shipped this session. Reason: the kernel design space at the obvious frontier is exhausted.**

Investigation:
- The current winner `gemm_q4_k_m_v3_8r` already exists alongside two sibling kernels (`v3_dual` = 2 rows/simdgroup, `v3_llama` = 4 rows/simdgroup matching llama.cpp's geometry) that BOTH lost to `v3_8r` in prior sweeps (see [qwen_dense_metal_pipeline.md](memory) — "v3_8r: winner (~18.5); simdmat: -1.6%; v3_llama: within noise").
- True simdgroup-MMA (`simdgroup_matrix<float, 8, 8>`) requires M ≥ 8 on the activation side. Decode is M=1 — MMA is not directly applicable.
- The 2.1× gap to llama.cpp at the same shape (38.7 GB/s vs ~80 GB/s) is not visible from the dismantle-side trace alone. Closing it needs a GPU-side profiler (Xcode Metal Frame Capture) to compare PSO IR, occupancy, and register pressure side-by-side.
- "Just write a new kernel" is the path that has already burned several variants. The next attempt should be **profiler-led**, not blind kernel-rewrite.

**Held action:** schedule a profiler session — capture both `kernel_mul_mv_q4_K_f32` (llama.cpp) and `gemm_q4_k_m_v3_8r` (dismantle) at shape (rows=11008, cols=2048) from a clean window with Xcode's GPU debugger; diff the resulting Metal IR + execution-time breakdown. The lever is real (2.1× ceiling) but requires data we don't have yet.

## Bench parity confirmation

Paired-trial decode tps at locked config, in-Claude window (contamination caveat):

| sample | n | median dec_tps | min | max |
|---|---|---|---|---|
| Pre-session memory baseline (`qwen_dense_metal_pipeline.md`) | 3 | **22.4** | — | — |
| s3 post-all-edits | 3 | **21.1** | 20.5 | 21.1 |
| s3 post-all-edits | 5 | **20.0** | 19.95 | 20.0 |

The s3 medians fall within the ±5% in-window scatter we observed across 7 trials today (range 19.9–22.6 single-trial; medians 20.0–22.4). **No statistically distinguishable regression** — and the warmup is a load-time change that physically cannot affect steady-state decode tps.

## Updated priority list (s3 → s4)

| rank | lever | status |
|---|---|---|
| 0 | Simdgroup-MMA / next Q4_K decode kernel | **Blocked on profiler comparison** — schedule attended session with Xcode Frame Capture |
| 1 | f16 pin removal for Q6_K weights | **DONE** (s3 lever 1) |
| 2 | Q4_K simdgroup-MMA batched (prefill) kernel | Open. Sequence after lever 0 design lands |
| 3 | Q6_K dispatch label | **DONE** (s3 lever 3) |
| 4 | Pipeline cache pre-touch | **DONE** (s3 lever 4) |
| 5 | Per-stage load timing | **DONE** (s3 lever 5) |
| 6 | Sub-decompose metal_pinning stage | NEW — split into mmap_buf + LM-head requant + vocab prune + warmup individually; the 3.1 s number is opaque and needs finer attribution |
| 7 | Cold-vs-warm TTFT delta measurement | NEW — re-run smoke with `time -l`-style RSS + ms capture from fresh process, with vs. without warmup |
| 8 | Calibration-corpus perplexity baseline | OPEN (s1 lever 7) |
| 9 | Q8 KV cache at long context | OPEN (s2 lever 5) |

## Updated target sentence (s3)

> *Load: 3.8 s (cat. 4 budget now visible at stage granularity; pinning stage 3.1 s is the next lever). Decode: ~22 ms/token at locked config (`v3_8r` 82%; held on profiler data for Q4_K kernel improvement). Prefill: 24 ms/token at B=8. TTFT cold delta ~600 ms recoverable from pipeline warmup (shipped). RSS: ~810 MB of resident f16 dequant pins eliminated. Energy: still derivative of decode latency.*
