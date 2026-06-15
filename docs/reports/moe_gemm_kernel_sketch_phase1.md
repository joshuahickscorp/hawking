# MoE GEMM kernel sketch — Phase 1 (Session J, 2026-05-22)

Sketch-phase output for a new custom Metal kernel targeting V2-Lite's
MoE down projection. Per `memory/per_kernel_time_breakdown.md`,
MoE GEMMs are 50.5% of decode encoder time and Q8_0 routed-down is the
single largest per-call cost (`q8_0_v2t` = 22.33 µs/call), so this is the
highest-cost-share lever still on the board after `gu_v2`.

## 1. Dominant shape (confirmed)

DeepSeek-V2-Lite-Chat config (`crates/dismantle-core/src/model/deepseek_v2.rs`):

| param | value |
|---|---|
| `hidden`              | 2048 |
| `moe_intermediate`    | 1408 |
| `n_routed_experts`    | 64 |
| `n_shared_experts`    | 2 |
| `top_k_routed`        | 6 |
| `first_k_dense_layers`| 1 (26 MoE layers) |

Per-token MoE GEMM dispatches (routed expert path, 26 MoE layers × 1 fused
`gu_v2` + 1 down):

| projection | shape (rows × cols) | dtype | kernel | µs/call (per breakdown) |
|---|---|---|---|---|
| gate + up (fused) | 1408 × 2048 | Q4_K | `…_v2t_gu_v2` | 6.07 |
| **down (routed)** | **2048 × 1408** | **Q8_0** | `…_q8_0_indexed_v2t` | **22.33** |
| down (shared)     | 2048 × 2816 | Q6_K | `…_q6_k_indexed_v2t` | 16.32 |

Shape `(rows=2048, cols=1408)` for the Q8_0 routed-down is the dominant
single per-call cost. With 6 routes × 26 layers × 22.33 µs ≈ **3.5 ms /
token** of encoder-attributed time for this kernel alone.

## 2. Current kernel + bottleneck

`moe_batched_gemm_q8_0_indexed_v2t` (`shaders/moe.metal:790`):

- TG = 256 threads (8 simdgroups × 32 lanes).
- 1 simdgroup owns 1 output row; 8 rows per TG.
- Cooperative x-cache preload (1408 floats = 5.6 KB shmem).
- Inner loop iterates `blocks_per_row = cols/32 = 44` Q8_0 blocks of 32
  values each. Per block: 1 fp16 scale + 1 signed-int8 weight per lane +
  1 `simd_sum`.

Grid for one call (down, routes=6, rows=2048):
- `n_tg_x = ceil(2048/8) = 256`  → 256 × 6 = **1536 TGs** × 256 threads
  = 393,216 threads.

Bandwidth budget per call (all 6 routes):
- Weights:  6 × 2048 × 44 × 34 B ≈ **18.4 MB**
- x_cache reload: 6 × 1408 × 4 B = 33 KB (1 preload per TG, but x_cache
  is amortized over 8 rows so only loaded once per 8 output rows per
  route → 6 × 256 = 1536 preloads = 8.4 MB DRAM reads of x).
- Effective DRAM if fully bandwidth-bound at M3-Pro ~150 GB/s: ~180 µs
  total per call (≈8 µs per route).

The kernel reports 22.33 µs/call (contaminated trace per memory note);
the gap between 22.33 µs and the BW-ceiling 180 µs suggests this kernel
is **not the GPU-time bottleneck** on its own — the per_kernel attribution
includes shared TCB encoder setup cost. **Real GPU time per call is
likely much smaller**, which caps the win available from kernel-shader
changes alone.

This is the cost-share gate from `memory/feedback_wall_clock_audit.md`:
the breakdown is encoder-time, not GPU-time, so per-kernel optimizations
are confounded by shared TCB overhead.

Likely true bottlenecks (in priority order):
1. **Per-TG x_cache preload** — loaded fresh by each TG even though all
   8 simdgroups in a TG share it. 256 TGs × 1408 floats per call.
2. **Scale-load granularity** — 1 fp16 scale per 32-value block; loaded
   per simdgroup per block. With 1 row/TG×simdgroup, a scale read is
   amortized over only 32 MADs.
3. **Encoder count** — the entire MoE block fires ~7 distinct kernel
   encoders per layer × 26 layers = 182 encoders/token. Compress here
   is the L7-style win, not shader-internal.

## 3. New kernel sketched: `_v2t_w2` (wide-2)

`shaders/moe.metal` — `moe_batched_gemm_q8_0_indexed_v2t_w2` (inserted
after `_v2t`).

**Geometry change** — 2 rows per simdgroup, 16 rows per TG:
- Halves `n_tg_x` (256 → 128 TGs per call), so the per-TG x_cache
  preload is amortized over 16 rows × 44 blocks = 704 simdgroup-blocks
  per TG (vs 352 in `_v2t`).
- Per-block inner work doubles: 2 fp16 scale loads, 2 int8 reads per
  lane, 2 MADs against 1 shared `xi`. Register pressure rises from
  ~4 floats/thread → ~7 floats/thread (well below the Phase Y kill at
  64 floats/thread that destroyed `moe_batched_gemm_q4_indexed_v3`).

**Hypothesis tested:** halving the TG count is "free" register-wise and
should reduce wave-launch overhead and improve x_cache amortization.

**Wiring** (`crates/dismantle-core/src/kernels/mod.rs`,
`encode_batched_gemv_indexed_tcb`): opt-in env var
`DISMANTLE_Q8_DOWN_W2=1`. Activates only when:
- kernel name is `moe_batched_gemm_q8_0_indexed_v2t`
- `rows % 16 == 0` (V2-Lite down has rows=2048, ✓)

Falls back to default `_v2t` in all other cases. **No defaults changed.**

## 4. Parity gate (Session J continuation, 2026-05-22)

Bit-identical greedy decode at `--seed 0`, two prompts × 8 tokens:

| prompt | tokens | off | DISMANTLE_Q8_DOWN_W2=1 | DISMANTLE_Q8_DOWN_W4=1 |
|---|---|---|---|---|
| "Once upon a time" | 3 | `", there was"` | `", there was"` | `", there was"` |
| "The quick brown fox jumps over the lazy" | 8 | `' dog"\n\n# 2.'` | `' dog"\n\n# 2.'` | `' dog"\n\n# 2.'` |

**Parity green** for both `_v2t_w2` and `_v2t_w4` against the default
`_v2t` baseline.

## 5. Microbench delta (3-trial + 6-trial paired wall-clock)

Setup: `dismantle generate --prompt "Once upon a time" --max-new-tokens 64 --seed 0`.
Same model, profile, same machine state (Claude live, paused-bench mode).

**First pass (3 trials each, post shader-hash bump):**

| variant       | trials                | mean  | Δ vs off |
|---|---|---:|---:|
| off (`_v2t`)  | 24.74, 24.57, 24.88   | 24.73 | —       |
| `_v2t_w2`     | 25.16, 25.14, 24.94   | 25.08 | +0.35   |
| `_v2t_w4`     | 25.00, 24.69, 24.87   | 24.85 | +0.12   |

**Second pass (off vs w2 only, 3 more trials each):**

| variant       | trials                | mean  |
|---|---|---:|
| off (`_v2t`)  | 24.84, 24.85, 24.79   | 24.83 |
| `_v2t_w2`     | 25.13, 25.19, 25.09   | 25.14 |

**Combined (6 trials each, w2 vs off):**

| variant       | mean  | stdev | Δ vs off          |
|---|---:|---:|---|
| off (`_v2t`)  | 24.78 | 0.11  | —                |
| `_v2t_w2`     | 25.11 | 0.09  | **+0.33 (+1.33%)** |
| `_v2t_w4`     | 24.85 | 0.16  | +0.07 (+0.3%)    |

`_v2t_w2` is consistently above off across 6 trials, with σ≈0.1 and Δ≈3×σ
(real, but small). `_v2t_w4` is within noise — quartering the TG grid
(64 TGs × 6 routes = 384 TGs) underfills the M3 Pro's ~18 cores and
loses the per-row launch savings. Sweet spot is `_v2t_w2` with 768 TGs
per call.

## 6. Recommendation for Phase 2

**Park `_v2t_w2` opt-in as-is** (env-gated, default unchanged). Wall-clock
delta is real but below the +1 tps gate, so it's not worth changing
defaults yet. Two follow-up directions for Phase 2 (ordered by EV):

1. **Apply the same `gu_v2` optimization-pack to single-matrix down
   kernels.** The largest gap observed in `per_kernel_time_breakdown.md`
   was `q4_gu_v2` being 2.68× faster than the raw `q4_v2t`. The shared
   gains were:
   - scale pre-load (one extraction per block, not per inner iter),
   - activation pre-load into registers,
   - sumy-trick correction (Q4_K only; Q8_0 has no min term so doesn't
     apply directly),
   - paired-nibble reads (Q4_K only).
   For Q8_0 `_v2t` down, the relevant transfer is **scale + activation
   pre-load** and possibly a per-block tail unroll. Apply to both the
   routed `_q8_0_v2t` (rows=2048, cols=1408) and the shared
   `_q6_k_v2t` (rows=2048, cols=2816) — together they're 38.65 µs/call
   × 26 layers = ≈1 ms/token, ~2-3% of decode.
2. **Pure-GPU re-measurement of `_v2t_w2`** via `DISMANTLE_TCB_TRACE=gpu_prod`
   + a per-kernel sink (the current trace doesn't dump to stderr; needs
   the L7 ProdCbGpu drain path wired to a CSV writer). Wall-clock +1.3%
   could be 5-10% of pure-GPU time on the routed_down call alone; the
   number we have today underestimates the kernel-level win because
   non-MoE encoders dilute the share.

**Phase 2 scope estimate:** 1-2 weeks for (1) — same shape of work as
this sketch, applied to the Q6_K shared-down kernel; (2) is ~1 day of
trace plumbing.

**Phase 2 kill criterion:** if `_q8_0_v2t_w2` with scale+activation
pre-load doesn't show ≥+1 dec_tps in paired wall-clock vs current `_v2t`,
the single-matrix down path is BW-bound and shader-level wins are
exhausted; pivot to encoder-count reduction (megakernel-style fusing
down + route_accumulate) instead.

## 7. Diff manifest (uncommitted, this worktree)

- `crates/dismantle-core/shaders/moe.metal` — added
  `moe_batched_gemm_q8_0_indexed_v2t_w4` kernel (4 rows / simdgroup).
- `crates/dismantle-core/src/kernels/mod.rs` — `DISMANTLE_Q8_DOWN_W4=1`
  env-var dispatcher branch; W4 takes precedence over W2.
- `profiles/deepseek-v2-lite-q4.m3pro18.json` — `shader_hash` bumped
  `92ba78831a4ad1d0abfacb70` → `fd1c7c108f9a4df33feabfb2`.
- This report (`reports/moe_gemm_kernel_sketch_phase1.md`) — sections
  4–7 filled in.

No defaults changed; no commits made.

---

## Caveats and limits

- All numbers in this report are encoder-attributed, not pure GPU.
  Real-GPU `_v2t` time per call is likely much smaller; therefore the
  shader-only ceiling for `_v2t_w2` is probably <0.5 ms/token at the
  hot-loop level. The Phase 2 conclusion must include a pure-GPU
  re-measurement before committing further iteration.
- This is a worktree-only sketch; no commits, default unchanged.
- `bench-kernel` (CLI) does NOT yet support MoE batched kernels (only
  single-row GEMVs in `crates/dismantle-core/src/kernel_bench.rs`).
  Direct microbench at the production shape requires either a new
  bench harness or end-to-end paired wall-clock with/without env var.
  Phase 2 should add an MoE bench entry point.
