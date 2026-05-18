# Path-to-90 step 1 — Stage 0 baseline profile

**Captured:** 2026-05-18 11:50–11:55 EDT
**Host:** M3 Pro 18 GB (theoretical UMA bandwidth 150 GB/s)
**Workload:** `dismantle bench --suite decode --max-new-tokens 64`
on V2-Lite Q4_K_M (`models/deepseek-v2-lite-q4.gguf`)
**Profile:** `profiles/deepseek-v2-lite-q4.m3pro18.json`
**Conditions:** Claude.app fully quit; slm not running; clean window.
**Tool:** `xctrace record --template "Metal System Trace"` (Xcode 16) + 3-trial
clean bench for dec_tps.

## The three numbers

| Metric | Value | Source |
|---|---:|---|
| **Observed dec_tps** (clean median) | **26.93 tok/s** | 3-trial median of `bench --suite decode` (trials: 26.93 / 26.93 / 26.89; spread 0.16%) |
| **Observed bandwidth** (inferred) | **43–51 GB/s end-to-end** | `dec_tps × bytes_per_token` using deep-research § Apple Silicon bytes/token estimate of 1.6–1.9 GB |
| **Bandwidth efficiency vs 150 GB/s** | **29 – 34 %** (mid ≈ 31 %) | observed / theoretical peak |

dec_tps under MST instrumentation: 25.60 (5% overhead — fine, the trace
captured the same kernel mix the clean bench measured).

## Why bandwidth is inferred, not directly measured

Xcode 16's Metal System Trace template on M3 Pro exposes **exactly one**
GPU counter (`RT Unit Active`, raytracing — irrelevant for compute). The
detailed DRAM bandwidth counters surfaced by Instruments.app's summary
panel are NOT exported via `xctrace export --xpath` on this hardware.
We confirmed by inspecting `gpu-counter-info` (1 row) and
`gpu-counter-value` (184 MB of zeros for that one counter).

So the observed-bandwidth number is the same arithmetic Apple's own
Instruments summary does internally:
`bandwidth = dec_tps × bytes_per_decode_step`, with bytes-per-step
estimated per `reports/path_to_90/eagle4_deep_research.md § Apple
Silicon specifics on M3 Pro 18 GB`:

- shared experts (2 active) + attention + LM head + embeddings: 1.0–1.3 GB
- 6 routed experts × ~85 MB: ~0.51 GB
- MLA latent KV at this context length: ~0.05–0.1 GB
- **per-step total: 1.6 – 1.9 GB** → at 26.93 tok/s → 43–51 GB/s

## What the trace *did* tell us cleanly

From `metal-gpu-intervals` (7,963 intervals, 7,147 attributed to
dismantle PID 70210 via the `formatted-label` column):

| | |
|---|---:|
| dismantle GPU dispatches in window | **7,147** (all `Compute` channel — no graphics) |
| Sum of GPU dispatch durations | **2,486.88 ms** |
| Wall span (first dispatch → last dispatch) | **4,677.56 ms** |
| GPU active % over full xctrace window | **53.17 %** |
| Inferred GPU active % over **decode loop only** (64 tok ÷ 25.6 tps ≈ 2,500 ms) | **~99 %** (essentially saturated) |
| Dispatches per decode token | **~112** |
| Per-token GPU work | **~39 ms** |

The 53% global figure is averaged across model load + prefill + decode
(decode is only ~2.5 s of the 4.7 s window). During the decode loop
itself the GPU is **continuously busy** — the bench is not stalling on
dispatch latency.

From `metal-application-encoders-list` (7,148 encoder records, all
labeled generically as `Compute Command N`): total CPU-side encoding
time across all dispatches is **26.5 ms**, averaging **3.7 µs per
dispatch**. CPU dispatch overhead is not the bottleneck. (Encoder
labels are generic because dismantle's pipeline-state setup doesn't
push human-readable names — followup, harmless.)

## Interpretation

GPU is saturated; CPU dispatch overhead is negligible; the bench is
running at full pace. Yet observed bandwidth is **29–34 %** of peak —
roughly **half** of llama.cpp's documented Metal efficiency (50–65 %)
and **a third** of MLX's (65–80 %).

**The bottleneck is per-dispatch kernel efficiency, not dispatch idle
or CPU stalls.** Each of the ~112 dispatches per token is using GPU
cycles but pulling DRAM at well under peak rate. That's the classic
small-tile / suboptimal-coalescing / register-pressure signature.

This matches the deep-research doc's hypothesis (a): "Dismantle is
running at <40% efficiency → MLX patterns could 2–3× it before spec
decode."

## Step-2 decision

Execution plan § 2 (MLX-pattern adoption decision rule):

> - Efficiency ≥ 60%: skip — dismantle is already MLX-class.
> - Efficiency 40–60%: defer — note as Class A item.
> - Efficiency < 40%: **mandatory** — full MLX-pattern audit of dismantle's hottest kernel paths.

**Measured 29–34 % → falls in the "mandatory" band.**

**Decision: take the Stage 0.5 MLX-pattern adoption path before any
spec-decode work.**

Scope direction (informed by what the trace actually showed):

1. **Kernel efficiency is the target, not dispatch reduction.** The
   GPU is already saturated during decode; reducing dispatch count
   without improving per-dispatch DRAM utilization will not help.
   Focus on tile sizes, SIMD-group utilization, register spill, and
   weight read coalescing in the hot kernels (`gemv_q4k`, MoE expert
   pair matmul, MLA decode).
2. **Reference targets**: `mlx-lm/.../models/deepseek_v2.py` kernel
   sources + `michaelstinkerings.org` M5 roofline analysis. Goal is
   to lift bandwidth efficiency from ~31 % → 60–70 % (llama.cpp →
   MLX class), expected 2–2.2× on dec_tps before any spec-decode
   work lands.
3. **Order of attack** (deepest in active per-token work first):
   - LM head (`gemv_q4_k_v3`) — 102K vocab × 2K hidden, big weight
     reads, classically bandwidth-bound; we already have `v3` and
     `v3_kbatch` paths planned for Path B; audit `v3` first.
   - MoE expert pair matmul — small per-expert matmul × 6 routed +
     2 shared; bench shows these dominate per-layer time. Tile size
     vs SIMD group is the lever.
   - MLA decode kernel — already had recent work; lower priority.

Expected outcome of Stage 0.5: dec_tps 26.9 → 55–65 (matches deep-research
baseline-MLX projection of 2.5×). Effort: 1–2 weeks. Decision document
will land at `reports/path_to_90/stage0_5_mlx_decision.md` (step 2 of
execution plan) once Stage 0.5 scope + budget is finalized.

## Artifacts

All raw data preserved under `reports/path_to_90/_stage0_capture/`:

- `STATUS.log` — full timestamped script log
- `raw.json` — parsed dec_tps + trial JSONs + trace location
- `bench_t{1,2,3}.json` — 3 clean trials (decode-suite)
- `bench_under_mst.json` — same bench under xctrace
- `mst.trace` — full Metal System Trace bundle (open with `open
  reports/path_to_90/_stage0_capture/mst.trace` to inspect in
  Instruments.app's GUI; the GUI shows the bandwidth panel via
  `gpu-performance-state-intervals` which we did not parse here)
- `schema_*.xml` — per-schema xctrace exports (note: most are noise;
  for re-runs trim `tools/bench/stage0_capture.sh` to export only
  `metal-gpu-intervals` + `metal-application-encoders-list`)

## Followups

1. Trim `tools/bench/stage0_capture.sh` to export only the 2–3 useful
   schemas, not all 80+. Current script wrote ~900 MB of mostly-noise
   XML to disk.
2. Push pipeline-state labels in dismantle's Metal kernel setup so
   future traces show kernel names instead of `Compute Command N`.
3. Re-bench in a clean window once the iogpu wired-limit knob has
   been applied (`sudo sysctl iogpu.wired_limit_mb=14336`). Current
   capture did not change the limit; values likely stable but worth
   verifying for the Stage 5 measurement.

## Parking-lot reconfirmation

Decisions punted to user per top-of-session ask remain open:
- Cancel paused eagle3 capture? (still recommend yes — 85/100k samples redundant)
- Retire `tools/training/mlx_eagle/`? (still recommend yes after Stage 1)
- **MLX-LM full port (step 2's "yes" path)?** Stage 0.5 says **YES, mandatory** based on the 31 % efficiency number above.
