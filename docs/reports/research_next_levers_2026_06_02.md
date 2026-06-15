# Next tps/energy levers — deep-research synthesis (2026-06-02)

> Source: `reports/research_next_levers_2026_06_02.json` (106-agent deep-research wave,
> 8 adversarially-verified findings, 3-vote). Fed the full dead-lever list so it would
> not re-propose killed axes. Baseline: ~30.5 tps / 0.197 J/tok clean (M3 Pro, Qwen-3B-Q4_K_M);
> llama.cpp = ~49 tps on the SAME machine (1.6× gap).

## Bottom line
- **TPS:** the 1.6× gap is **real and closable on M3 Pro** (llama proves 49 here), but it lives
  in the **runtime / command-buffer / GPU-saturation** layer — NOT bytes (dead) and NOT a faster
  GEMV (dead, at the memory-model optimum). The mechanism is **unproven on M3 Pro by literature**
  → the decisive next step is a **Metal System Trace diff of llama.cpp vs dismantle**, not a blind build.
- **ENERGY:** there is **no large clean J/tok win** — dismantle already sits near the DRAM floor
  (sequential weight-streaming avoids ~32× the row-buffer overfetch waste random access suffers).
  Energy gains must come from **race-to-idle** (= faster decode, aligns with tps) + precise instrumentation.
- **Off the single-stream table:** DWQ quant = quality-only (no speed); continuous batching =
  aggregate-only (no single-stream tps); ANE = dead for decode.

## TPS (primary) — ranked
1. **Close the runtime/graph gap [BIGGEST, but MEASURE FIRST].** MLX runs ~1.5× faster than llama.cpp
   on the same Apple GPU (>90% GPU util / <3% CPU); llama.cpp itself recovered **+21%** on M1 Ultra
   purely via graph restructuring (copy elimination, **no kernel change**). So a meaningful fraction of
   the gap is schedulable host/graph overhead + GPU-idle-between-dispatches (dismantle's own 0.2 trace:
   GPU-busy ≈ 76% → ~24% non-GPU gap). **Conflict check:** dismantle's "host per-dispatch overhead" kill
   was about *CPU-encode time* (0.51% of wall, dead) — this lever is *GPU saturation / command-buffer
   structure* (keep the GPU busy >90%), a different axis. **Caveat:** MLX numbers are M2/M1 Ultra (800 GB/s,
   76 cores), NOT M3 Pro; the specific llama copy-edits are arch-specific. **Decisive step:** Metal System
   Trace (Instruments) per-token timeline diff llama.cpp vs dismantle on M3 Pro → find the idle/scheduling
   delta. Effort: medium (profiling) then medium-high (the fix). Risk: the gap may partly be M3-Pro HW
   (lower BW/cores) → validate the ceiling with #3 first. [findings #1, #3]
2. **MTLResidencySet [CHEAP FIRST TRY].** Wire the weight buffers so the OS can't evict/idle-throttle them
   (llama.cpp PR #11427, macOS≥15; NOT confirmed in dismantle). Low effort (Metal API + version gate),
   modest gain (~250 ms/req on 7B → small on 3B), also a stability win. Build + paired A/B. [finding #2]
3. **MLX A/B reference on M3 Pro [MEASUREMENT].** Run MLX Qwen-3B-4bit decode on the M3 Pro, A/B vs
   dismantle's 30.5. If MLX ≈ llama's 49 → the gap is runtime-structural (closable) → port the winning
   structure. If MLX ≈ 30 → the gap is M3-Pro-HW-specific. Settles #1's ceiling cheaply. [finding #3]
4. **DWQ quant = NOT a tps lever.** MLX-native distilled-weight-quant from f16 improves 4-bit *quality*
   (8.85 vs 9.07 ppl) but decode speed is identical to standard 4-bit. Use only if pushing to ≤4-bit while
   keeping quality. Not a speed/gap lever. [finding #4]
5. **Continuous batching = aggregate-only.** vllm-mlx 441→1642 tok/s at 16 concurrent (3.7×) — real if
   pivoting to multi-stream serving, does nothing for single-stream batch-1 tps. [finding #5]

## ENERGY (close second) — ranked
1. **Instrument precisely + race-to-idle [BIGGEST realistic step].** No large clean J/tok win exists
   (memory-bound). Integrate **zeus-apple-silicon** (IOKit/IOReport, sub-ms, sudo-free, per-domain
   CPU/GPU/GPU-SRAM/DRAM/ANE) → replace the proxy in `phase_joules.sh` with MEASURED per-domain J/tok.
   Then race-to-idle: faster decode = less energy (tps and J/tok align here). [finding #6]
2. **Near the DRAM floor already.** Streaming ~1.9 GB contiguous weights/token is sequential/high-row-locality
   → avoids the dominant overfetch waste. "Fewer/better-laid-out bytes" won't give big J/tok wins; the only
   axis left is power-state. (The exact LPDDR5 pJ/byte floor was NOT found in the literature — unquantified.)
   [finding #7]
3. **DVFS controllability: unconfirmed (likely OS-gated).** The research found no user-space GPU clock/voltage
   control on Apple Silicon. Probe it, but expect it gated. [Family D, no positive finding]
4. **ANE = DEAD for decode.** ANE decode is slower than even CPU (170 vs 283 tok/s GPT-2 124M; ~2.3 ms
   IOSurface round-trip per dispatch, model-size-independent → worse at 3B); the 52–137 mJ/tok figure is
   PREFILL-only. Type-2 parked (impl limit, not fundamental) but no decode win today. Confirms silicon-audit #6.
   [finding #8]

## Recommended sequence ("knock these out consecutively")
1. **MTLResidencySet** — the one concrete buildable tps lever; build + A/B. (low risk)
2. **MLX A/B + Metal System Trace diff on M3 Pro** — settle WHERE the 1.6× lives + the ceiling. (measurement; gates the big build)
3. **zeus-apple-silicon J/tok instrumentation** — measured per-domain energy; race-to-idle is then the energy axis.
4. **Build the gap-closer the trace reveals** (likely GPU-saturation / command-buffer scheduling — push GPU-busy 76%→90%+). Gated on #2.
