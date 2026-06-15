# Closeout — aggregate-opt + clean-room + reduce-ms (2026-06-04)

Branch `paradigm/exec`, **31 commits ahead of origin, all LOCAL (not pushed).** M3 Pro /
macOS 26.5 / Qwen2.5-3B-Q4_K_M.

## What shipped this session (commit chain)

**Aggregate-opt R1→R3 (continuous-batching per-token cost):**
- `8aba79e` R1 — GPU-batched Q4_K LM head (opt-in `DISMANTLE_QWEN_Q4K_LMHEAD=1`); replaces B
  sequential CPU full-vocab matmuls with one v3w GEMM. **The dominant aggregate lever.**
- `88a00ef` R2 — batched RoPE (`rope_f32_batched_multiseq`), 2B→2 dispatches/layer, bit-identical.
- `113a9a8` R3 — batched KV scatter-append (`kv_scatter_append_multiseq`), 2B→1/layer, byte-identical.
- `8eaf627` C1 — single-TCB tail: stack returns the live command buffer, LM-head GEMM appends into
  it → one commit/step (was two). Bit-identical, ~5-10 ms/step.

**Bench / tooling:**
- `6137883` aggregate bench measures flag-OFF + R1-ON + the R1 delta.
- `f7ab8c0` `FAST=1` + per-section token floors for `clean_bench_queue.sh`.
- `844d7b1` clean-room trace no longer aborts on the macOS-26 xctrace quirk; f16-KV energy same-N.
- `three_way_bench.sh` (this commit) — dismantle vs llama.cpp vs MLX, tps + J/tok.

## Measured (clean room, 2026-06-04)
- Single-stream decode **32.65 dec_tps** (~31 anchor confirmed; ~39 was optimistic).
- Energy **0.196 J/tok — GPU 0.177 / DRAM 0.085** (the per-domain moat number).
- Aggregate **B=8 = 47.96 tok/s, 5.02× scaling** (R1+R2+R3, flag-ON) — inside the 3.5-5.6× ceiling.
  flag-OFF B=8 = 19.12 / 2.45×.
- **R1 clean delta = ×2.51 at B=8** (the Claude-open preview's ×5.66 was inflated — paired A/B only
  cancels contamination when both arms share a bottleneck class; R1 crosses CPU↔GPU).

## THE next lever (profiled, definitive) — v3w GEMM efficiency
Wave `wf_60586084` profiled the multiseq-B=1 gap (3.4× slower than single-stream):
**~80% is the layer projection GEMMs running `gemm_q4_k_m_batched_v3w` at low B — under-occupied
(8 rows/TG, no 2-row ILP), DRAM-latency-bound NOT bandwidth-bound** (predec ON/OFF identical at B=1).
It's the B-independent weight-read pass, so it caps B=8 too. The rest is marginal (LM-head 2nd commit
~9% [now fixed by C1], per-step embed/layer0 ~3-4%, CPU argmax <0.5%).

**Next session = the v3w rewrite (route b):** port the tuned single-token `gemv_q4_k_v4_predec_2r`'s
16-rows/TG + 2-row-ILP DRAM-latency-hiding into `gemm_q4_k_m_batched_v3w[_predec]`. Lifts the
weight-read at ALL B → shrinks the 96 ms fixed pass → drives B=8 from **47.96 toward ~80+** agg tps.
Parity-gated by `q4k_batched_gemm_parity` + the multiseq anchors. (Route a — swap to the existing
`_mma` kernel — is easier but underfills at low B and is numerically ≠ the single-stream gemv, so the
`multiseq_decode_parity` B=1-vs-single-stream anchor could flip an argmax; needs the anchor relaxed +
a paired clean measure.)

Marginal levers (profile-confirmed, NOT applied, available on request): A3 batch embed/layer0
(~1-3 ms@B8), batched GPU argmax (<0.5%@B1), A7 hoist allocs (~0.5 ms).

## Open / parked
- MST-vs-llama per-kernel DIFF: blocked — dismantle's `gpu_prod` labels are internal, not Metal
  `setLabel`, so Instruments exports zero dismantle kernel names. Low priority (adverse GEMV prior).
  `--skip trace` in the clean queue. dismantle's own `gpu_prod` tracer is the dismantle-only tool.
- Serving build (multi-seq prefill + HTTP concurrent loop + paged-KV for B>8) — the path to a real
  server; paged-KV (B1) is also the aggregate-ceiling lever beyond B=8.

## Bench commands
```
# clean room (Claude quit) — our stack, all diagnostics, fast/faithful:
FAST=1 tools/bench/clean_bench_queue.sh

# three-way vs the competition (clean room):
tools/bench/three_way_bench.sh                 # dismantle | llama.cpp | MLX : tps + J/tok
ONLY=dismantle,llama tools/bench/three_way_bench.sh
```

## In flight
- Deep research launched (`niches + moat + pull-from-competition`) — ranked, evidence-backed list of
  moats to deepen and llama/MLX techniques to port (esp. the low-batch Q4_K GEMV occupancy fix).

---
## Paste-able next-session opening prompt

> Continue dismantle on branch `paradigm/exec` (pure-Rust + Metal, M3 Pro, Qwen2.5-3B-Q4_K_M).
> 31 local commits ahead of origin (push-as-one when ready). Read
> `reports/closeout_aggregate_2026_06_04.md` + `reports/research_final_push_2026_06_03.md` + memory
> `paradigm_final_push_research_2026_06_03.md` first — they carry full state.
>
> STATE: continuous-batching decode is built + parity-green + clean-benched: single-stream 32.65
> dec_tps / 0.196 J/tok; aggregate B=8 = 47.96 tok/s at 5.02× (R1+R2+R3 + C1 landed). The profile is
> definitive: **~80% of the multiseq-B=1 gap is `gemm_q4_k_m_batched_v3w` being under-occupied at low
> B** (8 rows/TG, no 2-row ILP, DRAM-latency-bound). That fixed weight-read pass caps B=8 too.
>
> TASK (the genuine lever past 47.96): **rewrite `gemm_q4_k_m_batched_v3w[_predec]` (shaders/quant.metal)
> to port the tuned single-token `gemv_q4_k_v4_predec_2r` occupancy — 16 rows/TG + 2-row-ILP DRAM-
> latency-hiding.** Parity-gate with `q4k_batched_gemm_parity` (atol 1e-3, +rtol 1e-4 for the reorder)
> + keep `multiseq_decode_parity` / `multiseq_q4k_lmhead_parity` green. Then paired-measure the B=8
> aggregate (`tools/bench/batch_aggregate_bench.sh`) — target 47.96 → ~80+. Commits authored Joshua
> Hicks (-c inline), no AI attribution, LOCAL. Confirm GPU/RAM free first (`ps -Ao %cpu,%mem,comm -r`).
> Worktree agents branch off STALE origin/main — use non-worktree agents that RETURN diffs, apply
> serially. Re-run parity yourself before trusting any "passed".
