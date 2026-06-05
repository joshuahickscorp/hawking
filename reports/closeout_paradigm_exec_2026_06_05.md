# Closeout — paradigm/exec (2026-06-05)

Branch `paradigm/exec`, M3 Pro / macOS 26.5 / Qwen2.5-3B-Q4_K_M.

## What shipped this branch (full chain)

### Serving infrastructure (first wave)
- `b1d0edc` — continuous-batching loop, prefill_slot, arena lifecycle fix
- `7c34f7b` — admission queue, real metrics, 4096-token context per slot
- `6fa5349` — B=1 predec fast path, GPU LM head, serve optimization defaults
- `5c67435` — fix LM head predec scale mismatch (OOB → silent hang)
- `04098c0` — parallel prefill: amortize weight reads across B slots per position
- `81f593f` — prefill gather window (5 ms batch formation before engine lock)
- `e15f530` — hybrid prefill + prefix sharing + length-bucketed admission

### Aggregate-opt (per-token cost at B>1)
- `8aba79e` R1 — GPU-batched Q4_K LM head (×2.51 B=8 delta, clean)
- `88a00ef` R2 — batched RoPE (2B→2 dispatches/layer, bit-identical)
- `113a9a8` R3 — batched KV scatter-append (2B→1/layer, byte-identical)
- `8eaf627` C1 — single-TCB tail: LM-head folds into stack command buffer

### v4r GEMM kernel chain
- `a190b8d` — `gemm_q4_k_m_batched_v4r_predec` (barrier-free, B=2..4, predec-aware)
- `d9bdfab` — hybrid v4r/v3w routing: v4r for B≤4, v3w for B>4
- `4d78a21` — complete v4r routing at remaining ffn_down predec sites

### Tail polish (uncommitted at handoff → committed below)
- b16 kernel — `gemm_q4_k_m_batched_v3w_predec_b16`: extends v3w to B=1..16 (adds
  `partial_lo2/hi2` for slots 8..15; shmem up to 16 KiB at B=16, within M3 Pro limit)
- B=1 short-circuit in `forward_multiseq_batched` → routes to `forward_token_greedy_tcb`
- ffn_down routing ladder: B=1 → v4_predec GEMV; B≤4 → v4r; B>4 → v3w
- Serve: single-slot prefill uses `prefill_slot` instead of `prefill_slots_parallel`
- Bench: `REQUEST_TIMEOUT_SEC` cap on per-stream curl, MLX binary auto-detection, `RUN_TIMEOUT_SEC`

## Clean-room measurements at branch close
*(from `c28cb73`, verified clean-room)*

| Path | Value |
|---|---:|
| Single-stream decode | 32.65 dec_tps |
| Aggregate B=8 (R1–R3+C1 ON) | 47.96 tok/s (5.02×) |
| Aggregate B=8 (OFF) | 19.12 tok/s (2.45×) |
| R1 clean delta | ×2.51 at B=8 |
| Energy | 0.196 J/tok (GPU 0.177, DRAM 0.085) |

## What the three-way bench revealed (→ new plan)

`three_way_bench.sh` head-to-head:

| Engine | ~dec_tps | notes |
|---|---:|---|
| llama.cpp | ~55 | Q4_K_M same weights |
| dismantle default | ~26 | serve path, full logits |
| dismantle in-process fast | ~31–43 | --profile fast |
| dismantle serve B=8 aggregate | ~51 | with R1–R3 |

**Root cause of the gap** (profiled, from wave `wf_60586084`):
- ~80% of the multiseq-B=1 wall is layer projection GEMMs running v3w at low B —
  under-occupied, DRAM-latency-bound.
- Serve still asks for full logits + CPU sampling on every step; full `B×vocab×f32` readback.
- `--profile fast` is not the default serve path; most bench numbers used the slow path.
- `--kernel-profile` vs `--profile fast` naming creates easy confusion.

## What the new plan attacks

`plans/bleeding_edge_throughput_energy_moat_plan_2026_06_05.md` covers 10 tracks.
**Immediate Phase A (stop doing unnecessary work):**
1. Greedy token-only multiseq path (`forward_multiseq_greedy_tokens`)
2. Batched GPU argmax (read back B×u32, not B×vocab×f32)
3. Route temperature=0 serve requests through token-only lane
4. Add stats counters (readback bytes, lane used, dispatches)

Expected Phase A result: serve B=1 → ~40–55 t/s; B=8 aggregate → 100–130 t/s; J/tok drop.

## Open / parked
- v3w GEMM efficiency (16-row/TG + 2-row ILP, port from v4_predec): the structural fix for the
  weight-read wall at all B. The b16 extension leaves the inner loop unchanged; this is the next
  kernel work after Phase A landing.
- MST per-kernel diff vs llama blocked (dismantle GPU kernels have no Metal labels → Instruments
  exports nothing useful). Not worth fixing now.
- Paged-KV / B>8 serving: parked pending Phase A throughput baseline.

## Commits stayed local; no push.
