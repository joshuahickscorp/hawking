# Phase A Close — Multi-token batching

**Date:** 2026-05-04  
**Baseline dec_tps:** 2.207 (clean bench, post-v0.8.6 revert)  

## Wedges landed

| Wedge | Status | Commit |
|-------|--------|--------|
| A1: forward_tokens_batched scaffold | LANDED | 40c6741 |
| A2: mla_decode_kernel_batched (M=4) | LANDED | 152267e |
| A3: MoE dispatch batch-aware | STUCK | see v_a3_moe_batch_STUCK.md |
| A4: batched argmax | SKIPPED (depends on A3) | — |
| A5: generate() prefill batching | SKIPPED (depends on A3) | — |

## Infrastructure added

- `Engine::forward_tokens_batched_for_test` and `reset_kv_for_test` in Engine trait
- `DeepSeekV2::forward_tokens_batched` (token-first scaffold, A2-ready structure)
- `mla_decode_kernel_batched` Metal shader: M=4, grid (n_heads × TG, M, 1),
  causal mask per token (token m attends to max(base_seq + m, 1) entries)
- `mla_decode_metal_batched` Rust dispatcher

## In-session bench (contaminated)

Cannot run `clean_bench.sh` from within Claude session (Claude GPU process
inflates dec_tps 4-5×). Contaminated in-session numbers are not meaningful.

## Honest dec_tps assessment

**Phase A will NOT improve dec_tps.** Greedy decode is inherently sequential —
each token requires the previous token's output. Batch decode requires speculative
decoding (α=0.34 acceptance rate, closed for now). The A2 batched attention
kernel is useful infrastructure for future spec decoding but does not activate
in the current generate() loop.

Phase A only benefits **prefill speed** (first-token latency), and only partially
since A3 (MoE batching) was not implemented.

## What Phase A does enable (future)

- If speculative decoding acceptance rises (α > 0.5), `mla_decode_kernel_batched`
  can handle M=4 draft-token verification in one attention dispatch
- The scaffold in `forward_tokens_batched` needs A3 to be useful for layer-first batching

## MANAGER:CLEAN_BENCH_NEEDED — Phase A

No clean bench needed for Phase A. Phase A does not affect decode throughput.
The clean bench should be run only after Phase E (f16 residual) or Phase C
(MPSGraph attention) to capture real dec_tps changes.

## Next phases

Proceed directly to Phase E (f16 residual stream) — the highest expected
dec_tps improvement (~30-50% from 2× bandwidth reduction). Phase B/C
(MPSGraph) assessed next for feasibility.
