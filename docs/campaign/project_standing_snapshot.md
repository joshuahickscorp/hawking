# Hawking — Project Standing Snapshot (2026-06-22)

One page. Canonical numbers below; trial-level evidence in `test_matrix.md`, unproven items in
`open_risks_and_gates.md`, rejected ideas in `kill_ledger.md`.

## TL;DR
Hawking is an Apple-Silicon-first Rust/Metal LLM inference runtime. The validated strategic finding: **the RWKV-7 SSM
path removes the transformer long-context KV wall** — flat ~110–119 tps out to 8k vs Qwen's collapse to ~8.6 tps
(**~13–14× at 8k**). The transformer (Qwen) decode path is mature and near its config ceiling (`predec` is the headline
win). **RWKV serve is now correct end-to-end** (2026-06-22): admission + the GPU-slot recurrent-state handoff (the
`fresh`-flag fix, parity-gated) + the SSE stream terminator are all fixed — `ssm_serve_smoke.sh` is **fail=0** with coherent
multi-token output. The validated frontier shifts from *serve correctness* to *serve throughput* (the B=8 arena does
8-stream work for 1 active stream → ~7.8 vs ~119 single-stream tps) and *valid instruct quality eval* (now unblocked via
`/v1/chat/completions`).

## Speed table (warm, measured)
| Model / config | short | ~2.5k | ~8k | note |
|---|---|---|---|---|
| Qwen2.5-3B-Q4_K_M (default) | ~38–41 | ~18.8 | ~8.6 | transformer KV wall, ~4.6× / −78% drop |
| Qwen `--profile fast` | +3–7% (noisy) | — | — | mild quality trade (83–90% argmax-identity) |
| Qwen `predec` OFF | **−46.7%** | — | — | predec is the real Qwen decode win (default-ON) |
| Qwen `F16_KV` | ~0% | +1.9% (scales) | — | −50% KV footprint; quality high; long-ctx/footprint lever |
| **RWKV-7-0.4B-SFT** (overnight 3-trial) | **~114.6** | **~113.9** | **~110.5** | FLAT — SSM, no KV wall (~13–14× Qwen @8k) |
| RWKV-7-0.4B-SFT (orig single bench) | 118.6 | 110.6 | 119.4 | corroborates flatness |
| mamba2-370M | ~11 | ~11 | **0.00 (FAIL)** | unoptimized; 8k kernel bug; NOT product-ready |

## Status classification
- **PRODUCTION-READY:** Qwen `generate` decode (`predec` default-on); `--profile fast` (speed-priority, mild trade);
  `F16_KV` (long-ctx footprint); RWKV-7 single-stream `generate` (coherent, ~114–119 tps flat).
  `F16_KV` (long-ctx footprint); RWKV-7 single-stream `generate` (coherent, ~114–119 tps flat); **RWKV serve — correct
  end-to-end (admission + decode + stream terminator; `ssm_serve_smoke.sh` fail=0)**, throughput-limited (see below).
- **EXPERIMENTAL / partial:** per-channel int4-KV (numerics pass, NOT wired end-to-end); KD drafts (undertrained; KD>SFT,
  best 75M ~19.4% vs SFT 17.7%); **RWKV serve THROUGHPUT** (~7.8 vs ~119 single-stream — the B=8 arena does 8-stream work
  for 1 active stream; size the arena to active slots).
- **NEWLY UNBLOCKED:** valid instruct quality eval via `/v1/chat/completions` (the serve fix removed the gate).
- **BLOCKED:** mamba2 long-ctx (8k kernel bug).
- **DEAD (evidence in `kill_ledger.md`):** spec-decode for speed (per-cycle overhead wall); per-ROW int4-KV (collapse);
  FFN_DOWN_Q4K (cold-start artifact); Q6_K predec (int8 scales trivial); decode-kernel micro-opt (bible §3.0 —
  simdgroup-matrix dead at M=1, A10 layout −16.8%); Q3_K decode (compute-bound).

## Quality / validation
- Qwen lever argmax-identity gates are useful + trustworthy: `--profile fast` 83–90%, `F16_KV` 88–100%.
- Lib unit suite **182/182 pass**; representative parity **3/3** (per-channel int4-KV outlier numerics, Q6_K 2r/4r).
- **Raw `hawking generate` is NOT a valid instruct/Q&A quality gate** (no chat template → garbage on Q&A prompts).
  Valid instruct eval needs `/v1/chat/completions` (gated on the RWKV serve fix) or manual per-model templates.

## Exact next commands
```bash
# The single highest-value gate (RWKV serve decode correctness):
cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1
# After it is green, confirm the serve path end-to-end:
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf
# Low-risk repeatable validation (CPU + sequential GPU; skips the known-failing serve gate):
RUN_SERVE_SMOKE=0 tools/ci/overnight_hardening.sh
```

## The one fix that unlocked the most value — DONE (2026-06-22)
**R1 is FIXED.** Root cause was a stale bundle-wide `fresh` flag (NOT a layout/stride issue): `prefill_slot` now clears
`g.fresh` after the state copy (`rwkv7.rs:1222`). Parity gate `rwkv7_prefill_slot_multiseq_parity` GREEN; `ssm_serve_smoke.sh`
**fail=0** (16 coherent tokens + stats + `[DONE]`). The **next** highest-value item is serve **THROUGHPUT** — the B=8 multiseq
arena does 8-stream work for 1 active stream (~7.8 vs ~119 tps); size the arena to active slots. See run-log Phase 8.

## Final recommendation
- **Ship now:** Qwen `generate` with `predec` (default) as the quality/short-context default; offer `--profile fast` as the
  speed-priority option and `F16_KV` for long-context footprint. RWKV-7 single-stream `generate` as the **long-context
  throughput** option (it is coherent and ~13–14× faster than Qwen at 8k).
- **Keep researching:** per-channel int4-KV wiring (deeper KV compression, gated); RWKV-7 instruct quality per task class
  (needs the chat-template gate); a converged KD training campaign (separate effort).
- **Stop doing:** spec-decode-for-speed; decode-kernel micro-optimization (tapped); FFN_DOWN_Q4K / Q6_K-predec / per-ROW
  int4-KV / Q3_K-decode (all dead with evidence). Do not chase the 1.6× llama gap for decode — it is structural at M=1.
- **Single next technical fix (highest leverage):** RWKV serve THROUGHPUT — size the multiseq decode arena to the number of
  ACTIVE slots (it does B=8 work for 1 active stream → ~7.8 vs ~119 tps). The serve is now *correct* (R1 fixed, smoke fail=0);
  recovering throughput makes the SSM long-context moat shippable as a *server*, not just single-stream `generate`.

