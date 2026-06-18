# Expansion Wave Ingestion Index
Date: 2026-06-18

Start here. This file consolidates the current expansion-wave research so a new
Claude Code session can ingest the codebase without searching through every old
plan first.

## Primary Reading Order

Read in this order:

1. `docs/plans/expansion_wave_ingestion_index_2026_06_18.md` - this resolver.
2. `docs/plans/full_project_report_2026_06_18.md` - factual project snapshot and user-facing positioning.
3. `docs/plans/low_bit_rwkv7_strengthened_revision_2026_06_18.md` - low-bit/QAT promote ladder and watcher context.
4. `docs/plans/dismantle_expansion_wave_v2_2026_06_18.md` - new research and implementation queue.
5. `docs/plans/claude_code_prompt_expansion_wave_v2_2026_06_18.md` - direct integration prompt.
6. `tools/training/g1a_watcher.sh` - live monitor that launches post-G1a work.
7. `tools/training/g1a_phase2_chain.sh` - phase2 post-G1a chain.
8. `tools/training/g1a_v2_expansion_chain.sh` - new wider self-skipping expansion chain.

## New Work Is Here

The new part is not the older Eagle/QTIP/STRAND archive. The new part is:

| Area | New source of truth | Why |
|---|---|---|
| Overall expansion sequencing | `docs/plans/dismantle_expansion_wave_v2_2026_06_18.md` | Splits work into independent, artifact-gated, and clean-room lanes. |
| Claude integration | `docs/plans/claude_code_prompt_expansion_wave_v2_2026_06_18.md` | Gives a direct build order with current file paths and gates. |
| Automatic post-G1a checks | `tools/training/g1a_v2_expansion_chain.sh` | Adds compile, JSON, Mamba2, RWKV, TQ, optional 64k, and optional llama.cpp checks. |
| Phase2 handoff | `tools/training/g1a_phase2_chain.sh` | Now calls `mamba2_smoke` and launches the V2 expansion chain. |
| Mamba2 smoke gate | `crates/dismantle-core/tests/mamba2_smoke.rs` | Self-skipping deterministic greedy gate for Mamba2 GGUFs. |

If an old markdown conflicts with the V2 plan, follow the V2 plan unless the
code proves otherwise.

## Current Truth Table

| Surface | Current truth | Next action |
|---|---|---|
| G1a QAT | Live run/watcher exist; do not stop or restart them. | Let watcher reach step 25/final and launch chains. |
| TQ loader | `rwkv7::gpu::load_tq_artifact` exists behind `--features tq`. | Fix stale loader tests and compatibility gates; do not ask Claude to invent the loader from scratch. |
| TQ serving | `ProjWeight::Tq` exists but single/batched dispatch still `todo!`. | Implement a first-mile strict GPU/CPU path or return explicit unsupported errors, never a runtime panic. |
| TQ GPU | `tq_gpu.rs` and `kernels/mod.rs` have bitslice decode/GEMV/GEMM scaffolding. | Add resident buffers, strict compatibility checks, and CPU/GPU parity before serving claims. |
| Spec decode | Qwen has Eagle and user-ngram propose/verify loops; `speculate/replay_oracle.rs` exists. | Generalize draft sources and add exact verifier tests before RWKV runtime integration. |
| Eagle | Historical trained-head path was net-negative without a better head. | Use Eagle files as reference only; do not make it the main bet. |
| JSON mode | API flag, constraint state machine, and some engine hooks exist. | Verify every generation path masks logits correctly before sampling. |
| Embeddings | `/v1/embeddings` exists, but default engine embedding is still a logit-proxy style fallback unless overridden. | Add true hidden/state pooling per engine and document dimensions truthfully. |
| Mamba2 | Loader/generate path and smoke test exist. | Add HF/Transformers parity only if local weights are available and self-skip otherwise. |
| llama.cpp baseline | Not yet measured for RWKV-7. | Keep optional and clean-room gated via `G1A_V2_LLAMA_BASELINE=1`. |

## Old Expansion Plans To Search As Background

These are useful for archaeology, negative results, and context, but they are
not the first integration target:

| Topic | Background files |
|---|---|
| Dead levers / killed paths | `docs/dead_levers.md`, `docs/reports/kill_ledger_reconciliation.md` |
| Eagle/spec decode history | `docs/eagle5_qwen_port_plan.md`, `docs/plans/eagle_spec_handoff_2026_05_30.md`, `docs/plans/eagle_forward_parity_handoff.md`, `docs/reports/eagle5_v2_wiring_handoff.md`, `docs/reports/eagle5_phase_c_initial_bench.md`, `docs/reports/eagle5_phase_c_root_cause.md`, `docs/reports/spec_decode_runtime_NOT_broken_2026_05_22.md`, `docs/reports/spec_decode_runtime_cost_2026_05_22.md`, `docs/reports/oracle_small_draft_design.md` |
| QTIP/TQ/low-bit | `docs/plans/qtip_bytecut_design_2026_05_31.md`, `docs/reports/oracle_qtip_quality.md`, `docs/reports/qtip_colab_readiness.md`, `docs/reports/mixed_precision_quant_wiring_handoff.md` |
| STRAND archive | `docs/strand/STRAND-dismantle-wiring.md`, `docs/strand/STRAND-metal-kernel-impl.md`, `docs/strand/STRAND-production-status.md`, `docs/strand/STRAND-speed-roadmap.md`, `docs/strand/STRAND-quality-density-frontier.md`, `docs/strand/STRAND-vs-gguf-isobpw.md`, `docs/strand/STRAND-rung-allocator-design.md`, `docs/strand/research/2bit-frontier-SUMMARY.md` |
| Throughput and silicon | `docs/plans/throughput_bible_2026_05_30.md`, `docs/plans/bleeding_edge_throughput_energy_moat_plan_2026_06_05.md`, `docs/reports/path_to_50_gap_diagnosis_2026_05_29.md`, `docs/reports/scout_phase_2_1_gemv.md`, `docs/reports/scout_phase_2_2_dispatch_fusion.md`, `docs/reports/scout_phase_2_3_flash_decode_attn.md` |
| Long-context/stateful moat | `docs/plans/stateful_core_design_2026_05_30.md`, `docs/plans/stateful_moat_continuation_design_2026_05_31.md`, `docs/reports/oracle_prefix_cache.md`, `docs/reports/oracle_prefix_cache_githistory.md` |

## Conflict Resolver

Use this when old docs disagree:

| Conflict | Follow this |
|---|---|
| "Build Eagle first" vs user-ngram/RWKV custom speculation | Build `DraftSource` + replay/state-fork oracles first. Eagle is secondary unless a new oracle clears speed gates. |
| "TQ loader missing" vs current code | Loader exists behind `--features tq`; tests and dispatch are the missing work. |
| `time_mix_gate.weight` in old TQ loader tests | Correct RWKV projection name is `time_mix_output.weight`. |
| "Mamba2 not present" vs current code | Mamba2 loader/generate path exists; smoke/parity and Metal SSD speed are the missing work. |
| "Embeddings done" | Endpoint exists, but per-engine hidden/state pooling is the real feature gate. |
| "JSON mode done" | API/state machine exist, but runtime masking must be verified per engine. |
| "Bench immediately" | Heavy/full benches remain clean-room gated. Scripts should self-skip or require env toggles. |

## Custom Spec Decode Research Built Out

The open question was not just "add custom spec decode"; it was "what is
actually possible in this codebase?"

Answer:

1. Qwen already has the best generic verifier seam: `forward_tokens_verify` plus
   the user-ngram propose-first loop in `qwen_dense.rs`.
2. `speculate/user_ngram.rs` is already a real draft source; it should be lifted
   into a shared trait instead of duplicated.
3. `speculate/replay_oracle.rs` already scores user-ngram acceptance without
   touching GPU or model weights; extend that oracle to every draft source.
4. RWKV is the high-upside custom target because its state is constant-size and
   cloneable. A state-fork verifier can verify drafts without KV growth.
5. Cross-tokenizer drafts are the wrong first move. Same-tokenizer RWKV->RWKV or
   Mamba2->Mamba2 drafts are safer than RWKV draft for Qwen target.
6. The actual speed gate is not acceptance alone. The gate is accepted tokens per
   target forward, verifier overhead, total tokens/sec, and no drift.
7. Grammar/JSON masks must run before accepting speculative tokens; otherwise
   speculation can violate the constrained decoding contract.

The concrete build order is in
`docs/plans/dismantle_expansion_wave_v2_2026_06_18.md`.

## Script Chain Summary

Current post-G1a chain:

```text
g1a_watcher.sh
  -> g1a_phase2_chain.sh
      -> TQ export only if promote gate passes
      -> TQ build/parity hooks
      -> Mamba2 smoke
      -> RWKV flatness/tps legacy benches
      -> g1a_v2_expansion_chain.sh
          -> cargo check core/serve/bench
          -> cargo check core --features tq
          -> json_constrain unit tests
          -> mamba2 smoke
          -> RWKV Metal parity if model exists
          -> RWKV 16k flatness if model exists
          -> TQ trellis synthetic parity
          -> optional 64k flatness
          -> optional TQ artifact loader/bench
          -> optional llama.cpp RWKV baseline
```

Important: the V2 chain is meant to widen coverage without creating a single
all-or-nothing dependency on G1a/TQ. A missing artifact should produce a skip row,
not stop independent work.

## External Research Anchors

Primary references used for the V2 research:

| Topic | Source |
|---|---|
| Exact speculative decoding | https://arxiv.org/abs/2211.17192 |
| Big-Little Decoder | https://arxiv.org/abs/2302.07863 |
| EAGLE-3 | https://arxiv.org/abs/2503.01840 |
| Multi-token prediction | https://arxiv.org/pdf/2404.19737 |
| Mamba2 / SSD | https://arxiv.org/abs/2405.21060 |
| QTIP trellis quantization | https://arxiv.org/abs/2406.11235 |
| QuIP# | https://arxiv.org/abs/2402.04396 |
| BitNet inference | https://github.com/microsoft/BitNet |
| llama.cpp grammars | https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md |
| llama.cpp server | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| CubeCL | https://github.com/tracel-ai/cubecl |
