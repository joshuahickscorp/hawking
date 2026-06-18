# Dismantle Expansion Wave V2
Date: 2026-06-18

This plan extends the G1a low-bit chain with every high-upside lane that is
not blocked on the current QAT result. The goal is to make the post-G1a bench
wave measure more than "did ternary FFN survive": it should also measure
architecture breadth, llama.cpp parity surface, custom SSM speculation, TQ
serving readiness, JSON/grammar constraints, and Mamba2 viability.

## Research Anchors

Primary references checked during this wave:

| Topic | Source | Dismantle implication |
|---|---|---|
| QTIP / trellis quantization | https://arxiv.org/abs/2406.11235 | STRAND/TQ is aligned with the state of the art: trellis-coded weight-only quantization is a real path to sub-Q4 quality and speed if decode is lookup-light/fused. |
| QuIP# | https://arxiv.org/abs/2402.04396 | The RHT/incoherence idea is not optional decoration; it is the quality mechanism that makes extreme PTQ viable. Runtime success depends on folding or cheapening the transform. |
| Exact speculative decoding | https://arxiv.org/abs/2211.17192 | The lossless contract is verifier-emitted tokens. Dismantle should preserve this contract for every custom draft source. |
| Big-Little Decoder | https://arxiv.org/abs/2302.07863 | Small-draft/large-verifier pairings are credible, but only if fallback/rollback overhead is lower than normal decode. |
| Multi-token prediction | https://arxiv.org/pdf/2404.19737 | MTP heads are a strong match for verified RWKV proposals because they can produce K candidates without a second model. |
| Mamba2 / SSD | https://arxiv.org/abs/2405.21060 | Mamba2 is not just another architecture box to tick; SSD gives an SSM/attention bridge and a 2-8x algorithmic speed claim worth benchmarking against RWKV. |
| EAGLE-3 | https://arxiv.org/abs/2503.01840 | The latest speculative-decoding direction is direct token prediction plus multi-layer fusion, not only feature prediction. Dismantle should adapt this to RWKV states instead of copying transformer assumptions. |
| BitNet / 1.58-bit deployment | https://github.com/microsoft/BitNet | Native ternary/binary training plus special kernels are viable. Dismantle should treat ternary STE as a training pressure and STRAND-1/2 as the first deployable target; literal ternary kernels remain a separate bet. |
| llama.cpp grammar surface | https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md | JSON-mode is only the first rung. A GBNF-compatible parser/mask cache is the feature-parity target. |
| llama.cpp server surface | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | Embeddings and LoRA adapter controls are expected serve features; Dismantle now has a first embeddings route and should add real hidden-state pooling/hot adapters. |
| llama.cpp backend breadth | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md | The cross-platform gap is concrete: CPU, BLAS, Metal, SYCL, CUDA, HIP, Vulkan, CANN, OpenCL, Android, OpenVINO. Dismantle should not chase all at once; it needs one portable-kernel spike. |
| CubeCL | https://github.com/tracel-ai/cubecl | Best-looking Rust-native portability candidate: one low-risk spike should target RMSNorm + Q4-like GEMV shapes, not the whole engine. |

## Current Codebase Position

Already present or partially landed:

| Surface | Code surface | Status |
|---|---|---|
| G1a watcher | `tools/training/g1a_watcher.sh` | Watches step 25/final and launches `g1a_phase2_chain.sh`. |
| Phase2 post-G1a chain | `tools/training/g1a_phase2_chain.sh` | Runs TQ export when gate passes, TQ build/parity hooks, RWKV benches, and now launches V2 expansion. |
| V2 expansion chain | `tools/training/g1a_v2_expansion_chain.sh` | New wider chain: compile gates, JSON constraint tests, Mamba2 smoke, RWKV parity/flatness, optional TQ and llama.cpp baselines. |
| Mamba2 | `crates/dismantle-core/src/model/mamba2.rs` | Loader/CPU-reference-style decode exists; `tests/mamba2_smoke.rs` now gates deterministic greedy smoke. |
| Embeddings | `Engine::embed`, `/v1/embeddings` | First route exists, but current default is logit-proxy embedding; RWKV/Mamba should override with hidden/state pooling. |
| JSON constraints | `json_constrain.rs`, `GenerateRequest::json_mode` | Parallel work partially lands JSON-object mode. Engines still need to actually apply masks before sampling. |
| STRAND/TQ | `tq.rs`, `tq_gpu.rs`, `rwkv7_tq_*` tests | Decode, artifact parse, and bitslice scaffolding exist; true RWKV `ProjWeight::Tq` serving, compatibility fences, and stale loader tests remain blockers. |
| Spec decode | `speculate/eagle5*`, `user_ngram`, `replay_oracle`, Qwen verify loops | Transformer-oriented path exists; strongest current surface is user-ngram plus replay oracle. RWKV needs a state-aware verifier before runtime integration. |

Subagent audit update: the strongest existing speculation surface is the
user-draft propose-first loop in Qwen, not the trained Eagle path. Eagle5/Eagle
infrastructure is useful as a reference, but the observed accept-rate history
makes it a secondary bet until a better head is trained. The generic path should
therefore lift `UserNgramDraft` into a draft-source trait and reuse the existing
`forward_tokens_verify` verifier shell before introducing RWKV/SSM drafts.

TQ audit update: `load_tq_artifact` already exists under
`rwkv7::gpu` when built with `--features tq`, and `TqPreparedGpu` plus
`strand_bitslice_gemv/gemm` scaffolding are present. The blocker is no longer
"write a loader from scratch"; it is stale loader tests, strict compatibility
checks, GPU residency, and the `ProjWeight::Tq` single/batched `todo!` dispatch.

## Codebase Research Delta

This is the part that was still under-researched in the first pass.

| Question | Codebase answer | Consequence |
|---|---|---|
| Is there already a draft-source abstraction? | Not as a trait. There are concrete draft sources: `UserNgramDraft` and `Eagle5Head`. | Add a tiny trait around `propose`, `note_token`, `reset`, stats; route `UserNgramDraft` through it first. |
| Is there already an oracle? | Yes: `speculate/replay_oracle.rs` scores user-ngram acceptance using the real in-tree draft logic. | Extend it before adding runtime features; use it as the cheap gate for every custom draft source. |
| Is there already exact verifier machinery? | Yes for Qwen: `forward_tokens_verify` and the propose-first user draft loop. Shared helpers live in `speculate/shared.rs`. | Keep the verifier-emits-token invariant. Do not add any approximate accept path. |
| Why is RWKV special? | RWKV decode state is constant-size per stream, unlike transformer KV. | A state-fork verifier can clone/scratch/commit state without KV growth, making exact speculation unusually attractive. |
| Can a draft use a different tokenizer? | Technically possible through text bridging, but not safe as a first integration. | First drafts must share token ids with the verifier: RWKV->RWKV, Mamba2->Mamba2, Qwen->Qwen. |
| Does grammar/JSON interact with speculation? | Yes. `json_constrain.rs` masks logits by allowed token class. | Proposed draft tokens must be grammar-valid at their positions before acceptance. |
| Is TQ ready to be a draft model backend? | Not yet. Loader exists; dispatch still panics for `ProjWeight::Tq`. | TQ draft is high-upside but artifact/dispatch gated. |
| Is Mamba2 a speculative draft candidate? | Yes after parity, but current implementation is CPU/reference-style and not speed-proven. | Keep Mamba2 smoke/parity independent; do not claim speed until SSD Metal kernels exist. |

## Custom Spec Decode Feasibility Matrix

The goal is a Dismantle-specific speculative decoder, not a blind port of a
transformer design. Every row below must preserve exact verified output.

| Candidate | Target | Draft source | Tokenizer requirement | Current code seam | Blocker | First gate | Likely value |
|---|---|---|---|---|---|---|---|
| Generic `DraftSource` trait | Qwen first, then RWKV/Mamba | Existing `UserNgramDraft` | Same token ids as target | `speculate/user_ngram.rs`, `qwen_dense.rs` propose-first loop | Trait API and no-regression refactor | Qwen draft-on/off temp=0 token ids identical | High leverage, low risk. |
| Offline replay oracle expansion | Target-agnostic | Any draft that can propose token ids | Same token ids or pre-tokenized corpus | `speculate/replay_oracle.rs` | Generalize beyond `UserNgramDraft` | Corpus replay reports tau, acceptance histogram, governor behavior | Very high; no GPU, no training interruption. |
| Qwen user-ngram proposer | Qwen | Per-user n-gram/PLD | Qwen tokenizer | Existing live loop around `forward_tokens_verify` | Needs trait cleanup and grammar mask compatibility | Existing tests plus JSON-mode constrained replay | Cheap speed on repetitive/code-like streams. |
| RWKV state-fork verifier | RWKV-7 | User n-gram first | RWKV World tokenizer | `rwkv7.rs` state/arena plus shared verify helpers | Need test-only scratch state clone/commit path | Exact greedy equivalence vs no-spec over fixed prompts | High; constant-size state is the moat. |
| RWKV small-model draft | RWKV-7 0.4B target | 191M or future 50M/100M RWKV | Same World tokenizer | RWKV loader/generate path; future `DraftSource` | Need cheap draft runtime and state isolation | Replay plus live acceptance; accepted/target-forward >= 1.25 | Potentially very high for serving and STaR. |
| TQ/low-bit RWKV draft | Q4/FP RWKV verifier | STRAND-1/2 or ternary RWKV | Same World tokenizer | TQ loader + `ProjWeight::Tq` planned path | `ProjWeight::Tq` dispatch, resident buffers, quality gate | TQ draft finite logits, exact verifier output, tps above Q4 draft | Huge if TQ dispatch is actually fast. |
| RWKV multi-token heads | RWKV-7 | Auxiliary heads predicting t+2/t+3/t+4 | Same World tokenizer | Training stack, hidden/state access | Need head training/export and verifier harness | MTP proposals exact-verified; no-spec parity | High; avoids second model. |
| Self-spec / early-layer draft | RWKV or Qwen | Partial layers / skip layers | Same target tokenizer | Model internals, hidden captures | Needs stable partial-forward API and cost model | Partial draft accepts enough to beat full forward | Medium/high but more invasive. |
| Mamba2 small draft | Mamba2 target | Smaller or low-bit Mamba2 | Same Mamba tokenizer | `mamba2.rs`, `mamba2_smoke.rs` | Parity and Metal SSD speed absent | HF parity, then accepted/tps gate | Good architecture breadth, speed unknown. |
| Cross-architecture draft | Qwen/RWKV target | Mamba/RWKV text bridge | No direct token-id match | None safe today | Retokenization can break exactness/latency | Do not implement first; only offline text oracle | Low first-mile value; research only. |
| Grammar-aware speculation | Qwen/RWKV/Mamba | Any draft source | Same ids plus grammar token mask | `json_constrain.rs`, future GBNF cache | Need per-position grammar state replay over drafts | Draft accepted only if every accepted token is grammar-valid | Required for agent/server usefulness. |
| Chunked/flash-WKV verification | RWKV | Any multi-token draft | Same World tokenizer | Chunked-scan trainer/prefill ideas, RWKV GPU state | Need K-token verifier that is faster than K serial steps | K-window verify beats serial while exact | Make-or-break for large speedups. |

Do not do these first:

1. Do not revive trained Eagle as the main path unless a new oracle clears
   tau >= 2.5 and end-to-end tps improves.
2. Do not start with cross-tokenizer speculation. It will bury the exactness
   contract under retokenization edge cases.
3. Do not claim speculative speed from acceptance alone. The gate is
   accepted tokens/sec after verifier overhead.
4. Do not integrate into serve until draft-off and draft-on outputs are identical
   for greedy decoding and constrained modes.

Spec-decode gates:

| Gate | Required result |
|---|---|
| Exactness | Temp=0 output ids identical vs no-spec for fixed prompts and recorded transcripts. |
| Grammar safety | JSON/grammar mode output remains valid with speculation enabled. |
| Acceptance | Mean accepted lookahead > 0 and accepted/target-forward >= 1.25 before runtime integration. |
| Throughput | End-to-end tps greater than baseline on same model/hardware, not only fewer target forwards. |
| Overhead | Draft compute + verifier + state copy/rollback stays below saved target work. |
| Memory | RWKV state-fork scratch stays constant-size per slot; no hidden KV-like growth. |
| Batching | Continuous-batch B=2/4/8 does not regress non-spec slots. |
| Disable behavior | Governor backs off after reject streaks and recovers on low-entropy/repetitive spans. |

## V2 Principle

Split every task into one of three lanes:

| Lane | Rule | Examples |
|---|---|---|
| Independent now | Can be coded/tested while G1a trains | Mamba2 smoke, JSON mask tests, hidden-state embeddings, grammar mask cache, LoRA adapter manifest, CubeCL spike doc/tests. |
| Artifact-gated | Runs only if G1a/TQ artifact exists | TQ loader, TQ bpw ledger, TQ parity, TQ tps/PPL bench. |
| Clean-room gated | Requires no active training/desktop GPU noise | llama.cpp RWKV baseline, full 64k flatness, energy/J-token, aggregate B=8 serve sweep. |

The chain should never block independent tasks because artifact-gated tasks are
not ready. Skips are valid outcomes; silent non-measurement is not.

## Custom Speculative Decoding Arc

Do not simply port transformer speculative decoding. RWKV/SSM state changes the
game: state is small, cloneable, and constant-size. That enables a Dismantle-
specific verifier.

### A. State-Fork Speculation

Mechanism:

1. At decode step `t`, clone the target RWKV state into a scratch state.
2. A draft source proposes `K` tokens:
   - n-gram/user draft for free proposals,
   - 191M RWKV draft for cheap same-tokenizer proposals,
   - Mamba2 draft once Mamba2 parity/speed is known,
   - future 50M/100M distilled RWKV draft.
3. Verify proposed tokens on the target using the scratch state.
4. Accept the longest prefix whose target greedy/top-p decision agrees.
5. Commit only the accepted scratch state back to the live slot.
6. If zero accepted, fall back to normal target decode.

Why this is different: transformer verification has KV growth and batch-shape
costs; RWKV state verification is constant memory and can be rolled back by
state pointer swap/copy. That makes exact-lossless speculation more attractive
for Dismantle than for a normal KV engine.

First implementation:

| Step | Deliverable |
|---|---|
| S1 | Add a small `DraftSource` trait: observe tokens, propose `K` ids, reset, stats. |
| S2 | Refactor existing user-ngram propose-first path through `DraftSource` with no behavior change. |
| S3 | Add a mock draft source parity test: draft-on output must equal draft-off at temperature 0. |
| S4 | Add offline replay oracle for arbitrary draft sources using recorded token streams. |
| S5 | Add RWKV state-fork verifier as an offline/test-only source, separate from serving state. |
| S6 | Bench accept length and overhead on recorded transcripts. Gate: `accepted_tokens / target_forwards >= 1.25` before runtime integration. |
| S7 | Add 191M RWKV or Mamba2 draft source only after S1-S6 are green. |

### B. Spec-Topper Governor

The custom "topper" is a per-token policy that chooses the cheapest acceleration
that is safe for the current entropy/format:

| Signal | Decision |
|---|---|
| JSON mode active | Grammar mask first, speculation only if proposed tokens stay grammar-valid. |
| Low entropy / repeated format | Try n-gram/user draft. |
| Medium entropy | Try small RWKV draft. |
| High entropy or recent reject streak | Disable speculation for N steps. |
| Long prompt prefill | Prefer chunked/flash-WKV prefill, not speculation. |
| TQ artifact serving | Prefer lower-bit draft model; verify on Q4/FP target. |

This makes speculation composable with grammar, TQ, and continuous batching
instead of being a single on/off feature.

### C. Multi-Token Prediction for RWKV

Train auxiliary heads on RWKV hidden state for `t+2`, `t+3`, `t+4`. Use them
only as proposal sources and verify with the target path. This borrows the
"direct token prediction" lesson from EAGLE-3 without depending on transformer
layer features. Gate: exact verified output must match no-spec output.

## Quantization Expansion

### A. TQ Serving Blockers

| Blocker | File | Fix |
|---|---|---|
| Stale artifact tests | `rwkv7_tq_loader.rs` tests | Wire ignored tests to existing `rwkv7::gpu::load_tq_artifact`, and fix expected projection names from `time_mix_gate.weight` to `time_mix_output.weight`. |
| Compatibility fence | `tq.rs`, `tq_gpu.rs`, `rwkv7.rs` | First serving MVP should explicitly accept only `RhtMode::None`, scalar `vec_dim == 1`, aligned columns, and no OUTL; every other artifact returns an unsupported error. |
| Runtime dispatch | `rwkv7.rs::ProjWeight` | Replace `ProjWeight::Tq` single/batched `todo!` with an error-gated or working path; no claimed runtime path may panic. |
| GPU residency | `tq_gpu.rs`, `RwkvDecodeArena` | Move payload/table/LUT/scratch into resident buffers instead of re-uploading per call. |
| GPU matvec | `tq_gpu.rs`, `kernels/mod.rs` | Finish `strand_bitslice_gemv` parity against CPU `StrandTensor::matvec` for first-mile compatible artifacts. |
| bpw ledger | new report/test | Record actual bits-per-weight including sideinfo/outliers; compare Q4_K_M bytes. |
| Quality gate | Python eval + Rust fixture | PPL <= promote threshold and deterministic greedy sanity. |

### B. Low-Bit Ladder After G1a

| Rung | Dependency | Purpose |
|---|---|---|
| G1a | running | Last-8 FFN ternary survival. |
| G1a-TQ | G1a gate pass | Export STRAND-2 and measure artifact. |
| G1b | only if Silver | All-layer FFN ternary recovery. |
| G2a | independent once trainer idle | Time-mix-only ternary sensitivity. |
| G2b | after G2a | Mixed FFN/time rung map. |
| G3 | after TQ serving | STRAND-1/2 mixed export, literal ternary kernel only if STRAND loses. |

### C. Quality Improvements

Priority order:

1. Teacher top-k logit capture (`rwkv7_capture_teacher_logits.py`) for KD.
2. Per-tensor sensitivity map using held-out PPL deltas, not just module names.
3. Mixed-rung allocator: protect `r_proj`, lm_head, LoRA micro-matrices; attack FFN bulk first.
4. RHT folding/offline rotation so runtime TQ is not paying a transform tax.
5. Outlier side-channel only where bpw ledger still beats Q4_K_M.

## llama.cpp Gap Closure

| llama.cpp win | Dismantle response |
|---|---|
| 30+ architectures | Add an architecture triage harness: Mamba2 first, then any GGUF architecture with simple dense block layout; do not add dead loaders without smoke tests. |
| Cross-platform | Keep CPU reference correct; spike CubeCL for RMSNorm + one GEMV shape; only graduate when perf and maintenance beat hand-porting. |
| Grammar sampling | Land JSON mode fully, then build GBNF parser + token mask cache. |
| Flash attention | Transformers: only revisit if Qwen gap returns to attention-bound. SSM: prioritize flash-WKV/chunked prefill. |
| Lazy loading | GGUF mmap exists; next improvement is residency/pinning policy and optional tensor prefetch, not "mmap" itself. |
| Embeddings | Replace logit-proxy default with per-engine hidden pooling; expose dimensions/usage correctly. |
| LoRA hot-swap | Add adapter registry and per-request scales. RWKV LoRA micro-matrices make this especially natural. |

## Extended Chain

The watcher now reaches:

```text
g1a_watcher.sh
  -> g1a_phase2_chain.sh
      -> TQ export/build/parity if gate passes
      -> Mamba2/RWKV benches
      -> g1a_v2_expansion_chain.sh
          -> compile gates
          -> JSON constraint tests
          -> Mamba2 smoke
          -> RWKV parity + quick flatness
          -> optional full 64k flatness
          -> optional llama.cpp RWKV baseline
          -> optional TQ artifact gates
```

Environment toggles:

| Env | Meaning |
|---|---|
| `G1A_V2_FULL_BENCH=1` | Run full 64k RWKV flatness sweep. |
| `G1A_V2_LLAMA_BASELINE=1` | Run llama.cpp RWKV head-to-head if llama.cpp is installed and clean-room conditions hold. |
| `RWKV7_TQ_MODEL=/path/model.tq` | Override TQ artifact for loader/bench gates. |
| `DISMANTLE_RWKV7_GGUF=/path/model.gguf` | Override RWKV baseline model. |

## Go/No-Go Gates

| Gate | Go | No-go action |
|---|---|---|
| JSON mode | Valid JSON object smoke plus no compile regressions | Keep API flag but disable runtime mask until engine integration is correct. |
| Mamba2 | Deterministic greedy smoke and finite logits | Keep loader hidden from release docs until parity is real. |
| TQ CPU | `StrandTensor::matvec` parity with exported artifact | Do not wire `ProjWeight::Tq`. |
| TQ GPU | GPU bitslice GEMV matches CPU within fixed tolerance | CPU-only TQ allowed for correctness, not speed claims. |
| TQ quality | PPL within gate and coherent fixtures | Run G1b/G2 mixed ladder. |
| Spec-state fork | Exact greedy equivalence vs no-spec | Keep as oracle/bench only. |
| Spec speed | accepted/target-forward ratio >= 1.25 and tps > baseline | Integrate governor; otherwise use for research only. |
| 64k flatness | +/-2% from depth 0 | Investigate state copy/residency or timing contamination. |
| llama.cpp RWKV baseline | Dismantle >= 1.3x llama.cpp on same model/hardware | If below, profile ggml RWKV path before claiming #1. |

## Next Implementation Queue

1. Finish JSON-mode runtime masking inside RWKV/Qwen/Mamba sampling loops.
2. Override `Engine::embed` for RWKV hidden-state pooling; fix token usage.
3. Fix `rwkv7_tq_loader` ignored tests to call existing `rwkv7::gpu::load_tq_artifact`; expected projection names must use `time_mix_output.weight`, not `time_mix_gate.weight`.
4. Add strict first-mile TQ serving compatibility checks: `RhtMode::None`, scalar `vec_dim == 1`, aligned columns, no OUTL, explicit unsupported errors.
5. Implement `ProjWeight::Tq` CPU/GPU MVP only for the supported first-mile case; never allow the existing `todo!` dispatch to panic in a claimed runtime path.
6. Add GPU-resident TQ buffers and TCB-compatible bitslice GEMV only after synthetic CPU/GPU GEMV parity is green.
7. Add generic `DraftSource` and route user-ngram through it with no behavior change.
8. Generalize `speculate/replay_oracle.rs` beyond `UserNgramDraft`; add mock parity tests that prove exact verifier output.
9. Add RWKV state-fork verification oracle with user-ngram proposals, scratch state, commit/rollback, and exact greedy equivalence.
10. Add grammar-aware speculation tests: JSON mode plus draft proposals must still emit valid constrained output.
11. Add Mamba2 parity harness against HF/Transformers on a tiny prompt, self-skipping when weights/tooling are absent.
12. Add CubeCL spike doc/test harness for one RMSNorm and one GEMV shape.

The north star is not feature-count parity with llama.cpp. The north star is a
local training-inference-compression flywheel where new capabilities feed each
other: QAT creates TQ artifacts, TQ creates faster drafts, faster drafts make
STaR/GRPO cheaper, and grammar/spec-governed serving makes the model useful in
agentic settings without cloud infrastructure.
