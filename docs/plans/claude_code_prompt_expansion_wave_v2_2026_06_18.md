# Claude Code Prompt - Dismantle Expansion Wave V2

You are working in `/Users/scammermike/Downloads/dismantle`.

The user wants the post-G1a chain expanded as aggressively as possible without
interrupting the currently running QAT process. Do not stop or restart training.
Do not run clean-room benches while training is active unless explicitly told to.

Read these files first:

1. `docs/plans/expansion_wave_ingestion_index_2026_06_18.md`
2. `docs/plans/full_project_report_2026_06_18.md`
3. `docs/plans/low_bit_rwkv7_strengthened_revision_2026_06_18.md`
4. `docs/plans/dismantle_expansion_wave_v2_2026_06_18.md`
5. `tools/training/g1a_watcher.sh`
6. `tools/training/g1a_phase2_chain.sh`
7. `tools/training/g1a_v2_expansion_chain.sh`

Use the ingestion index as the conflict resolver. Older Eagle/QTIP/STRAND docs
are background and negative-result evidence; the V2 expansion plan is the active
implementation queue.

Current live process shape:

- G1a QAT writes to `artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8`.
- Watcher polls for `step_000025/state_dict.pt` and `final/state_dict.pt`.
- Watcher launches `tools/training/g1a_phase2_chain.sh` after final.
- Phase2 now launches `tools/training/g1a_v2_expansion_chain.sh`.

Do the work in this order:

1. Verify the current worktree before editing. There may be user/parallel edits,
   especially around `json_constrain.rs`, `GenerateRequest::json_mode`,
   `/v1/embeddings`, and Mamba2. Preserve them.
2. Run lightweight compile gates only:
   - `cargo check -p dismantle-core`
   - `cargo check -p dismantle-serve`
   - `cargo check -p dismantle-bench`
3. Finish JSON-mode runtime integration if it is not already complete:
   - Build or cache a `JsonVocabIndex`.
   - When `GenerateRequest::json_mode` is true, mask logits before sampling.
   - Add one small unit/integration test that proves `response_format={"type":"json_object"}` produces structurally valid JSON for a deterministic tiny prompt if model weights exist; otherwise test the state machine/token mask only.
4. Replace the default logit-proxy embeddings with a real engine override where practical:
   - RWKV should pool hidden/state output if accessible.
   - If hidden states are not exposed yet, add an explicit TODO and keep the route but do not overclaim quality.
5. Fix TQ loader and serving groundwork independent of G1a result:
   - `rwkv7::gpu::load_tq_artifact` already exists behind `--features tq`; do
     not re-invent it.
   - Replace stale stub panics in `tests/rwkv7_tq_loader.rs` with real ignored
     tests that call the loader when `RWKV7_TQ_TEST_ARTIFACT` is set.
   - Fix expected names: use `time_mix_output.weight`, not
     `time_mix_gate.weight`.
   - Add first-mile compatibility checks before serving: `RhtMode::None`,
     scalar `vec_dim == 1`, aligned columns, and no OUTL. Unsupported artifacts
     must return explicit errors.
   - Replace `ProjWeight::Tq` `todo!` dispatch paths with either a working
     first-mile dispatch or a clean unsupported error. No runtime path should
     panic under a claimed feature.
6. Add the custom speculation foundation:
   - Add a tiny `DraftSource` trait and route `UserNgramDraft` through it with
     no behavior change.
   - Generalize `speculate/replay_oracle.rs` so it can score any draft source.
   - Add a mock draft parity test proving exact verifier output equals no-spec.
7. Add the RWKV custom speculation oracle:
   - Start with a test-only state-fork verifier using user-ngram proposals.
   - Clone/scratch/commit RWKV state; never mutate the live state until accepted.
   - Prove exact greedy equivalence vs no-spec.
   - Add gates for accepted/target-forward, overhead, tps, and memory.
   - Do not integrate into serve until those gates exist and pass.
8. Make speculation grammar-safe:
   - When `json_mode` or a future grammar mask is active, a draft token may only
     be accepted if it is legal at that grammar position.
   - Add a test around JSON constraint state replay over a draft window.
9. Improve Mamba2 cautiously:
   - Keep `tests/mamba2_smoke.rs` green.
   - Add parity against HF/Transformers only if the local HF model is available and the test self-skips otherwise.
   - Do not claim speed until a Metal SSD kernel exists.
10. Update `g1a_v2_expansion_chain.sh` with any new gates, but make each heavy/artifact-dependent step self-skip.

Important sequencing:

- Independent tasks must still run if TQ export fails or G1a PPL is marginal.
- TQ serving must not be enabled unless the artifact loads, CPU parity is green,
  GPU parity is green, and PPL/fixture gates pass.
- Speculative decoding must be exact by verification; never trade quality for
  speed silently.
- Same-tokenizer drafts come first. Do not start with cross-tokenizer
  RWKV/Mamba/Qwen text-bridge speculation.
- Eagle is not the main path unless a new oracle clears tau >= 2.5 and e2e tps
  improves. Use it as reference infrastructure, not the default bet.
- Full 64k flatness and llama.cpp RWKV baseline are clean-room gates. Put them
  behind env toggles such as `G1A_V2_FULL_BENCH=1` and
  `G1A_V2_LLAMA_BASELINE=1`.

Definition of done:

- `cargo check -p dismantle-core`, `cargo check -p dismantle-serve`, and
  `cargo check -p dismantle-bench` pass or failures are explained.
- New tests self-skip cleanly when weights/artifacts are missing.
- `tools/training/g1a_phase2_chain.sh` still parses with `bash -n`.
- `tools/training/g1a_v2_expansion_chain.sh` still parses with `bash -n`.
- The final report explains what was added to the chain, what is still gated,
  and exactly what the watcher will do after G1a final.

Do not remove any user work. Do not revert unrelated files.
