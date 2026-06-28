# Claude Moonshot Goal Prompt - SSM Production Campaign

> **SUPERSEDED (2026-06-22):** use
> `docs/campaign/hawking_ship_finalization_prompt.md` plus
> `docs/campaign/hawking_ship_goal_prompts.md` for future ship-finalization
> work. This file is kept for provenance and older SSM campaign context.

Copy/paste this into Claude when the current focused loop finishes too quickly
and you want a longer autonomous production run. For the final condensation pass
before stopping the loop, use
`docs/campaign/claude_final_hardening_condensation_goal_prompt.md`.

````text
/goal Continue the Hawking autonomous production-hardening campaign in:

  /Users/scammermike/Downloads/hawking

Moonshot mission:
Turn the validated RWKV/SSM long-context speed moat into a production-grade
Hawking path, not just a benchmark result. Keep iterating aggressively across
serve correctness, quality evaluation, model-selection policy, validation
infrastructure, and commit hygiene. If one lane completes, immediately move to
the next lane below. Stop only for destructive/irreversible choices, secrets,
external spending, or hot-path Rust/Metal changes that lack a parity/quality
gate.

Read first:
1. docs/campaign/autonomous_run_log.md
2. docs/campaign/test_matrix.md
3. docs/campaign/ssm_productionization.md
4. docs/campaign/roadmap.md
5. docs/campaign/change_manifest.md
6. docs/campaign/claude_goal_prompt.md
7. docs/env_flags.md
8. docs/architecture.md

Current state to preserve:
- Spec decode/Event Horizon is lossless but speed-negative. Do not reopen unless
  a new design removes verifier/proposer overhead.
- RWKV-7 is the strategic frontier: ~119 tps at ~8k vs Qwen ~8.6 tps. The product
  opportunity is long-context throughput plus bounded recurrent state.
- A small RWKV serve admission patch is already applied in
  `crates/hawking-core/src/model/rwkv7.rs`: `encode_prompt_for_batch`,
  `decode_token_for_batch`, and `eos_id_for_batch`.
- Admission gap evidence:
  - Before: `reports/serve-smoke/20260622T022233Z/` had queued=1, admitted=0,
    tokens=0, and timed out after 180s.
  - After: `reports/serve-smoke/20260622T023814Z/` has admitted=1, queued=0,
    no 180s hang, but generation returns one empty token and no final stats.
- Downstream serve bug:
  - Single-stream `hawking generate` is coherent and fast.
  - `hawking serve` now admits RWKV but emits `{"text":"","tok_index":0}`.
  - The likely fault is RWKV serve prefill's CPU -> GPU multiseq recurrent-state
    handoff (`copy_cpu_state_to_gpu_slot`) or immediate-EOS handling.
  - Treat fixes in RWKV/Metal state layout as higher-care hot-path work. Add a
    parity gate first.
- Active external work:
  - An `overnight_hardening.sh` run is active at
    `reports/overnight/20260622T025008Z/`.
  - It skipped serve smoke and is collecting SSM/Qwen bench evidence. It was
    observed on `mamba2_long`. Do not start another GPU/model job until this
    exits. Harvest its logs and update campaign docs when complete.
- The worktree is dirty with multiple lanes. Never revert changes you did not
  make. Use `docs/campaign/change_manifest.md` before staging anything.

Rails:
- No destructive git: no reset, checkout, revert, branch deletion, or history
  rewrite.
- Keep GPU jobs sequential. If a `hawking generate`, `hawking serve`, cargo
  build, or overnight runner is active, observe or wait; do not contend.
- Record every material result in `docs/campaign/autonomous_run_log.md` and
  `docs/campaign/test_matrix.md`.
- Reports under `reports/` are evidence and should not be staged.
- Sonnet is fine for docs, harnesses, log harvesting, CLI validation, and small
  reversible patches.
- Use higher-care review for hot-path Rust/Metal, model-output defaults, RWKV
  recurrent-state layout, or any change that could alter generation semantics.

Lane 0 - harvest live evidence first:
1. Check active processes:
   `ps ax -o pid= -o ppid= -o etime= -o %cpu= -o %mem= -o command= | rg 'hawking (generate|serve)|overnight_hardening|cargo (build|check|test)|mamba2|rwkv7' | rg -v 'rg '`
2. If `reports/overnight/20260622T025008Z/` is still running, wait or observe.
3. When it completes, read:
   - `reports/overnight/20260622T025008Z/summary.md`
   - `reports/overnight/20260622T025008Z/commands.log`
   - all `rwkv7_*`, `mamba2_*`, `ratios_*`, and `preflight_fast.log` files
4. Update `docs/campaign/test_matrix.md` and
   `docs/campaign/autonomous_run_log.md`.
5. If `preflight_fast` failed, inspect and classify it. Fix only if it is
   in-scope and low-risk; otherwise record the exact failure.

Lane 1 - make RWKV serve correct:
1. Add a focused RWKV parity/diagnostic test before touching state-copy code.
   Goal: prove `prefill_slot(slot_id, prompt_ids)` followed by
   `forward_multiseq_batched(... region=slot_id ...)` produces the same next
   token/logits as RWKV single-stream GPU `generate`/`forward_token_routed`.
2. If direct logits comparison is awkward, start with a token-level test that
   asserts the first served token is non-empty and matches/does not immediately
   EOS on a stable prompt. Prefer a real parity oracle over a weak smoke.
3. Only after the test reproduces the bug, inspect and fix
   `copy_cpu_state_to_gpu_slot` layout or replace CPU-prefill handoff with a
   per-slot GPU prefill path that preserves slot isolation.
4. Rebuild release and rerun:
   `tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf`
5. Success gate: health/models/metrics pass; native SSE emits useful text,
   final stats, `[DONE]`, admitted>=1, queued=0, tokens_generated>1.

Lane 2 - make HTTP admission robust:
1. Audit every `.ok().flatten()` around `driver.admit`.
2. Distinguish `Err(Unimplemented)` / tokenization failures from `Ok(None)`.
   Unsupported engines should return a clear structured SSE error or use a
   deliberate single-request `generate` fallback; they must not silently queue
   forever.
3. Add a small unit test or fake-engine serve-path test for "admit error does
   not enter wait_queue".
4. Keep Qwen's continuous-batch behavior unchanged.

Lane 3 - build the SSM product gate:
1. Add or extend a script such as `tools/ci/ssm_product_gate.sh` that can run:
   - RWKV serve smoke
   - RWKV short/mid/long/optional-16k speed matrix
   - representative quality prompts
   - request-isolation/repeated-request smoke
   - cancellation or client-disconnect smoke if easy
2. The gate should write one timestamped report under `reports/ssm-product/`.
3. Include exact commands, model paths, prompt sizes, median tps, pass/fail,
   and recovery instructions.

Lane 4 - quality, not just speed:
1. Create a long-context quality suite with real tasks:
   - evidence near the beginning of a long context
   - code-file summarization and bug identification
   - JSON fact extraction
   - instruction-following with format constraints
   - multilingual and math sanity prompts
2. Compare RWKV/SSM against Qwen or a clear rubric. Do not claim product
   readiness from tps alone.
3. Record failures by prompt class and propose fallback routing.

Lane 5 - model selection and hybrid path:
1. Draft an operator-facing decision table:
   `ctx_tokens`, quality mode, latency mode, prompt class -> Qwen/RWKV/Mamba/hybrid.
2. Prototype a hybrid lane only if low-risk:
   SSM summarizes/extracts long context -> Qwen produces final high-precision
   answer. Gate on end-to-end correctness and latency.
3. Keep model-output defaults conservative. No silent semantic changes.

Lane 6 - commit hygiene and packaging:
1. Keep lanes separate:
   - RWKV serve admission patch
   - RWKV serve correctness/handoff patch, if any
   - CI/harness/docs artifacts
   - training/KD/RWKV experiment changes
2. Run `git diff --check`.
3. Run the smallest relevant tests for each touched lane.
4. Do not stage generated reports.
5. Commit only when the scope is intentional and does not mix unrelated user or
   agent changes.

Lane 7 - optional transformer compression after SSM lane is moving:
Only after RWKV serve correctness or its gate is clearly advanced, consider
per-channel int4-KV wiring behind a new default-off flag
`HAWKING_QWEN_INT4_KV_PC`. Use the roadmap wiring spec and require:
- parity test
- long-context coherence
- real-model perplexity spot-check
Do not treat int4-KV as a substitute for SSM productionization.

Useful commands:

```bash
git status --short
git diff --stat

ps ax -o pid= -o ppid= -o etime= -o %cpu= -o %mem= -o command= \
  | rg 'hawking (generate|serve)|overnight_hardening|cargo (build|check|test)|mamba2|rwkv7' \
  | rg -v 'rg '

sed -n '1,260p' reports/serve-smoke/20260622T023814Z/summary.md
sed -n '1,260p' reports/overnight/20260622T025008Z/summary.md
tail -120 reports/overnight/20260622T025008Z/commands.log

bash -n tools/ci/overnight_hardening.sh
bash -n tools/ci/ssm_serve_smoke.sh
cargo check
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf
```

Deliverables before stopping:
- Updated `docs/campaign/autonomous_run_log.md`.
- Updated `docs/campaign/test_matrix.md`.
- A clear next command for the next agent.
- If code changed: `git diff --check` and the smallest relevant passing gate.
- A short final summary of what is now proven, what failed, and what should run
  next.
````
