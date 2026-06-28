# Claude Goal Prompt - Hawking Autonomous Production Run

> **SUPERSEDED (2026-06-22):** use
> `docs/campaign/hawking_ship_finalization_prompt.md` plus
> `docs/campaign/hawking_ship_goal_prompts.md` for future ship-finalization
> work. This file is kept for provenance and older campaign context.

Copy/paste this into Claude when starting or resuming the next autonomous run.
For a longer, more aggressive campaign prompt, use
`docs/campaign/claude_moonshot_goal_prompt.md`.

````text
You are continuing the Hawking autonomous production-hardening campaign in:

  /Users/scammermike/Downloads/hawking

Mission:
Keep making aggressive, evidence-backed progress toward production viability without breaking the tree. Prioritize low-risk infrastructure, validation breadth, benchmark hygiene, and reversible gated implementation. The campaign should keep iterating, but any hot-path Rust/Metal change must be behind a default-off flag and pass parity/quality gates before being considered commit-worthy.

Model/agent guidance:
- Sonnet is acceptable for docs, benchmark orchestration, artifact hygiene, issue triage, harness work, and running/recording validation.
- Use Opus/Codex-level care for hot-path code, Metal kernel wiring, nontrivial Rust architecture changes, or any commit that could alter model outputs.
- If using Sonnet overnight, bias it toward running existing scripts, improving reports, and producing small reversible patches. Do not let it freestyle hot-path rewrites.

Read first, in this order:
1. docs/campaign/findings_summary.md
2. docs/campaign/roadmap.md
3. docs/campaign/test_matrix.md
4. docs/campaign/autonomous_run_log.md
5. docs/campaign/kill_ledger.md
6. docs/env_flags.md
7. docs/architecture.md

Current validated state:
- Spec decode/Event Horizon is lossless but speed-negative; router committed/default-off. Do not chase speculative decode speed unless a new verifier/proposer removes the per-cycle overhead wall.
- The biggest validated strategic win is the RWKV-7/SSM long-context path: RWKV-7 0.4B stays roughly flat at ~119 tps at ~8k while Qwen falls to ~8.6 tps.
- Qwen decode config levers are mostly at the noise floor. `--profile fast` is only ~+3-7% and has mild quality trade.
- `predec` is the real Qwen decode win and is already default-on.
- F16_KV is a clean footprint/long-context lever. Per-channel int4-KV numerics pass parity but the e2e hot-path wiring is still intentionally deferred.
- Production-hardening infrastructure now exists:
  - `tools/ci/overnight_hardening.sh`
  - `tools/ci/ssm_serve_smoke.sh`
  - `tools/ci/README.md`
  - `docs/campaign/change_manifest.md`
  - `docs/campaign/ssm_productionization.md`
- CPU-only hardening passed in `reports/overnight/20260622T020448Z/`: `cargo check`, core lib tests (182 passed, 0 failed, 1 ignored), and 3 parity tests.
- RWKV serve smoke found a real production gap in `reports/serve-smoke/20260622T022233Z/`: server loads and `/healthz`, `/v1/models`, `/metrics` pass, but `/v1/hawking/generate` times out after 180s. Metrics after the request show `queued_requests=1`, `requests_admitted=0`, `tokens_generated=0`. Treat this as the top serve-path bug to characterize/fix.
- The prior Claude-launched mamba2 corroboration process appears to have exited, but no saved stdout/log was found under `/tmp`, `reports`, or the app terminal. Do not record new mamba numbers unless you can directly verify them from output.
- The working tree may contain user/agent changes. Never revert changes you did not make.

Rails:
- No destructive git: no `git reset --hard`, no checkout/revert of user changes, no history rewrite.
- Before editing, inspect `git status --short` and the relevant diffs.
- Keep GPU/model jobs sequential. Do not start a new `hawking generate` while another is active unless using the runner's `GPU_WAIT=1`.
- Record every material result in `docs/campaign/test_matrix.md` or `docs/campaign/autonomous_run_log.md`.
- If a result is not directly verified from logs/output, label it pending or unverified.
- Leave reports under `reports/`; they are ignored and should not be staged.
- Do not commit unless the staged scope is clean, intentional, and excludes unrelated user changes.

Priority order:
1. If a mamba2/RWKV/overnight process is running, wait or observe; do not contend with it.
2. Characterize/fix the SSM serve admission gap:
   - Reproduce with `tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf`.
   - Start from report `reports/serve-smoke/20260622T022233Z/`.
   - Inspect why a native generation request enters `wait_queue` but is never admitted/decode-stepped.
   - Keep any code change small, default-preserving, and covered by the smallest relevant serve/batch test or smoke.
3. Run and/or improve the non-destructive validation lane:
   - `RUN_GPU=0 tools/ci/overnight_hardening.sh`
   - then, when GPU is idle, `tools/ci/overnight_hardening.sh`
   - for a focused SSM server gate, `tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf`
4. Turn validation output into durable artifacts:
   - update `docs/campaign/test_matrix.md`
   - update `docs/campaign/autonomous_run_log.md`
   - add exact commands and pass/fail summaries
5. Productionize the SSM long-context path:
   - serving docs
   - workload bench matrix
   - model-selection defaults/recommendations
   - quality checks on representative long-context prompts
6. Only after that, consider per-channel int4-KV wiring:
   - add a new default-off `HAWKING_QWEN_INT4_KV_PC` flag
   - mirror the documented wiring spec in `docs/campaign/roadmap.md`
   - run parity, long-context coherence, and real-model perplexity before any default-on or commit
7. Treat MLX-diff/Q4_K GEMV structural work as research only. Prototype behind flags, measure GB/s, and stop if the delta is not clean.

Useful commands:

```bash
git status --short
ps ax -o pid= -o etime= -o %cpu= -o %mem= -o command= | rg 'hawking generate|mamba2-370m|rwkv7|cargo test|cargo build|overnight_hardening' | rg -v 'rg '

# CPU-only production hardening. Safe while GPU is busy.
RUN_GPU=0 tools/ci/overnight_hardening.sh

# Full unattended validation. Use only when GPU/model slot is free, or let GPU_WAIT=1 wait.
tools/ci/overnight_hardening.sh

# Focused SSM serve smoke: health, models, native SSE generation, metrics, shutdown.
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf

# Inspect the latest failed serve-smoke report.
sed -n '1,240p' reports/serve-smoke/20260622T022233Z/summary.md
sed -n '1,120p' reports/serve-smoke/20260622T022233Z/metrics_after.log

# Recovery from latest report.
ls -td reports/overnight/* | head
sed -n '1,240p' "$(ls -td reports/overnight/* | head -1)/summary.md"
```

Deliverables before stopping:
- A short status update in `docs/campaign/autonomous_run_log.md`.
- Any new validated numbers in `docs/campaign/test_matrix.md`.
- A clear next command for the next agent.
- If code was changed, run the smallest relevant gate plus `git diff --check`.

Stop and ask the user before:
- deleting or reverting code,
- changing defaults for model output,
- spending external money/cloud credits,
- starting an irreversible long training run,
- committing hot-path changes without parity/quality evidence.
````
