# Change Manifest - Current Dirty Tree

> **SUPERSEDED (2026-06-22) by `commit_plan.md`** — that file is now the canonical dirty-tree lane split.
> Kept for provenance; do not delete.

Purpose: keep commit/review hygiene sane while the autonomous campaign continues. This
file classifies the current worktree by intent so the next agent does not mix unrelated
lanes or accidentally revert user/agent work.

## Likely campaign-infra / artifact lane

These are the production-hardening artifacts created or consolidated by the current
campaign. They are good candidates for one focused "campaign infra + validation
artifacts" commit after one final `git diff --check` and scope review:

- `docs/campaign/`
  - `findings_summary.md`
  - `kill_ledger.md`
  - `roadmap.md`
  - `test_matrix.md`
  - `autonomous_run_log.md`
  - `claude_goal_prompt.md`
  - `claude_moonshot_goal_prompt.md`
  - `claude_final_hardening_condensation_goal_prompt.md`
  - `change_manifest.md`
  - `ssm_productionization.md`
- `docs/architecture.md`
- `docs/env_flags.md`
- `docs/plans/ratios_roadmap_2026_06_21.md`
- `docs/plans/q6k_predec_design.md`
- `tools/bench/ratios.sh`
- `tools/ci/README.md`
- `tools/ci/preflight.sh`
- `tools/ci/overnight_hardening.sh`
- `tools/ci/ssm_serve_smoke.sh`

Notes:
- `reports/overnight/*` is generated evidence and ignored by git. Do not stage it.
- `reports/serve-smoke/*` is generated evidence and ignored by git. Do not stage it.
- `tools/ci/overnight_hardening.sh` has been syntax-checked and no-GPU smoke-tested.
- `tools/ci/ssm_serve_smoke.sh` has been syntax-checked; RWKV serve smoke exposed an SSM serve admission/decode gap.
- `docs/campaign/claude_goal_prompt.md` is the restart prompt for Claude/Sonnet.

## Test-only diagnostic code lane

- `crates/hawking-core/src/model/qwen_dense.rs`

Current diff is a `#[cfg(test)]` / diagnostic module (`bsize_verify_diag` /
verify-cost curve work), not a half-wired int4-KV hot-path patch. It should be
reviewed as a diagnostic/testing commit, not bundled with production wiring.

Before committing this lane:
- confirm the diff is still test-only,
- run the smallest relevant test target,
- do not present it as int4-KV wiring.

## Training / KD / RWKV experiment lane

These files are modified from prior agent work and should not be bundled with the
campaign-infra commit unless intentionally reviewed as a training experiment change:

- `docs/plans/g1a_v2_expansion_results_2026_06_20.md`
- `docs/plans/rwkv7_competitive_scorecard_2026_06_20.md`
- `docs/plans/rwkv7_spec_hardening_2026_06_20.md`
- `docs/plans/throughput_pivot_campaign.md`
- `tools/training/g1a_v2_expansion_chain.sh`
- `tools/training/launch_draft_sweep.sh`
- `tools/training/rwkv7_competitive_scorecard.py`
- `tools/training/rwkv7_custom_configs.py`
- `tools/training/rwkv7_spec_hardening.py`
- `tools/training/diag_draft_rollout.py`
- `tools/training/diag_draft_step1.py`

Treat these as a separate training/KD lane. Read their diffs before editing; do not
revert them while doing infrastructure work.

## Recommended commit split

1. Campaign infra/artifacts:
   - `docs/campaign/*`
   - `docs/architecture.md`
   - `docs/env_flags.md`
   - `docs/plans/ratios_roadmap_2026_06_21.md`
   - `docs/plans/q6k_predec_design.md`
   - `tools/bench/ratios.sh`
   - `tools/ci/*`

2. Diagnostic tests:
   - `crates/hawking-core/src/model/qwen_dense.rs`

3. Training/KD/RWKV experiment updates:
   - all `tools/training/*` and related `docs/plans/*` above, after a separate
     review and validation pass.

## Current safety status

- No hot-path int4-KV-PC wiring is present.
- Per-channel int4-KV remains design-ready and default-off until implemented.
- The earlier mamba2 SSM bench is no longer visible in `ps`, but its final stdout/log was not found. Treat final numbers as unverified.
- The next safe validation command is:

```bash
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf
```
