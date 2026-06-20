# Session Handoff — Hawking rename executed + chain restarting (2026-06-20)

Paste the "Opening prompt" at the bottom into a new chat to resume.
NOTE: the on-disk folder is still `/Users/scammermike/Downloads/dismantle` (the
repo-folder + GitHub-repo rename is the deliberate LAST step — see "Rename remaining").

## One-paragraph state

**G1a is DONE** — RWKV-7 0.4B ternary-QAT hit its EMA gate (loss_ema 5.92 ≤ 6.0) at
step 90, exported a full HF model (candidate ppl 3.449 vs fp32 base 3.356 = **+2.7%**).
**The `dismantle → hawking` rename is EXECUTED** across crates, tool-crates, tooling
env/binary refs, and public docs — `cargo check --workspace` + 167 lib tests green.
**Storage lightened** (`cargo clean`, dead artifacts pruned: 36 GB → 20 GB). The
**master chain is being restarted** (draft-sweep bug fixed) to train the 7 spec-decode
drafts. Only the folder/GitHub-repo rename remains.

## What the rename covered (committed, build-green)

- `crates/{dismantle,-core,-serve,-bench}` → `crates/hawking{,-core,-serve,-bench}`
  (git mv); all Rust imports `dismantle_core::`→`hawking_core::`, Cargo package names +
  path deps + lockfile, the 3 tool-crates (q4k_fast/awq_bake/tq_bake), and renamed
  source files (`competitors/hawking.rs`, etc.).
- CLI command `dismantle`→`hawking`; `DISMANTLE_*`→`HAWKING_*` env (code + tools);
  `/v1/dismantle/*`→`/v1/hawking/*`; `dismantle_*`→`hawking_*` metrics; `.dismantle`
  sidecar ext→`.hawking` (no persisted files existed); `target/release/dismantle`→
  `…/hawking` in tools; cargo `-p dismantle-core`→`-p hawking-core` in chain/bench.
- Public docs (README, CONTRIBUTING, serve/autotune/profile guides) rebranded to Hawking.
- Commits: `c0133c7` (crates), `949b252` (tools+public docs), `6f5bbec` (cargo refs in
  tools), plus the draft-sweep fix `499e566`. Safety tag: `pre-hawking-rename`.

### NOT renamed on purpose
- The on-disk folder `/Users/.../Downloads/dismantle` and the GitHub repo (clone URL).
- Dated internal docs (`docs/plans|reports|strand`) — historical records, carry
  absolute paths; left as-is.
- Shader/golden comment mentions of `dismantle-core` paths (cosmetic only).
- Deep logic refactor: deliberately NOT done — prior dead-code audit found the codebase
  curated (Metal kernels are string-referenced, look dead but aren't); condensing would
  risk parity with no bisect. Storage was the safe "lightening" win.

## Rename remaining (the final Phase-6 step — do when ready)
```bash
# 1. rename folder (nothing must be running; closes this repo path)
cd ~/Downloads && mv dismantle hawking && cd hawking
# 2. GitHub repo: rename on github.com, then:
git remote set-url origin https://github.com/joshuahickscorp/hawking.git
# 3. optional: tag the old name  git tag dismantle-final-alias
# 4. fix any absolute /Downloads/dismantle paths in internal docs/fixtures if desired
```

## Master chain (restarted this phase)

Sequential (one MPS job; 18 GB), detached (`nohup caffeinate`), chunked + seeded(1337),
watermark 0.0. Stages (each soft-fails):
1. cargo check core/serve/bench/tq (now `-p hawking-*`), json/mamba2 tests, rwkv7 parity
   + flatness, TQ parity, TQ artifact gates, Qwen3B llama baseline (soft; needs release
   binary which is not built — will skip).
2. **Draft sweep** — `draft_35m_probe 50m 75m 100m 150m 200m 300m`, from scratch,
   chunked, EPOCHS=1. The bash-3.2 empty-array bug that made the prior run false-complete
   in 1s is FIXED (`499e566`). Results → `runs/custom_<v>/eval_log.jsonl`.
3. `rwkv7_spec_hardening.py` → `rwkv7_competitive_scorecard.py`.

### Monitor
```bash
tail -f artifacts/lowbit_rwkv7/master_chain.log
tail -f artifacts/lowbit_rwkv7/draft_sweep.log        # MUST show real [ep0 opt=N] step lines, not instant abort
ls -t artifacts/lowbit_rwkv7/runs/custom_*/eval_log.jsonl
pgrep -fl "g1a_v2_expansion_chain|rwkv7_train_draft"
```
Re-launch just the sweep if it dies:
`DRAFT_VARIANTS="<remaining>" EPOCHS=1 SEED=1337 USE_CHUNKED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 bash tools/training/launch_draft_sweep.sh`

## G1a output (finished 0.4B model — banked)
`artifacts/lowbit_rwkv7/hawking_arc/ema6p0_step_000090_20260620_041225/` — `hf/`
(loadable export), `report.md`, `ppl.jsonl`, `samples.jsonl`, `frontier_queue.sh`.

## Open items
- [ ] Let the chain finish (~20–30h); confirm the draft sweep ACTUALLY trains (watch
      draft_sweep.log for real step lines this time).
- [ ] Build release binary to validate end-to-end: `cargo build --release` → `target/release/hawking`.
- [ ] Folder + GitHub repo rename (commands above).
- [ ] 9 stale feature branches still un-decided (21–31 behind main).
- [ ] Re-run draft accept-rate vs the real 3B/7B target once downloaded.

## Opening prompt (paste into a new chat)

> Resuming the Hawking project (folder still at ~/Downloads/dismantle; repo rename
> pending). Read `docs/plans/SESSION_HANDOFF_2026_06_19.md` first. State: G1a 0.4B QAT
> is done + exported; the dismantle→hawking code/docs rename is executed and build-green
> (167 lib tests pass); storage cleaned; the master chain (7-model draft sweep) was
> restarted with the bash-3.2 bug fixed. First check the chain is really training:
> `tail -n 40 artifacts/lowbit_rwkv7/draft_sweep.log` (must show `[ep0 opt=N]` lines,
> not instant aborts) and `pgrep -fl rwkv7_train_draft`. Then either babysit the sweep,
> build the release binary, or do the final folder/GitHub repo rename.
