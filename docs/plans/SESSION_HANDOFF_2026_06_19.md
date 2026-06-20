# Session Handoff — Hawking rename COMPLETE (incl. folder + GitHub) — 2026-06-20

This is the continuity anchor. The previous session renamed everything to **Hawking**,
including the on-disk folder (`~/Downloads/hawking`) and the GitHub repo
(`joshuahickscorp/hawking`). Open your new chat **inside `~/Downloads/hawking`**.

## FIRST THING TO DO (the chain is ALREADY RUNNING — just verify)

The 7-model draft-sweep chain was relaunched from the renamed folder and is running
detached (`nohup caffeinate`, reparented to launchd, survives terminal close). Do NOT
launch a second one. Verify it's alive + actually training:
```bash
cd ~/Downloads/hawking
pgrep -fl "g1a_v2_expansion_chain|rwkv7_train_draft"
tail -n 30 artifacts/lowbit_rwkv7/draft_sweep.log   # expect [ep0 opt=N] lines (real training)
tail -n 10 artifacts/lowbit_rwkv7/master_chain.log  # current stage
```
If it died, relaunch:
```bash
DRAFT_VARIANTS="draft_35m_probe draft_50m_probe draft_75m_probe draft_100m draft_150m draft_200m draft_300m" \
DRAFT_EPOCHS=1 DRAFT_ACCEPT_SEQS=50 SEED=1337 USE_CHUNKED=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHON=.venv-rwkv/bin/python \
  nohup caffeinate -dimsu bash tools/training/g1a_v2_expansion_chain.sh 3.4489 pass \
  > artifacts/lowbit_rwkv7/master_chain.log 2>&1 &
```

## State

- **G1a DONE**: RWKV-7 0.4B ternary-QAT hit ema 5.92 ≤ 6.0 at step 90; exported HF model,
  candidate ppl 3.449 vs fp32 base 3.356 (**+2.7%**, near-lossless).
- **Rename FULLY COMPLETE**: code, CLI, env (`HAWKING_*`), endpoints (`/v1/hawking/*`),
  metrics (`hawking_*`), `.hawking` sidecar, public docs, **the folder (`~/Downloads/hawking`),
  and the GitHub repo (`github.com/joshuahickscorp/hawking`)**. `cargo check --workspace`
  + 167 lib tests green at rename time.
- **Storage cleaned**: `cargo clean` + dead-artifact prune (was 36 GB → 20 GB; `target/`
  will rebuild on first cargo run).
- **Chain**: relaunch it (above). Draft-sweep bash-3.2 empty-array bug fixed + verified.

## Key locations (all under ~/Downloads/hawking)

- Finished 0.4B model: `artifacts/lowbit_rwkv7/hawking_arc/ema6p0_step_000090_20260620_041225/`
  (`hf/` loadable export, `report.md`, `ppl.jsonl`, `samples.jsonl`).
- Training tools: `tools/training/` (`g1a_v2_expansion_chain.sh`, `launch_draft_sweep.sh`,
  `rwkv7_train_draft.py`, `rwkv7_progress.py`, `autocycle_step50_ozempic.sh`,
  `hawking_after_ema.py`).
- Rename kit/plan (history): `docs/plans/hawking_rename_execution_kit_2026_06_19.md`,
  `docs/plans/hawking_total_rename_plan_2026_06_19.md`.

## Monitor the chain
```bash
tail -f artifacts/lowbit_rwkv7/master_chain.log     # stage-by-stage
tail -f artifacts/lowbit_rwkv7/draft_sweep.log      # current draft model + step lines
ls -t artifacts/lowbit_rwkv7/runs/custom_*/eval_log.jsonl
```
Chain = cargo checks (cold rebuild ~10 min, also revalidates the rename) → 7-model draft
sweep (chunked, seeded, EPOCHS=1, ~20–30h) → spec hardening → competitive scorecard.

## Git
- Local `main` == `origin/main`, remote = `github.com/joshuahickscorp/hawking.git`.
- Safety tag `pre-hawking-rename` marks the last pre-rename commit if you ever need it.
- 9 stale feature branches remain (21–31 behind main) — decide merge/delete.

## Verify the rename build (optional but recommended)
```bash
cd ~/Downloads/hawking
cargo build --release            # produces target/release/hawking
./target/release/hawking --help
cargo test --workspace --lib     # expect ~167 pass
```

## What was deliberately NOT done
- Deep logic refactor — prior dead-code audit shows the codebase is curated (Metal kernels
  are string-referenced; they look dead to the compiler but aren't). Condensing logic risks
  parity. Storage was the safe "lightening".
- Dated internal docs (`docs/plans|reports|strand`) keep old `dismantle` mentions + absolute
  `/Downloads/dismantle` paths as historical records. Fix opportunistically if desired.

## Open items
- [ ] Restart + babysit the chain (above) — confirm real training this time.
- [ ] `cargo build --release` to validate the renamed binary end-to-end.
- [ ] Decide on the 9 stale feature branches.
- [ ] Re-run draft accept-rate vs the real 3B/7B target once downloaded.
- [ ] (cosmetic) sweep `/Downloads/dismantle` → `/Downloads/hawking` in internal docs/fixtures.

## Opening prompt (paste into a new chat opened in ~/Downloads/hawking)

> Resuming the Hawking project (renamed from dismantle last session — folder is now
> ~/Downloads/hawking, GitHub repo is joshuahickscorp/hawking). Read
> `docs/plans/SESSION_HANDOFF_2026_06_19.md` first. G1a 0.4B QAT is done + exported; the
> full dismantle→hawking rename (code + folder + repo) is complete and was build-green
> (167 lib tests). The 7-model draft-sweep chain was stopped for the folder rename and
> needs restarting — run the "FIRST THING TO DO" command block in the handoff, then
> confirm it trains via `tail artifacts/lowbit_rwkv7/draft_sweep.log` (expect [ep0 opt=N]
> lines). After that: build the release binary, then keep an eye on the sweep.
