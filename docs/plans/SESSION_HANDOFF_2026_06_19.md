# Session Handoff — Hawking (rename finalized + docs consolidated) — 2026-06-20

Continuity anchor. The project is **Hawking** (folder `~/Downloads/hawking`, repo
`github.com/joshuahickscorp/hawking`). Open new chats **inside `~/Downloads/hawking`**.

## FIRST THING TO DO — the chain is RUNNING, just verify (do NOT relaunch)

The 7-model RWKV-7 draft-sweep chain is running detached (`nohup caffeinate`, survives
terminal close). Confirm it's alive + actually training:
```bash
cd ~/Downloads/hawking
pgrep -fl "g1a_v2_expansion_chain|rwkv7_train_draft"      # expect 2-3 procs
tail -n 30 artifacts/lowbit_rwkv7/draft_sweep.log         # expect [ep0 opt=N] loss=… lines
tail -n 10 artifacts/lowbit_rwkv7/master_chain.log
```
If (and only if) it died, relaunch with the AUTO_BATCH preset (fast + no OOM — docs/SPEED.md):
```bash
AUTO_BATCH=1 BATCH_SIZE=16 MEM_CEILING_GB=17 GRAD_CKPT=1 GRAD_ACCUM=1 \
DRAFT_VARIANTS="draft_35m_probe draft_50m_probe draft_75m_probe draft_100m draft_150m draft_200m draft_300m" \
DRAFT_EPOCHS=1 DRAFT_ACCEPT_SEQS=50 SEED=1337 USE_CHUNKED=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHON=.venv-rwkv/bin/python \
  nohup caffeinate -dimsu bash tools/training/g1a_v2_expansion_chain.sh 3.4489 pass \
  > artifacts/lowbit_rwkv7/master_chain.log 2>&1 &
```
To FULLY stop the sweep, killing the top bash is NOT enough — `launch_draft_sweep.sh`
is a separate child that will advance to the next variant. Kill all of:
`pkill -9 -f 'g1a_v2_expansion_chain|launch_draft_sweep|rwkv7_train_draft|rwkv7_draft_watcher'`.
The chain runs cargo/parity gates → 7-model draft sweep (chunked, seeded, EPOCHS=1,
~20–30h) → spec hardening → competitive scorecard. The draft sweep launches
**unconditionally** (a Rust gate failure cannot block it — by design).

## State (2026-06-20)

- **G1a DONE**: RWKV-7 0.4B ternary-QAT hit ema 5.92 ≤ 6.0 at step 90; exported HF model,
  candidate ppl 3.449 vs fp32 base 3.356 (**+2.7%**, near-lossless).
- **Rename is now TRULY complete.** The prior session renamed code/CLI/env(`HAWKING_*`)/
  endpoints/metrics/`.hawking` sidecar/folder/GitHub repo, but the **folder rename left
  stale state** that this session found and fixed:
  - **44 runtime-breaking dead refs in `tools/` scripts** (`--backend dismantle`,
    `cargo build -p/--bin dismantle`, dead `/Downloads/dismantle` paths, `dismantle_*`
    Prometheus metric greps). Fixed: **78 precise substitutions across 50 files**; all
    `bash -n` / `py_compile` / JSON-valid.
  - **The RWKV-7 GPU parity test was a false-negative**, not a regression — see GOTCHA below.
- **Binary validated**: `cargo build --release` green; `hawking --help` / `version` OK.
- **RWKV-7 GPU↔CPU parity green**: 2/2, argmax mismatches=0, worst |Δlogit|=0.00015.
- **Docs consolidated 214 → 21** for a clean slate. Pruned working-logs are indexed in
  `docs/ARCHIVE_INDEX.md` and fully recoverable (`git checkout pre-hawking-rename -- <path>`).
- **Training is now ~16x faster** (merged to main). The trainer was batch=1 + per-example
  `empty_cache()`; now batched + RAM-hungry. The CURRENT sweep was restarted with the fast
  preset. Details: `docs/SPEED.md` + [[../SPEED.md]]. The draft sweep was RESTARTED fresh
  (old slow run archived under `artifacts/lowbit_rwkv7/_slow_batch1_archive/`).

## ⚠️ GOTCHA — folder rename leaves a stale Cargo build cache

Renaming the project folder moves `target/` and its fingerprints intact, so `cargo` thinks
everything is up-to-date and does **not** recompile. Any test that locates fixtures via
`env!("CARGO_MANIFEST_DIR")` / `file!()` then keeps the **old baked path** and false-fails
(e.g. `rwkv7_metal_parity` panicked reading `/Downloads/dismantle/.../fixtures/...`). Fix =
force a rebuild of the affected crate/test (`touch` the source or `cargo clean -p hawking-core`)
so the path re-bakes. After any future folder move, do a clean rebuild before trusting tests.

## Key locations (under ~/Downloads/hawking)

- Finished 0.4B model: `artifacts/lowbit_rwkv7/hawking_arc/ema6p0_step_000090_20260620_041225/`
  (`hf/` export, `report.md`, `ppl.jsonl`, `samples.jsonl`).
- Training tools: `tools/training/` (`g1a_v2_expansion_chain.sh`, `launch_draft_sweep.sh`,
  `rwkv7_train_draft.py`, `rwkv7_progress.py`, `hawking_after_ema.py`).
- Current chain reports: `docs/plans/{g1a_v2_expansion_results,rwkv7_competitive_scorecard,rwkv7_spec_hardening}_2026_06_20.md`.
- Throughput strategy: `docs/plans/bible_active.md` (+ `bible_archive.md`). Kill-ledger
  (do-not-respawn): `docs/dead_levers.md`.

## Git

- All this session's work is on `main`, remote = `github.com/joshuahickscorp/hawking.git`.
- Safety tag `pre-hawking-rename` = the last pre-rename commit; holds all pruned docs +
  the deleted feature branches' content.
- **Branches cleaned to a clean slate**: the remote now has **only `main`** (16 stale
  pre-rename feature/worktree-agent branches deleted). 6 LOCAL-only experiments were kept
  (unpushed unique work): `bench/dense-frontier`, `bench/ssm-moe`, `perf/dispatch-fusion`,
  `rwkv7/posttrain-opt`, `rwkv7/posttrain-prep`, `ssm/derisk` — review/delete at leisure.

## Open items

- [ ] Babysit the sweep to completion (~20–30h); winner ranked by accept-rate in
      `artifacts/lowbit_rwkv7/runs/custom_*/eval_log.jsonl`.
- [ ] Decide on the stale feature branches (inventory above).
- [ ] Re-run draft accept-rate vs the real 3B/7B target once that model is downloaded.
- [ ] (optional) Sweep remaining `dismantle` mentions in `crates/` **code comments** +
      dated `docs/strand` history — deliberately left (cosmetic; editing `.metal` comments
      churns the shader-hash that kernel-profile JSON validates against).

## Opening prompt (paste into a new chat opened in ~/Downloads/hawking)

> Resuming Hawking. Read `docs/plans/SESSION_HANDOFF_2026_06_19.md` first. G1a 0.4B QAT is
> done + exported; the dismantle→hawking rename is fully complete (code + folder + repo +
> tools + docs), build-green, parity green, docs consolidated. The 7-model draft-sweep
> chain is running detached — run "FIRST THING TO DO" to verify it's training (expect
> `[ep0 opt=N]` lines), then babysit it to completion.
