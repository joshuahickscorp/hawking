# Phase F.2 — DEFERRED → RELAUNCHED (full run cooking overnight)

**Date:** 2026-05-19 → 2026-05-20 00:05 (relaunch)
**Branch:** `claude/dreamy-golick-d54ff8`
**Status update 2026-05-20 00:05:** RAM cleared, stale ckpts removed
(`medusa_v1_retune_slope2` + empty `medusa_v1` stub), F.2 full run
relaunched as background task `bvyopbvq5` with `--head-weight-slope 2.0`.
Confirmed running at `[train] e0 s50 ... rows/s=102`. ETA 8–12 hr.
Tomorrow's session writes the final closeout (rename this file to
`phase_f2_done.md` or `phase_f2_negative.md` based on
`eagle4/checkpoints/medusa_v1/best_eval.json`).

**Earlier history (preserved for audit trail):**
Subset complete (1 epoch best at top1_mean=0.156); retune subset
killed; first full-run attempt killed by user (RAM pressure). All
those training processes confirmed terminated, no zombies.

## What ran

| Run | Status | Best result |
| --- | --- | --- |
| Subset (default hyperparams, 100k × 3 epochs) | Complete (46 min) | e1 best: top1_mean=0.156, head 0=60.1%, mean(head[1..7])=9.2% |
| Retune subset (slope=2.0, 100k × 3 epochs) | Killed mid-flight | partial ckpt (1.5 GB) — discard |
| Full run (slope=2.0, 491k × 10 epochs) | Killed pre-flight (RAM) | none |

## Subset eval table (canonical F.2 data point so far)

`eagle4/checkpoints/medusa_v1_subset/best_eval.json` (preserved on disk):

| i | top1 | top4 | top10 |
| --- | --- | --- | --- |
| 0 | 60.1% | 79.1% | 87.3% |
| 1 | 24.7% | 40.9% | 52.8% |
| 2 | 12.2% | 24.8% | 35.1% |
| 3 | 7.7% | 19.1% | 29.4% |
| 4 | 6.2% | 15.1% | 25.5% |
| 5 | 5.4% | 14.5% | 24.2% |
| 6 | 4.3% | 13.0% | 22.1% |
| 7 | 4.1% | 13.0% | 21.9% |

Gate against F.2 acceptance criteria (top1[0] ≥ 40% AND top1[K-1] ≥ 15%):
- top1[0] = 60.1% ✓ (well clear)
- top1[K-1] = 4.1% ✗ (need 15%)

The 100k subset cannot meet acceptance — too few rows for late heads to
learn. The full 491k × 10-epoch run is the necessary path forward.

## Pickup state for next session

**On disk (NOT committed):**
- `eagle4/checkpoints/medusa_v1_subset/` (3.1 GB — keep best.npz + best_eval.json,
  can delete epoch_*.npz if disk-pressured)
- `eagle4/checkpoints/medusa_v1_retune_slope2/` (1.5 GB partial — safe to `rm -rf`)
- `eagle4/checkpoints/medusa_v1/` (0 B empty dir — safe to `rmdir`)
- `training_data/medusa_v1_subset_train.log` (full subset training trace)
- `training_data/medusa_v1_retune_slope2_train.log` (partial retune trace)
- `reports/path_to_90/plans/methodology_distilled_post_f2.md` (NEW)
- `reports/path_to_90/plans/path_to_100_retool.md` (NEW)
- `reports/path_to_90/closeouts/phase_f2_deferred.md` (this file)

**Modified (NOT committed):**
- `reports/path_to_90/plans/phase_f_medusa.md` (+18 lines: prereq reading block)
- `reports/path_to_90/plans/phase_l5_chain_pipeline.md` (+11 lines: prereq reading)
- `reports/path_to_90/plans/phase_l7_kernel_rewrites.md` (+9 lines: prereq reading)

**Diagnostic edits intact (do not touch):**
- `crates/dismantle-core/src/engine.rs` (+10)
- `crates/dismantle-core/src/kernels/mod.rs` (+13)
- `crates/dismantle-core/src/model/deepseek_v2.rs` (+4)

## What unblocks F.2 ship

Run the full 10-epoch overnight training with RAM headroom. Command lives in
`reports/path_to_90/plans/phase_f2_setup.md` — add `--head-weight-slope 2.0`
based on subset findings (late heads at 4–12% top1 need accelerated weighting).

```bash
cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
nice -n 19 taskpolicy -b /Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python eagle4/medusa_head.py train \
  --parquet 'training_data/c2_hidden/eagle4_v0_medusa/shard_*.parquet' \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir eagle4/checkpoints/medusa_v1 \
  --K 8 --batch-size 128 --epochs 10 --lr 3e-4 \
  --heldout-shards 2 \
  --head-weight-slope 2.0 \
  > training_data/medusa_v1_train.log 2>&1 &
```

RAM footprint observed: ~2 GB in-RAM cache + adapter weights. Should be safe
with ≥4 GB free at launch.

## Followups when F.2 ships

- Single commit with this closeout (renamed to `phase_f2_done.md` or
  `phase_f2_negative.md`) PLUS the two new plan docs PLUS the three plan-doc
  edits. Inline Joshua Hicks identity, no trailers.
- Strip-restore the +27 diagnostic edits via `git stash push` / `pop` around
  the commit.
- F.3 Rust port gated on top1[K-1] ≥ 15%. See `phase_f_medusa.md` (now
  references `methodology_distilled_post_f2.md` patterns 11, 12, 16, 20).

## Methodology lessons from this aborted attempt

- Pattern 19 (cooperative scheduling) is necessary but not sufficient: RAM
  pressure from concurrent workloads can still block a launch even with
  `nice -n 19 taskpolicy -b`. Future overnight launches should check
  `vm_stat | grep free` for ≥4 GB before kicking off.
- Pattern 13 (smoke → subset → full): the subset gate correctly caught
  that 100k rows can't make late heads cross 15%. The full run is the
  ONLY path; retune subsets are diagnostic, not productive.
- Bash harness cwd is NOT persistent across calls. Every long-running
  background launch MUST `cd <worktree> &&` inline.
