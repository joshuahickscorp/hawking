# Phase F.2 — medusa training infrastructure — READY

**Date:** 2026-05-19
**Branch:** `claude/dreamy-golick-d54ff8`
**Script:** [eagle4/medusa_head.py](../../../eagle4/medusa_head.py)

Infrastructure built per the F.2 audit. Smoke-tested end-to-end (train + ckpt
save/load + eval). Ready for the next session to launch a real training run.

## Audit levers baked in

All ten levers from the F.2 thought-bubbles in
[reports/path_to_90/plans/phase_f_medusa.md](phase_f_medusa.md) are
implemented in [eagle4/medusa_head.py](../../../eagle4/medusa_head.py):

  Speed
    A. Tied LM head & output_norm to V2-Lite frozen weights — per-head
       adapter is hidden→hidden only (~8M params per head vs 213M naive).
    B. Batched K dispatch — stack adapter outputs into (K,B,hidden), one
       GEMM through tied lm_head.
    C. In-RAM hidden_high cache after first read.
    D. Column-projected dataloader (`pq.read_table(columns=[...])`).
    E. No V2-Lite forward — training script does not load mlx-lm at all.
       Runs concurrent with other GPU work (no clean window required).
    F. Single optimizer step over all K heads (one scalar loss).

  Quality
    A.1 MSE auxiliary on `hidden_high_+i` (β_mse=0.1 default).
    A.2 Per-head loss weighting w_k = 1 + slope·k/(K-1)  (slope=1 default).
    B.3 KL distillation from frozen lm_head at +i positions (α_kl=0.5 default).
    B.4 Per-head adapter is `RMSNorm → SwiGLU → residual gate(0.05)` —
        richer than `Linear→SiLU→Linear` per the plan skeleton, lighter
        than `_Block` (no attention; row-level training works without a
        windowed dataloader).

  Eval
    Reports per-head top-{1,4,10} on a held-out shard so F.5 can make
    informed K_inference choices on real branch-quality data, not just
    top-1.

## Smoke-test result

30 gradient steps × 2000 training rows × K=8, eval on 1000 held-out rows:

| i | top1 | top4 | top10 |
|---|------|------|-------|
| 0 | **53.9%** | 76.1% | 85.6% |
| 1 | 8.1% | 22.9% | 35.7% |
| 2 | 2.6% | 10.3% | 21.6% |
| 3 | 3.3% | 8.0% | 16.3% |
| 4 | 2.5% | 8.0% | 14.5% |
| 5 | 2.6% | 9.9% | 17.4% |
| 6 | 1.9% | 6.0% | 12.4% |
| 7 | 2.6% | 6.4% | 12.1% |

Head 0 already clears the plan's ≥40% target on essentially-untrained
weights. Late heads will improve with full training (10+ epochs over
500k rows). The top-{4,10} columns confirm the eval-lever rationale:
head 7 top10 ≈ 12% means tree-decode with B=10 lookahead picks up
head 7 ~12% of the time even when its top1 is weak — under-measured
by top1-only metrics.

## Recommended launch (next session)

**Subset-first iteration (warmup, ~30 min):**

```bash
cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
nice -n 19 taskpolicy -b /Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python eagle4/medusa_head.py train \
  --parquet 'training_data/c2_hidden/eagle4_v0_medusa/shard_*.parquet' \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir eagle4/checkpoints/medusa_v1_subset \
  --K 8 --batch-size 128 --epochs 3 --lr 3e-4 \
  --max-rows 100000 --heldout-shards 2 \
  > training_data/medusa_v1_subset_train.log 2>&1 &
```

GO/NO-GO decision after subset: if `top1_mean` after 3 epochs clears
≥20% mean across heads and head 0 clears ≥45% top1, kick off full run.

**Full run (8-12 hr est. — overnight or daytime, no clean window needed):**

```bash
nice -n 19 taskpolicy -b /Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python eagle4/medusa_head.py train \
  --parquet 'training_data/c2_hidden/eagle4_v0_medusa/shard_*.parquet' \
  --frozen eagle4/v2lite_frozen.npz \
  --ckpt-dir eagle4/checkpoints/medusa_v1 \
  --K 8 --batch-size 128 --epochs 10 --lr 3e-4 \
  --heldout-shards 2 \
  > training_data/medusa_v1_train.log 2>&1 &
```

Per-epoch heldout eval runs automatically; best ckpt at `best.npz`
when `top1_mean` improves. JSON summary at `best_eval.json`.

**Eval-only against a specific shard:**

```bash
/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python eagle4/medusa_head.py eval \
  --ckpt eagle4/checkpoints/medusa_v1/best.npz \
  --frozen eagle4/v2lite_frozen.npz \
  --parquet 'training_data/c2_hidden/eagle4_v0_medusa/shard_00060.parquet' \
  --K 8
```

## Tunable hyperparams (audit defaults shown)

| Flag | Default | Notes |
|---|---|---|
| `--K` | 8 | Capture-wide; can also train smaller K and ship narrower |
| `--batch-size` | 128 | Larger = better M3 Pro util; cap at memory |
| `--lr` | 3e-4 | AdamW default; bump to 1e-3 if first epoch barely moves |
| `--alpha-kl` | 0.5 | KL distillation strength. 0 disables. |
| `--beta-mse` | 0.1 | MSE aux strength. 0 disables. |
| `--head-weight-slope` | 1.0 | 0 = uniform, larger = more aggressive late-head weighting |
| `--max-rows` | None | Cap for subset iterations |
| `--heldout-shards` | 2 | Last N shards reserved for per-epoch eval |

## What NOT to do this session

- **Don't write the Rust port (F.3) yet.** F.3 requires F.2 to have
  produced a checkpoint with per-head top1 ≥40% / ≥15%. Wait for that
  decision point.
- **Don't re-run the Phase F.1 capture rewrite.** The medusa shards at
  `training_data/c2_hidden/eagle4_v0_medusa/` are correct and complete.
- **Don't pursue tree-decode (the failed Phase E).** F.5 hybrid tree-of-
  medusa is a separate later phase; F.2 is medusa alone.

## Acceptance criteria (re-stated from F.5 plan)

For F.2 ship:
- Per-head top1 ≥ 40% at i=0
- Per-head top1 ≥ 15% at i=K-1
- All existing parity gates still pass (none touched by this work)

If both top1 criteria met → F.3 (Rust port).
If head 0 OK but late heads fall short → try `--head-weight-slope 2.0`
or `--alpha-kl 1.0` before declaring F.2 dead.
