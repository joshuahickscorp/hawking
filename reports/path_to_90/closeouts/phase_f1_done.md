# Phase F.1 — capture pipeline — DONE

**Date:** 2026-05-19
**Branch:** `claude/dreamy-golick-d54ff8`
**Commit anchor:** see Phase F.1 commit
**Verdict:** **SHIPPED.** Output at `training_data/c2_hidden/eagle4_v0_medusa/`.

## Result in one paragraph

Phase F.1 shipped in 39 seconds via window-join rewrite of the
existing 62 eagle4_v0 shards, instead of the 12-hr V2-Lite re-capture
the plan budgeted. The existing shards already contain
`(sample_id, position, next_token, hidden_high)` at every position;
the K-position-ahead medusa targets for row `(S, p)` are just the
`next_token` and `hidden_high` of row `(S, p+i)` within the same
sample. A streaming pass over the 62 shards (sorted by sample_id by
construction of [eagle4/capture.py](../../../eagle4/capture.py))
materializes the new columns with `-1` / zero-bytes sentinels for
tail rows whose `+i` sibling falls past end-of-sample.

## Output

| Property | Value |
|---|---|
| Path | `training_data/c2_hidden/eagle4_v0_medusa/` |
| Shards | 62 |
| Rows | 500,000 (matches input — no drops, tail rows kept with sentinels) |
| Disk | 21 GB (vs 8.6 GB input; +12 GB for 7×hidden_high_+i columns) |
| Schema | 10 base + 7 `next_token_p{1..7}` + 7 `hidden_high_p{1..7}` = 24 fields |
| K (wide-capture) | 8 — see "capture wide, ship narrow" below |

Tail-sentinel counts per `i`: p1=1963, p2=3926, p3=5889, p4=7852,
p5=9815, p6=11778, p7=13741 — perfectly linear, consistent with
~1963 distinct samples × 255 positions each (`max_ctx=256`).

## Correctness verification

Verified on `sample_0` (255 rows) in output shard_00000:

- `next_token_p{1..7}` at row 0 bit-equal to `next_token` at rows 1..7. ✓
- `hidden_high_p1` at row 0 byte-equal to `hidden_high` at row 1. ✓
- Last row of sample (pos=254): all `next_token_p{i}` = `-1`,
  `hidden_high_p1` all zeros. ✓
- Total rows in == rows out (500,000 == 500,000). ✓

## Capture wide, ship narrow

K=8 was chosen for capture as pure optionality: the wide-capture cost
is just extra parquet columns (+12 GB disk, +30 sec rewrite time, no
GPU). F.2 trains 8 heads and evaluates per-head top1; F.5 picks
`K_inference ∈ {4,5,6,7,8}` empirically based on which heads earn
their slot. We are NOT committed to K=8 at inference. The information-
theoretic ceiling on head accuracy decays monotonically with i —
late heads will be weak — but in tree-decode head 7 only fires when
heads 0-6 also accept, so weak late heads contribute multiplicatively
and don't hurt strong early heads.

## Why this happened — what the plan doc missed

[phase_f_medusa.md § F.1](../plans/phase_f_medusa.md) prescribed a
12-hr overnight re-capture on a clean window. The implicit assumption
was that capturing `+i` targets required running V2-Lite forward at
each `+i` position to obtain the targets. That assumption is wrong:
the `+i` token (and hidden state) at position `p` IS the existing
captured value at position `p+i` for the same sample. The existing
shards weren't missing data — they just weren't materialized in the
shape F.2 wants.

The audit caught this before launching the overnight run. Wall-time
saved: ~12 hr. Pattern: when a phase requires "richer training data",
the first question is "is this richer data derivable from existing
data via reshape or join, or does it genuinely require new
observations?" Re-capture is the rare answer.

## What's NOT addressed

- **Cross-shard ordering**: handled by streaming per-sample (rows
  within a shard are in (sample_id, position) order by construction;
  samples can span shards, state carries across).
- **Tail rows**: kept with `-1` / zero-bytes sentinel rather than
  dropped, so heads 0..K-1 can train on tail rows up to their own
  available depth. F.2 dataloader uses `next_token_p{i} != -1` as the
  per-head loss mask.
- **Other auxiliary columns** (e.g. shared_hidden_+i, hidden_low_+i):
  not materialized. The `hidden_high_+i` columns cover the MSE
  auxiliary the plan called out; F.2 can request more via the same
  window-join trick if needed (~30 sec per added column-set).

## Next-action recommendation

Per [phase_f_medusa.md § F.2](../plans/phase_f_medusa.md):
implement `eagle4/medusa_head.py` (K=8 independent prediction heads
on shared frozen-V2-Lite representation, K cross-entropy losses).
Train on the new shards, evaluate per-head top1. Decision point: per
the plan, ship F.3 (Rust port) only if per-head top1 looks
reasonable on a small training run.

## Artifacts

- `eagle4/rewrite_medusa.py` (~135 LoC) — reusable, idempotent rewrite
  script. Streaming per-sample with `-1` / zero-bytes sentinel for
  tail rows. Defaults to K=8.
- `training_data/c2_hidden/eagle4_v0_medusa/` — 62 shards, 21 GB,
  500k rows.
