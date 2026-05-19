# Session handoff — 2026-06-03 (post-L7 + Phase E v2)

**Supersedes:** `HANDOFF.md` from 2026-05-19. That doc framed the
session that just ended; this doc frames the next one.

**Branch:** `claude/dreamy-golick-d54ff8` (97+ commits ahead of
origin; NOT pushed — user has not authorized push)
**Worktree:** `/Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8`

## What this session shipped

Six commits in commit-chain order:

| Commit  | Subject |
|---------|---------|
| c0fc428 | L7 Stage 0.5 — `gemm_q4_k_m_v3_xtg_sumy` kernel + parity (4 cases @ 1e-5 rel) |
| 8073a9e | L7.2 mixed-quant `moe_expert_pair_fused` kernel + parity (2 cases @ 1e-3 rel) |
| e5c5435 | L7 closeout v1 (bench queued) |
| f5788a4 | L7 Stage 0.5 bench — sumy is 5% slower than v3_xtg at LM head, **negative** |
| 332979b | L7.2 bench — fused is 3.7-3.8× slower than chained at every N tested, **negative** |
| 84e4a71 | Phase E plan v2 — pre-validation gate, L7-lessons-aware |

## Key results in one paragraph

L7 phase shipped **two parity-verified kernels that both lost their
benches**. The lesson: parity-only validation is insufficient for
kernel rewrites in this codebase. The matched-pair bench fixture
pattern (see `bench_moe_expert_pair_chained` vs
`bench_moe_expert_pair_fused` in [kernel_bench.rs](crates/dismantle-core/src/kernel_bench.rs))
is the correct template — write it FIRST, before any integration
code. The Phase E v2 plan adopts this discipline explicitly via the
new E.0 pre-validation milestone.

## What's NOT shipped

- No live wiring of `moe_expert_pair_fused` (it would not help —
  bench rules it out).
- No `gemm_q4_k_schedule_per_shape` flips for `v3_xtg_sumy` (bench
  rules it out at both shapes tested).
- No Phase E code yet — the v2 plan recommends E.0 first.
- No push to origin (user has not authorized).

## User diagnostic edits — preserved

`+27 lines / 3 files` at end of session, identical to entry state:

```
crates/dismantle-core/src/engine.rs            | 10 ++++++++++
crates/dismantle-core/src/kernels/mod.rs       | 13 +++++++++++++
crates/dismantle-core/src/model/deepseek_v2.rs |  4 ++++
```

Strip-restore was applied across the 6 commits. Verified clean via
`git diff --stat` against those 3 files.

## State at end-of-session

- **L8 monitoring session:** completed. iter4_k2_vector HALTed at
  0% K=2 chain accept (regression from step-400 mid-flight's 33.3%).
  Background pids 43980 and 52414 both exited. No new L8 iterations
  queued by this session.
- **Active profile:** unchanged. `deepseek-v2-lite-q4.m3pro18.json`
  still uses `gemm_q4_k_schedule = "v2t_gu_v2"` (the L7 kernel
  additions are opt-in and dormant).
- **shader_hash:** `1d71174fdbc56412996f9c19` — current,
  reflecting both new kernels (sumy + fused) compiled into the
  shader library.

## Next session — pick a phase

### Recommended: Phase E v2, milestone E.0.a

The cheapest informative experiment: extend the existing
mla_decode_kernel_fc_kbatch bench fixture from K=4 to K=8/16 and
check whether attention verifier cost scales flat. **~half a day,
~150 LoC.** If gates pass, graduate to E.0.b (chunked-K MoE). If
gates fail, write `phase_e0_negative.md` and pivot — this prevents
sinking 1-2 weeks into a phase that physics may not support.

Plan: [phase_e_tree_decode_v2.md](reports/path_to_90/plans/phase_e_tree_decode_v2.md)

### Alternative: revisit L8 training

L8 iter4 finished at 0% K=2 chain accept — this kills Phase L5 AND
Phase E.1 (head training) as currently planned. A new L8 iteration
that produces ≥20% K=4 chain accept unblocks both. The L8 monitoring
session would normally own this; check its closeout for any followup
the autoiter wrote when iter 4 HALTed.

### Alternative: Phase F (medusa multi-token head)

2-4 weeks, structurally independent of L8 chain accept. F.1 still
needs a clean overnight capture window (62 shards). High leverage,
high commitment.

## Hard rules to carry forward (unchanged)

- Commits via inline `git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com'`. NEVER `git config`.
- NEVER `Co-Authored-By: Claude` or `Generated with Claude Code` trailers.
- `cargo build --release -p dismantle-core` must be clean every commit.
- Strip-restore the user's +27 / 3 files diagnostic edits on every
  commit. Verified via `git diff --stat` end-state.
- For any new kernel rewrite: matched-pair bench fixture is part of
  the milestone's deliverable, NOT deferred. (Hard lesson from L7.)
- Bench under contention (Claude alive) is acceptable for RATIO data;
  use `clean_bench.sh` only when an absolute number is needed for an
  external comparison.

## Bench fixture cookbook

For the next session writing E.0.a, the template is:

1. In `crates/dismantle-core/src/kernel_bench.rs`, add a
   `bench_mla_kbatch` fixture mirroring `bench_q4k_v3_xtg`:
   - Allocate synthetic Metal buffers at the production MLA shape
   - Dispatch the kernel N times in a tight loop
   - Time each dispatch with `Instant::now()`
2. Add the kernel name to `ALL_KERNEL_NAMES` and the `run_kernel`
   match arm.
3. Make K configurable via env var (mirror
   `DISMANTLE_MOE_BENCH_N_ROUTES` pattern from this session).
4. Run with `nice -n 19 ./target/release/dismantle bench-kernel
   --kernel <name> --shape <shape> --iterations 200 --no-history`
   for ratio data.
5. Sweep K and check the gates.

Cost: ~half a day. Returns: a go/no-go for the entire +30-50 dec_tps
Phase E lever.

## File pointers

- **Plans:** [PHASES.md](reports/path_to_90/plans/PHASES.md) (index),
  [phase_e_tree_decode_v2.md](reports/path_to_90/plans/phase_e_tree_decode_v2.md) (this session's update),
  [phase_l5_chain_pipeline.md](reports/path_to_90/plans/phase_l5_chain_pipeline.md) (gated on L8),
  [phase_f_medusa.md](reports/path_to_90/plans/phase_f_medusa.md)
- **Closeouts:** [phase_l7_closeout.md](reports/path_to_90/closeouts/phase_l7_closeout.md) (full bench tables + verdicts)
- **Bench fixture template:** [kernel_bench.rs](crates/dismantle-core/src/kernel_bench.rs) (lines 183-617 are the L7 + L7.2 fixtures)

## The one-line version

L7 phase shipped, parity green, bench red. Phase E plan v2 reflects
the lessons. Start the next session at Phase E milestone E.0.a — the
cheapest gate that decides whether Phase E is worth the next 1-2
weeks.
