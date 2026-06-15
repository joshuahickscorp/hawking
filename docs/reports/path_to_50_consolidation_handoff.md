# path-to-50 consolidation — handoff snapshot

**Date:** 2026-05-22
**Purpose:** Hand state to a second chat that will pick up perf gating once
the 24h compute slot opens.

## TL;DR

Three Rust workstreams (A / B / C) launched in parallel. **B is consolidated
in main**; **A and C still running in their own worktrees** at handoff time.
**All compute deferred to the overnight pipeline** via two new scripts.

## What's in main right now (architectural code, no commits made)

### Session B — spec-decode runtime cost reduction (DONE)

Surgical edits in `crates/dismantle-core/src/model/deepseek_v2.rs`:

| Line  | Change |
|-------|--------|
| ~877  | `DecodeArena::new` `max_batch_size`: 8 → 17 (K=16 verify-window now hits single-TCB fast path) |
| ~1124 | Replaced `ExactShared` serial verify loop with single `forward_tokens_batched(&batch_tokens, &batch_positions)` call — same structure as the existing `NGram` path |
| ~2560 | Marked the now-unused `forward_token_argmax` with `#[allow(dead_code)]` |

**Cost driver eliminated:** ExactShared verify was doing K+1 separate
`tcb.commit_and_wait()` round-trips (~150–200 µs each on M3 Pro). That alone
explained the 50–60 % regression at 47 % acceptance rate.

**Writeup:** `reports/spec_decode_runtime_cost_2026_05_22.md`

**Projected gain:** +8–15 dec_tps once spec-decode flips to net-positive at K=4.

## What's pending (running in isolated worktrees)

### Session A — mixed-precision wedge (RUNNING)

- Worktree: `.claude/worktrees/agent-a13b89bf62bacca0e`
- Branch: `worktree-agent-a13b89bf62bacca0e`
- Expected files: edits to `crates/dismantle-core/src/model/deepseek_v2.rs`
  (expert weight loader swap), a new
  `crates/dismantle-core/tests/mixed_precision_parity.rs`, and possibly
  `crates/dismantle-core/src/engine.rs` plumbing.
- Projected gain: +3–5 dec_tps
- When the agent finishes, its worktree behaves like B's did (changes
  auto-merge into main's working tree). If that doesn't happen, the diff
  is recoverable via `git diff worktree-agent-a13b89bf62bacca0e`.

### Session C — Q8 KV production wiring (RUNNING)

- Worktree: `.claude/worktrees/agent-a87a64ce0ab4022d7`
- Branch: `worktree-agent-a87a64ce0ab4022d7`
- Expected files: `crates/dismantle-core/src/cache/mod.rs`,
  `crates/dismantle-core/src/engine.rs` (adds `q8_kv: bool` + `--q8-kv` CLI
  flag), `crates/dismantle-core/src/model/deepseek_v2.rs`,
  `crates/dismantle-core/src/attn/mod.rs` (route MLA decode through
  `mla_decode_q8kv`), and the existing
  `crates/dismantle-core/tests/q8_kv_parity.rs`.
- Projected gain: +2–5 dec_tps

## What's wired for the 24h pipeline (architectural — landed in main)

Two new files, both opt-in, both refuse to run with Claude live:

### 1. `tools/bench/path_to_50_matrix.sh`

Stand-alone bench matrix. Runs every lever combination and writes a Markdown
delta report.

```
L0     baseline (B always-on, no other flags)
L1     --vocab-prune-path
L2     --quant-tier-map-path                          (Session A's flag)
L3     --q8-kv                                        (Session C's flag)
L4     --speculate exact-shared --verify-window 4     (validates B's fix)
STACK  L1 + L2 + L3 together
```

Picks the most recent `artifacts/vocab_prune/*.json` and
`artifacts/calibration/tier_maps/*.json` automatically; envs let you pin
specific files. Skips a lever cleanly if its input file is missing.

**Output:** `artifacts/runs/path_to_50_matrix/<utc>/report.md` plus a
`latest` symlink.

### 2. `tools/training/overnight_path_to_50_bench.sh`

Wraps the matrix in the existing overnight-pipeline status-JSON pattern.
Pre-flight gates:

1. Reads `artifacts/runs/overnight/extended_status.json`; refuses unless
   training shows `current_stage == complete && state == done`.
2. Refuses if Claude.app is live.
3. Runs `cargo build --release`.
4. Runs parity tests:
   - `v1_1_phase4D_spec_exact_mode` (B)
   - `integration_greedy_64` (always-on sanity)
   - `mixed_precision_parity` (only if test file exists — auto-detected for A)
   - `q8_kv_parity` (only if test file exists — auto-detected for C)
5. Runs the bench matrix.
6. Writes heartbeats to `artifacts/runs/overnight/path_to_50_bench_status.json`
   so the cron monitor can report progress.

## Compute status at handoff

- Eagle5 v3 training: **complete** as of 2026-05-22T21:02 UTC (per
  `extended_status.json`).
- Eagle5 v4 training: status TBD (check the same file).
- Free disk reported `0 GB` — needs human verification (`df -g .`); could be
  a stat glitch or genuine pressure.

## Build + microbench sanity check (this session, Claude live)

Rebuilt with B's edits: `cargo build --release -p dismantle` → 39.73s, clean.

Microbench at MoE-down production shape (1408x2048, 200 iters, no history append):

| Kernel | mean | p50 | p99 |
|---|---:|---:|---:|
| `gemv_q4_k_m_v2_pinned_tcb` | 156.7 µs | 138.0 µs | 602.5 µs |
| `gemv_q3_k_pinned_tcb` | 173.2 µs | 145.2 µs | 1026.8 µs |
| `gemv_f16_metal_pinned` | 317.3 µs | 271.5 µs | 1298.5 µs |

Numbers are illustrative only (Claude was running during this bench) but
confirm the new binary dispatches Metal kernels successfully. The
authoritative numbers come from `overnight_path_to_50_bench.sh` (Claude
quit, strict mode).

## Recommended next actions for the second chat

1. Verify A and C have landed their diffs (either auto-merged into main, or
   pull from their worktrees). Run `git status` to confirm new files exist.
2. Confirm free disk is healthy (`df -g .`).
3. Quit Claude.app and any CLI Claude sessions.
4. Run `bash tools/training/overnight_path_to_50_bench.sh`.
5. Read `artifacts/runs/path_to_50_matrix/latest/report.md`.

## Constraints to honor (carried from session prompts)

- **No Claude git attribution** anywhere. User's hard rule.
- **No autonomous commits.** All landed diffs are uncommitted in their
  worktrees / in main's working tree. The user reviews and commits.
- **All new levers are opt-in via flags.** Default behavior is unchanged
  except for Session B's always-on TCB-batched verify (which is part of
  the spec-decode path that itself requires `--speculate`).

## Projected gains if all three land

| Lever stack            | Projected dec_tps |
|------------------------|-------------------|
| Baseline (post-B)      | ~24–27            |
| + L1 vocab-prune       | ~26–29            |
| + L2 mixed-precision   | ~28–32            |
| + L3 q8-kv             | ~30–35            |
| + L4 spec-decode K=4   | **~35–45** (B's fix flips spec-decode net-positive) |

Stretch target after eagle5-head deployment (separate session): ~50–55.
