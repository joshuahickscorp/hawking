# Step 2A setup — eagle4 K=1 tax allocation harness

**Session:** N+1 of the [path_to_100_repath.md](../plans/path_to_100_repath.md) dual-track plan.
**Status:** SETUP COMPLETE — measurement deferred to user post-Cmd-Q.
**Goal:** Allocate the 8.9 dec_tps eagle4-K=1 tax (off=26.87 → eagle4-K=1=18.01, per [_bench_20260520T143008/summary.txt](../_bench_20260520T143008/summary.txt)) across four candidates.

This session was scoped to **diagnosis prep only** — no kernel implementation, no wiring, no destructive edits. The user (or a later session) runs the harness in a clean window and pastes the medians back for allocation analysis.

## What's been built

### 1. Per-phase timing instrumentation — `deepseek_v2.rs`

Added to the K=1 loop in `forward_token`'s `SpeculateMode::Eagle4` branch (lines ~1812-1953, the `chain_k < 2` path). Four `Instant::now()` markers wrap the four candidates:

| Marker | Phase | Candidate |
|---|---|---|
| `t_step_a0` → `t_after_capture` | `forward_token_argmax` with `eagle4_capture_active=true` + capture-buffer drain | **(a) capture forward** |
| `t_after_capture` → `t_after_hshared` | `h_shared_norm2` test + either GPU read or `cpu_shared_expert_forward(26, ...)` fallback | **(b) h_shared compute** (`hshared_fallback` flag) |
| `t_after_hshared` → `t_after_head` | `head.forward_full_{amx,metal,cpu}_no_lm_head` | **(c) head propose** |
| `t_after_head` → `t_after_argmax` | `gemv_f16_argmax_dispatch` | **(d) head argmax** |

Gated entirely on `DISMANTLE_SPEC_LOG=1`. When set, a new log line is emitted per token:

```
[spec/eagle4-step] step=N backend=amx|metal|cpu capture_us=… hshared_us=… hshared_fallback=true|false head_us=… argmax_us=… total_us=…
```

The existing `[spec/eagle4]` line (draft/v2/calib) is preserved; the new line is additive. `Instant::now()` calls are unconditional (sub-10 ns each), so the breakdown is bit-faithful to the un-gated K=1 timing.

### 2. Harness script — `tools/bench/path_to_100_step2a.sh`

Two-phase clean-window bench, executable, refuses if `pgrep -i Claude` returns alive (exit 2). Mirrors the `path_to_125_bench.sh` conventions (nice -n 19, identical model/profile/draft head paths, summary.txt + raw.jsonl output).

**Phase 1** — dec_tps median collection. 3 prompts × 3 trials × 4 configs = 36 trials total:

| # | Config | EAGLE4_BACKEND env | Head forward dispatch |
|---|---|---|---|
| 1 | off / seq / K=1 | n/a | n/a (control) |
| 2 | eagle4 / seq / K=1 | unset (default) | `forward_full_amx_no_lm_head` |
| 3 | eagle4 / seq / K=1 | `=metal` | `forward_full_metal_no_lm_head_on` |
| 4 | eagle4 / seq / K=1 | `=cpu` | `forward_full_no_lm_head` (CPU gemv_f32) |

**Phase 2** — per-step `[spec/eagle4-step]` capture. One prompt × 32 tokens × 3 eagle4 configs. Lines are parsed into `step_breakdown.csv` and per-backend µs medians are printed (skipping step=0 to drop cold-cache outliers).

Outputs land in `reports/path_to_90/_bench_step2a_<TS>/{summary.txt, raw.jsonl, spec_log_eagle4-{amx,metal,cpu}.txt, step_breakdown.csv}`.

## Allocation matrix (post-measurement)

Read this once the user pastes back the numbers. Each row maps an observation pattern to which sub-step owns the K=1 tax — and which lever from the path-to-100 backlog becomes the implementation target.

| Phase-1 / Phase-2 observation | Implicated sub-step | Recommended next lever | Expected K=1 recoverable |
|---|---|---|---|
| `capture_us` >> all other phases on all backends. AMX/Metal/CPU all hover near 18 tps. | **(a) capture forward** — per-layer-commit TCB drain costs the comment's "~4 ms" toll | Ship `eagle4_stats_off` flag (~30 LoC) that bypasses `eagle4_capture_active` when stats aren't needed. K=1 with stats-off becomes ≡ off-mode. | +5-9 tps (entire tax) |
| `hshared_fallback=true` on most steps AND `hshared_us` is the dominant phase | **(b) h_shared GPU capture** is broken; CPU fallback `cpu_shared_expert_forward(26, ...)` owns the cost | Fix the moe_shared_out_buf read at layer 26 (check whether dense-layer routing actually emits to the captured slot); GPU h_shared should be free | +3-7 tps |
| `head_us` very different across backends (metal >> amx; cpu in middle) | **(c) head propose** dominates; Metal dispatch is the slow path. AMX is fine in production. | No K=1 win available without changing the default; AMX is already chosen on macOS. Move to Step 2B (chain-K=4) where Metal head matters more — L5 Lever B (chain-step pipelining) becomes relevant. | ~0 for production (AMX); Metal path becomes Step 2B target |
| `head_us` large in all three backends (AMX ~equal to metal/cpu within 30%) | **(c) head propose** dominates structurally; head's compute density is the wall | Head distillation / smaller head architecture. Architectural — not a kernel problem. Path-to-100 K=1 dead. | ~0 |
| `argmax_us` >> all other phases | **(d) head argmax** dispatch (gemv_f16_argmax_dispatch) is the cost — separate Metal commit + sync | **L5 Lever A** wiring: the dormant `eagle4_rmsnorm_residual_gate` kernel already covers this (commit [8578d7e](../../../../../crates/dismantle-core/src/kernels/mod.rs)). Becomes the wiring target. | +3-6 tps |
| All four phases roughly equal (within 30%) | **architectural wall**: K=1's 8.9-tps tax is spread across capture + h_shared + head + argmax with no single fat phase | No K=1 win available; document and move directly to Step 2B (chain-K=4 acceptance diagnosis) without expecting K=1 recovery | 0; path-to-100 K=1 closed; Step 2B is the only remaining knob |

## What success unlocks

Per [path_to_100_repath.md §Sequencing recommendation](../plans/path_to_100_repath.md):

- **Tax recoverable** (rows 1, 2, 5) → ship the targeted fix, K=1 should land near 27 tps, then proceed to Step 2B (chain-K=4) with a clean K=1 baseline.
- **Tax architectural** (rows 3, 4, 6) → write `phase_step2a_negative.md` documenting the dead lever, skip to Step 2B directly. Chain-K=4 acceptance becomes the only remaining path to multiplier > 1.

## Out of scope this session

- Kernel implementation (L7.D inner-block, L5 Lever A wiring, any new Metal shader)
- Modifying `eagle4_head.rs`'s forward path beyond reading
- Step 2B (chain-K=4 acceptance distribution) — that's session N+2
- Any change requiring stash-pop / strip-restore of the existing diagnostic edits on `engine.rs` / `kernels/mod.rs` / `deepseek_v2.rs`

## How to run

```bash
# 1. Cmd-Q Claude.app  (the script refuses while pgrep finds Claude alive)
# 2.
cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
./tools/bench/path_to_100_step2a.sh

# Expected wall-clock: ~6-8 min (36 trials × ~4 s/trial + 3 spec_log captures × ~10 s)
# Output: reports/path_to_90/_bench_step2a_<TS>/summary.txt
```

Paste the `=== Phase 1 RESULT ===` table and the `=== Phase 2 RESULT ===` per-phase µs medians back into the session that triages this. Then walk the allocation matrix above.

## Verification this session

| Check | Result |
|---|---|
| `cargo build --release --workspace` | clean (pre-existing warnings only) |
| `cargo test --workspace --lib` | 59 lib tests pass (5 + 45 + 9 across crates) |
| `bash -n tools/bench/path_to_100_step2a.sh` | syntax OK |
| `tools/bench/path_to_100_step2a.sh --help` | exits 0, prints config table |
| Claude-running refusal | exits 2 with explanatory error |
| Diagnostic edits intact | `git diff --stat` confirms +10/+13/+4 unchanged (instrumentation block is additive to deepseek_v2.rs and lives in the K=1 loop, not the test-helper region at line 2055) |
