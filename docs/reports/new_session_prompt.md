# New session prompt — path-to-75, picking up from 2026-05-23 (post 6h chain)

Open a fresh Claude Code session at `/Users/scammermike/Downloads/dismantle`.
Copy everything between `--- BEGIN PROMPT ---` and `--- END PROMPT ---`.

--- BEGIN PROMPT ---

You are picking up the dismantle path-to-75 effort. An autonomous 6h
chain was launched at the end of the previous session and should be
done (or near done) by the time you read this. Your first job is to
review what it produced, NOT to launch new work.

## First 10 minutes — read these in this order

1. `artifacts/runs/overnight_6h/status.json` — is the chain still
   running? If yes, `ps -p $(cat artifacts/runs/overnight_6h/pid)` and
   `tail -50 artifacts/runs/overnight_6h/chain.log` before doing
   anything else.
2. `artifacts/runs/overnight_6h/WRAP.md` — synthesis of all 6 modules.
3. `artifacts/runs/overnight_6h/M5_high_conf_stack.md` — the
   release-grade TRIALS=20 × 3 prompts × 4 configs matrix. This is
   the most important deliverable; it tells you the deployable best.
4. `artifacts/runs/overnight_6h/M4_autotune.md` — autotune sweep over
   4 schedule fields. If `M4_candidate_profile.json` exists AND paired
   validation showed positive delta, adopt it by
   `cp artifacts/runs/overnight_6h/M4_candidate_profile.json profiles/deepseek-v2-lite-q4.m3pro18.json`.
5. `artifacts/runs/overnight_6h/M3_kernel_hot_spots.md` — top kernels
   by GPU time (re-run captured with `DISMANTLE_TCB_TRACE=gpu`, the
   first attempt's `--trace-dispatch`-only path produced empty
   gpu_us). These are the kernel-sketch targets for the next 2-4 weeks.
6. `artifacts/runs/overnight_6h/M2_q8_kv_3way.md` — patch apply
   result. The patch is known to be structurally divergent from main
   on 17 files; expect this module to confirm Q8 KV remains UNWIRED
   and that the next step is hunk-by-hunk port (NOT another patch
   apply attempt).
7. `artifacts/runs/overnight_6h/M1_commit_plan.md` — commit dry-run.
   Tells you exactly which files would land in the first safe commit.

## Then read the standing context

1. `reports/session_wrap_2026-05-23.md` — 149-file audit, 6-commit
   sequence with **non-attributed** messages
2. `reports/path_to_75_v2.md` — 5 levers ranked, ~46 tps gap to 75
3. `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/`
   especially `path_to_50_complete.md`, `per_kernel_time_breakdown.md`,
   `q8_kv_production.md`, `feedback_kernel_parity_gate.md`,
   `bench_contamination.md`, `feedback_bench_with_claude_open.md`
4. `reports/all_parallel_session_prompts.md` — Session C-completion
   prompt is your reference for the Q8 KV hunk port
5. `reports/dead_levers.md` — Sessions D (MLA Phase 4) and I (CPU+GPU
   pipelining) are confirmed DEAD; don't relitigate

## State of the world (verified end of 2026-05-23 session)

- **Production deployable today:** L1 vocab-prune alone (+1.55 dec_tps,
  σ=0.12, n=30). Confirm against M5 numbers when the chain finishes.
- **Working tree:** 149 files modified. Safe commit sequence is
  pre-written in `session_wrap_2026-05-23.md`. The chain's M1 runs in
  DRY-RUN by default — re-run with `CHAIN_AUTO_COMMIT=1` to land the
  first safe commit (new modules + tests).
- **Q8 KV (Session C):** `reports/patches/session_C_q8_kv_wiring.patch`
  has been confirmed structurally divergent from current main on 17
  files. Next step is HUMAN-DRIVEN hunk port, not another patch retry.
  See M2 output for the file list.
- **MoE GEMM:** 50.5% of decode time. Single biggest lever. M3's hot-
  spot list (with real GPU timings this time) tells you which shapes
  to attack. Only J w2 has shipped (+1.33% single shape, env-gated).
- **Spec-decode:** K's revert holds the line at K=4. Exact-shared at
  K≥8 still net-negative. N-gram drafts add nothing on top of L1.
- **Dead levers:** Session D (MLA Phase 4), Session I (CPU+GPU
  pipelining). See `reports/dead_levers.md`.

## Hard rules (user's globals — DO NOT VIOLATE)

- **No Claude git attribution** anywhere. No `Co-Authored-By`, no
  "Generated with Claude" footers. Commit messages look exactly like
  a human wrote them.
- **No autonomous commits** without explicit user OK or `CHAIN_AUTO_COMMIT=1`.
- **Bench discipline:** paired deltas Claude-open OK
  (`feedback_bench_with_claude_open.md`); absolute numbers need
  Claude quit (`bench_contamination.md`).
- **TRIALS≥15** for any decision-grade bench. TRIALS=20-30 for release.
- **Max 2 concurrent cargo builds** on M3 Pro 18 GB.
- **Don't add a Pages-free-based RAM watcher** — macOS uses RAM for
  caching, that signal is always low and produces false-positive
  pauses. The 6h chain learned this the hard way.

## Chain controls (in case the chain is still running)

- Status:  `cat artifacts/runs/overnight_6h/status.json`
- Log:     `tail -f artifacts/runs/overnight_6h/chain.log`
- Pause:   `bash tools/bench/pause_bench.sh`
- Resume:  `bash tools/bench/resume_bench.sh`
- Stop:    `kill $(cat artifacts/runs/overnight_6h/pid)`

If the chain is still running and you want fresh numbers without
contention, pause it before running your own benches.

## Suggested first 2 hours of session

1. **Read WRAP.md + the 5 module outputs** (15 min)
2. **Adopt M4 candidate profile** if positive (1 min) — `cp` the
   candidate over the live profile, rebuild, verify with 5-trial
   paired bench.
3. **With user approval, land commit 1** by setting
   `CHAIN_AUTO_COMMIT=1` and re-running M1 (or just doing the commit
   manually — the staged file list is in the dry-run output).
4. **Plan Session C-completion** by reading
   `reports/all_parallel_session_prompts.md` and the M2 file list.
   The patch's q8_kv plumbing exists; it just doesn't merge against
   current main. Hunk-port it.
5. **Pick a kernel from M3's top-3** and design a sketch variant.
   This is multi-day work, not one-session.

## Long-term roadmap (multi-week, NOT autonomous)

Per `reports/path_to_75_v2.md` ranking:

| Tier | Lever | Time | Gain |
|---|---|---|---|
| 1 | Commit safe sequence | minutes | — (hygiene) |
| 1 | Q8 KV hunk-port (Session C-completion) | 3-5 days | +1-5 tps |
| 2 | MoE GEMM kernel sketch (top-3 shapes from M3) | 1-2 weeks each | +3-8 tps |
| 2 | RMSNorm fusion remaining 5 sites | 2-3 weeks | +3-7 tps |
| 3 | Spec-decode profile-driven fix | 1-2 weeks | +5-15 tps (gated) |
| 3 | Smaller draft head | 1-2 weeks | +5-10 tps (gated on spec-decode) |

**75 tps = ~6-10 weeks of focused supervised work** on top of today's
~28.7 deployable floor. There is no shortcut.

## What to NOT do

- Don't restart eagle5 training (failed 4× in prior session; corpus
  gone; dict-unwrap + shape-mismatch + silent OOM all unresolved)
- Don't launch >2 parallel worktree agents (OOM-killed prior runs)
- Don't trust TRIALS=3 results (M1 baseline ranged 21.89-24.89)
- Don't believe "patch applied" without verifying the artifact (CLI
  flag present, kernel registered, etc.) — the M2 output proves this
- Don't pursue Session D (MLA Phase 4) or Session I (CPU+GPU
  pipelining) — both measured DEAD
- Don't re-run the patch apply for Q8 KV expecting different results;
  17 files diverge structurally. Port hunks.

--- END PROMPT ---
