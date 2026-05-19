# Path-to-150 phase work — handoff to a fresh session

**This session** (the one that broke the 7% chain-accept ceiling) is
busy monitoring L8 iter 4/5/6 training + auto-iter sequencer. It
should NOT do kernel/code work in parallel because:
- iter 4-6 training is GPU/CPU-contention sensitive
- The autoiter is sequencing 2-3 hr of training work
- New kernel parity tests would compete for the GPU

**A fresh session** should pick up the phase work (L5 → L7 → E → F)
in parallel, on a different branch or worktree, with no GPU work
that fights the training.

This doc is the prompt to paste into that fresh session.

---

## Copy below this line as the first message of a new session

```
=== PROMPT BEGIN ===
```

You are picking up `dismantle` path-to-150 phase work on branch
`claude/dreamy-golick-d54ff8`. The session that broke the 7% chain-
accept ceiling is still running (monitoring L8 iter 4→5→6 training).
Your job is the post-L8 engineering phases. **Stay out of GPU-
contention activities** until the user confirms training is done.

## State at session start

- HEAD: `448c87e` (phase plans landed)
- Branch: `claude/dreamy-golick-d54ff8`
- Worktree: `/Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8`
- Background processes you do NOT touch:
  - pid 43980 (iter 4 training — finishing soon)
  - pid 52414 (autoiter watch_chain — will auto-launch iter 5 then iter 6)
  - cron `a01ceb3d` (20-min status pings)
- User diagnostic edits in `engine.rs` / `kernels/mod.rs` /
  `deepseek_v2.rs` MUST be preserved (+27 lines / 3 files,
  identical to entry). Use strip-restore on every commit you make.

## Read these BEFORE doing anything

1. `reports/path_to_90/plans/PHASES.md` — phase dependency graph
2. `reports/path_to_90/plans/acceleration_patterns.md` — 10 patterns
   from L8's troubleshoot loop. **Apply the pre-launch checklist
   before each phase.**
3. The plan doc for the phase you're working on (see § Pick a phase
   below).

## Pick a phase — by dependency / ROI

Phases are independent once their prerequisites are met. Recommended
order (highest ROI per hour first):

### Option A — Phase L5 (start here for fastest win)

- Plan: `reports/path_to_90/plans/phase_l5_chain_pipeline.md`
- Estimate: 3-6 hr code, +3-8 dec_tps
- Why first: infrastructure (SharedEvent, TCB-on-secondary) is
  already shipped from commits `8c51a02` and `e482463`. The work
  is purely a chain-decode loop restructure in
  `crates/dismantle-core/src/model/deepseek_v2.rs` around line 1611.
- Prerequisite: iter 5 must confirm K=4 chain accept ≥ 25% first.
  Check `tools/l8_autoiter.sh status` before starting; if iter 5's
  K=4 accept is < 25%, hold L5 until iter 6 (with chain-reg) lands.

### Option B — Phase L7 (kernel rewrites)

- Plan: `reports/path_to_90/plans/phase_l7_kernel_rewrites.md`
- Estimate: 2-3 days code, +15-30 dec_tps
- Why: 3 new shaders with high blast radius (MoE expert pipeline +
  LM head Q4_K_M). The single biggest dec_tps lever available.
- Prerequisite: NONE. Can start immediately.
- Risk: GPU-contention work. **Wait until L8 training finishes**
  (autoiter will write final result to
  `reports/path_to_90/_levers/l8_iter_results.jsonl`).

### Option C — Phase E (tree decode, longest)

- Plan: `reports/path_to_90/plans/phase_e_tree_decode.md`
- Estimate: 1-2 weeks code, +30-50 dec_tps
- Why: research-engineering hybrid; payoff is large but speculative.
- Prerequisite: NONE. Tree decode reduces to chain decode at B=1
  (parity gate). Can start in parallel with L7.

### Option D — Phase F (medusa multi-token head)

- Plan: `reports/path_to_90/plans/phase_f_medusa.md`
- Estimate: 2-4 weeks code, +20-40 dec_tps
- Why: structural alternative to chain decode entirely.
- Prerequisite: capture pipeline rewrite (F.1) requires clean
  overnight window to re-capture all 62 shards. Don't start F.2-F.5
  until F.1 is done.

## Hard rules (from project CLAUDE.md)

- Commit as `Joshua Hicks <joshuahicksboba@gmail.com>` via inline
  `git -c user.name=... user.email=...`. Never `git config`. Never
  Co-Authored-By or Generated-With trailers.
- Bit-identical greedy parity gate must pass at every commit:
  ```
  EAGLE4_PARITY_TEST=1 DISMANTLE_EAGLE4_GREEDY_TOKENS=16 cargo test \
    --release -p dismantle-core --test eagle4_decode_parity -- --ignored
  ```
- shader_hash regen on any Metal shader change. Use
  `cargo run --release -p dismantle-core --example print_shader_hash`
  to print the new hash; update `profiles/deepseek-v2-lite-q4.m3pro18.json`
  in the same commit.
- All commits must run `cargo build --release -p dismantle-core` clean.
- User's diagnostic edits across `engine.rs`/`kernels/mod.rs`/
  `deepseek_v2.rs` must remain +27 lines / 3 files at the end of
  every commit. Use the strip-restore pattern (see
  `acceleration_patterns.md` Pattern 8).

## Coordination with the L8 monitoring session

While your phase work runs:

- DO NOT modify `tools/eagle4_v4_fromscratch.sh`,
  `tools/eagle4_iter5_k4_vector.sh`,
  `tools/eagle4_iter6_k4_chainreg.sh`,
  `tools/l8_autoiter.sh`, or `tools/l8_midflight_eval.sh` (active
  scripts).
- DO NOT touch `eagle4/checkpoints/eagle4_v4_fromscratch/` (active
  ckpt dir) or any `_iter*` archive dirs.
- DO NOT change Eagle4 training code in `eagle4/eagle4.py` while
  iter 5 or iter 6 is running. The script reads the file's current
  state; mid-flight changes to constants would silently load wrong
  hyperparameters.
- DO use a different working file scope when possible (new shaders
  in new files; new tests in new test files; etc.).
- DO check `reports/path_to_90/_levers/l8_iter_results.jsonl`
  before benching anything yourself — bench only in a CLEAN window
  (no training, no other Claude session).

## Bench window protocol

L7 and L5 need clean-window benches. To get one:
1. Check `tools/l8_autoiter.sh status` — confirm all iters
   `verdict != "ADVANCE"` (i.e., queue ended).
2. `kill <autoiter_pid>` if autoiter is still sleeping (it auto-
   exits on completion but be safe).
3. Cmd-Q Claude if user can spare it. (User has typically said no
   to closing Claude; you may need to bench under contention and
   call out the variance in your report.)
4. `tools/bench/path_to_125_bench.sh`. Paste output back into the
   plan docs' "results" section.

## Final output expected from each phase

Each shipped phase should end with:
- A commit (or series) implementing the plan
- All parity gates green
- Clean-window bench result OR documented "bench queued"
- A short closeout in `reports/path_to_90/closeouts/phase_<name>_closeout.md`
  with:
  - What shipped (commit hashes)
  - What was deferred (and why)
  - Net dec_tps delta
  - Surprises / corrections to the original plan
  - Next-phase recommendation

## Start command

```
1. cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
2. git pull   (if other sessions might commit)
3. tools/l8_autoiter.sh status   (check L8 final state)
4. cat reports/path_to_90/plans/PHASES.md
5. cat reports/path_to_90/plans/acceleration_patterns.md
6. cat reports/path_to_90/plans/phase_<chosen>.md
7. Start the phase.
```

The plan docs are self-contained execution playbooks. Read them
end-to-end before writing code. Apply the acceleration_patterns
checklist before launching any subprocess.

```
=== PROMPT END ===
```

---

## Notes the L8 monitoring session can read

The monitoring session (the one that wrote this doc) should keep
running until the autoiter queue completes. It will:
- Pick up cron pings every 20 min
- Confirm iter 4 / 5 / 6 results
- Verify the autoiter's `l8_iter_results.jsonl` records make sense
- Write a final `reports/path_to_90/closeouts/l8_final_closeout.md`
  once the queue is empty

The phase session can read both the autoiter results file and the
L8 closeout to know which path-to-150 phases to attack and which
are blocked.
