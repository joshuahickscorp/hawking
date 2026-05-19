# NEXT-SESSION STARTER PROMPT — path-to-125 final lever exhaustion

**Copy everything below the `=== PROMPT BEGIN ===` line as the FIRST message to the
next Claude Code session.** Do not edit it. It is calibrated to keep the
agent moving end-to-end without intermediate "let me stop and summarize"
breaks.

```
=== PROMPT BEGIN ===
```

You are picking up `dismantle` path-to-125 mid-pursuit on branch
`claude/dreamy-golick-d54ff8` HEAD `b5afd84`. Read
`reports/path_to_90/session_closeout_2026-05-19b.md` end-to-end first.
That document is the only context you need; it folds in everything
load-bearing from prior sessions.

## HARD RULE — DO NOT RETURN UNTIL THE ENTIRE LEVER QUEUE IS EXHAUSTED

The user explicitly wants every remaining lever attempted back-to-back
in this session. You may NOT return control to the user with a "good
stopping point" / "session summary" / "I'll continue next time"
message. Return control ONLY when one of these is true:

1. Every lever in §3 has either SHIPPED (commit landed + parity passed)
   or has a documented "DID NOT SHIP" entry in
   `reports/path_to_90/_levers/exhaustion_report.md` explaining why
   (build broken, architectural blocker, dependency missing, etc.) AND
   the next lever has been attempted regardless.
2. A hard halt condition fires (§5): build broken AND can't fix in 30 min,
   bit-identical parity fails AND no root-cause in 30 min, RSS > 5 GB
   sustained 5+ min, OR user explicitly says "stop".

DO NOT stop for: "context is getting long", "this is a natural
checkpoint", "I should write a closeout", "let me ask the user", "the
next lever is risky", "the previous one is already a lot". The
user has authorized full autonomous execution to the end.

Between levers, commit immediately and start the next one with the same
turn. Do not pause to ask if the user wants you to continue. They do.

## 1. State at session start

Branch: `claude/dreamy-golick-d54ff8`
HEAD:   `b5afd84`

What is SHIPPED (do not redo):

| commit | thing |
|---|---|
| `5aec2bf` | Branch 1: K-batched MLA wired into `forward_tokens_batched_parallel_k`. Bit-identical at K=4. ngram-K4=26.71 vs Off 26.78. |
| `4c772e8` `70eb0c9` `db36908` | Branch 2 (A1.2): MoE expert-union kernels + pipeline wrapper + Phase C wire-up. parity-safe, ~1% within-noise on V2-Lite (caching hides projected gain). |
| `888b3d2` `9cdbe49` `acfa2ea` | Branch 3 patches: `--chain-h-high`, position-shifted multi-step loss, `--multi-step-aux-decay`. Validated 5 iter runs; chain accept floor at 7% with v3 head, gate-init bump alone makes it worse (commit b5afd84 lever 2). |
| `c1d4be1` | Cross-lever auto-orchestrator (`tools/path_to_125_levers.sh`). |

What is PENDING (work this session):

| Lever | Effort | Expected dec_tps gain |
|---|---|---|
| L4 — Phase F1 AMX extend to V2-Lite projection gemvs | 4-8 hours | +5-10 |
| L5 — Phase F5 multi-queue Metal scheduling | 4-6 hours | +3-8 |
| L6 — Phase F3 async verify-start (DEFERRED last session; revisit if F5 lands) | 8-24 hours | +5-8 |
| L7 — Stage 0.5 MLX kernel rewrites (gemv_q4_k_v3 + MoE expert pair) | 6-10 hours | +10-20 |
| L8 — Eagle4 head retrain from scratch (no v3 warm-start) | 1-day compute + setup | +5-15 IF architecture supports |
| L9 — Full headline clean-window bench after every lever lands | ~10 min × N | (measurement) |

You will NOT finish all of these in one session. The instruction is to
attempt EACH in order, ship what works, document what doesn't, and keep
going until truly exhausted.

## 2. Inviolable rules (same as prior sessions)

- Bit-identical greedy gate (`eagle4_decode_parity` test) MUST pass at
  every commit. This is the safety net.
- Production K=1 path (`forward_tokens_batched_tcb`) MUST remain
  unchanged in behavior.
- Commit as `Joshua Hicks <joshuahicksboba@gmail.com>` via inline
  `git -c user.name=... user.email=...`. Never `git config`. Never
  Co-Authored-By / Generated-With trailers.
- User's diagnostic edits in `engine.rs` / `kernels/mod.rs` /
  `deepseek_v2.rs` MUST be preserved across every commit. The diff is
  exactly `+10 / +13 / +4 lines` across those 3 files. Verify with
  `git diff --stat` before and after every commit. Use strip-restore:
  ```
  cp <file> /tmp/<file>.bak
  git checkout HEAD -- <file>
  # apply YOUR edits
  git add <file>
  git commit -m "..."
  # re-apply user hunks via Edit tool
  ```

## 3. Lever sequence (execute in this order, do not skip)

### L4 — Phase F1: AMX extend to V2-Lite projections

Eagle4 head already uses direct Accelerate.framework `cblas_sgemv`
(commit `d1d50fb`, prior session). Extend the SAME pattern to
V2-Lite's smaller projection gemvs where matrix shape fits AMX's
sweet spot (rows ≤ 1024, cols ≤ 2048).

Concrete targets in `crates/dismantle-core/src/model/deepseek_v2.rs`:
- `q_a_proj` (rows = q_lora_rank = 1536, cols = hidden = 2048) — fits.
- `kv_a_proj_with_mqa` (rows = kv_lora_rank + qk_rope_head_dim = 576, cols = 2048) — fits well.
- `q_b_proj` (rows = n_heads × q_head_dim = 3072, cols = q_lora_rank = 1536) — borderline; A/B.
- `kv_b_proj` (rows = n_heads × (qk_nope + v_head) = 4096, cols = kv_lora_rank = 512) — try.

For each:
1. Add an AMX path in the dispatcher (mirror Eagle4 head's `forward_full_amx_no_lm_head`
   pattern — load f32 weight matrix once, call cblas_sgemv per call).
2. Gate by profile flag `attn_proj_amx = true|false` (default false).
3. A/B parity: synthetic GEMV result must match the existing Metal
   path within `atol=1e-3 f32`.
4. Wall-clock A/B in a brief contended smoke. If ≥3% improvement on
   the projection's wall time, ship as default-on at profile flag.
   If <3%, keep code, gate to false, document as "AMX shape edge".

Commit per-projection: `path-to-125 L4.1 AMX q_a_proj`, etc.

### L5 — Phase F5: multi-queue Metal scheduling

Current `MetalContext` uses ONE `MTLCommandQueue`. Multi-queue lets
draft (Eagle4 head) and verify (V2-Lite forward) overlap.

Concrete plan:
1. Add `secondary_queue: Option<metal::CommandQueue>` to MetalContext.
2. Add `TokenCommandBuffer::new_on_secondary(ctx)` constructor.
3. In Eagle4 chain decode loop, dispatch head propose on secondary
   queue while verifier's first-layer kv_append + MLA Phase A runs
   on primary. Synchronize via `MTLSharedEvent` (already in
   `metal/sync.rs` per memory note `sharedevent_amx_facts`).
4. A/B with profile flag `multi_queue = true|false`.

Parity: bit-identical to single-queue. The only change is dispatch
scheduling, not math.

Commit: `path-to-125 L5 multi-queue Metal scheduling`.

### L6 — Phase F3 async verify-start (revisit)

Only after L5 lands. F3 uses the multi-queue infrastructure to
overlap head's last propose step with verifier's first-layer
expert prefetch.

If L5 doesn't ship (regression), SKIP L6 and document.

### L7 — Stage 0.5 MLX kernel rewrites

Goal: lift Off baseline by adopting MLX-LM kernel patterns.

Targets (highest-leverage first):
1. `gemv_q4_k_v3_mlx` — rewrite the Q4_K_M GEMV against MLX-LM's
   `mlx_lm/models/deepseek_v2.py` LM head kernel pattern. New shader
   `crates/dismantle-core/shaders/gemv_q4_k_v3_mlx.metal`. Parity
   `atol=1e-3 fp16` vs CPU dequant+gemv reference. A/B vs existing
   `gemm_q4_k_m_fused_v2` on V2-Lite expert-projection shape (rows=
   10944, cols=2048). Ship if ≥10% wall improvement.
2. `moe_expert_pair_mlx` — fuse gate+up+down with shared SIMD-group
   register state, per MLX-LM's MoE forward.

Each shader change requires shader_hash regen (pitfall #2 from prior
sessions) and profile update.

### L8 — Head retrain from scratch (no v3 warm-start)

Critical experiment per b5afd84's finding: gate growth alone makes
chain accept worse if the block parameters aren't trained for it.
Full from-scratch training is the way to test if the architecture
CAN learn chain decode at K=4.

Setup:
1. Patch `eagle4/eagle4.py` `EagleHead.__init__` to take a
   `gate_init` keyword (default 0.05); orchestrator passes 0.1.
2. Run training WITHOUT `--resume`:
   ```
   python eagle4/eagle4.py train \
     --parquet training_data/c2_hidden/eagle4_v0/shard_*.parquet \
     --frozen eagle4/v2lite_frozen.npz \
     --ckpt-dir eagle4/checkpoints/eagle4_v4_fromscratch \
     --epochs 2 \
     --multi-step-k 4 \
     --multi-step-decay 0.7 \
     --chain-h-high \
     --target-warmup-steps 500 \
     --multi-step-aux-decay 0.3
   ```
3. This runs ~10-15 hours wall-clock contended (or ~3-4 hours clean).

If the user has Claude open, training runs at ~10× slowdown. Launch
via `nohup ... &` and ALSO write status to
`reports/path_to_90/_levers/l8_status.json` every 200 steps so the
agent can poll without blocking.

τ-eval + chain-decode smoke after training. If chain accept rate
climbs past 25%, the experiment validates that the architecture can
learn chain decode given proper training. If still ~7%, the v3
architecture has a hard ceiling and the path-to-125 needs Medusa or
EAGLE-3-style head replacement (out of scope).

### L9 — Headline bench (run after each shipped lever)

Use `tools/bench/path_to_125_bench.sh`. If Claude is open when L9
fires, the bench will refuse to run — that's correct behavior.

In that case, write a status note "bench queued — user action: Cmd-Q
Claude and run `tools/bench/path_to_125_bench.sh`" to
`reports/path_to_90/_levers/l9_bench_queued.md` and proceed to the
next lever WITHOUT BLOCKING.

## 4. Working pattern (do this every commit)

```
1. pick smallest task from active lever
2. write code
3. cargo build --release -p dismantle-core   # MUST pass
4. cargo test --release -p dismantle-core --lib   # 45/45 MUST pass
5. cargo test --release -p dismantle-core --test path_b_parity   # active MUST pass
6. EAGLE4_PARITY_TEST=1 DISMANTLE_EAGLE4_GREEDY_TOKENS=16 cargo test
     --release -p dismantle-core --test eagle4_decode_parity
     -- --ignored --nocapture
   # bit-identical greedy gate MUST pass at every commit (load-bearing)
7. strip-restore user diff (pitfall #6)
8. commit as Joshua Hicks via inline git -c
9. restore user diffs to working tree
10. IMMEDIATELY pick next task (do not pause for "summary")
```

## 5. Halt criteria (NARROW — most things are NOT halts)

Halt = write `reports/path_to_90/_levers/halt_<lever>_<short>.md` and
return control to user. Only for:

- Bit-identical greedy gate fails AND cannot root-cause in 30 min.
- `cargo build --release --workspace` fails AND error is outside your
  edits (upstream dep, OS issue).
- RSS > 5 GB sustained for > 5 min on a dismantle process.
- User explicitly types "stop".
- ALL §3 levers attempted (shipped or did-not-ship documented).

NOT halts:
- Lever shows regression — document, revert that lever's code only,
  proceed to next.
- Wall-clock A/B disappointing — ship behind opt-in flag, proceed.
- Parity test FAILS but you can root-cause it — fix, recommit.
- "Context is filling up" — keep going. Compact via commits + closeout
  docs, not by stopping.
- "I should ask the user about X" — make the call yourself per the
  prior session's autonomy charter (`AUTONOMOUS_PLAN.md` §3.5).

## 6. Final return value

When ALL levers in §3 have been attempted (shipped or skipped with
documented reason), write ONE final report at
`reports/path_to_90/_levers/exhaustion_report.md` summarizing:

- Levers shipped (commit hashes + dec_tps deltas where bench
  available)
- Levers attempted but reverted (with reason)
- Levers deferred (with concrete blocker)
- Final clean-window bench numbers if any
- Updated trajectory estimate to 125 dec_tps

Then commit the exhaustion report and return control to the user with
a brief message: "All levers exhausted. See exhaustion_report.md."

That is the ONLY acceptable return-of-control until then.

## 7. Reference files

- `reports/path_to_90/AUTONOMOUS_PLAN.md` — the master plan, with the
  `PLAN AMENDMENT 2026-05-19` block load-bearing for context.
- `reports/path_to_90/session_closeout_2026-05-19b.md` — prior block's
  closeout with the chain-accept-plateau diagnosis.
- `reports/path_to_90/_levers/status.json` — orchestrator state
  tracking (used by `tools/path_to_125_levers.sh`).
- `reports/path_to_90/a12_moe_union_design.md` — A1.2 design (shipped
  in commits 4c772e8/70eb0c9/db36908; here for cross-reference if you
  need to revisit the kernel internals).
- `tools/bench/path_to_125_bench.sh` — clean-window bench script
  (refuses Claude-open; that's a feature not a bug).
- `tools/path_to_125_levers.sh` — cross-lever orchestrator scaffold;
  extend it as you ship new levers.

## 8. Open question to handle at start (just one)

Q: User said "do not stop until all levers exhausted". They also need
to keep Claude open to work on other projects. The bench script
refuses Claude-open runs. How should you handle bench gates?

A: When a bench gate fires, write the queued status under
`reports/path_to_90/_levers/l9_bench_queued.md` and proceed without
blocking. Do NOT prompt the user to close Claude. Bench numbers are
nice-to-have, not required-for-shipping; each lever's parity gate +
synthetic A/B is sufficient for ship decisions. If the user closes
Claude later, they re-run the queued bench and paste back.

DO NOT ask this question; the answer is baked in. Just proceed.

## 9. Begin

Read §1 + §2 + §3, then start with L4 (Phase F1 AMX extend, target
q_a_proj first). Do not summarize what you read; do not ask which to
do first; do not stop to "plan." Just code, commit, and move to the
next lever.

The user has authorized full execution. Honor that.

```
=== PROMPT END ===
```
