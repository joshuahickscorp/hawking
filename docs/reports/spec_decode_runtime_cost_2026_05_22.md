# Spec-decode runtime cost reduction — T2.16 — 2026-05-22

Status: **CODE LANDED, BENCH PENDING** (eagle5_v3 training holds MPS — cannot
run clean bench without contaminating training run).

## Cost driver identified

In `ExactShared` (the so-called "eagle4 K=4") spec-decode mode, **the verify
pass runs K+1 separate full-model `forward_token` calls**, each paying its
own Metal command-buffer commit+wait round-trip.

`crates/dismantle-core/src/model/deepseek_v2.rs` (pre-T2.16, lines
1121-1138): the verify loop drove
`crate::speculate::shared::verify_draft_ids_until_mismatch` with a closure
that called `self.forward_token_argmax(tmp_last, pos + k, …)` per draft
token, followed by a bonus `forward_token_argmax`. Each call ran
`forward_token` → `forward_token_final_norm_maybe_read` → `tcb.commit_and_wait()`.

By contrast, the `NGram` path (line 1283) calls `forward_tokens_batched`
which routes to `forward_tokens_batched_tcb` — **a single TCB commit** for
all K+1 tokens. That structural difference explains why NGram doesn't
exhibit the same regression and why ExactShared regresses ~50-60% even at
47% acceptance.

### Per-step commit accounting

| Mode | Draft commits | Verify commits | Total commits | Tokens emitted (avg) |
|---|---|---|---|---|
| `off` | 0 | 1 | 1 | 1 |
| `exact-shared` K=4 (pre-T2.16) | 4 | up to 5 (early-exit) | 5-9 | 1.5-2 |
| `exact-shared` K=4 (T2.16) | 4 | **1** | 5 | 1.5-2 |
| `ngram` K=4 | 0 | 1 | 1 | 1.5-2 |

At ~150-200µs per commit+wait on M3 Pro (per memory:
`per_kernel_time_breakdown.md`), eliminating 3-4 verify commits per step
saves ~0.6-0.8ms per spec step. At the bench's measured step rate, this
should close most of the gap.

The remaining 4 draft commits are the next lever (see "Remaining
headroom" below).

## Fix applied

`crates/dismantle-core/src/model/deepseek_v2.rs`:

1. **Lines ~1118-1145** (the `SpeculateMode::ExactShared` arm): replaced
   the per-token `forward_token_argmax` verify loop with a batched
   `forward_tokens_batched(&[last_id, d0, …, dK-1], &[pos, pos+1, …, pos+K])`
   call, mirroring the structure of the `NGram` arm. Argmax over the
   returned `Vec<Vec<f32>>` reproduces the prior accept/reject logic
   bit-for-bit (greedy temp=0, argmax over original-space ids).

2. **Line 877** (`DecodeArena::new` call): bumped `max_batch_size`
   from 8 to 17 so that `verify-window=16` (the largest validated value)
   actually hits the single-TCB fast path. Pre-T2.16 K=16 silently fell
   back to the sequential per-token loop because K+1=17 > max_batch_size=8.

3. **`forward_token_argmax`** is now dead code; marked
   `#[allow(dead_code)]` and left in place as a diagnostic fallback.

### Trade-off note

The pre-T2.16 path had early-exit-on-first-mismatch; T2.16 always runs
K+1 forwards. At avg_accept ≈ 0.5 that's roughly +2.5 wasted forward
positions per step — but a *single* batched TCB at K=5 is still cheaper
than 2-3 individual TCB commits, because intra-TCB encoders amortize the
~150µs commit overhead. NGram mode already made this same trade-off and
wins big at K=4.

## Parity argument

The batched verify path uses exactly the same primitive (`forward_tokens_batched_tcb`)
that the NGram path uses and that has parity coverage in
`crates/dismantle-core/tests/v1_1_phase4D_spec_exact_mode.rs::{repetitive_prompt_spec_matches_greedy,
natural_prompt_spec_matches_greedy}` (these test NGram vs off — they
exercise the batched verify against off-mode reference under
greedy/temp=0/rep-penalty=1).

The new T2.16 ExactShared path now uses the same primitive, so an
existing test simply switching `SpeculateMode::NGram` →
`SpeculateMode::ExactShared` provides equivalent coverage. To make this
explicit, the existing `v1_1_phase4D_spec_exact_mode.rs` should grow
two more tests calling `load_engine(SpeculateMode::ExactShared)` (one
repetitive, one natural). **Not landed this session** — needs the bench
window to actually run end-to-end.

Compile-time check passed: `cargo test --release -p dismantle-core
--test v1_1_phase4D_spec_exact_mode --no-run` succeeds clean (only the
two pre-existing objc warnings).

## What was NOT touched

- **Draft path** — still K serial `forward_token_shared_only_gpu_argmax`
  commits. This is the obvious next lever: build a
  `forward_tokens_shared_only_batched_tcb` analog of `forward_tokens_batched_tcb`
  that runs K shared-only forwards in one TCB. Projected savings: another
  ~3 commits/step at K=4.
- **eagle4 head weights** — frozen per session prompt. The runtime treats
  "exact-shared" as a synonym for "shared-only-FFN-only draft," there is
  no separate eagle4 head loaded today.
- **Off-mode path** — unmodified.
- **NGram path** — unmodified.

## Bench plan (run after eagle5_v3 training completes)

1. Verify training done: `cat artifacts/runs/overnight/extended_status.json`
   shows `current_stage == complete && state == done`.
2. Quit Claude Code (per memory: `bench_contamination.md` — for absolute
   tps numbers, not for relative paired deltas).
3. `bash tools/bench/spec_decode_sweep.sh`.
4. Compare against the pre-T2.16 baseline in
   `artifacts/runs/overnight/spec_decode_sweep.md`:
   - `Once upon a time…` — pre: 9.26 dec_tps (24/130). Target: ≥ 24.04 (off).
   - `Capital of France…` — pre: 12.07 dec_tps (42/46). Target: ≥ 22.25.
   - `def fibonacci…` — pre: 11.73 dec_tps (38/61). Target: ≥ 21.80.
5. Run the parity test live:
   `cargo test --release -p dismantle-core --test v1_1_phase4D_spec_exact_mode`.

## Done-condition

- [ ] Re-run sweep shows ExactShared K=4 ≥ off-mode on ≥2/3 prompts.
- [ ] Spec parity tests green (existing NGram tests + planned ExactShared
      copies).
- [x] Writeup landed at `reports/spec_decode_runtime_cost_2026_05_22.md`.
- [ ] Memory note at `memory/spec_decode_cost_reduced.md` (write after
      bench validates).

## Remaining headroom (post-T2.16)

1. **Batched shared-only draft** — biggest remaining lever. Build
   `forward_tokens_shared_only_batched_tcb` modeled on the batched verify
   path. Saves K-1 draft commits per step. Projected: another +3-5
   dec_tps on top of T2.16.

2. **Speculative draft + verify pipelining** — once both draft and verify
   are single-TCB, draft commit N can issue concurrently with verify
   commit N-1. M-series Metal supports overlapping CB execution. Saves
   roughly one commit-wait latency per step.

3. **LM-head fold for batched verify** — `forward_token_final_norm_maybe_read`
   already folds the LM head + argmax into the global TCB for the
   single-token path (`lm_head_foldable`). The batched-TCB path
   (`forward_tokens_batched_tcb`) still does K separate CPU-side
   `gemv_f16_dispatch` calls in a post-commit loop (lines 2410-2426).
   Folding K LM-head gemvs into the global TCB would save the K
   `batch_x_norm_buf` GPU→CPU copies and the K CPU gemv calls.

4. **Higher avg_accept via eagle5 head deployment** — handoff in
   `reports/eagle5_v2_wiring_handoff.md`. Independent of T2.16.

## Cross-references

- `reports/spec_decode_runtime_NOT_broken_2026_05_22.md` — the runtime
  audit that ruled out structural bugs.
- `memory/path_to_100_repath.md` — places this lever inside the
  dual-track plan.
- `memory/per_kernel_time_breakdown.md` — commit-overhead size estimate.

## Files changed

- `crates/dismantle-core/src/model/deepseek_v2.rs` (~30 lines diff)

No commits made (per session rules — user reviews diff first).
