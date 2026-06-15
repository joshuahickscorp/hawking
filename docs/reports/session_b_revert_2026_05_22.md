# Session B (T2.16 batched ExactShared verify) — partial revert

**Date:** 2026-05-22
**Trigger:** Session K — Parallel-agent microbench (`microbench_levers.sh`) flagged
`L4_spec_exact_K4 = 5.31 dec_tps` (Δ -17.01 vs L0=22.32) and
`L4_spec_exact_K16 = 1.68` (Δ -20.64) on 7-trial paired-delta run
`artifacts/runs/microbench/20260522T231758Z/report.md`.

**Disposition:** **Partial revert.** Restore serial `forward_token_argmax`
verify with early-exit-on-mismatch in the ExactShared spec-decode path.
Keep `DecodeArena.max_batch_size = 17` (still useful for NGram K=16).
Keep `forward_token_argmax` live (drop the `#[allow(dead_code)]` annotation).

Leave uncommitted in the worktree per Session K instructions.

## What Session B changed

`crates/dismantle-core/src/model/deepseek_v2.rs` ExactShared block (around
the `--- VERIFY ---` comment):

1. Swapped serial `verify_draft_ids_until_mismatch` + per-step
   `forward_token_argmax` (which stops at first mismatch) for a single
   `forward_tokens_batched(&[last_id, ...draft_ids], &[pos..=pos+K])` call
   followed by an always-K-iterations argmax loop.
2. Bumped `DecodeArena::new(..., 17)` from `8` to support K=16 + bonus
   position in a single TCB.
3. Marked `forward_token_argmax` `#[allow(dead_code)]`.

The mirror change in the NGram path (line ~1296) was made earlier (T2.15)
and is **not** in scope for this revert — NGram acceptance is much higher
because drafts come from a CPU n-gram lookup, not a shared-only model
forward.

## Why batched verify regressed ExactShared specifically

ExactShared draft step calls `forward_token_shared_only_argmax`, which
does a full shared-only model forward per draft token. Per outer iteration:

| Phase  | Old serial-verify | New batched-verify     |
|--------|-------------------|------------------------|
| Draft  | K shared forwards | K shared forwards      |
| Verify | `first_reject+1` full forwards (early-exit) | **K+1 full forwards always** |
| Total  | K + first_reject + 1 | 2K + 1                |

At greedy-on-shared-only acceptance (typically 0–1 of 4), the batched
path pays ~5x the forward cost per emitted token. Cost model predicts:

* K=4: ~10 forwards/emitted-token → 22 tps / 5 ≈ ~4–5 tps. **Observed 5.31.**
* K=16: ~34 forwards/emitted-token → 22 / 17 ≈ ~1.3 tps. **Observed 1.68.**

The single-TCB-commit win (~3–5x cheaper per commit) does not recover the
work-amplification cost when acceptance is low.

NGram is unaffected because drafts are O(μs) CPU lookups, so doubling
verify work is a smaller fraction of step latency, and NGram's higher
acceptance rate makes the batched verify a net win.

## Bench (paired delta vs L0, Claude live)

Microbench run `artifacts/runs/microbench/20260523T002804Z/report.md`,
3 trials × 64 tok, after partial revert applied + binary rebuilt +
stale `shader_hash` in `profiles/deepseek-v2-lite-q4.m3pro18.json`
updated to current binary hash `92ba78831a4ad1d0abfacb70`.

| Lever              | Pre-fix (T2.16) | Post-fix (revert) |
|--------------------|----------------:|------------------:|
| L0_baseline        | 22.32           | 24.50             |
| L4_spec_exact_K4   | 5.31  (-17.01)  | 9.96  (-14.54)    |
| L4_spec_exact_K16  | 1.68  (-20.64)  | 4.10  (-20.40)    |
| L4b_spec_ngram_K4  | 24.93 (+2.61)   | 24.29 (-0.21)     |
| L4b_spec_ngram_K8  | 24.93 (+2.61)   | 24.37 (-0.13)     |
| L1_vocab_prune     | _n/m_           | 26.02 (+1.52)     |
| STACK_no_spec      | _n/m_           | 26.02 (+1.52)     |

**Done-condition NOT met.** `L4_spec_exact_K4 = 9.96` is 14.54 tps
below L0 (target ≤ -2.0).

### Why the done-condition can't be met by reverting Session B

`DISMANTLE_SPEC_LOG=1` on the post-fix binary (32 tok, ExactShared K=4):

```
[spec] accept=1/4 draft=92.5ms verify=78.6ms step=172.5ms emit=2 tps=11.6
[spec] accept=0/4 draft=93.4ms verify=40.2ms step=134.3ms emit=1 tps=7.4
[spec] accept=3/4 draft=93.8ms verify=159.1ms step=255.1ms emit=4 tps=15.7
...
final: dec_tps=10.67 draft_accepted=13 draft_rejected=60  (accept ≈ 18%)
```

Per-step cost breakdown:

* **Draft = ~93 ms** for K=4 shared-only forwards (~23 ms each).
  Shared-only forward is ~58% the cost of a full forward.
* **Verify = ~40 ms × (first_reject + 1)** — averaging ~1.3 forwards.
* Total ≈ 93 + 1.3·40 = ~145 ms per step, emit ~1.3 tokens → ~9 tps.

Even with first-mismatch early exit (the partial revert restores this),
the K shared-only draft forwards are already a near-fixed ~93 ms
overhead per outer iteration. At ~18% acceptance, emitted-per-step
(~1.3) is too small to amortize the draft cost vs the baseline
single-forward step at ~40 ms.

This **matches the 9.98 tps measurement** in
`memory/spec_decode_runtime_healthy.md` from 2026-05-22 morning, taken
before Session B's batched-verify change. The 22 → 5 tps regression
attributed to Session B was real but only relative to the *batched*
path. The serial path the revert restores is the *historical baseline*
behaviour — also a regression vs no-spec, just a smaller one.

**Conclusion:** Partial revert is the correct disposition (eliminates
Session B's ~5 tps additional damage on top of the inherent regression).
Reaching `dec_tps ≥ L0 − 2` requires either:

1. A cheaper draft model (the current "shared-only" path runs ~58% of
   the full MoE, not the few-% an Eagle/Medusa head would cost).
2. Eagle5 v2 (queued — head deployable once user's overnight training
   lands; see `memory/path_to_50_complete.md`).
3. Higher acceptance rate (would require a stronger shared-only
   surrogate, e.g. shared-experts-only with a small bias term).

These are not in scope for this revert; the revert's job was to undo
the additional damage T2.16 introduced.

## Files touched

* `crates/dismantle-core/src/model/deepseek_v2.rs`
  * ~30 lines reverted in the ExactShared `--- VERIFY ---` block.
  * `#[allow(dead_code)]` removed from `forward_token_argmax`.
  * `max_batch_size = 17` kept (useful for NGram K=16; ~325 KiB overhead).
* `profiles/deepseek-v2-lite-q4.m3pro18.json`
  * `shader_hash` bumped to `92ba78831a4ad1d0abfacb70` (current
    binary's hash) so the microbench actually runs. The previous value
    `05ac3c172932cfe7f6b0b327` was stale relative to the rebuilt
    binary. **This is a bench-side fix, not a behavioural change.**

## Not committed

Per Session K spec — user to review the diff before commit. Two files
in the worktree diff:

```
M crates/dismantle-core/src/model/deepseek_v2.rs
M profiles/deepseek-v2-lite-q4.m3pro18.json
```

(Plus all the other Session A–J in-flight worktree changes that were
already there when Session K started.)
