# Phase F — EAGLE-3 medusa-style multi-token head

**Goal:** +20-40 dec_tps. Stretch to 130-150.
**Engineering est:** 2-4 weeks focused work.
**Confidence:** LOW — research-grade architectural change.

## What code is missing (Pattern 9)

| File | Lines (est) | Purpose |
|---|---|---|
| `eagle4/medusa_head.py` | ~400 | New head: K parallel prediction heads instead of chain |
| `eagle4/capture.py` (updates) | ~50 | Capture targets at K future positions instead of next-only |
| `crates/dismantle-core/src/speculate/medusa_head.rs` | ~500 | Rust port of medusa head |
| `crates/dismantle-core/src/model/deepseek_v2.rs` | ~200 | Verifier loop integration |
| `crates/dismantle-core/tests/medusa_parity.rs` | ~150 | Multi-head parity tests |
| `crates/dismantle-core/shaders/medusa_lmhead.metal` | ~120 | Batched K-head LM dispatch |

**Net:** ~1420 lines, fundamentally new head architecture.

## Why this lever

Eagle-style chain decode generates K candidates SEQUENTIALLY by
auto-regressively rolling the head. Each step depends on the previous,
so K-fold accept caps at `1 + p + p² + ...`.

Medusa-style heads instead have K PARALLEL prediction heads, each
specialized to predict position `+i` given the current state. No
chain rollout; all K predictions come out in one forward pass.

Benefits:
- No chain rollout dependency → each head can be perfectly optimized
  for its own offset (head i sees real i-step-ahead positions during
  training)
- No "gate collapse" failure mode that plagued chain training
- Inference is one forward pass instead of K sequential

Cost:
- More params (K independent prediction heads vs 1)
- Each head must learn independently — more training data needed
- Tree-decode compatibility is non-trivial

Published Medusa results: 1.5-2× over chain decode on Llama. Combined
with tree decode (Phase E), 2-3× over baseline.

## What's already shipped

Nothing for medusa specifically. The infrastructure that helps:
- Eagle4 training pipeline (eagle4.py) — pattern can be ported
- K-batched verifier kernels — already exist, work for any draft
  source (chain or medusa)
- Mid-flight eval helper

## Concrete plan

### Milestone F.1 — capture pipeline (3-5 days)

Modify `eagle4/capture.py` to capture targets at the next K positions
per row, not just the next-only. The parquet schema changes:
- `next_token_+0`, `next_token_+1`, ..., `next_token_+K-1`
- `hidden_high_+0`, ..., `hidden_high_+K-1` (optional, for MSE
  auxiliary)

Re-run capture on all 62 training shards. Estimated wall time: ~12 hr
clean window (capture is V2-Lite forward inference).

This is the hardest part because it's wall-clock-bound, not
engineering-bound. Plan a clean overnight window.

### Milestone F.2 — Medusa head architecture (1 week)

New file `eagle4/medusa_head.py`:
```python
class MedusaHead(nn.Module):
    def __init__(self, frozen, n_heads=4, hidden=2048, vocab=102400):
        # Each prediction head is a small MLP on top of the shared
        # representation from frozen V2-Lite.
        self.heads = [
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, vocab),
            )
            for _ in range(n_heads)
        ]

    def __call__(self, h):  # h: V2-Lite's last-layer hidden
        return [head(h) for head in self.heads]  # K × (B, S, V) logits
```

Training: K independent cross-entropy losses against `next_token_+i`
targets.

Eval: top-1 accuracy per head against ground-truth K-position-ahead
tokens. Healthy head: ≥40% top1 at i=0, decaying to ≥15% at i=K-1.

### Milestone F.3 — Rust port (3-5 days)

`crates/dismantle-core/src/speculate/medusa_head.rs`:
- NPZ loader for K-head weights
- Forward: K parallel MLPs → K vocab logits → K argmax = K drafts
  in ONE forward pass

The forward is structurally simpler than Eagle4Head's chain because
no rolling, no gate, no draft_hidden propagation. Pure parallel.

### Milestone F.4 — Verifier integration (2-3 days)

Adapt the existing parallel-k-union verifier (Branch 2 step 4) to
accept K independent drafts from the medusa head. Already works at
K=4 by construction — no kernel changes needed.

### Milestone F.5 — bench + tree-decode hybrid (3-5 days)

Headline bench: medusa K=4 vs chain K=4 on V2-Lite.

If medusa wins, evaluate tree-of-medusa hybrid: each medusa head's
top-B candidates fed to a tree-batched verifier (Phase E
infrastructure). Cumulative multiplier: medusa × tree.

## Risks + mitigations

1. **Capture pipeline doesn't have +K targets in current shards.**
   *Mitigation:* re-run capture. Costly but mechanical. Schedule
   overnight.

2. **Medusa head's K=4 head can't learn (sparse signal at high i).**
   *Mitigation:* head 0 carries most weight; later heads are bonus.
   Even partial medusa (heads 0,1 useful; 2,3 weak) beats chain at K=2.

3. **Medusa+tree integration complex — combinatorial state explosion.**
   *Mitigation:* ship medusa alone first (Phase F). Tree integration
   (Phase G) is a separate phase.

4. **Vocab is large (102400) — K=4 medusa = 4× LM head GEMV.**
   *Mitigation:* the existing K-batched LM head kernel
   (gemv_f16_lmhead_kbatch) already batches across K. Medusa K=4 uses
   K=4 lookup → same path as chain K=4. No extra cost beyond what
   chain already pays.

## Acceleration patterns applied

- Pattern 1: per-head top1 eval at each training checkpoint
- Pattern 2: synthetic parity / per-head top1 / full chain bench
- Pattern 5: medusa is the structural alternative to chain — apply
  acceleration_patterns Pattern 5 (architectural fix before more
  hyperparam tuning).
- Pattern 9: 1420 lines, mostly new code

## Acceptance criteria

- Per-head top1 ≥ 40% at i=0, ≥ 15% at i=K-1
- Full K=4 medusa chain_accept (using parallel-k verifier) ≥ 50%
- Clean bench: ≥1.3× dec_tps over chain decode at matched K
- All existing parity gates still pass

## Open research questions

These need experimental answers, not just engineering:
1. Optimal number of heads K — 4 is the standard but 6 or 8 might
   work given V2-Lite's MoE capacity.
2. Independent vs shared head trunks — pure medusa vs medusa-with-shared-trunk.
3. Training schedule — can we warm-start from a chain-trained head?

## Decision point at each milestone

F.1 is wall-clock-bound; just do it.
F.2 ships if per-head top1 looks reasonable on a small training run.
F.3-F.4 ship after Rust parity tests pass.
F.5 is the GO/NO-GO for medusa as a whole.

## Next-session quickstart

Phase F is the longest. Recommend starting with milestone F.1
during a clean overnight window. F.2 can be parallel-developed in
a different worktree.

```
# Clean window required for F.1:
# 1. Cmd-Q Claude (capture is V2-Lite forward, contention-sensitive)
# 2. Modify capture.py to write +K targets
# 3. Run capture overnight on all 62 shards
# 4. Verify schema in resulting parquet
# 5. Next day: start F.2 medusa head implementation
```
