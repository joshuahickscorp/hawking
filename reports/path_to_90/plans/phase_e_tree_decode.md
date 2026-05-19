# Phase E — tree decode

**Goal:** +30-50 dec_tps (1.5-1.8× chain speculation multiplier).
**Engineering est:** 1-2 weeks focused work.
**Confidence:** MEDIUM-LOW — bigger architectural change with empirical
unknowns on V2-Lite shape.

## What code is missing (Pattern 9)

| File | Lines (est) | Purpose |
|---|---|---|
| `crates/dismantle-core/shaders/mla_tree_decode_kernel.metal` | ~250 | Tree-batched MLA verifier |
| `crates/dismantle-core/shaders/moe_tree_batched.metal` | ~280 | Tree-batched MoE expert dispatch |
| `crates/dismantle-core/src/model/deepseek_v2.rs` | ~300 | Tree decode loop + branch verifier |
| `crates/dismantle-core/src/speculate/eagle4_head.rs` | ~150 | Tree-output draft head (top-k per chain step) |
| `crates/dismantle-core/src/metal/decode_arena.rs` | ~80 | Tree-shape arena buffers |
| `eagle4/eagle4.py` | ~200 | Tree-aware training (multi-token target per chain step) |
| `crates/dismantle-core/tests/tree_decode_parity.rs` | ~150 | Bit-identical at branch=1 (= chain decode) |

**Net:** ~1410 lines across 7 files. Substantial new work; not pure
compute.

## Why this lever

Chain decode at K=4 produces ONE draft sequence per verifier forward.
With chain accept p per step, expected accepted tokens = `1 + p + p² + p³ + p⁴`.

Tree decode produces MULTIPLE draft branches per verifier forward.
With B branches per step, the verifier checks B^K candidates and
accepts the longest valid path. Expected accepted tokens scales
roughly as `1 + Bp + B²p² + ...` — exponentially better when p is
moderate.

At p=0.6, K=4:
- Chain B=1: 1 + 0.6 + 0.36 + 0.216 + 0.130 = **2.31× tokens/verify**
- Tree B=2: 1 + 1.2 + 0.72 + 0.43 = ~3.4× (but more verifier work)
- Tree B=4: 1 + 2.4 + 1.44 + ... = ~5× (much more verifier work)

The trade-off is verifier compute per token. Tree decode at branch=2
typically nets ~50% more accepted tokens than chain at same verifier
overhead. Published results (Medusa, EAGLE-2): 1.5-1.8× on Llama
3-class models. V2-Lite's MoE makes branch overhead steeper because
MoE expert dispatch grows with branch count.

## What's already shipped

- Parallel-K verify (Branch 1): the verifier can batch K queries per
  forward. The K-batched kernels exist for MLA + Q4_K_M + MoE.
- Branch 2 step 4 (parallel-k-union): MoE expert union amortization
  across K queries.

Tree decode generalizes parallel-K from "K sequential candidates" to
"K * B parallel candidates with branch verification."

## Concrete plan

This is the longest phase plan. Breaking into sub-milestones.

### Milestone E.1 — tree-shape draft head (3-4 days)

Change Eagle4Head to emit `B` candidates per chain step instead of 1
(argmax). At each chain step k, the head's `token_logits` (vocab-sized)
gives top-B candidates; each one is rolled forward separately for
step k+1. Output is a tree of shape `B^K` candidate sequences.

Files:
- `eagle4/eagle4.py`: add `--branch-factor` CLI flag; chain training
  uses top-B targets at each step
- `crates/dismantle-core/src/speculate/eagle4_head.rs`: new
  `propose_tree(...)` method that returns B^K candidate sequences

Parity: at B=1 must be bit-identical to current chain decode.

### Milestone E.2 — tree-batched MLA kernel (3-4 days)

Generalize `mla_decode_kernel_fc_kbatch` to handle K-batch with tree
structure: each of K positions has B^k candidates. The kernel needs
to know the tree structure to share computation across siblings.

`crates/dismantle-core/shaders/mla_tree_decode_kernel.metal`:
- Input: tree structure metadata (parent pointers, branch ids)
- Output: per-leaf attention outputs
- Geometry: same as MLA kbatch but K → K*B

Parity: at B=1 reduces to the existing kbatch kernel; bit-equivalent.

### Milestone E.3 — tree-batched MoE (2-3 days)

Generalize `moe_routed_union_pipeline_tcb` from K queries to K*B
queries with shared routing across siblings.

`crates/dismantle-core/shaders/moe_tree_batched.metal`:
- Reuses the union pipeline pattern
- Tree-aware route table sharing for sibling positions

### Milestone E.4 — tree-decode verify loop (2 days)

`crates/dismantle-core/src/model/deepseek_v2.rs`:
- New `forward_tokens_tree(drafts: TreeDrafts, ...)` method
- Wraps the K*B verifier forward + branch acceptance logic
- Returns the longest valid prefix (leaf path through the tree)

### Milestone E.5 — parity + bench (1-2 days)

`tests/tree_decode_parity.rs`:
- At B=1, K=4: bit-identical to existing chain decode
- At B=2, K=4: token IDs match a reference implementation
- Test at multiple seed prompts (12+)

Clean-window bench: tree vs chain at matched K, multiple prompts.

## Risks + mitigations

1. **Tree overhead exceeds branch gain on V2-Lite MoE shape.**
   *Mitigation:* benchmark per-branch carefully; B=2 first. If <1.2×
   over chain, stop. Don't push B=4 without B=2 winning.

2. **Implementation complexity introduces parity bugs.**
   *Mitigation:* B=1 parity gate (must equal chain decode). If B=1
   passes but B=2 produces wrong tokens, it's the tree-batched kernel.

3. **Tree training (E.1) is novel — head may not learn branch-quality.**
   *Mitigation:* train head on top-B-vs-target loss; if B=2 accept
   rate is similar to chain accept, ship. If much worse, training
   pipeline needs more iteration.

4. **Memory pressure** — tree drafts cost K*B activation buffers vs
   chain's K. At K=4, B=2: 8 buffers per layer. Verify within decode
   arena budget.

## Acceleration patterns applied

- Pattern 1: parity test at B=1 = mid-flight signal during development
- Pattern 2: B=1 parity / synthetic kernel parity / clean bench levels
- Pattern 4: branch=2 first; don't curriculum-ramp B
- Pattern 5: tree structure is architectural, not hyperparameter; if
  the chain accept itself is the limit, fix that first (already
  done in L8)
- Pattern 9: 1410 lines across 7 files — substantial engineering

## Acceptance criteria

- B=1, K=4 tree decode bit-identical to chain decode at K=4
- B=2, K=4 chain accept ≥ 60% (vs chain's projected ~45-55%)
- Clean bench: ≥1.3× dec_tps over chain decode at matched K
- All existing parity gates still pass

## Decision point AT each milestone

E.1 ships if head training at B=1 produces same chain accept as
current. (No regression risk.)
E.2 ships if B=1 tree path is bit-identical to chain.
E.3 ships if B=1 MoE tree path is bit-identical.
E.4 ships if full B=2 produces sensible output.
E.5 is the GO/NO-GO for tree decode as a whole.

Each milestone is independently shippable — don't merge E.4+E.5 in
the same commit; the bench numbers decide ship.

## Next-session quickstart

```
# Phase E is big; recommend starting with E.1 (head training only):
# 1. Read this + acceleration_patterns.md
# 2. Patch eagle4.py for --branch-factor
# 3. Train one short run (1 epoch) at branch=2; verify training works
# 4. Eval branch=2 head's per-step top-B accept rate; if reasonable,
#    proceed to E.2
```

Phase E is also the right time to revisit the L8 efficiency patterns
— vector gate likely needs to extend to per-branch as well (each
branch's draft_hidden may need different gate values to differentiate).
