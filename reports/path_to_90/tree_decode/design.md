# Tree decoding for EAGLE-3 spec-decode (design)

**Status:** design only; head + engine implementations sketched.
**Goal:** boost spec-decode token-yield by replacing linear K-token chains
with branching trees that explore multiple continuation hypotheses per
target verify call.
**Estimated impact:** **1.5-2× over linear K=4 spec-decode** per EAGLE-3
paper §4.2; brings effective acceptance from 70% (linear) to ~85%
(tree-effective) on same hardware, same trained head.
**Estimated implementation:** 3-4 weeks elapsed AFTER Path B + first
trained head land.

## Why trees beat chains

Linear K=4 spec-decode propose one chain of 4 tokens. If position 2 is
wrong, you lose positions 3 and 4 too — even if those would have been
right under a different position-2 choice. Acceptance drops geometrically
with K.

Trees propose MULTIPLE candidates at each level. At branching factor
B=2, depth D=4, you have 1 + 2 + 4 + 8 + 16 = 31 candidate positions
verified in ONE target forward (vs 4 for linear). Acceptance becomes
"longest matching path through the tree" — much more forgiving of
single-position mispredictions.

Empirically (EAGLE-3 paper Table 4 on Vicuna-7B):
- Linear K=4:   70% accept → 2.4 tokens/verify
- Tree (32 nodes): same head → **3.8 tokens/verify**
- Tree (64 nodes): **4.5 tokens/verify**

That's a 1.6-1.9× boost on top of whatever linear spec-decode achieves,
**using the same trained head with no retraining required.**

## Tree topology

EAGLE-3's published tree topology is hand-tuned (greedy maximum of
expected tokens-per-verify on a held-out set). A reasonable starting
tree for K~24:

```
                     root
        ┌─────────────┼─────────────┐
       t1a           t1b           t1c       (top-3 at depth 1)
     ┌─┴─┐         ┌─┴─┐         ┌─┴─┐
    t2a t2b       t2a t2b       t2a t2b      (top-2 at depth 2)
     │   │         │   │         │   │
   t3a  t3a      t3a  t3a      t3a  t3a      (top-1 at depth 3+)
     │   │         │   │         │   │
   t4a  t4a      t4a  t4a      t4a  t4a
```

That's 3 + 6 + 6 + 6 = 21 nodes. The (B_d, depth) profile is a tunable
trained on a held-out slice to maximize tokens-per-verify.

The tree shape is FIXED at inference time per topology. Picking the
topology is an offline calibration step (~10 hr per topology to
sweep on the held-out slice).

## Forward (head emits a tree)

At each spec-decode step, the head produces a tree of (K, vocab) logits
rather than a single (K, vocab) chain. Two ways to do this:

**Option A: K independent head forwards** — one per tree node, each
seeded from its parent's predicted hidden state.
- Cost: K × head_forward (~K ms on M3 Pro for the small head)
- Simple to implement
- ~21 × 1 ms = 21 ms — non-trivial fraction of one target forward (~40 ms),
  so the draft-head's "essentially free" assumption gets shaky at large K

**Option B: Batched tree forward in one head call** — the head's
attention extends to the full tree-attention mask (block-diagonal causal
between parent-child paths), one forward emits all K logits in parallel.
- Cost: ~2-3 × single-token head_forward (sub-linear in K)
- More implementation work but better at large K
- Mirrors how the target verify pass works (parallel-K via Path B kernels)

**Decision:** ship Option A first (~2-3 days), upgrade to Option B
once tree topology is fixed (~1 week). Option A is enough to validate
the win exists at K~21.

### Head-side changes (Option A)

Adds a method to `EagleHead` that runs N child-forwards from a single
parent hidden state, predicting top-B tokens at the next position.

```python
class EagleHead(nn.Module):
    ...
    def propose_tree(
        self,
        prev_token: int,
        hidden: mx.array,           # (H,) target hidden at the committed position
        topology: list[int],        # e.g. [3, 2, 1, 1] = (depth → branching)
    ) -> dict[str, list]:
        """Returns:
          - node_tokens : list[int]   length = sum_d prod(topology[:d+1])
          - node_parents: list[int]   parent node-index for each (root parent = -1)
          - node_depths : list[int]   depth in tree (root depth = 0)
          - node_paths  : list[list[int]]  full path of tokens from root for each node
        """
```

Implementation walks the tree breadth-first; at each node, runs the head
with the parent's predicted hidden + last token, takes top-B from the
output, spawns B children. Each child caches its parent's predicted
hidden state (= head's draft_hidden output at that position).

## Tree-attention verify (engine side)

The target model verifies the WHOLE tree in one forward. Tokens at
different depths attend through the tree-structured mask: each node
attends to its ancestors + itself, NOT to its siblings or cousins.

For a tree of N nodes, the attention mask is (N, N) with:
- mask[i, j] = 0 if node j is an ancestor of node i (or i==j)
- mask[i, j] = -inf otherwise

Combined with Path B's parallel-K MLA kernel, this is a small generalization:
the K positions are now arranged in a tree structure rather than a linear
chain. The mask changes; the dispatch graph is the same shape.

Engine changes:

```rust
// crates/dismantle-core/src/speculate/tree.rs (new)
pub struct TreeProposal {
    pub node_tokens: Vec<u32>,
    pub node_parents: Vec<i32>,   // -1 for root
    pub node_paths: Vec<Vec<u32>>,
}

pub fn tree_attention_mask(parents: &[i32]) -> Vec<Vec<bool>> { ... }
pub fn longest_matching_path(
    proposal: &TreeProposal,
    verifier_argmax_per_node: &[u32],
) -> Vec<u32> { ... }

// In speculate/draft_head.rs (extend existing DraftHead trait)
pub trait DraftHead: Send + Sync {
    fn propose(&mut self, prev_token: u32, hidden: &[f32], k: usize) -> Result<Vec<u32>>;
    fn propose_tree(
        &mut self,
        prev_token: u32,
        hidden: &[f32],
        topology: &[usize],
    ) -> Result<TreeProposal> {
        // Default: degenerate to linear via single-branch tree
        Err(crate::Error::Unimplemented("propose_tree"))
    }
    ...
}

// crates/dismantle-core/src/kernels/parallel_k.rs — extend
pub fn mla_decode_kernel_fc_kbatch_masked(
    ...
    attention_mask: &[f32],  // (K, K) tree mask
);
```

The MLA kernel gets a per-(K, K) mask argument. For linear spec-decode
the mask is causal; for tree spec-decode it's the tree-structured mask
above. Same kernel surface, different mask data.

## Acceptance algorithm

After verify, the target produces argmax-per-node logits at all N tree
nodes. The accept algorithm walks the tree and finds the longest path
where every node's argmax matches the proposed token AT THAT NODE.

Pseudocode:
```python
def longest_matching_path(proposal, verifier_argmax_per_node):
    # tree_paths[i] = path from root to node i (token list)
    best = []
    for node_idx, path in enumerate(proposal.node_paths):
        match_len = 0
        for k, expected_tok in enumerate(path):
            actual_argmax = verifier_argmax_per_node[ancestor_at_depth_k]
            if actual_argmax == expected_tok:
                match_len += 1
            else:
                break
        if match_len > len(best):
            best = path[:match_len]
    return best
```

Plus the standard +1 (the target's argmax at the position AFTER the
last accepted node is a bonus token, same as linear spec-decode).

## Topology calibration

The topology `[B_0, B_1, ..., B_D]` (branching per depth) is hand-tuned
per model/dataset. We add a calibration script that runs the trained
head on a held-out 500-prompt slice with various topologies, measures
mean tokens-per-verify, picks the topology that maximizes it within a
node-count budget.

```
tools/training/mlx_eagle/calibrate_tree.py
  Args: --ckpt, --held-out-shard, --node-budgets [16, 32, 64, 128]
  Output: reports/path_to_90/tree_decode/topology.json
    {"topology": [3, 2, 2, 1, 1], "tokens_per_verify": 4.7, "n_nodes": 21}
```

Cost: ~10 hr wall on M3 Pro per topology sweep (head forwards are cheap;
the search space is small).

## Where the wins come from numerically

Linear K=4 at p=0.7: expected tokens = 1 + p + p² + p³ = 2.53
Tree-21 at p=0.7 (per published numbers, ~similar acceptance dynamics):
  expected tokens = ~3.8

Improvement: 3.8 / 2.53 = **1.5×** on token-yield per verify call.

Combined with Path B (verify cost ~1.5× single-forward):
- Linear K=4 + Path B: 2.53 / 1.5 = 1.69× speedup
- Tree-21 + Path B:   3.8 / 1.5 = **2.53× speedup**

Combined with engine improvements (asymptote ~28-50 dec_tps for engine
work), the full stack target becomes:

| Component | dec_tps |
|---|---|
| Engine + KV-quant ceiling (no spec) | ~50-55 |
| + Path B linear K=4 (70% accept head) | ~85 |
| + tree-21 instead of linear | ~120 |
| + Path B parallel-K verify of full tree | (compound) |

## Implementation order

1. **Head-side `propose_tree`** (~3-5 days) — Option A naive implementation.
2. **Engine `TreeProposal` + tree-attention mask + accept algo** (~3 days).
3. **Path B MLA kernel mask argument** (~2-3 days) — extends Path B
   kernel signature to accept (K, K) mask.
4. **Topology calibration script + first sweep** (~1 day script + 1 day sweep).
5. **End-to-end integration test** (~2-3 days).
6. **Option B batched tree forward in head** (~1 week) — bigger optimization
   once Option A validates the win.

## Risks

| Risk | Mitigation |
|---|---|
| Topology calibration overfits to held-out slice → poor in-prod acceptance | Cross-validate on 3 disjoint held-out slices; use median, not max |
| Tree mask doesn't compose with Path B's TG memory budget | TG memory grows ~linearly with K; tree-21 vs linear-4 = 5× more. May need tile-size reduction or different kernel for large trees |
| Tree-attention mask construction overhead dominates at small K | Cache common topologies; tree-N <= 32 uses precomputed masks |
| Head's "predicted hidden state" used to seed children diverges from target's actual hidden at depth d | Bounded by the head's own quality — same factor that drives linear acceptance |

## Dependencies (must land first)

- Path B parallel-K MLA kernel (otherwise tree verify is K × single-forward,
  same as linear regression)
- A trained head with ≥50% linear acceptance (validates the head is at all
  useful before paying for tree implementation)
- spec_decode_stub.py's K-parallel verify estimate already in shape — tree
  is a generalization of K-parallel

## What this design does NOT do

- Doesn't change head architecture — same 60M-param EAGLE-3
- Doesn't require re-training — same checkpoint works for linear or tree
- Doesn't change target model — only verify mask and dispatch shape change
- Doesn't address speculative sampling (stochastic — needed for top-p
  generation). Greedy-only here; speculative sampling is a separate
  research project per EAGLE-3 paper §3.4

## Files (to be created)

```
crates/dismantle-core/src/speculate/tree.rs                       (new)
crates/dismantle-core/src/speculate/draft_head.rs                 (modified — propose_tree default)
crates/dismantle-core/shaders/parallel_k_attn.metal               (modified — mask arg)
tools/training/mlx_eagle/model.py                                 (modified — propose_tree method)
tools/training/mlx_eagle/calibrate_tree.py                        (new)
reports/path_to_90/tree_decode/topology.json                      (new, after calibration)
reports/path_to_90/tree_decode/close.md                           (new, after impl)
```

## Headline target after all of this lands

With everything composed:
- 500K-trained EAGLE-3 head → ~78% linear acceptance
- + Tree-21 topology → ~88% effective acceptance
- + Path B parallel-K verify (tree-mask variant)
- + Existing engine + KV-quant work asymptoting

Target: **~120-150 dec_tps on M3 Pro Q4_K_M**. llama.cpp's basic spec-decode
on same hardware: ~70-80 dec_tps. Margin: **~1.5-2× ahead of llama.cpp.**

Engineering effort to get there from today: ~2-3 months elapsed, mostly
background compute (capture + train) + ~150 hours of focused engineering
across Path B + tree decoding + C3 wire-up.

---

# DySpec dynamic tree decode (path-to-90 step 20)

**Status:** design only. Supersedes the fixed-topology Sequoia
approach above for the EAGLE-4 + MoE regime.
**Why dynamic, not fixed:** the Qwen3.6-A3B llama.cpp benchmark
(April 2026) showed zero spec-decode configurations achieve net
speedup on Qwen3-MoE despite 100 % draft acceptance — because fixed-
topology trees activate too many experts per verify pass. The expert-
union grows roughly linearly with tree size, eating the bandwidth
that spec decode was supposed to save. EAGLE-4's `calib_logit` gives
us a per-position confidence signal that fixed trees ignore.

DySpec (arXiv 2403.12428, NeurIPS 2024) replaces the static tree
shape with a per-token confidence-driven shape function. On a high-
confidence position, branch wider (high acceptance probability — more
candidates pay off). On a low-confidence position, narrow or skip
spec decode entirely (low acceptance — extra candidates are wasted
bandwidth). For V2-Lite MoE this is the difference between a working
spec-decode regime and a regression.

## (a) Tree-shape function f(calib) → (depth, width)

`calib` here is `eagle4_head.propose(...).calib`, post-sigmoid. The
shape function:

```rust
fn tree_shape(calib: f32) -> TreeShape {
    if calib >= 0.85 {
        TreeShape { depth: 5, width: [1, 4, 4, 2, 2] }   // wide, deep
    } else if calib >= 0.65 {
        TreeShape { depth: 4, width: [1, 3, 3, 2] }      // moderate
    } else if calib >= 0.50 {
        TreeShape { depth: 3, width: [1, 2, 2] }         // narrow
    } else {
        TreeShape { depth: 1, width: [1] }               // skip spec
    }                                                     // (autoregressive)
}
```

Total node counts: 35 nodes (wide), 19 (moderate), 7 (narrow), 1
(off). Expert-union growth (V2-Lite, 6 routed per token):
- 35-node tree: ~28 unique experts/layer (worst case, of 64)
- 19-node tree: ~22 experts/layer
- 7-node tree:  ~12 experts/layer
- 1-node:        6 experts/layer (baseline routed top-6)

Thresholds 0.85 / 0.65 / 0.50 are starting points; the right values
land empirically from a sweep against Spec-Bench MT-Bench. Until
then, default to MODERATE (`calib ≥ 0.65 → 19-node`) and treat the
shape function as a profile parameter (not hard-coded in the kernel).

## (b) Tree attention mask construction

A tree with N total nodes generates an N×N attention mask where
`mask[i,j] = 0` iff j is an ancestor of i in the tree (or i == j),
else `-1e9`. Each row corresponds to one verify-position; the
softmax-masked attention only attends to its ancestors + self.

Concrete for the 7-node narrow tree:

```
positions:  [root, c1, c2, c1a, c1b, c2a, c2b]
ancestors:
    root: ∅          → mask row: [0, -, -, -, -, -, -]
    c1:   root       → mask row: [0, 0, -, -, -, -, -]
    c2:   root       → mask row: [0, -, 0, -, -, -, -]
    c1a:  root, c1   → mask row: [0, 0, -, 0, -, -, -]
    c1b:  root, c1   → mask row: [0, 0, -, -, 0, -, -]
    c2a:  root, c2   → mask row: [0, -, 0, -, -, 0, -]
    c2b:  root, c2   → mask row: [0, -, 0, -, -, -, 0]
```

`-` here = `-1e9`. Encoded as a packed `(N, N)` f32 buffer passed
into `forward_tokens_batched_tree(tokens, positions, mask)`. The
existing Wedge C path's MLA decode needs a tree-mask variant; this
is exactly what Path B's `mla_decode_kernel_fc_kbatch` becomes when
extended with an attention mask buffer (one extra input). The
masked-verify kernel from step 12 (the MoE variant) gets a similar
extension for the FFN side — though MoE's per-position routing is
already independent across positions, so the tree-mask only matters
for attention.

## (c) Verify-side accept / reject across tree branches

Linear K-spec accept rule: longest matching greedy prefix. Tree
version: longest matching ROOT-TO-LEAF path under V2-Lite's argmax
at each position. Implementation:

```rust
fn accept_tree(tree: &Tree, v2_argmaxes: &[u32]) -> Vec<u32> {
    let mut accepted = vec![tree.root.token];
    let mut node = &tree.root;
    while let Some(children) = node.children() {
        let v2_token = v2_argmaxes[node.position];
        match children.iter().find(|c| c.token == v2_token) {
            Some(matched_child) => {
                accepted.push(matched_child.token);
                node = matched_child;
            }
            None => break,  // V2-Lite disagrees with all children →
                            // emit v2_token as correction and stop.
        }
    }
    accepted
}
```

Bit-identicality preserved: every emitted token is either a tree
node V2 accepted, or V2's own correction at the first divergence.
The longest-matching-path is a generalization of the longest-
matching-prefix rule; same correctness argument applies (per Sequoia
§3.1).

KV rollback after accept: KV slots for the accepted path stay; KV
slots for non-accepted siblings get truncated. With the tree mask
approach, all N positions in the tree get their KV computed in the
single verify forward, but only the longest accepted path's KV is
retained.

## Integration with EAGLE-4 + Path B kernels

EAGLE-4's `propose()` currently returns a flat K-best list at one
position. For trees we need `propose_tree(inputs, shape) -> TreeNodes`
returning a tree structure: at each tree level, propose
`width[level]` candidates per parent. That requires K iterations of
the head's autoregressive draft loop (eagle4 head fed its own
`draft_hidden` as the next position's `h_high` — see eagle4.py's
training-side multi-step-K dance).

Implementation order, INSIDE step 21 once Path B kernels are in
place:

1. Extend `Eagle4Head` with `propose_tree(inputs, shape) -> TreeNodes`.
   Reuses `forward_full`'s plumbing K-times with shape-driven branching.
2. Extend `forward_tokens_batched_for_test` to accept an attention
   mask buffer. Mask-aware Path B kernels do the actual work.
3. Decode loop in `model/deepseek_v2.rs::generate`:
   `SpeculateMode::Eagle4` branch routes to tree decode when
   `calib ≥ 0.50` and a positive shape comes back; else falls back
   to K=1 verify-by-comparison (step 8's current path).
4. Bit-identical regression: same as step 9 but with tree shape
   variations — sequences across shapes must all match Off greedy.

## What's the expected MoE multiplier?

Per deep-research (`eagle4_deep_research.md` § "MoE spec-decode is a
documented minefield"):

- Dense Vicuna trees: 1.4–1.8× over chain spec decode.
- MoE V2-Lite trees: 1.2–1.5× expected (less than dense because
  expert-union still grows with tree size; mitigated but not
  eliminated by DySpec's narrow-tree behavior on low-calib).

Stage 4 measurement target (step 22): 70–95 tok/s. Computed from
Stage 3's 55–75 tok/s × 1.2–1.5 MoE tree multiplier with DySpec
mitigating the worst-case expert-union explosion.

## Dependencies + ordering against the foundation halt

Tree decode (steps 20–22) is gated on:

- CPU `attention()` divergence fix (foundation halt; chip queued
  2026-05-18). Tree-mask MLA decode is meaningless if the mask-less
  MLA decode is already wrong.
- Path B kernel substrate (steps 12–17). The tree-mask kernels are
  Path B kernels with one extra buffer; design them together for
  shared body / shared parity tests.
- Routing recall fine-tune (step 11). DySpec's narrow-tree fallback
  uses calib to decide width; calib accuracy depends on the recall
  fine-tune that also rebalances the head's loss weighting.

If all three land cleanly, tree decode is ~1 week of focused
implementation (per step 21's estimate). If any are blocked, defer
tree decode and pursue parallelizable Stage 5 hardware-path work
instead.

## Concrete file deltas (step 20 design only, no code in this commit)

- `reports/path_to_90/tree_decode/design.md` — extended with this
  DySpec section.
- No new code files. Implementation lands in step 21 as
  `crates/dismantle-core/src/speculate/tree.rs` + the tree-mask
  extensions to Path B kernels in `parallel_k_attn.metal` /
  `parallel_k_moe_masked.metal`.
